# =========================================================================
# beliefMCTS.jl — Phase 6: belief-space MCTS with the chance constraint wired in
# (Phase 5 was the miss-distance baseline; Phase 6 replaces that reward with Pc).
#
# A HAND-ROLLED custom MCTS (architecture doc §4/§6), NOT POMCPOW / any
# POMDPs.jl-ecosystem solver. The whole reason this is hand-rolled is that the
# belief (μ, Σ) must stay visible to the reward at EVERY simulated step — which
# POMCPOW's reward/isterminal interface structurally cannot expose (§6). Here
# the belief is carried on every node and the reward is just code we control.
#
# WHAT IT DOES (architecture §4, one simulated step, steps 1–7):
#   1. Propagate the sampled TRUE state via POMDPs.transition (Phase 0 dynamics,
#      Δv kick on the SC if MANEUVER).
#   2. PREDICT the belief forward one dt step  (Phase 4 `predict`: Σ⁻ = Φ Σ Φᵀ).
#   3. SAMPLE a real observation z of the true next state (Phase 4
#      `sample_observation`, a genuine random draw).
#   4. CORRECT the belief with z (Phase 4 `correct_linear` — the RUNTIME path;
#      correct_brahe is a test-only cross-val oracle, see beliefTracker.jl).
#   5. EVALUATE Pc-AT-TCA from the resulting belief (μ⁺, Σ⁺): propagate both
#      sub-beliefs forward to TCA (mean + the node's ACCUMULATED Σ via brahe's
#      STM, exactly as Phase 4 `predict` does), then call the Phase 1 `chan_pc`
#      on the two ECI states + two ECI covariances + combined HBR. The Σ used is
#      the node's own tracked covariance (NOT a fresh P0) — see the "UNCERTAINTY
#      MODEL" section header for why. The belief stays two independent 6×6
#      sub-beliefs (no 12×12) — which is what chan_pc wants.
#   6. CHECK THE CHANCE CONSTRAINT, right here, EVERY step, at every depth: if
#      Pc > pomdp.pc_threshold, apply a large penalty and/or terminate the branch
#      (`constraint_mode`, below). This is the whole point of the hand-rolled
#      loop — a once-per-node hook (POMCPOW estimate_value) was explicitly
#      rejected (§6); the check is part of the per-step reward we control.
#   7. BACK UP the return as a running average per node/action.
#
# REWARD (Phase 6, replaces the Phase 5 miss-distance placeholder wholesale):
#   step reward = −MCTS_PC_REWARD_WEIGHT · Pc(child)          (lower Pc is better)
#                 − pomdp.maneuver_cost                        (per MANEUVER burn)
#                 − MCTS_PC_VIOLATION_PENALTY  if Pc > threshold  (§4 step 6)
# leaf/cutoff value (§7) = −MCTS_PC_REWARD_WEIGHT · Pc-at-TCA(node)  (+ the same
#   violation penalty if it's over threshold). BOTH the true leaf (reached TCA)
#   and the computational-budget cutoff reuse the SAME Pc-at-TCA computation —
#   Chan already propagates to TCA regardless of "now", so there is no separate
#   heuristic. Accumulated fuel cost is carried through the per-step maneuver
#   costs during backup, not re-added at the leaf.
#
# CONSTRAINT MODE (`constraint_mode`, planner arg — supports the with/without
# ablation Grace asked for):
#   :penalize (default) — add the violation penalty but keep expanding the branch.
#       Softer, less brittle: far from TCA the covariance ellipsoid is huge (Phase
#       3: debris along-track 1σ ~30 km @24h) so Pc is naturally high there, but a
#       future tracking measurement shrinks Σ — a branch that violates early and is
#       driven safe by TCA is a LEGITIMATE (often optimal) wait-and-measure plan.
#       Penalizing (not amputating) lets the backup/UCB math discourage sustained
#       / at-TCA violations without deleting recoverable branches.
#   :terminate — penalize AND mark the branch terminal (search stops past it). The
#       aggressive arm: this CAN prune away good wait-then-measure branches (it
#       reads early-horizon uncertainty as risk), biasing toward over-maneuvering.
#       Kept for the ablation, not the default.
#   :off — no penalty at all (the Phase-5-style no-constraint baseline arm).
#
# OBSERVATION HANDLING — DOUBLE PROGRESSIVE WIDENING (Grace's call):
# each simulation draws its own sampled true state + observation z; a node
# accumulates observation-children across visits under a widening rule, and the
# backup averages over them. This is the standard handling for stochastic /
# continuous observations. We BORROW POMCPOW's widening ALGORITHM (not its
# solver): add a new observation-child while  n_children ≤ k_o · n^α_o , else
# reuse an existing child at random (POMCPOW solver2.jl line 74). Only the two
# discrete actions {WAIT, MANEUVER} exist, so we enumerate actions (UCB over
# both) rather than widening on actions.
#
# UCB action selection (POMCPOW criteria.jl MaxUCB): pick the action maximizing
#   Q(a) + c · √( ln(N_node) / n_a ) ,   unvisited actions taken first.
#
# CONSTANTS (all in CONSTANTS.md): the widening k_o / α_o, simulation budget
# (tree_queries), and max_depth keep POMCPOW's published defaults. The UCB `c`,
# the Pc reward weight, and the violation penalty are NEW Phase-6 tuning knobs
# scaled to the Pc reward magnitude (Pc is O(1e-5), whereas Phase 5's
# miss-distance reward was O(10–10³), so the Phase 5 c=1 is wildly off). All are
# flagged TODO: needs source — the principled tuning is the deferred efficiency
# pass, but sane values are needed here for the search to behave.
#
# dt: swappable (defaults pomdp.dt) — NEVER hard-coded (Phase 3 convention).
#
# Σ(τ) TABLE (optional, on the planner): passing a `build_covariance_table`
# result lets step 5 look Σ up by time-remaining instead of re-propagating it.
# Phase 4 proved predict-Σ == table-Σ to <1e-9, so the lookup is exact at grid
# τ. Valid ONLY under noiseless maneuvers (§5/§8) — fine now, breaks at Phase 8.
# The mean state is always propagated directly (the table stores only Σ).
# =========================================================================

using LinearAlgebra
using Random

# --- search tuning constants (see CONSTANTS.md) --------------------------------
# Widening / budget knobs keep POMCPOW's published defaults (Phase 5).
const MCTS_K_OBS        = 10.0     # POMCPOW k_observation default
const MCTS_ALPHA_OBS    = 0.5      # POMCPOW alpha_observation default
const MCTS_TREE_QUERIES = 1000     # POMCPOW tree_queries default (sim budget)
const MCTS_MAX_DEPTH    = 24       # 24-hr window / 1 decision-per-step cap

# --- Phase 6 Pc-reward tuning constants (NEW; see CONSTANTS.md) ----------------
# The reward is now Pc-based (O(1e-5)), NOT miss-distance (O(10–10³)). These are
# scaled to that magnitude so exploration and the constraint penalty are on
# comparable footing. TODO: needs source (tune in the deferred efficiency pass).
# Scales chosen so the three reward terms are commensurate at the operating
# point (Pc threshold = 1e-5, maneuver_cost default = 10):
#   • PC_REWARD_WEIGHT · threshold = 1e6 · 1e-5 = 10  → an at-threshold Pc costs
#     ~one maneuver, so the shaping gradient is on the same scale as the fuel it
#     would spend to reduce it (not swamped by, nor swamping, the burn cost).
#   • VIOLATION_PENALTY = 100 ≫ maneuver_cost (10): crossing the threshold is
#     always worth a burn to avoid, so the constraint actually binds.
#   • UCB_C = 10: exploration bonus commensurate with the O(10–100) reward spread
#     (Phase 5's c=1 assumed O(1) rewards; the miss-distance reward was O(10–10³),
#     so this is the Pc-reward re-scaling the Phase 5 note flagged).
const MCTS_UCB_C               = 10.0     # exploration constant, rescaled for the Pc reward
const MCTS_PC_REWARD_WEIGHT    = 1.0e6    # reward = −weight·Pc (at-threshold Pc ≈ one maneuver)
const MCTS_PC_VIOLATION_PENALTY = 100.0   # penalty when Pc > threshold (≫ maneuver_cost)
const MCTS_CONSTRAINT_MODE     = :penalize   # :penalize | :terminate | :off

# Σ-propagation mode for the Pc eval (efficiency pass, 2026-07-23):
#   :fast  — precompute the branch-invariant DEBRIS Σ-at-TCA per depth ONCE per
#            plan, look it up per node; propagate only the debris MEAN + the full
#            satellite belief per node. Pc-EXACT (debris Σ is bitwise-invariant),
#            ~2× faster. VALID ONLY under noiseless maneuvers (breaks at Phase 8).
#   :exact — propagate every node's full belief (mean+Σ, both objects) to TCA.
#            The correctness oracle; the ONLY valid mode once Phase 8 adds
#            maneuver noise. See node_pc_at_tca / the FAST Σ PATH section header.
const MCTS_SIGMA_MODE = :fast

# =========================================================================
# Tree node.  Reuses Phase 4's `Belief` (two 6×6 sub-beliefs + time-remaining)
# as the belief — NOT a new (μ,Σ) container.  Carries the sampled true CAState
# this node was reached with, MCTS visit/value bookkeeping, and children.
#
# Children are stored as: for each action a, a list of observation-children
# (each itself a BeliefNode reached by predict→sample z→correct with that a).
# This is the observation progressive-widening structure.
# =========================================================================
mutable struct BeliefNode
    belief::Belief                 # tracked (μ,Σ) at this node (Phase 4 Belief)
    s_true::CAState                # sampled true state at this node
    N::Int                         # total visits to this node
    Na::Dict{CAAction,Int}         # per-action visit counts
    Qa::Dict{CAAction,Float64}     # per-action running-average value
    children::Dict{CAAction,Vector{BeliefNode}}   # obs-children per action
    is_terminal::Bool
    # --- asymmetric measurement cadence (TODOS "measurement realism") ---------
    # Seconds elapsed since each object last got a correction. A child expanded
    # from this node predicts one dt step, adding dt to each timer; when a timer
    # crosses the object's cadence a correction fires (shrinks Σ) and the timer
    # resets. sat (GPS) and debris (TLE) run on independent schedules.
    since_sc::Float64
    since_debris::Float64
    # --- Phase 6 Pc instrumentation (for the constraint + the ablation) -------
    pc::Float64                    # Pc-at-TCA from this node's belief (NaN = not yet computed)
    violated::Bool                 # did this node's Pc exceed pomdp.pc_threshold?
end

function BeliefNode(belief::Belief, s_true::CAState, is_terminal::Bool;
                    since_sc::Real = 0.0, since_debris::Real = 0.0)
    return BeliefNode(belief, s_true, 0,
                      Dict{CAAction,Int}(), Dict{CAAction,Float64}(),
                      Dict{CAAction,Vector{BeliefNode}}(), is_terminal,
                      Float64(since_sc), Float64(since_debris), NaN, false)
end

# =========================================================================
# Pc-AT-TCA from a node's belief (architecture §4 step 5, §7).
#
# UNCERTAINTY MODEL (corrected 2026-07-22 with Grace — the earlier "Option 2"
# fresh-P0 approach was WRONG and is reverted): a node's Pc uses that node's OWN
# ACCUMULATED belief Σ — the covariance the Kalman predict/correct cycle actually
# produced getting to this node (predict-grown, measurement-shrunk) — propagated
# the rest of the way to TCA:
#
#     Σ_at_TCA = Φ(now → TCA) · Σ_belief(now) · Φ(now → TCA)ᵀ
#
# where Σ_belief(now) = node.belief.{sc,debris}.Σ. This is "Pc at TCA if we stop
# measuring now and coast" — a conservative, physically-honest risk number.
#
# WHY NOT a fresh P0 (the reverted approach): fresh-P0 assumed a measurement each
# step re-anchors Σ to P0. But `correct_linear` does Σ⁺ = (I−K)Σ⁻ with
# K = Σ⁻(Σ⁻+R)⁻¹, and R ≳ P0 in magnitude (σ_debris²=1e4 vs P0_debris pos 2500;
# σ_sc²=100 = P0_sc pos 100), so one measurement only PARTIALLY shrinks Σ — it
# never resets to P0. Fresh-P0 simply IGNORES the tracked belief and substitutes
# a different covariance, so its Pc is wrong — in EITHER direction, depending on
# geometry: at a deep node it regrows the full P0 over the remaining coast while
# the real belief Σ has been measurement-shrunk AND has less time to grow, so
# fresh-P0 there OVERstates Pc by many orders of magnitude (measured: 2.1e-3 vs
# the accumulated-Σ 6e-20 at a depth-1 node). The point is not a fixed bias sign
# — it is that only the accumulated Σ is the covariance the planner actually holds.
#
# WHY THE ACCUMULATED Σ IS THE RIGHT / SUFFICIENT THING: the MCTS rollout already
# does predict → sample z → correct at every expanded step (see expand_child), so
# a node's belief Σ genuinely accumulates the measurements taken along its path.
# Rollouts run full-depth (max_depth = nsteps), so a true leaf AT TCA already
# carries a Σ that absorbed every measurement to TCA — no fictional
# future-measurement model is needed; the rollout IS the measurement sequence.
# Because Σ is currently z-independent and maneuver-independent (no process noise,
# §8), Σ_at_TCA is ~identical across branches at a given depth, so Pc differs
# across branches essentially through the MEAN (i.e. through maneuvers) — the
# constraint acts on "did the burn move the mean far enough," with Σ as a fixed
# backdrop. (Phase 8's maneuver noise will make Σ genuinely branch-dependent.)
#
# The MEAN is propagated to TCA under the accurate force model (a maneuver moves
# it). chan_pc combines the two objects' ECI states + ECI covariances and
# projects onto the encounter plane. Two independent 6×6 sub-beliefs — no 12×12.
#
# COST: one brahe propagation per object per Pc eval (mean+Σ in a single
# propagate, 2 per node); ~156 ms of the ~171 ms/node is the Σ propagation
# (measured 2026-07-22). A propagate-once-and-interpolate speedup is captured for
# the deferred efficiency pass (TODOS Phase-5 extensions) — valid because Σ_at_TCA
# is branch-invariant today; breaks at Phase 8.
#
# CONSERVATIVE-vs-TRACKED TAIL: "coast" (above) vs. assuming measurements continue
# past a truncated rollout only matters once the efficiency pass TRUNCATES
# rollouts (cutoffs don't occur at full depth today) — deferred there as a
# leaf_pc_mode toggle, default :coast.
# =========================================================================

# Below this time-remaining (s) a node is treated as already at TCA (no prop).
const PC_TAU_MATCH_ATOL = 1.0

"""
    _grow_belief_to_tca(pomdp, μ, Σ, objParams, t) -> (μ_tca, Σ_tca_eci)

Propagate one object's belief `(μ, Σ)` from the node epoch (time-remaining `t`)
forward to TCA under the accurate force model: the mean `μ` is carried to TCA,
and the ACCUMULATED belief covariance `Σ` is grown to TCA as Σ(TCA) = Φ Σ Φᵀ,
read back from brahe's covariance_gcrf (ECI). One brahe propagation does both.
(This is the node's real tracked Σ — NOT a fresh P0; see the section header.)
"""
function _grow_belief_to_tca(pomdp::SpacecraftCAPOMDP, μ::AbstractVector,
                             Σ::AbstractMatrix, objParams::AbstractVector, t::Real)
    bh = get_brahe()
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - Float64(t)
    prop, ep0 = eci2orb_brahe(collect(float.(μ)), epoch_to_tuple(epoch_current),
                              objParams, pomdp.forceModel; initial_covariance = Matrix(Σ))
    ep_tca = ep0 + Float64(t)
    prop.propagate_to(ep_tca)
    μ_tca = collect(prop.current_state()[1:6])
    Σ_tca = _sym(collect(prop.covariance_gcrf(ep_tca)))
    return μ_tca, Σ_tca
end

"""
    node_pc_at_tca(pomdp, node) -> Float64

Pc-at-TCA for `node` (architecture §4 step 5 / §7), using the node's ACCUMULATED
belief Σ (see the section header): propagate both sub-beliefs' `(μ, Σ)` to TCA
(mean carried forward, the tracked Σ grown as Φ Σ Φᵀ) and evaluate `chan_pc` on
the two ECI states + two ECI covariances + combined hard-body radius. If the node
is already at/after TCA (τ ≤ PC_TAU_MATCH_ATOL) the belief `(μ, Σ)` is used
directly (no propagation). Returns Pc in [0, 1].
"""
function node_pc_at_tca(pomdp::SpacecraftCAPOMDP, node::BeliefNode)
    b = node.belief
    t = b.t
    hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    if t <= PC_TAU_MATCH_ATOL
        # already at TCA — use the belief (μ, Σ) directly (no growth)
        return chan_pc(b.sc.μ, b.debris.μ, b.sc.Σ, b.debris.Σ, hbr)
    end
    μ_sc_tca, Σ_sc_tca = _grow_belief_to_tca(pomdp, b.sc.μ,     b.sc.Σ,     pomdp.satParams,    t)
    μ_db_tca, Σ_db_tca = _grow_belief_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, t)
    return chan_pc(μ_sc_tca, μ_db_tca, Σ_sc_tca, Σ_db_tca, hbr)
end

# =========================================================================
# FAST Σ PATH — precompute the DEBRIS Σ-at-TCA once per DEPTH, index by depth
# (efficiency pass, 2026-07-23). Cuts the per-node Pc cost by removing the
# DEBRIS STM/covariance propagation — one of the two expensive brahe propagations
# per node. (Measured: a mean+Σ grow is ~11× a mean-only grow; the STM/covariance
# history is essentially the whole cost of a grow. So replacing the debris mean+Σ
# grow with a table lookup + a mean-only grow roughly halves the per-node Pc cost:
# ~680 → ~370 ms/node ≈ 1.8× on this machine.)
#
# SCOPE DECISION (Grace, 2026-07-23): precompute ONLY the debris Σ (which is
# BITWISE branch-invariant — see below), and keep the SATELLITE Σ propagated
# EXACTLY per node. The satellite Σ is only weakly branch-invariant (a maneuver
# perturbs its STM ~1.7–9% across a WAIT vs MANEUVER path at equal depth); rather
# than approximate it, we propagate it exactly, so the fast path's Pc is EXACT
# (no satellite approximation anywhere). The cost we give up is that the surviving
# satellite propagation is still the expensive STM kind — hence ~2×, not the ~11×
# a both-objects precompute would give. This was the deliberate accuracy-over-speed
# call; revisit if the satellite propagation later becomes the bottleneck.
#
# WHY THE DEBRIS PRECOMPUTE IS EXACT (verified 2026-07-23; see the session note +
# TODOS Phase-5 extensions): the accumulated debris belief Σ at a given tree DEPTH
# is a pure function of depth, NOT of the branch —
#   • predict's Σ⁻ = Φ Σ Φᵀ is maneuver-independent (noiseless maneuvers, §8) and
#     the debris never gets a maneuver kick, so its STM reference trajectory (its
#     mean) is identical on every branch between TLE fixes;
#   • the cadence correct's Σ⁺ = (I−K)Σ⁻ is z-independent (the load-bearing
#     linear-Gaussian property, beliefTracker.jl), and the cadence timer schedule
#     (which steps get a fix) is itself deterministic in depth.
# The result is a SAWTOOTH under the ~8 h cadence (Σ grows between TLE fixes, snaps
# at each fix), NOT a smooth Φ P0 Φᵀ curve — so we replay the exact predict/correct
# Σ sequence ONCE along a reference WAIT trajectory and index by depth. Verified:
# WAIT-path vs MANEUVER-path debris Σ-at-TCA agree to 0.0 rel diff at every depth.
#
# WHAT STAYS PER-NODE: both MEANS (a maneuver moves the SC mean; each branch's
# sampled GPS/TLE observations nudge the means, μ⁺ depends on z) AND the full
# satellite belief (mean+Σ). The DEBRIS mean is propagated mean-only (no
# initial_covariance ⇒ no STM history) — the ~11×-cheaper propagation — since its
# Σ now comes from the table.
#
# ⚠️ BREAKS AT PHASE 8 (maneuver execution noise). Process noise Q(a) makes even
# the DEBRIS Σ genuinely action-sequence-dependent — the per-depth table is then
# no longer branch-invariant and this fast path is INVALID. Phase 8 must switch
# back to per-node Σ (sigma_mode = :exact — kept as the oracle for exactly this
# reason). Do NOT silently rely on :fast past Phase 8.
# =========================================================================

"""
    SigmaTCATable

Precomputed DEBRIS Σ-at-TCA per tree depth (efficiency pass). `Σ_db[d+1]` is the
debris ECI covariance-at-TCA for a node at depth `d` (`d = 0` is the root). Built
once per plan by `build_sigma_tca_table`; consumed by `node_pc_at_tca_fast` under
`sigma_mode = :fast`. The spacecraft Σ is NOT tabled — it is propagated exactly
per node (see the section header). Valid only under noiseless maneuvers + the
linear-Gaussian update (breaks at Phase 8). `dt` / `cadence_*` / `correct_at_root`
record the schedule the table was built for so a stale table can be detected.
"""
struct SigmaTCATable
    Σ_db::Vector{Matrix{Float64}}
    dt::Float64
    cadence_sc::Float64
    cadence_debris::Float64
    correct_at_root::Bool
end

"""
    _grow_mean_to_tca(pomdp, μ, objParams, t) -> μ_tca

Propagate one object's belief MEAN from time-remaining `t` to TCA under the
accurate force model, WITHOUT the covariance/STM history — the ~11×-cheaper
propagation (measured: mean-only ~30 ms vs. mean+Σ ~340 ms). Used by the fast Pc
path for the DEBRIS, whose Σ comes from the precomputed per-depth table instead.
"""
function _grow_mean_to_tca(pomdp::SpacecraftCAPOMDP, μ::AbstractVector,
                           objParams::AbstractVector, t::Real)
    bh = get_brahe()
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - Float64(t)
    prop, ep0 = eci2orb_brahe(collect(float.(μ)), epoch_to_tuple(epoch_current),
                              objParams, pomdp.forceModel)   # no initial_covariance ⇒ no STM
    ep_tca = ep0 + Float64(t)
    prop.propagate_to(ep_tca)
    return collect(prop.current_state()[1:6])
end

"""
    _grow_sigma_to_tca(pomdp, μ_ref, Σ, objParams, t) -> Σ_tca_eci

Grow one object's belief covariance `Σ` from time-remaining `t` to TCA along the
reference mean trajectory `μ_ref` (which sets the STM), returning Σ-at-TCA (ECI).
Same brahe path as `_grow_belief_to_tca` but returns only Σ. Used to build the
debris per-depth table along the (branch-invariant) debris WAIT trajectory.
"""
function _grow_sigma_to_tca(pomdp::SpacecraftCAPOMDP, μ_ref::AbstractVector,
                            Σ::AbstractMatrix, objParams::AbstractVector, t::Real)
    bh = get_brahe()
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - Float64(t)
    prop, ep0 = eci2orb_brahe(collect(float.(μ_ref)), epoch_to_tuple(epoch_current),
                              objParams, pomdp.forceModel; initial_covariance = Matrix(Σ))
    ep_tca = ep0 + Float64(t)
    prop.propagate_to(ep_tca)
    return _sym(collect(prop.covariance_gcrf(ep_tca)))
end

"""
    build_sigma_tca_table(pomdp, root; dt=..., cadence_sc=..., cadence_debris=...,
                          max_depth=MCTS_MAX_DEPTH) -> SigmaTCATable

Precompute the DEBRIS Σ-at-TCA per depth for the fast Pc path (efficiency pass).
Replays the belief-Σ evolution the tree produces for the debris — `predict` (Σ
grows one dt step) then the ~8 h cadence `correct` (Σ shrinks when a TLE fix is
due) — along a reference WAIT trajectory from `root`, capturing the accumulated
debris Σ at each depth `d`, then grows each depth's Σ to TCA. Because the debris Σ
update is maneuver-independent (predict) and z-independent (correct), and the
cadence schedule is deterministic in depth, this single WAIT replay reproduces the
per-depth debris Σ EXACTLY on every branch (verified 0.0 rel diff WAIT vs MANEUVER).

The correction uses a zero-innovation observation (`z = μ⁻`): Σ⁺ is z-independent,
so this yields the identical Σ the tree would while keeping the reference mean on
the deterministic WAIT trajectory (which sets the STM). Only the debris sub-belief
Σ is captured; the reference sat mean advances too (needed to keep `predict`
well-formed) but its Σ is discarded (the sat Σ is propagated exactly per node).

Cost: ~`max_depth` STM propagations, ONCE per plan (vs. one debris propagation per
node before).
"""
function build_sigma_tca_table(pomdp::SpacecraftCAPOMDP, root::BeliefNode;
                               dt::Real = pomdp.dt,
                               cadence_sc::Real = pomdp.cadence_sc,
                               cadence_debris::Real = pomdp.cadence_debris,
                               max_depth::Int = MCTS_MAX_DEPTH)
    Σ_db = Matrix{Float64}[]

    b = root.belief
    since_sc     = root.since_sc
    since_debris = root.since_debris

    # depth 0 (root): grow the root debris Σ to TCA along its own mean.
    push!(Σ_db, b.t <= PC_TAU_MATCH_ATOL ? Matrix(b.debris.Σ) :
          _grow_sigma_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, b.t))

    for _ in 1:max_depth
        # PREDICT one dt step along the WAIT spine (a == WAIT: no maneuver kick;
        # Σ grows regardless of action, so WAIT is the right reference).
        b = predict(pomdp, b, WAIT; dt = dt)

        # asymmetric cadence: correct the object(s) whose fix is due this step.
        # Σ⁺ = (I−K)Σ⁻ is z-independent, so a zero-innovation z (= μ⁻) gives the
        # exact tree Σ while keeping the reference mean on the WAIT trajectory.
        since_sc     += Float64(dt)
        since_debris += Float64(dt)
        correct_sc     = since_sc     >= cadence_sc
        correct_debris = since_debris >= cadence_debris
        if correct_sc || correct_debris
            z = vcat(b.sc.μ, b.debris.μ)      # zero-innovation observation
            if correct_sc
                b = correct_linear_sc(pomdp, b, z);     since_sc = 0.0
            end
            if correct_debris
                b = correct_linear_debris(pomdp, b, z); since_debris = 0.0
            end
        end

        push!(Σ_db, b.t <= PC_TAU_MATCH_ATOL ? Matrix(b.debris.Σ) :
              _grow_sigma_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, b.t))
    end

    return SigmaTCATable(Σ_db, Float64(dt), Float64(cadence_sc),
                         Float64(cadence_debris), pomdp.correct_at_root)
end

"""
    node_pc_at_tca_fast(pomdp, node, depth, table) -> Float64

Fast Pc-at-TCA (efficiency pass): propagate the SATELLITE belief exactly (mean+Σ
to TCA — the sat Σ is not tabled), propagate only the DEBRIS mean (mean-only, the
~11×-cheaper propagation) and read the debris Σ-at-TCA for this `depth` from the
precomputed `table`, then call `chan_pc`. Numerically EXACT vs. `node_pc_at_tca`
(`:exact`): the debris Σ is bitwise branch-invariant (table lookup == per-node
propagation) and the satellite is propagated per node, so no approximation is
introduced. `depth` is the node's tree depth (root = 0); deeper than the table was
built for falls back to the exact path.
"""
function node_pc_at_tca_fast(pomdp::SpacecraftCAPOMDP, node::BeliefNode,
                             depth::Int, table::SigmaTCATable)
    b = node.belief
    t = b.t
    hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    idx = depth + 1
    if idx > length(table.Σ_db)
        return node_pc_at_tca(pomdp, node)          # out of table range → exact
    end
    if t <= PC_TAU_MATCH_ATOL
        return chan_pc(b.sc.μ, b.debris.μ, b.sc.Σ, table.Σ_db[idx], hbr)
    end
    μ_sc_tca, Σ_sc_tca = _grow_belief_to_tca(pomdp, b.sc.μ, b.sc.Σ, pomdp.satParams, t)  # sat exact
    μ_db_tca           = _grow_mean_to_tca(pomdp, b.debris.μ, pomdp.debrisParams, t)     # debris mean-only
    return chan_pc(μ_sc_tca, μ_db_tca, Σ_sc_tca, table.Σ_db[idx], hbr)
end

"""
    node_pc(pomdp, node; depth=nothing, table=nothing, sigma_mode=MCTS_SIGMA_MODE) -> Float64

Dispatch the Pc-at-TCA computation for `node` between the fast per-depth path
(`:fast`, requires `depth` and a `table` from `build_sigma_tca_table`) and the
exact per-node path (`:exact`). Falls back to `:exact` if `:fast` is requested but
no `table`/`depth` is supplied (so a caller without a table still works). This is
the single entry point the reward + leaf value call, so switching modes is one
flag on the planner.
"""
function node_pc(pomdp::SpacecraftCAPOMDP, node::BeliefNode;
                 depth::Union{Int,Nothing} = nothing,
                 table::Union{SigmaTCATable,Nothing} = nothing,
                 sigma_mode::Symbol = MCTS_SIGMA_MODE)
    if sigma_mode == :fast && table !== nothing && depth !== nothing
        return node_pc_at_tca_fast(pomdp, node, depth, table)
    end
    return node_pc_at_tca(pomdp, node)
end

# =========================================================================
# Reward — Pc-BASED + the per-step chance constraint (Phase 6; architecture §4
# step 5–6, §7). Replaces the Phase 5 miss-distance placeholder wholesale.
#
#   step reward = −MCTS_PC_REWARD_WEIGHT · Pc(child's belief)   (lower Pc better)
#                 − pomdp.maneuver_cost                          (per MANEUVER)
#                 − MCTS_PC_VIOLATION_PENALTY  if Pc > threshold (constraint, §4.6)
#
# The Pc term + constraint check are evaluated on EVERY simulated step (every
# child), at every depth — not once per node. `constraint_mode` selects how a
# violation is handled (see module header): :penalize (default), :terminate, :off.
# =========================================================================

"""
    miss_distance(s::CAState) -> Float64

Relative position magnitude (m) between spacecraft and debris (RTN position
block). Retained for diagnostics / the end-to-end fixture; NOT the reward metric
anymore (Phase 6 rewards on Pc, not miss distance).
"""
miss_distance(s::CAState) = norm(get_x_rel(s)[1:3])

"""
    step_reward(pomdp, a, child;
                constraint_mode=MCTS_CONSTRAINT_MODE,
                pc_weight=MCTS_PC_REWARD_WEIGHT,
                pc_penalty=MCTS_PC_VIOLATION_PENALTY) -> (r::Float64, pc::Float64, violated::Bool)

Pc-based step reward for reaching `child` under action `a` (architecture §4
steps 5–6). Computes Pc-at-TCA from the child's belief (caching it onto the
node), charges a maneuver cost per burn, rewards lower Pc, and applies the
per-step chance-constraint penalty when Pc exceeds `pomdp.pc_threshold`. Returns
the reward together with the Pc and the violation flag (used to instrument the
tree for the ablation, and — under `:terminate` — to mark the branch terminal).
"""
function step_reward(pomdp::SpacecraftCAPOMDP, a::CAAction, child::BeliefNode;
                     constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                     pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                     pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY,
                     depth::Union{Int,Nothing} = nothing,
                     table::Union{SigmaTCATable,Nothing} = nothing,
                     sigma_mode::Symbol = MCTS_SIGMA_MODE)
    pc = isnan(child.pc) ?
         node_pc(pomdp, child; depth = depth, table = table, sigma_mode = sigma_mode) :
         child.pc
    child.pc = pc
    violated = pc > pomdp.pc_threshold
    child.violated = violated

    r = -pc_weight * pc
    if a == MANEUVER
        r -= pomdp.maneuver_cost
    end
    if violated && constraint_mode != :off
        r -= pc_penalty
    end
    return r, pc, violated
end

# =========================================================================
# UCB action selection over {WAIT, MANEUVER}.
#   score(a) = Q(a) + c · √( ln(N_node) / n_a )
# Any action with n_a == 0 is taken first (infinite exploration bonus).
# =========================================================================

"""
    ucb_select(node, actions; c=MCTS_UCB_C) -> CAAction

Pick the action maximizing the UCB score. Unvisited actions are selected before
any explored action (standard UCB). `c` is the exploration constant.
"""
function ucb_select(node::BeliefNode, actions::AbstractVector{CAAction};
                    c::Real = MCTS_UCB_C)
    best_a = actions[1]
    best_score = -Inf
    logN = log(max(node.N, 1))
    for a in actions
        na = get(node.Na, a, 0)
        if na == 0
            return a                        # explore unvisited action first
        end
        q = get(node.Qa, a, 0.0)
        score = q + c * sqrt(logN / na)
        if score > best_score
            best_score = score
            best_a = a
        end
    end
    return best_a
end

# =========================================================================
# Observation progressive widening (POMCPOW algorithm, borrowed not adopted).
# Add a new observation-child while  n_children ≤ k_o · n_a^α_o , else reuse an
# existing child at random.  `n_a` is the per-action visit count (POMCPOW keys
# widening off the action-node visit count).
# =========================================================================

"""
    should_widen(n_children, na; k=MCTS_K_OBS, α=MCTS_ALPHA_OBS) -> Bool

POMCPOW observation-widening test: `true` (add a new observation-child) while
`n_children ≤ k · na^α`, else `false` (reuse an existing child).
"""
should_widen(n_children::Int, na::Int; k::Real = MCTS_K_OBS, α::Real = MCTS_ALPHA_OBS) =
    n_children <= k * (max(na, 1)^α)

# =========================================================================
# Expansion: one action's predict→sample z→correct + true-state transition.
# Produces a child BeliefNode and the step reward for reaching it.
# =========================================================================

"""
    expand_child(pomdp, node, a, rng; dt=pomdp.dt, cadence_sc=..., cadence_debris=...,
                 constraint_mode=MCTS_CONSTRAINT_MODE, pc_weight=..., pc_penalty=...)
        -> (child::BeliefNode, r::Float64)

Take action `a` from `node`: propagate the sampled true state (Phase 0
`POMDPs.transition`), predict the belief (Phase 4 `predict`), then apply the
ASYMMETRIC measurement cadence (TODOS "measurement realism"). The satellite is an
own-asset GPS fix (~10 m, every `cadence_sc` ≈ 2 h) and the debris is an SSN/TLE
fix (~1 km, every `cadence_debris` ≈ 8 h) — so on any one dt step we correct the
satellite, the debris, both, or NEITHER, depending on how much time has elapsed
since each object was last fixed. Between fixes an object is predict-only and its
Σ GROWS (no measurement). A correction fires for an object when its elapsed-time
timer (parent's `since_*` + this step's dt) reaches its cadence, at which point
the timer resets; otherwise it carries forward. Only the object(s) actually being
measured are corrected (`correct_linear_sc` / `correct_linear_debris`), each
against a freshly sampled observation of the true state.

Then evaluate the Pc-based step reward + per-step chance constraint on the child
(architecture §4 steps 5–6): the child's Pc-at-TCA and violation flag are cached
on it. Under `constraint_mode == :terminate`, a violating child is additionally
marked terminal so the search stops expanding past it. Returns the new
observation-child node and its step reward. `dt` and both cadences are swappable
(default to the POMDP values).
"""
function expand_child(pomdp::SpacecraftCAPOMDP, node::BeliefNode, a::CAAction,
                      rng::AbstractRNG; dt::Real = pomdp.dt,
                      cadence_sc::Real = pomdp.cadence_sc,
                      cadence_debris::Real = pomdp.cadence_debris,
                      constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                      pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                      pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY,
                      depth::Union{Int,Nothing} = nothing,
                      table::Union{SigmaTCATable,Nothing} = nothing,
                      sigma_mode::Symbol = MCTS_SIGMA_MODE)
    # 1. propagate the sampled true state (Phase 0 dynamics)
    sp = rand(rng, POMDPs.transition(pomdp, node.s_true, a))

    # 2. predict the belief forward one dt step (Phase 4). Σ GROWS this step.
    b = predict(pomdp, node.belief, a; dt = dt)

    # 3–4. asymmetric cadence: correct only the object(s) whose fix is due.
    # A fix is due for an object when the time since its last fix has reached its
    # cadence. Timers carry the parent's elapsed time forward by one dt step.
    since_sc     = node.since_sc     + Float64(dt)
    since_debris = node.since_debris + Float64(dt)
    correct_sc     = since_sc     >= cadence_sc
    correct_debris = since_debris >= cadence_debris

    if correct_sc || correct_debris
        z = sample_observation(pomdp, a, sp, rng)   # one genuine draw of the true state
        if correct_sc
            b = correct_linear_sc(pomdp, b, z)
            since_sc = 0.0                           # fix taken → reset the timer
        end
        if correct_debris
            b = correct_linear_debris(pomdp, b, z)
            since_debris = 0.0
        end
    end

    child = BeliefNode(b, sp, isterminal(pomdp, sp);
                       since_sc = since_sc, since_debris = since_debris)

    # 5–6. Pc-at-TCA + per-step chance constraint (§4 step 6). Caches pc/violated.
    r, _, violated = step_reward(pomdp, a, child;
                                 constraint_mode = constraint_mode,
                                 pc_weight = pc_weight, pc_penalty = pc_penalty,
                                 depth = depth, table = table, sigma_mode = sigma_mode)
    if constraint_mode == :terminate && violated
        child.is_terminal = true       # amputate the branch past a violation
    end
    return child, r
end

# =========================================================================
# leaf / cutoff value (architecture §7) — Pc-at-TCA, single consistent metric.
#   True leaf (reached TCA / collision): Pc-at-TCA from the node's belief.
#   Cutoff (depth budget hit before TCA): reuse the SAME Pc-at-TCA computation —
#   Chan already propagates to TCA regardless of "now", so there is no separate
#   heuristic. Accumulated fuel cost is carried through the per-step maneuver
#   costs during backup, not re-added here.
# =========================================================================

"""
    leaf_value(pomdp, node; constraint_mode=..., pc_weight=..., pc_penalty=...)
        -> Float64

Value estimate for a node the rollout does not expand past — either a true leaf
(reached TCA / collision) or a computational-budget cutoff. BOTH use the same
Pc-at-TCA metric (architecture §7): value = −pc_weight·Pc, minus the violation
penalty if Pc exceeds `pomdp.pc_threshold` (unless `constraint_mode == :off`).
Caches the node's Pc / violation flag. No fuel term here — fuel is charged per
step during backup.
"""
function leaf_value(pomdp::SpacecraftCAPOMDP, node::BeliefNode;
                    constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                    pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                    pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY,
                    depth::Union{Int,Nothing} = nothing,
                    table::Union{SigmaTCATable,Nothing} = nothing,
                    sigma_mode::Symbol = MCTS_SIGMA_MODE)
    pc = isnan(node.pc) ?
         node_pc(pomdp, node; depth = depth, table = table, sigma_mode = sigma_mode) :
         node.pc
    node.pc = pc
    violated = pc > pomdp.pc_threshold
    node.violated = violated
    v = -pc_weight * pc
    if violated && constraint_mode != :off
        v -= pc_penalty
    end
    return v
end

# =========================================================================
# One MCTS simulation (recursive): selection → expansion (with widening) →
# recurse → backup as a running average.  Returns the return from this node.
# =========================================================================

"""
    simulate!(pomdp, node, depth, rng; dt=pomdp.dt, c=..., k=..., α=...,
              table=nothing, constraint_mode=..., pc_weight=..., pc_penalty=...) -> Float64

Run one MCTS simulation from `node` (architecture §4 steps 1–7). Selects an
action by UCB, adds or reuses an observation-child under progressive widening,
evaluates the Pc-based step reward + per-step chance constraint (§4 steps 5–6)
on the child, recurses to `depth-1`, and backs up a running-average value.
Returns the (discounted) return seen from this node. Terminal nodes and the
depth==0 cutoff return the Pc-at-TCA `leaf_value` (§7). The `constraint_mode`,
`pc_weight`, `pc_penalty` args are threaded into the reward and the leaf value.
"""
function simulate!(pomdp::SpacecraftCAPOMDP, node::BeliefNode, depth::Int,
                   rng::AbstractRNG; dt::Real = pomdp.dt,
                   cadence_sc::Real = pomdp.cadence_sc,
                   cadence_debris::Real = pomdp.cadence_debris,
                   c::Real = MCTS_UCB_C, k::Real = MCTS_K_OBS, α::Real = MCTS_ALPHA_OBS,
                   constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                   pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                   pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY,
                   node_depth::Int = 0,
                   table::Union{SigmaTCATable,Nothing} = nothing,
                   sigma_mode::Symbol = MCTS_SIGMA_MODE)
    if node.is_terminal || depth <= 0
        return leaf_value(pomdp, node; constraint_mode = constraint_mode,
                          pc_weight = pc_weight, pc_penalty = pc_penalty,
                          depth = node_depth, table = table, sigma_mode = sigma_mode)
    end

    node.N += 1
    acts = POMDPs.actions(pomdp)
    a = ucb_select(node, acts; c = c)

    na = get(node.Na, a, 0)
    kids = get!(node.children, a, BeliefNode[])

    # observation progressive widening (POMCPOW rule). The child sits one tree
    # level deeper than `node`, so its per-depth Σ table index is node_depth + 1.
    child_depth = node_depth + 1
    local child::BeliefNode
    local r::Float64
    if should_widen(length(kids), na; k = k, α = α)
        child, r = expand_child(pomdp, node, a, rng; dt = dt,
                                cadence_sc = cadence_sc, cadence_debris = cadence_debris,
                                constraint_mode = constraint_mode,
                                pc_weight = pc_weight, pc_penalty = pc_penalty,
                                depth = child_depth, table = table, sigma_mode = sigma_mode)
        push!(kids, child)
    else
        child = rand(rng, kids)
        # step reward for the reused edge — the child's Pc is already cached
        # (step_reward reuses it), so this is cheap and consistent with expand.
        r, _, _ = step_reward(pomdp, a, child;
                              constraint_mode = constraint_mode,
                              pc_weight = pc_weight, pc_penalty = pc_penalty,
                              depth = child_depth, table = table, sigma_mode = sigma_mode)
    end

    q = r + POMDPs.discount(pomdp) * simulate!(pomdp, child, depth - 1, rng;
                                               dt = dt, cadence_sc = cadence_sc,
                                               cadence_debris = cadence_debris,
                                               c = c, k = k, α = α,
                                               constraint_mode = constraint_mode,
                                               pc_weight = pc_weight, pc_penalty = pc_penalty,
                                               node_depth = child_depth,
                                               table = table, sigma_mode = sigma_mode)

    # running-average backup on the per-action value
    node.Na[a] = na + 1
    node.Qa[a] = get(node.Qa, a, 0.0) + (q - get(node.Qa, a, 0.0)) / node.Na[a]
    return q
end

# =========================================================================
# Top-level planner: build a fresh tree from a root belief + sampled true
# state, run `n_iterations` simulations, return the best action by value.
# =========================================================================

"""
    MCTSPlanner(pomdp; n_iterations, max_depth, c, k, α, dt,
                constraint_mode, pc_weight, pc_penalty)

Configuration for the Phase-6 chance-constrained belief-space MCTS planner. `dt`
defaults to `pomdp.dt` and is swappable (run both grids to compare). The
measurement cadences `cadence_sc` (~2 h GPS) / `cadence_debris` (~8 h TLE) are
also swappable (default to the POMDP values) and drive the asymmetric per-object
correction schedule in `expand_child` (TODOS "measurement realism"). The search
knobs (`c`, `k`, `α`, `n_iterations`, `max_depth`) and the Pc-reward knobs
(`pc_weight`, `pc_penalty`) default to the CONSTANTS.md values. `constraint_mode`
(`:penalize` | `:terminate` | `:off`) selects how a per-step Pc violation is
handled — see the module header (supports the with/without ablation).
"""
struct MCTSPlanner
    pomdp::SpacecraftCAPOMDP
    n_iterations::Int
    max_depth::Int
    c::Float64
    k::Float64
    α::Float64
    dt::Float64
    cadence_sc::Float64
    cadence_debris::Float64
    constraint_mode::Symbol
    pc_weight::Float64
    pc_penalty::Float64
    sigma_mode::Symbol
end

function MCTSPlanner(pomdp::SpacecraftCAPOMDP;
                     n_iterations::Int = MCTS_TREE_QUERIES,
                     max_depth::Int = MCTS_MAX_DEPTH,
                     c::Real = MCTS_UCB_C,
                     k::Real = MCTS_K_OBS,
                     α::Real = MCTS_ALPHA_OBS,
                     dt::Real = pomdp.dt,
                     cadence_sc::Real = pomdp.cadence_sc,
                     cadence_debris::Real = pomdp.cadence_debris,
                     constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                     pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                     pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY,
                     sigma_mode::Symbol = MCTS_SIGMA_MODE)
    return MCTSPlanner(pomdp, n_iterations, max_depth, Float64(c),
                       Float64(k), Float64(α), Float64(dt),
                       Float64(cadence_sc), Float64(cadence_debris),
                       constraint_mode, Float64(pc_weight), Float64(pc_penalty),
                       sigma_mode)
end

"""
    plan(planner, root::BeliefNode, rng) -> (best_action, root)

Run the planner's simulations from `root` and return the best action by
per-action value `Qa`. The mutated `root` (with visit/value stats and the built
tree, including per-node Pc / violation instrumentation) is returned too for
inspection/testing.
"""
function plan(planner::MCTSPlanner, root::BeliefNode, rng::AbstractRNG)
    # Efficiency pass: under :fast, precompute the branch-invariant debris
    # Σ-at-TCA per depth ONCE (see the FAST Σ PATH section header), then look it
    # up per node instead of re-propagating the debris covariance every node.
    table = planner.sigma_mode == :fast ?
            build_sigma_tca_table(planner.pomdp, root; dt = planner.dt,
                                  cadence_sc = planner.cadence_sc,
                                  cadence_debris = planner.cadence_debris,
                                  max_depth = planner.max_depth) : nothing
    for _ in 1:planner.n_iterations
        simulate!(planner.pomdp, root, planner.max_depth, rng;
                  dt = planner.dt, cadence_sc = planner.cadence_sc,
                  cadence_debris = planner.cadence_debris,
                  c = planner.c, k = planner.k, α = planner.α,
                  constraint_mode = planner.constraint_mode,
                  pc_weight = planner.pc_weight, pc_penalty = planner.pc_penalty,
                  node_depth = 0, table = table, sigma_mode = planner.sigma_mode)
    end
    acts = POMDPs.actions(planner.pomdp)
    best_a = acts[1]
    best_q = -Inf
    for a in acts
        q = get(root.Qa, a, -Inf)
        if q > best_q
            best_q = q
            best_a = a
        end
    end
    return best_a, root
end

"""
    root_from_pomdp(pomdp, s0::CAState; correct_at_root=pomdp.correct_at_root) -> BeliefNode

Build a root node whose belief is anchored at the true state `s0` with the
POMDP's P0 covariances (Phase 4 `belief_from_pomdp`) and whose sampled true
state is `s0` itself. `t` (time-remaining) comes from `s0.t`.

The cadence timers (`since_sc` / `since_debris`) are initialized per
`correct_at_root`:
  • `true`  — both objects were just fixed at detection (P0 IS that fix), so the
    timers start at 0 and the first in-tree correction for each object fires one
    full cadence later. (Default.)
  • `false` — timers start pre-loaded at each object's cadence, so a fix can land
    on the very first step (a fix that arrives soon after detection). A phase
    parameter for whether the debris TLE / GPS fix is "fresh" or "stale" at the
    root; swappable so we can test the effect (TODOS "measurement realism").
"""
function root_from_pomdp(pomdp::SpacecraftCAPOMDP, s0::CAState;
                         correct_at_root::Bool = pomdp.correct_at_root)
    b0 = belief_from_pomdp(pomdp, s0.sc_eci, s0.debris_eci, s0.t)
    since_sc     = correct_at_root ? 0.0 : pomdp.cadence_sc
    since_debris = correct_at_root ? 0.0 : pomdp.cadence_debris
    return BeliefNode(b0, s0, isterminal(pomdp, s0);
                      since_sc = since_sc, since_debris = since_debris)
end
