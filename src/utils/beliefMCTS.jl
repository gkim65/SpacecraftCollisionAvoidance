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
    # --- Phase 6 Pc instrumentation (for the constraint + the ablation) -------
    pc::Float64                    # Pc-at-TCA from this node's belief (NaN = not yet computed)
    violated::Bool                 # did this node's Pc exceed pomdp.pc_threshold?
end

function BeliefNode(belief::Belief, s_true::CAState, is_terminal::Bool)
    return BeliefNode(belief, s_true, 0,
                      Dict{CAAction,Int}(), Dict{CAAction,Float64}(),
                      Dict{CAAction,Vector{BeliefNode}}(), is_terminal,
                      NaN, false)
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
                     pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY)
    pc = isnan(child.pc) ? node_pc_at_tca(pomdp, child) : child.pc
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
    expand_child(pomdp, node, a, rng; dt=pomdp.dt,
                 constraint_mode=MCTS_CONSTRAINT_MODE, pc_weight=..., pc_penalty=...)
        -> (child::BeliefNode, r::Float64)

Take action `a` from `node`: propagate the sampled true state (Phase 0
`POMDPs.transition`), predict the belief (Phase 4 `predict`), sample a genuine
observation of the new true state (`sample_observation`), and correct the belief
(`correct_linear`, the runtime path). Then evaluate the Pc-based step reward +
per-step chance constraint on the child (architecture §4 steps 5–6): the child's
Pc-at-TCA and violation flag are cached on it. Under `constraint_mode ==
:terminate`, a violating child is additionally marked terminal so the search
stops expanding past it. Returns the new observation-child node and its step
reward. `dt` is swappable (defaults pomdp.dt).
"""
function expand_child(pomdp::SpacecraftCAPOMDP, node::BeliefNode, a::CAAction,
                      rng::AbstractRNG; dt::Real = pomdp.dt,
                      constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                      pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                      pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY)
    # 1. propagate the sampled true state (Phase 0 dynamics)
    sp = rand(rng, POMDPs.transition(pomdp, node.s_true, a))

    # 2. predict the belief forward one dt step (Phase 4)
    b_pred = predict(pomdp, node.belief, a; dt = dt)

    # 3–4. sample a real observation of sp, then correct (runtime path)
    z = sample_observation(pomdp, a, sp, rng)
    b_post = correct_linear(pomdp, b_pred, z)

    child = BeliefNode(b_post, sp, isterminal(pomdp, sp))

    # 5–6. Pc-at-TCA + per-step chance constraint (§4 step 6). Caches pc/violated.
    r, _, violated = step_reward(pomdp, a, child;
                                 constraint_mode = constraint_mode,
                                 pc_weight = pc_weight, pc_penalty = pc_penalty)
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
                    pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY)
    pc = isnan(node.pc) ? node_pc_at_tca(pomdp, node) : node.pc
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
                   c::Real = MCTS_UCB_C, k::Real = MCTS_K_OBS, α::Real = MCTS_ALPHA_OBS,
                   constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                   pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                   pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY)
    if node.is_terminal || depth <= 0
        return leaf_value(pomdp, node; constraint_mode = constraint_mode,
                          pc_weight = pc_weight, pc_penalty = pc_penalty)
    end

    node.N += 1
    acts = POMDPs.actions(pomdp)
    a = ucb_select(node, acts; c = c)

    na = get(node.Na, a, 0)
    kids = get!(node.children, a, BeliefNode[])

    # observation progressive widening (POMCPOW rule)
    local child::BeliefNode
    local r::Float64
    if should_widen(length(kids), na; k = k, α = α)
        child, r = expand_child(pomdp, node, a, rng; dt = dt,
                                constraint_mode = constraint_mode,
                                pc_weight = pc_weight, pc_penalty = pc_penalty)
        push!(kids, child)
    else
        child = rand(rng, kids)
        # step reward for the reused edge — the child's Pc is already cached
        # (step_reward reuses it), so this is cheap and consistent with expand.
        r, _, _ = step_reward(pomdp, a, child;
                              constraint_mode = constraint_mode,
                              pc_weight = pc_weight, pc_penalty = pc_penalty)
    end

    q = r + POMDPs.discount(pomdp) * simulate!(pomdp, child, depth - 1, rng;
                                               dt = dt, c = c, k = k, α = α,
                                               constraint_mode = constraint_mode,
                                               pc_weight = pc_weight, pc_penalty = pc_penalty)

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
defaults to `pomdp.dt` and is swappable (run both grids to compare). The search
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
    constraint_mode::Symbol
    pc_weight::Float64
    pc_penalty::Float64
end

function MCTSPlanner(pomdp::SpacecraftCAPOMDP;
                     n_iterations::Int = MCTS_TREE_QUERIES,
                     max_depth::Int = MCTS_MAX_DEPTH,
                     c::Real = MCTS_UCB_C,
                     k::Real = MCTS_K_OBS,
                     α::Real = MCTS_ALPHA_OBS,
                     dt::Real = pomdp.dt,
                     constraint_mode::Symbol = MCTS_CONSTRAINT_MODE,
                     pc_weight::Real = MCTS_PC_REWARD_WEIGHT,
                     pc_penalty::Real = MCTS_PC_VIOLATION_PENALTY)
    return MCTSPlanner(pomdp, n_iterations, max_depth, Float64(c),
                       Float64(k), Float64(α), Float64(dt),
                       constraint_mode, Float64(pc_weight), Float64(pc_penalty))
end

"""
    plan(planner, root::BeliefNode, rng) -> (best_action, root)

Run the planner's simulations from `root` and return the best action by
per-action value `Qa`. The mutated `root` (with visit/value stats and the built
tree, including per-node Pc / violation instrumentation) is returned too for
inspection/testing.
"""
function plan(planner::MCTSPlanner, root::BeliefNode, rng::AbstractRNG)
    for _ in 1:planner.n_iterations
        simulate!(planner.pomdp, root, planner.max_depth, rng;
                  dt = planner.dt, c = planner.c, k = planner.k, α = planner.α,
                  constraint_mode = planner.constraint_mode,
                  pc_weight = planner.pc_weight, pc_penalty = planner.pc_penalty)
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
    root_from_pomdp(pomdp, s0::CAState) -> BeliefNode

Build a root node whose belief is anchored at the true state `s0` with the
POMDP's P0 covariances (Phase 4 `belief_from_pomdp`) and whose sampled true
state is `s0` itself. `t` (time-remaining) comes from `s0.t`.
"""
function root_from_pomdp(pomdp::SpacecraftCAPOMDP, s0::CAState)
    b0 = belief_from_pomdp(pomdp, s0.sc_eci, s0.debris_eci, s0.t)
    return BeliefNode(b0, s0, isterminal(pomdp, s0))
end
