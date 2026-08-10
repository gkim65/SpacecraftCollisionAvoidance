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
using Distributed

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

# --- REWARD MODE (2026-08-09 redesign; see CONSTANTS.md + the REWARD section) ---
# Selects WHERE the Pc term is charged (the over-maneuvering fix, memory
# chance-constraint-framing-tension + the root_decision_probe diagnostic):
#   :per_step (LEGACY) — charge −pc_weight·Pc at EVERY step and sum it down the
#       path (the original Phase-6 shaping). Punishes a branch for its wide EARLY
#       belief even when measurement resolves it by TCA → sinks the wait-and-measure
#       branch (Q(WAIT) ≈ −19.5k vs Q(MANEUVER) ≈ −13 on SWIFT/JILIN). Kept for the
#       ablation + the synthetic suites.
#   :terminal (DEFAULT) — Pc is a TERMINAL quantity: the per-step reward is fuel
#       cost ONLY (no Pc term), and the WHOLE Pc term lives in `leaf_value` at TCA.
#       A branch is valued by its Pc-AT-TCA (the resolved outcome), not the summed
#       early-belief Pc it passed through — so WAIT-that-goes-safe (fuel 0,
#       Pc→~1e-12) beats MANEUVER (fuel>0) on a feasible case. This is standard
#       chance-constrained planning (trajectory feasibility, not a running penalty)
#       and sharpens the ACAS-X parallel (reason about the resolved event).
const MCTS_REWARD_MODE = :terminal        # :terminal (default) | :per_step (legacy)

# Terminal soft over-δ penalty (only in :terminal mode). A leaf whose Pc-at-TCA
# exceeds pomdp.pc_threshold takes this flat penalty ON TOP of −pc_weight·Pc — a
# SOFT terminal constraint: a barely-over branch is still COMPARED (ranked by its
# Pc), not amputated, so the search can still prefer the least-infeasible option
# when NO feasible one exists (the debris control case correctly stays MANEUVER).
# Sized ≫ the per-step maneuver_cost (10) and ≫ a feasible branch's −pc_weight·Pc
# (≤ pc_weight·δ = 1e6·1e-5 = 10) so crossing δ always dominates a fuel burn, i.e.
# the constraint binds. TODO: needs source (a tuning knob, like pc_weight /
# maneuver_cost; scaled to bind at δ, not an independently-measured quantity).
const MCTS_TERMINAL_PENALTY = 1.0e4       # flat penalty when leaf Pc-at-TCA > threshold

# Σ-propagation mode for the Pc eval (efficiency pass, 2026-07-23):
#   :fast  — precompute the branch-invariant DEBRIS Σ-at-TCA per depth ONCE per
#            plan, look it up per node; propagate only the debris MEAN + the full
#            satellite belief per node. Pc-EXACT (debris Σ is bitwise-invariant),
#            ~2× faster. VALID ONLY under noiseless maneuvers (breaks at Phase 8).
#   :exact — propagate every node's full belief (mean+Σ, both objects) to TCA.
#            The correctness oracle; the ONLY valid mode once Phase 8 adds
#            maneuver noise. See node_pc_at_tca / the FAST Σ PATH section header.
const MCTS_SIGMA_MODE = :fast

# --- root-parallel MCTS knobs (efficiency pass, 2026-07-23) --------------------
# Multiprocess root parallelization: split the simulation budget across N Julia
# WORKER PROCESSES (Distributed), each with its OWN Python/brahe interpreter, then
# merge the per-action visit counts + Q (Na-weighted running-average combine). See
# the ROOT-PARALLEL MCTS section header for the why (PyCall's GIL rules out
# in-process threading) and the soundness argument for pick-best-by-Q.
#   MCTS_PARALLEL   — default on/off for the parallel path (serial when false).
#   MCTS_N_WORKERS  — how many worker processes to split the budget across when on;
#                     `nothing` ⇒ use all currently-attached Distributed workers.
const MCTS_PARALLEL  = false
const MCTS_N_WORKERS = nothing

# =========================================================================
# ADAPTIVE EVENT-DRIVEN DECISION GRID (2026-08-09 redesign, Part 2).
#
# WHY: the fixed-dt grid plans at ~33 uniform steps over a 33 h horizon (~263 s /
# plan) and, worse, makes the TERMINAL Pc reward ineffective — a full-depth rollout
# rarely reaches TCA within the depth budget, so the leaf value at TCA is seldom
# seen. Between measurements, though, the belief only GROWS deterministically
# (predict, Σ⁻ = Φ Σ Φᵀ); nothing DECISION-relevant happens, so planning there is
# wasted. This grid instead jumps epoch→epoch, where a decision epoch is a
# MEASUREMENT time (per the class cadence) plus TCA. A rollout then reaches TCA in
# a handful of steps → leaves are cheap AND actually reached → the terminal reward
# works. The two redesign parts are mutually enabling.
#
# WHAT IT IS: plain DATA the caller builds and hands to the planner, so timings are
# fully swappable per run and can be NON-UNIFORM along the horizon (coarse far from
# TCA, fine near it). `grid === nothing` ⇒ the legacy fixed-dt / cadence-timer path,
# byte-for-byte unchanged (the synthetic suites' path). The executor re-plans every
# executed step (receding horizon), so rebuilding the grid from the CURRENT
# time-to-TCA each step makes it adapt as the episode marches down the horizon.
#
# STRUCTURE: a node at tree DEPTH d (root = 0) sits at epoch `t_epochs[d+1]` (time
# remaining). Taking an action steps to depth d+1 by predicting over the gap
# `dts[d+1] = t_epochs[d+1] − t_epochs[d+2]`, then correcting the object(s) whose
# measurement lands at the new epoch (`correct_sc[d+1]` / `correct_debris[d+1]`).
# The last epoch is 0 (TCA), so a rollout that runs to the grid's end IS at TCA.
# MANEUVER is available at every epoch (the burn-candidate rule Grace chose for now;
# adaptive burn density is a later layer).
# =========================================================================

"""
    DecisionGrid

An adaptive event-driven decision schedule for one `plan` call. Element `i`
(1-based, `i = depth + 1`) describes the STEP taken FROM depth `d = i-1`:
`dts[i]` is the time gap (s) to the next epoch, and `correct_sc[i]` /
`correct_debris[i]` say whether the spacecraft / debris gets a measurement
correction at the epoch reached by that step. `t_epochs` lists the epoch
time-remaining values (descending, ending at 0 = TCA) for reference / logging;
`t_epochs[i]` is the time-remaining at depth `i-1`. Build one with
`decision_grid` (auto-generated from cadences, optionally non-uniform) or
construct directly for a fully-custom schedule. Length of `dts`/`correct_*` is
`length(t_epochs) - 1` (one step between each pair of epochs).
"""
struct DecisionGrid
    t_epochs::Vector{Float64}     # epoch time-remaining (s), descending, last = 0
    dts::Vector{Float64}          # step gap (s) from depth d to d+1
    correct_sc::Vector{Bool}      # sc measurement lands at the epoch reached
    correct_debris::Vector{Bool}  # debris measurement lands at the epoch reached
end

"""
    grid_depth_count(grid) -> Int

Number of steps (== max usable tree depth) the grid defines: `length(grid.dts)`.
A rollout of this many steps from the root reaches the final epoch (TCA).
"""
grid_depth_count(grid::DecisionGrid) = length(grid.dts)

"""
    decision_grid(horizon; cadence_secondary, schedule=nothing, atol=1.0) -> DecisionGrid

Auto-generate an event-driven `DecisionGrid` over `[horizon, 0]` (time-remaining,
s). The model (Grace 2026-08-09): the PRIMARY is the own asset with continuous
GPS-level tracking — it is not an *event*, so it does NOT define decision epochs;
its belief stays tight and it is simply corrected at EVERY epoch. The SECONDARY
(the risky object) is measured on its class ground-pass cadence, and THOSE
measurement times ARE the events a decision should react to. So:

    decision epochs = the SECONDARY's measurement times + TCA
    correct_sc = true at every epoch     (continuous GPS primary)
    correct_db = true at every epoch     (each epoch IS a secondary measurement)

`cadence_secondary` (s) is the secondary's measurement cadence (its class value —
payload ~2 h, debris/RB ~8 h). Epochs land at `H, H−cad, H−2·cad, …, 0`. Pass a
`schedule` (a vector of `(t_lo, t_hi) => Δt` pairs, s) instead for a NON-UNIFORM
secondary schedule along the horizon (e.g. denser near TCA); the first matching
interval (`t_lo ≤ t < t_hi`) sets the local gap. `atol` (s) merges epochs closer
than that (avoids a degenerate ~0-length step). For a fully-custom schedule,
construct a `DecisionGrid` directly.
"""
function decision_grid(horizon::Real;
                       cadence_secondary::Union{Real,Nothing} = nothing,
                       schedule::Union{AbstractVector,Nothing} = nothing,
                       atol::Real = 1.0)
    H = Float64(horizon)
    H > 0 || error("decision_grid: horizon must be positive (got $H)")

    # local secondary-cadence gap at time-remaining t (schedule wins over scalar).
    function step_at(t::Float64)
        if schedule !== nothing
            for (rng, Δt) in schedule
                lo, hi = Float64(rng[1]), Float64(rng[2])
                lo <= t < hi && return Float64(Δt)
            end
            error("decision_grid: schedule does not cover time-remaining $t")
        end
        cadence_secondary === nothing &&
            error("decision_grid: pass `cadence_secondary` or `schedule`")
        return Float64(cadence_secondary)
    end

    # Epochs = the secondary's measurement times, walking DOWN from H toward TCA.
    epochs = Float64[H]
    t = H
    guard = 0
    while t > atol
        Δt = step_at(t)
        Δt > 0 || error("decision_grid: non-positive cadence gap $Δt at t=$t")
        t = max(0.0, t - Δt)
        (t <= atol || abs(epochs[end] - t) > atol) && push!(epochs, t)
        guard += 1
        guard > 100_000 && error("decision_grid: too many epochs (cadence too small?)")
    end
    epochs[end] > atol && push!(epochs, 0.0)   # ensure TCA is the final epoch

    # Every epoch IS a secondary measurement, and the continuous primary is fixed at
    # every epoch too → both correction flags are true on every step.
    n = length(epochs)
    dts = Float64[epochs[i] - epochs[i + 1] for i in 1:(n - 1)]
    csc = trues(n - 1)
    cdb = trues(n - 1)
    return DecisionGrid(epochs, dts, csc, cdb)
end

"""
    wait_spine_pc(pomdp, b0, s_true, epochs) -> Vector{Float64}

Compute Pc-at-TCA along the WAIT spine at each epoch in `epochs` (descending
time-remaining, the first entry = the root). Walks `predict → correct(both, every
epoch)` from `b0` — the SAME schedule the grid uses — and evaluates
`node_pc_at_tca` at each. z-independent Σ update ⇒ a zero-innovation observation
(`z = μ⁻`) gives the exact belief Σ. Cheap (no MCTS): one propagation chain of
`length(epochs)` steps. This is the CROSSING DETECTOR the adaptive grid probes
with — it reveals WHERE (if anywhere) WAIT's Pc crosses the threshold, so the grid
can refine only there. Returns a Pc per epoch (`length(epochs)` entries).
"""
function wait_spine_pc(pomdp::SpacecraftCAPOMDP, b0::Belief, s_true::CAState,
                       epochs::AbstractVector{<:Real})
    pcs = Float64[]
    b = b0
    n0 = BeliefNode(b, s_true, isterminal(pomdp, s_true))
    push!(pcs, node_pc_at_tca(pomdp, n0))
    for i in 1:(length(epochs) - 1)
        dt = Float64(epochs[i]) - Float64(epochs[i + 1])
        b = predict(pomdp, b, WAIT; dt = dt)
        z = vcat(b.sc.μ, b.debris.μ)                 # zero-innovation (Σ⁺ z-independent)
        b = correct_linear_sc(pomdp, b, z)
        b = correct_linear_debris(pomdp, b, z)
        nd = BeliefNode(b, s_true, false)
        push!(pcs, node_pc_at_tca(pomdp, nd))
    end
    return pcs
end

"""
    adaptive_decision_grid(pomdp, b0, s_true, horizon; cadence_secondary,
                           coarse=cadence_secondary, fine=cadence_secondary,
                           truncate_safe=false, safe_margin=coarse,
                           safe_factor=1e3, atol=1.0) -> (DecisionGrid, crossing_t)

Build a CROSSING-ADAPTIVE grid: probe the WAIT spine at the `coarse` cadence to
find where WAIT's Pc-at-TCA crosses below `pomdp.pc_threshold` (the WAIT-becomes-
safe point), then build the real grid COARSE everywhere EXCEPT a bracket around
that crossing, where it uses the `fine` step. This is self-tuning per case: a
DEBRIS case whose Pc never crosses stays coarse everywhere (few steps, fast);
a payload case densifies only around its crossing (cheap + captures the
decision-relevant region). Returns the grid and the crossing time-remaining
(`NaN` if no crossing — grid is uniformly coarse).

Refinement bracket = `[t_x + coarse, t_x − coarse]` (one coarse step on each side
of the crossing epoch `t_x`), filled at the `fine` cadence; outside it the coarse
cadence. Both correction flags are true every epoch (primary continuous; each
epoch a secondary measurement), same as `decision_grid`.

`truncate_safe` (PURE-EFFICIENCY, value-preserving — distinct from the deferred
provably-infeasible PRUNE, which is CONSTRAINT logic that CHANGES decisions): once
the WAIT-spine Pc is DEEPLY safe (`< pc_threshold / safe_factor`) AND monotone-
decreasing, further expansion cannot change any branch's terminal value (Pc only
falls further to TCA — verified monotone past the crossing on well-tracked cases,
maneuver_effectiveness_findings). So the grid STOPS at the first such epoch plus a
`safe_margin` cushion; the leaf THERE still evaluates Pc-at-TCA (propagated to TCA
regardless of node position — the `:coast` leaf), so the terminal quantity is
preserved. This ONLY saves compute after the decision is settled; it does not
prune infeasible branches and does not alter Q (up to the monotone assumption).
Default `false` (full depth to TCA). NB monotonicity is the guard — if a case's Pc
straddles the threshold (a ripple regime), leave `truncate_safe=false`.
"""
function adaptive_decision_grid(pomdp::SpacecraftCAPOMDP, b0::Belief,
                                s_true::CAState, horizon::Real;
                                cadence_secondary::Real,
                                coarse::Real = cadence_secondary,
                                fine::Real = cadence_secondary,
                                truncate_safe::Bool = false,
                                safe_margin::Real = coarse,
                                safe_factor::Real = 1e3,
                                atol::Real = 1.0)
    H = Float64(horizon)
    coarse = Float64(coarse); fine = Float64(fine)

    # 1. coarse probe epochs (secondary-cadence spine at the coarse step).
    probe = decision_grid(H; cadence_secondary = coarse, atol = atol)
    te = probe.t_epochs
    pcs = wait_spine_pc(pomdp, b0, s_true, te)

    # 2. find the crossing: first epoch where Pc drops to/below threshold.
    thr = pomdp.pc_threshold
    xi = findfirst(p -> p <= thr, pcs)
    if xi === nothing || xi == 1
        # never crosses (debris) — or already safe at root: uniform coarse grid.
        return probe, NaN
    end
    t_x = te[xi]                       # crossing epoch (time-remaining)

    # 2b. EFFICIENCY TRUNCATION (value-preserving; see docstring). Find the first
    # epoch where Pc is DEEPLY safe AND monotone-decreasing from there; stop the
    # horizon a `safe_margin` past it. Guard: require the tail from that epoch to be
    # non-increasing (no straddle) — else do not truncate.
    t_stop = 0.0
    if truncate_safe
        deep = thr / Float64(safe_factor)
        si = findfirst(i -> pcs[i] < deep, eachindex(pcs))
        if si !== nothing && si < length(pcs)
            tail = pcs[si:end]
            monotone = all(tail[j + 1] <= tail[j] + eps() for j in 1:(length(tail) - 1))
            monotone && (t_stop = max(0.0, te[si] - Float64(safe_margin)))
        end
    end

    # 3. build the refined epoch set: coarse everywhere, fine in the bracket
    #    [t_x + coarse, t_x − coarse]. Walk down from H at the local step, stopping
    #    at `t_stop` (0.0 = full depth to TCA when not truncating).
    lo = max(0.0, t_x - coarse); hi = t_x + coarse
    epochs = Float64[H]; t = H; guard = 0
    while t > t_stop + atol
        step = (t <= hi + atol && t >= lo - atol) ? fine : coarse
        t = max(t_stop, t - step)
        (abs(t - t_stop) <= atol || abs(epochs[end] - t) > atol) && push!(epochs, t)
        guard += 1; guard > 100_000 && error("adaptive_decision_grid: too many epochs")
    end
    # ensure the final epoch is exactly t_stop (TCA=0 in the non-truncating case).
    abs(epochs[end] - t_stop) > atol && push!(epochs, t_stop)

    n = length(epochs)
    dts = Float64[epochs[i] - epochs[i + 1] for i in 1:(n - 1)]
    return DecisionGrid(epochs, dts, trues(n - 1), trues(n - 1)), t_x
end

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
                             Σ::AbstractMatrix, objParams::AbstractVector, t::Real;
                             q_rtn::AbstractVector = zeros(3), dt::Real = pomdp.dt)
    bh = get_brahe()
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - Float64(t)

    # Q = 0 ⇒ single-shot Φ Σ Φᵀ (byte-identical to the pre-Phase-8 path, fast).
    if all(iszero, q_rtn) || Float64(t) <= 0.0
        prop, ep0 = eci2orb_brahe(collect(float.(μ)), epoch_to_tuple(epoch_current),
                                  objParams, pomdp.forceModel; initial_covariance = Matrix(Σ))
        ep_tca = ep0 + Float64(t)
        prop.propagate_to(ep_tca)
        return collect(prop.current_state()[1:6]),
               _sym(collect(prop.covariance_gcrf(ep_tca)))
    end

    # Q ≠ 0 ⇒ step the covariance in dt sub-steps so the SNC accumulation matches
    # the belief tree's stepped `predict` exactly (a one-shot STM accumulation is
    # NOT equivalent — later Φ Σ Φᵀ steps keep amplifying each step's added Q).
    # The final partial step (t not a multiple of dt) uses the remainder.
    μc = collect(float.(μ)); Σc = Matrix(Σ); tt = 0.0
    while tt < Float64(t) - 1e-6
        step = min(Float64(dt), Float64(t) - tt)
        prop, ep0 = eci2orb_brahe(μc, epoch_to_tuple(epoch_tca - (Float64(t) - tt)),
                                  objParams, pomdp.forceModel; initial_covariance = Σc)
        prop.propagate_to(ep0 + step)
        Σc = _sym(collect(prop.covariance_gcrf(ep0 + step)) .+ snc_q_eci(q_rtn, step, μc))
        μc = collect(prop.current_state()[1:6])
        tt += step
    end
    return μc, Σc
end

"""
    _backprop_object_to_detection(pomdp, μ_tca, Σ_tca, objParams, t_horizon;
                                  q_rtn, dt) -> (μ_det, Σ_det)

Seed a DETECTION-epoch belief `(μ_det, Σ_det)` for one object such that
forward-growing it to TCA (via `_grow_belief_to_tca`, the planner's own path)
reproduces the given TCA belief `(μ_tca, Σ_tca)`. This is the covariance-fix
back-propagation (audit F2/F3, step 2b): a real CDM is a single TCA snapshot,
but the POMDP needs a detection→TCA belief history; a single conjunction's OD
covariance is SMALLER at detection and GROWS toward TCA, so we back-propagate
the CDM's TCA covariance to a tighter detection seed that forward-grows back.

MEAN: back-propagated by reverse dynamics (one brahe propagation TCA→detection).
Exact to the integrator-step floor (~0.05 m at 1 h, ~105 m at 33 h on the
default tolerance — an integrator artifact, not physics: Keplerian, which is
time-reversible, shows the same; tighten `NumericalPropagationConfig` tol to
shrink it, at ~15× propagation cost). The ~105 m shifts recovered Pc by ~3 %.

COVARIANCE: a single backward brahe propagation reverses the STM growth,
Σ_det = Φ⁻¹ Σ_tca Φ⁻ᵀ (read back via `covariance_gcrf` at the detection epoch).
This is the Q=0 reverse; when `q_rtn ≠ 0` the forward grow adds Q_acc, so the
seed is `reverse(Σ_tca − Q_acc)` — computed by first estimating Q_acc from the
forward grow of the Q=0 seed and reversing the Q-reduced endpoint ONCE. We do
NOT iterate an affine residual correction: Φ⁻¹ over a multi-hour arc amplifies a
small ECI residual enormously (the in-track/velocity inverse coupling), so
back-propagating a residual matrix is numerically unstable and can push the seed
non-PD. The single Q-reduced reverse is stable and recovers Σ_tca to a small
relative error (test_cdm_scenario.jl); the residual is the integrator-asymmetry
floor, the same ~3 % Pc effect as the mean's ~105 m.
"""
function _backprop_object_to_detection(pomdp::SpacecraftCAPOMDP,
                                        μ_tca::AbstractVector, Σ_tca::AbstractMatrix,
                                        objParams::AbstractVector, t_horizon::Real;
                                        q_rtn::AbstractVector = zeros(3),
                                        dt::Real = pomdp.dt)
    bh = get_brahe()
    epoch_tca = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    th = Float64(t_horizon)

    # One backward propagation from TCA carries the mean (reverse dynamics) and
    # the Q=0 reverse of a covariance (Φ⁻¹ Σ Φ⁻ᵀ).
    function reverse_meancov(μ, Σ)
        prop, ep0 = eci2orb_brahe(collect(float.(μ)), epoch_to_tuple(epoch_tca),
                                  objParams, pomdp.forceModel; initial_covariance = Matrix(Σ))
        prop.propagate_to(ep0 - th)
        return collect(prop.current_state()[1:6]),
               _sym(collect(prop.covariance_gcrf(ep0 - th)))
    end

    μ_det, Σ_det0 = reverse_meancov(μ_tca, Σ_tca)
    all(iszero, q_rtn) && return μ_det, Σ_det0

    # Q ≠ 0: estimate the forward-accumulated process noise Q_acc from the Q=0
    # seed's forward grow, subtract it from the endpoint, and reverse that ONCE.
    # (Q_acc is the ADDED part, ~independent of the seed magnitude for a fixed
    # trajectory, so one subtraction is a good approximation without iterating.)
    _, Σ_fwd_q0 = _grow_belief_to_tca(pomdp, μ_det, Σ_det0, objParams, th; q_rtn = zeros(3), dt = dt)
    _, Σ_fwd_q  = _grow_belief_to_tca(pomdp, μ_det, Σ_det0, objParams, th; q_rtn = q_rtn,   dt = dt)
    Q_acc = _sym(Σ_fwd_q .- Σ_fwd_q0)
    # Guard: if the accumulated process noise exceeds the endpoint covariance in
    # some direction, Σ_tca − Q_acc goes non-PD (the physical signal that q is too
    # large for this lead time — the process noise alone would over-fill the CDM's
    # TCA covariance). Clip eigenvalues to a small positive floor so the seed stays
    # a valid covariance; the clip firing is a red flag that q needs recalibrating.
    Σ_reduced = _clip_pd(_sym(Matrix(Σ_tca) .- Q_acc))
    _, Σ_det = reverse_meancov(μ_tca, Σ_reduced)
    return μ_det, _clip_pd(Σ_det)
end

"""
    _clip_pd(Σ; floor_rel=1e-12) -> Σ_pd

Snap a symmetric matrix to the nearest PD matrix by clipping eigenvalues up to
`floor_rel · λ_max` (a small positive floor). Used to keep a back-propagated /
Q-reduced covariance seed a valid covariance when an over-large q would otherwise
drive it indefinite (see `_backprop_object_to_detection`). No-op on an already-PD
matrix to roundoff.
"""
function _clip_pd(Σ::AbstractMatrix; floor_rel::Real = 1e-12)
    S = Symmetric(_sym(Σ))
    vals, vecs = eigen(S)
    λmax = maximum(vals)
    λmax <= 0 && return Matrix(floor_rel * I, size(Σ)...)  # degenerate; tiny isotropic
    floorλ = floor_rel * λmax
    any(vals .< floorλ) || return Matrix(S)
    vals_clipped = max.(vals, floorλ)
    return _sym(vecs * Diagonal(vals_clipped) * transpose(vecs))
end

"""
    backprop_belief_to_detection(pomdp, μ_sc_tca, Σ_sc_tca, μ_db_tca, Σ_db_tca,
                                 t_horizon) -> Belief

Build a detection-epoch `Belief` (time-remaining `t_horizon`) that forward-grows
to the given TCA belief for BOTH objects, using each object's `pomdp.q_rtn_*`.
The loader uses this to turn a single CDM TCA snapshot into a runnable
detection→TCA scenario (see `_backprop_object_to_detection`).
"""
function backprop_belief_to_detection(pomdp::SpacecraftCAPOMDP,
                                      μ_sc_tca::AbstractVector, Σ_sc_tca::AbstractMatrix,
                                      μ_db_tca::AbstractVector, Σ_db_tca::AbstractMatrix,
                                      t_horizon::Real)
    μ_sc, Σ_sc = _backprop_object_to_detection(pomdp, μ_sc_tca, Σ_sc_tca,
                                               pomdp.satParams, t_horizon; q_rtn = pomdp.q_rtn_sc)
    μ_db, Σ_db = _backprop_object_to_detection(pomdp, μ_db_tca, Σ_db_tca,
                                               pomdp.debrisParams, t_horizon; q_rtn = pomdp.q_rtn_debris)
    return Belief(ObjBelief(μ_sc, Σ_sc), ObjBelief(μ_db, Σ_db), Float64(t_horizon))
end

"""
    node_pc_at_tca(pomdp, node) -> Float64

Pc-at-TCA for `node` (architecture §4 step 5 / §7), using the node's ACCUMULATED
belief Σ (see the section header): propagate both sub-beliefs' `(μ, Σ)` to TCA
(mean carried forward, the tracked Σ grown as Φ Σ Φᵀ) and evaluate `elrod_pc` on
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
        return elrod_pc(b.sc.μ, b.debris.μ, b.sc.Σ, b.debris.Σ, hbr)
    end
    μ_sc_tca, Σ_sc_tca = _grow_belief_to_tca(pomdp, b.sc.μ,     b.sc.Σ,     pomdp.satParams,    t;
                                             q_rtn = pomdp.q_rtn_sc)
    μ_db_tca, Σ_db_tca = _grow_belief_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, t;
                                             q_rtn = pomdp.q_rtn_debris)
    return elrod_pc(μ_sc_tca, μ_db_tca, Σ_sc_tca, Σ_db_tca, hbr)
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
                               max_depth::Int = MCTS_MAX_DEPTH,
                               grid::Union{DecisionGrid,Nothing} = nothing)
    Σ_db = Matrix{Float64}[]

    b = root.belief
    since_sc     = root.since_sc
    since_debris = root.since_debris

    # depth 0 (root): grow the root debris Σ to TCA along its own mean.
    push!(Σ_db, b.t <= PC_TAU_MATCH_ATOL ? Matrix(b.debris.Σ) :
          _grow_sigma_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, b.t))

    # On the adaptive grid the debris Σ per depth is STILL branch-invariant (the
    # predict/correct schedule is a pure function of depth via the grid), so the
    # same WAIT-spine replay works — but the step size and which object is corrected
    # come from the grid at each depth instead of a fixed dt + cadence timers.
    n_steps = grid === nothing ? max_depth : min(max_depth, grid_depth_count(grid))

    for d in 1:n_steps
        step_dt = grid === nothing ? Float64(dt) : grid.dts[d]
        # PREDICT one step along the WAIT spine (Σ grows regardless of action).
        b = predict(pomdp, b, WAIT; dt = step_dt)

        # Which object is corrected this step: grid flags on the adaptive path,
        # cadence timers on the fixed path. Σ⁺ = (I−K)Σ⁻ is z-independent, so a
        # zero-innovation z (= μ⁻) gives the exact tree Σ.
        if grid === nothing
            since_sc     += Float64(dt)
            since_debris += Float64(dt)
            correct_sc     = since_sc     >= cadence_sc
            correct_debris = since_debris >= cadence_debris
        else
            correct_sc     = grid.correct_sc[d]
            correct_debris = grid.correct_debris[d]
        end
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
precomputed `table`, then call `elrod_pc`. Numerically EXACT vs. `node_pc_at_tca`
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
        return elrod_pc(b.sc.μ, b.debris.μ, b.sc.Σ, table.Σ_db[idx], hbr)
    end
    μ_sc_tca, Σ_sc_tca = _grow_belief_to_tca(pomdp, b.sc.μ, b.sc.Σ, pomdp.satParams, t)  # sat exact
    μ_db_tca           = _grow_mean_to_tca(pomdp, b.debris.μ, pomdp.debrisParams, t)     # debris mean-only
    return elrod_pc(μ_sc_tca, μ_db_tca, Σ_sc_tca, table.Σ_db[idx], hbr)
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
    # The :fast per-depth debris Σ table is built WITHOUT process noise and its
    # satellite grow is Q=0; once any q_rtn ≠ 0 (Phase-8 SNC growth fix) it is no
    # longer valid — force the exact per-node path. (This is the documented
    # "process noise breaks :fast" guard; see the FAST Σ PATH section header.)
    q_on = !all(iszero, pomdp.q_rtn_sc) || !all(iszero, pomdp.q_rtn_debris)
    if !q_on && sigma_mode == :fast && table !== nothing && depth !== nothing
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
                     reward_mode::Symbol = MCTS_REWARD_MODE,
                     depth::Union{Int,Nothing} = nothing,
                     table::Union{SigmaTCATable,Nothing} = nothing,
                     sigma_mode::Symbol = MCTS_SIGMA_MODE)
    # --- :terminal (default) — per-step reward is FUEL COST ONLY. No Pc term and
    # no per-step violation penalty: Pc is a TERMINAL quantity charged once in
    # `leaf_value` at TCA (the over-maneuvering fix). We still compute + cache the
    # child's Pc / violation flag for the tree instrumentation (and so :terminate
    # can amputate on it if a caller opts in), but they do NOT enter the reward.
    if reward_mode == :terminal
        pc = isnan(child.pc) ?
             node_pc(pomdp, child; depth = depth, table = table, sigma_mode = sigma_mode) :
             child.pc
        child.pc = pc
        violated = pc > pomdp.pc_threshold
        child.violated = violated
        r = a == MANEUVER ? -Float64(pomdp.maneuver_cost) : 0.0
        return r, pc, violated
    end

    # --- :per_step (legacy) — charge −pc_weight·Pc every step + the per-step
    # violation penalty (the original Phase-6 path-summed shaping).
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
# VARIABLE-dt TRUE-STATE TRANSITION (adaptive grid, Part 2).
#
# `POMDPs.transition` (transitions.jl) hardcodes the step at `pomdp.dt`. On the
# adaptive grid the true state must advance by the SAME variable epoch gap the
# belief does, so the truth and belief stay on one timeline (the executor's
# dt-consistency invariant). `transition_dt` is `POMDPs.transition` with an
# explicit `dt`: apply the maneuver Δv kick (identical convention), propagate both
# objects forward by `dt` under the accurate force model, and return a Deterministic
# next CAState. When `dt == pomdp.dt` it is exactly `POMDPs.transition`.
# =========================================================================

"""
    transition_dt(pomdp, s::CAState, a::CAAction, dt) -> Deterministic(CAState)

Advance the true state `s` by `dt` seconds under action `a` (same dynamics /
maneuver convention as `POMDPs.transition`, which is the `dt == pomdp.dt` case).
Used by the adaptive-grid MCTS so the true state steps by the variable epoch gap.
"""
function transition_dt(pomdp::SpacecraftCAPOMDP, s::CAState, a::CAAction, dt::Real)
    isterminal(pomdp, s) && return Deterministic(s)
    bh = get_brahe()
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - s.t
    epoch_next    = epoch_tca - (s.t - Float64(dt))

    sc_eci_start = copy(s.sc_eci)
    if a == MANEUVER
        v     = sc_eci_start[4:6]
        v_hat = v / norm(v)
        sc_eci_start[4:6] += pomdp.Δv * v_hat
    end

    epoch_current_tuple = epoch_to_tuple(epoch_current)
    prop_sc, _     = eci2orb_brahe(sc_eci_start, epoch_current_tuple,
                                   pomdp.satParams, pomdp.forceModel)
    prop_debris, _ = eci2orb_brahe(s.debris_eci, epoch_current_tuple,
                                   pomdp.debrisParams, pomdp.forceModel)
    prop_sc.propagate_to(epoch_next)
    prop_debris.propagate_to(epoch_next)

    sc_eci_next     = collect(prop_sc.current_state()[1:6])
    debris_eci_next = collect(prop_debris.current_state()[1:6])

    t_next     = s.t - Float64(dt)
    R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    x_rel_next = collect(bh.state_eci_to_rtn(sc_eci_next, debris_eci_next))
    terminal   = t_next <= 0.0 || norm(x_rel_next[1:3]) < R_combined

    return Deterministic(CAState(sc_eci_next, debris_eci_next, t_next, terminal))
end

# =========================================================================
# SHARED per-step belief update (predict → asymmetric cadence correct).
#
# This is the single source of truth for "advance a belief one dt step under an
# action, applying the asymmetric measurement cadence." BOTH the MCTS expansion
# (`expand_child`, which simulates this step in the tree) AND the real-world
# closed-loop executor (`beliefExecutor.jl`, which does it for real once per
# executed step) call this, so the belief the planner assumes internally and the
# belief actually tracked in execution are updated by IDENTICAL code — they
# cannot silently drift (the correctness point of the receding-horizon driver).
#
# It returns the updated belief AND the advanced cadence timers, so the caller
# can carry `since_sc`/`since_debris` forward. The observation `z` is drawn HERE
# (a genuine random draw of the true next state, architecture §4 step 3) only
# when at least one object is due for a fix, so the sampling is identical on both
# paths. Only the belief is touched — tree bookkeeping (nodes, Pc caching,
# terminal marking, reward) stays in the callers, which is not a belief concern.
# =========================================================================

"""
    step_belief(pomdp, b, a, s_true_next, rng; dt=pomdp.dt, cadence_sc=..., cadence_debris=...,
                since_sc, since_debris, p_arrival=1.0) -> (b′::Belief, since_sc′, since_debris′)

Advance belief `b` one `dt` step under action `a` (architecture §4 steps 2–4),
applying the asymmetric measurement cadence. `predict` first (Σ grows one step;
maneuver kicks the SC mean), then correct the object(s) whose fix is due this
step against a freshly sampled observation of the true next state `s_true_next`.
An object's fix is due when its elapsed-time timer (`since_* + dt`) reaches its
cadence, at which point the timer resets to 0; otherwise the timer carries the
step forward and that object is predict-only. Returns the updated belief and the
advanced timers. Shared by `expand_child` (in-tree simulation) and the
closed-loop executor (real-world update) so the two stay bit-for-bit consistent.

PROBABILISTIC MEASUREMENT ARRIVAL (`p_arrival`): a SCHEDULED debris (SSN/TLE) fix
ARRIVES with probability `p_arrival` (a Bernoulli draw from `rng`); on a
non-arrival it is treated as if no measurement came — the debris is predict-only
that step (Σ grew, no innovation) and the cadence timer STILL RESETS (the fetch
was scheduled, it just returned nothing — the next one is a full cadence away).
The satellite GPS fix is an own-asset continuous track and is NOT gated (always
arrives when due). `p_arrival` is applied IDENTICALLY on both the grid and the
cadence-timer path, and — because `step_belief` is the single shared update — the
MCTS rollout (`expand_child`) and the real executor use the SAME arrival model,
so the planner genuinely reasons about arrival rather than assuming p=1. When
`p_arrival == 1.0` the Bernoulli draw is SKIPPED ENTIRELY (no `rand` consumed), so
the RNG stream — and therefore every synthetic-suite result — is byte-identical to
the pre-arrival code.
"""
function step_belief(pomdp::SpacecraftCAPOMDP, b::Belief, a::CAAction,
                     s_true_next::CAState, rng::AbstractRNG;
                     dt::Real = pomdp.dt,
                     cadence_sc::Real = pomdp.cadence_sc,
                     cadence_debris::Real = pomdp.cadence_debris,
                     since_sc::Real, since_debris::Real,
                     p_arrival::Real = 1.0,
                     grid::Union{DecisionGrid,Nothing} = nothing,
                     grid_depth::Union{Int,Nothing} = nothing)
    # ADAPTIVE GRID PATH: the step size and which object(s) are corrected come from
    # `grid` at this depth, not from the cadence timers. `grid_depth` is the CHILD's
    # tree depth (root child = 1); the step from parent depth d to child depth d+1
    # uses grid index d+1 == grid_depth. predict over that variable gap, then
    # correct the object(s) whose measurement lands at the reached epoch. Timers are
    # not used on this path (kept unchanged, so a mixed caller is well-defined).
    if grid !== nothing && grid_depth !== nothing && grid_depth >= 1 &&
       grid_depth <= grid_depth_count(grid)
        gdt = grid.dts[grid_depth]
        b = predict(pomdp, b, a; dt = gdt)
        do_sc = grid.correct_sc[grid_depth]
        # probabilistic arrival gates the DEBRIS fix only (SSN/TLE fetch may not
        # land); the satellite GPS fix always arrives. On a non-arrival the debris
        # is predict-only this step (Σ already grew above, no innovation applied).
        do_db = grid.correct_debris[grid_depth] && _debris_arrives(rng, p_arrival)
        if do_sc || do_db
            z = sample_observation(pomdp, a, s_true_next, rng)
            do_sc && (b = correct_linear_sc(pomdp, b, z))
            do_db && (b = correct_linear_debris(pomdp, b, z))
        end
        return b, since_sc, since_debris   # timers untouched on the grid path
    end

    # predict the belief forward one dt step (Phase 4). Σ GROWS this step.
    b = predict(pomdp, b, a; dt = dt)

    # asymmetric cadence: correct only the object(s) whose fix is due. A fix is
    # due when the time since the object's last fix has reached its cadence; the
    # timer resets on a fix, else carries forward.
    since_sc     = Float64(since_sc)     + Float64(dt)
    since_debris = Float64(since_debris) + Float64(dt)
    correct_sc      = since_sc     >= cadence_sc
    debris_due      = since_debris >= cadence_debris
    # probabilistic arrival: a scheduled debris fix arrives with prob p_arrival.
    # On a non-arrival the debris is predict-only this step, but the timer STILL
    # resets — the fetch was scheduled and simply returned nothing, so the next
    # scheduled fix is a full cadence away (a missed fetch is not retried early).
    # The satellite GPS fix (own-asset continuous track) is never gated.
    debris_arrives  = debris_due && _debris_arrives(rng, p_arrival)

    if correct_sc || debris_arrives
        z = sample_observation(pomdp, a, s_true_next, rng)  # one genuine draw
        if correct_sc
            b = correct_linear_sc(pomdp, b, z)
        end
        if debris_arrives
            b = correct_linear_debris(pomdp, b, z)
        end
    end
    correct_sc && (since_sc = 0.0)
    debris_due && (since_debris = 0.0)   # timer resets on a scheduled fix, arrived or not
    return b, since_sc, since_debris
end

# Bernoulli arrival gate for a scheduled DEBRIS measurement. Returns true when the
# fix arrives. `p_arrival == 1.0` short-circuits WITHOUT drawing from `rng`, so the
# guaranteed-measurement path consumes no random number and stays byte-identical to
# the pre-arrival stream (the p=1.0 regression anchor). Only drawn for p<1.
_debris_arrives(rng::AbstractRNG, p_arrival::Real) =
    p_arrival >= 1.0 || rand(rng) < p_arrival

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
                      reward_mode::Symbol = MCTS_REWARD_MODE,
                      p_arrival::Real = 1.0,
                      depth::Union{Int,Nothing} = nothing,
                      table::Union{SigmaTCATable,Nothing} = nothing,
                      sigma_mode::Symbol = MCTS_SIGMA_MODE,
                      grid::Union{DecisionGrid,Nothing} = nothing)
    # 1. propagate the sampled true state (Phase 0 dynamics). On the adaptive grid
    #    the true state advances by the SAME variable epoch gap the belief does, so
    #    the truth and belief stay on one timeline (see DECISION GRID). `depth` is
    #    the child's tree depth, i.e. grid index d+1.
    eff_dt = (grid !== nothing && depth !== nothing && depth >= 1 &&
              depth <= grid_depth_count(grid)) ? grid.dts[depth] : Float64(dt)
    sp = rand(rng, transition_dt(pomdp, node.s_true, a, eff_dt))

    # 2–4. predict + measurement correct over this step — the SHARED per-step
    # belief update (see `step_belief`). On the fixed grid (grid === nothing) the
    # asymmetric cadence timers decide which object is fixed; on the adaptive grid
    # the correction is dictated by `grid` at this depth (which epoch = which fix).
    b, since_sc, since_debris = step_belief(pomdp, node.belief, a, sp, rng;
                                            dt = dt, cadence_sc = cadence_sc,
                                            cadence_debris = cadence_debris,
                                            since_sc = node.since_sc,
                                            since_debris = node.since_debris,
                                            p_arrival = p_arrival,
                                            grid = grid, grid_depth = depth)

    child = BeliefNode(b, sp, isterminal(pomdp, sp);
                       since_sc = since_sc, since_debris = since_debris)

    # 5–6. Pc-at-TCA + reward. Under :terminal the step reward is fuel-only and Pc
    # is charged at the leaf; under :per_step this applies the legacy shaping +
    # per-step constraint. Either way the child's pc/violated are cached.
    r, _, violated = step_reward(pomdp, a, child;
                                 constraint_mode = constraint_mode,
                                 pc_weight = pc_weight, pc_penalty = pc_penalty,
                                 reward_mode = reward_mode,
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
                    reward_mode::Symbol = MCTS_REWARD_MODE,
                    terminal_penalty::Real = MCTS_TERMINAL_PENALTY,
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
    if reward_mode == :terminal
        # SOFT terminal constraint: a flat over-δ penalty ON TOP of −pc_weight·Pc.
        # A barely-over branch is still ranked by its Pc (compared, not amputated),
        # so the search prefers the least-infeasible option when none is feasible.
        # (Independent of constraint_mode, which governed the legacy per-step path;
        # :off still suppresses it for the no-constraint ablation arm.)
        if violated && constraint_mode != :off
            v -= terminal_penalty
        end
    else
        # legacy :per_step — the leaf mirrors the per-step violation penalty.
        if violated && constraint_mode != :off
            v -= pc_penalty
        end
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
                   reward_mode::Symbol = MCTS_REWARD_MODE,
                   terminal_penalty::Real = MCTS_TERMINAL_PENALTY,
                   p_arrival::Real = 1.0,
                   node_depth::Int = 0,
                   table::Union{SigmaTCATable,Nothing} = nothing,
                   sigma_mode::Symbol = MCTS_SIGMA_MODE,
                   grid::Union{DecisionGrid,Nothing} = nothing)
    # On the adaptive grid, a node is a LEAF once it reaches the final epoch (TCA),
    # i.e. its tree depth has consumed the whole grid — regardless of the depth
    # budget. This makes a rollout reach TCA in `grid_depth_count` steps so the
    # terminal reward is actually seen (the point of Part 2).
    grid_exhausted = grid !== nothing && node_depth >= grid_depth_count(grid)
    if node.is_terminal || depth <= 0 || grid_exhausted
        return leaf_value(pomdp, node; constraint_mode = constraint_mode,
                          pc_weight = pc_weight, pc_penalty = pc_penalty,
                          reward_mode = reward_mode, terminal_penalty = terminal_penalty,
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
                                reward_mode = reward_mode, p_arrival = p_arrival,
                                depth = child_depth, table = table, sigma_mode = sigma_mode,
                                grid = grid)
        push!(kids, child)
    else
        child = rand(rng, kids)
        # step reward for the reused edge — the child's Pc is already cached
        # (step_reward reuses it), so this is cheap and consistent with expand.
        r, _, _ = step_reward(pomdp, a, child;
                              constraint_mode = constraint_mode,
                              pc_weight = pc_weight, pc_penalty = pc_penalty,
                              reward_mode = reward_mode,
                              depth = child_depth, table = table, sigma_mode = sigma_mode)
    end

    q = r + POMDPs.discount(pomdp) * simulate!(pomdp, child, depth - 1, rng;
                                               dt = dt, cadence_sc = cadence_sc,
                                               cadence_debris = cadence_debris,
                                               c = c, k = k, α = α,
                                               constraint_mode = constraint_mode,
                                               pc_weight = pc_weight, pc_penalty = pc_penalty,
                                               reward_mode = reward_mode,
                                               terminal_penalty = terminal_penalty,
                                               p_arrival = p_arrival,
                                               node_depth = child_depth,
                                               table = table, sigma_mode = sigma_mode,
                                               grid = grid)

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
    parallel::Bool
    n_workers::Union{Int,Nothing}
    reward_mode::Symbol
    terminal_penalty::Float64
    # probability a SCHEDULED debris (SSN/TLE) measurement actually arrives; on a
    # non-arrival the debris is predict-only that step. 1.0 = the original
    # guaranteed-measurement behavior (Bernoulli draw skipped, byte-identical).
    p_arrival::Float64
    grid::Union{DecisionGrid,Nothing}
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
                     sigma_mode::Symbol = MCTS_SIGMA_MODE,
                     parallel::Bool = MCTS_PARALLEL,
                     n_workers::Union{Int,Nothing} = MCTS_N_WORKERS,
                     reward_mode::Symbol = MCTS_REWARD_MODE,
                     terminal_penalty::Real = MCTS_TERMINAL_PENALTY,
                     p_arrival::Real = 1.0,
                     grid::Union{DecisionGrid,Nothing} = nothing)
    # On the adaptive grid the max usable depth is the number of grid steps (a
    # rollout reaches TCA there); cap max_depth to it so the depth budget never
    # cuts a rollout short of the terminal epoch it was built to reach.
    md = grid === nothing ? max_depth : min(max_depth, grid_depth_count(grid))
    return MCTSPlanner(pomdp, n_iterations, md, Float64(c),
                       Float64(k), Float64(α), Float64(dt),
                       Float64(cadence_sc), Float64(cadence_debris),
                       constraint_mode, Float64(pc_weight), Float64(pc_penalty),
                       sigma_mode, parallel, n_workers,
                       reward_mode, Float64(terminal_penalty),
                       Float64(p_arrival), grid)
end

"""
    run_sims!(planner, root, rng; n_iterations=planner.n_iterations, table=<built>)
        -> (root, table)

Run `n_iterations` MCTS simulations from `root` (building the fast Σ table once if
`sigma_mode == :fast` and none is supplied), mutating `root` in place. This is the
serial search kernel shared by the single-process `plan` and by each worker of the
root-parallel path — factoring it out keeps the two paths bit-for-bit identical
given the same `rng` state and iteration count. Returns the mutated `root` and the
Σ table used (so a caller can reuse or ship it).
"""
function run_sims!(planner::MCTSPlanner, root::BeliefNode, rng::AbstractRNG;
                   n_iterations::Int = planner.n_iterations,
                   table::Union{SigmaTCATable,Nothing} = nothing)
    # :fast precomputes ONE branch-invariant debris Σ-at-depth table (the whole
    # trick is that the debris Σ schedule is deterministic in depth). Probabilistic
    # arrival (p_arrival < 1) makes each scheduled debris fix a random draw, so the
    # debris Σ becomes genuinely branch-dependent — the table is no longer valid
    # (same failure mode as Phase-8 maneuver noise). Use :exact under p_arrival < 1.
    if planner.sigma_mode == :fast && planner.p_arrival < 1.0
        error("build_sigma_tca_table (:fast) assumes a branch-invariant debris Σ, " *
              "but p_arrival=$(planner.p_arrival) < 1.0 makes debris arrival stochastic " *
              "per node. Use sigma_mode = :exact with probabilistic measurement arrival.")
    end
    # Efficiency pass: under :fast, precompute the branch-invariant debris
    # Σ-at-TCA per depth ONCE (see the FAST Σ PATH section header), then look it
    # up per node instead of re-propagating the debris covariance every node.
    if table === nothing && planner.sigma_mode == :fast
        table = build_sigma_tca_table(planner.pomdp, root; dt = planner.dt,
                                      cadence_sc = planner.cadence_sc,
                                      cadence_debris = planner.cadence_debris,
                                      max_depth = planner.max_depth,
                                      grid = planner.grid)
    end
    for _ in 1:n_iterations
        simulate!(planner.pomdp, root, planner.max_depth, rng;
                  dt = planner.dt, cadence_sc = planner.cadence_sc,
                  cadence_debris = planner.cadence_debris,
                  c = planner.c, k = planner.k, α = planner.α,
                  constraint_mode = planner.constraint_mode,
                  pc_weight = planner.pc_weight, pc_penalty = planner.pc_penalty,
                  reward_mode = planner.reward_mode,
                  terminal_penalty = planner.terminal_penalty,
                  p_arrival = planner.p_arrival,
                  node_depth = 0, table = table, sigma_mode = planner.sigma_mode,
                  grid = planner.grid)
    end
    return root, table
end

"""
    best_action_by_q(actions, Qa) -> best_action

Pick the action with the highest per-action value `Qa` (the MCTS decision rule).
An action absent from `Qa` scores `-Inf`. Shared by the serial and merged
(root-parallel) decisions so both decide identically from a `Qa` dict.
"""
function best_action_by_q(actions::AbstractVector{CAAction}, Qa::AbstractDict)
    best_a = actions[1]
    best_q = -Inf
    for a in actions
        q = get(Qa, a, -Inf)
        if q > best_q
            best_q = q
            best_a = a
        end
    end
    return best_a
end

"""
    plan(planner, root::BeliefNode, rng) -> (best_action, root)

Run the planner's simulations from `root` and return the best action by
per-action value `Qa`. The mutated `root` (with visit/value stats and the built
tree, including per-node Pc / violation instrumentation) is returned too for
inspection/testing. When `planner.parallel` is set, dispatches to the
root-parallel multiprocess path (`plan_parallel`) instead — see that function and
the ROOT-PARALLEL MCTS section header. `rng` must be seeded by the caller for a
reproducible result (the parallel path derives its per-worker substreams from it).
"""
function plan(planner::MCTSPlanner, root::BeliefNode, rng::AbstractRNG)
    if planner.parallel
        return plan_parallel(planner, root, rng)
    end
    run_sims!(planner, root, rng)
    return best_action_by_q(POMDPs.actions(planner.pomdp), root.Qa), root
end

# =========================================================================
# ROOT-PARALLEL MCTS (multiprocess; efficiency pass, 2026-07-23)
#
# WHY MULTIPROCESS, NOT THREADS: ~95% of per-node cost is the brahe propagation,
# which runs through PyCall and holds the Python GIL — in-process @threads
# SERIALIZE at the brahe call (no speedup) and multi-thread PyCall is not reliably
# safe (segfault/corruption; the from_orbits port hit this). So we split the
# simulation budget across N Julia WORKER PROCESSES (Distributed), each with its
# OWN Python/brahe interpreter (get_brahe() lazy-inits one per process — no GIL
# contention, PyCall stays single-threaded per worker).
#
# ROOT PARALLELIZATION (the standard, sound-for-pick-best-by-Q scheme): every
# worker builds an INDEPENDENT tree from a COPY of the same root, runs its share of
# the budget (n_iterations split as evenly as possible), and reports its per-action
# visit counts Na and values Qa. We then MERGE across workers with an Na-WEIGHTED
# running-average of Q (`merge_action_stats`) — i.e. the merged Q(a) is the same
# number a single serial run of the combined budget would converge to for that
# action's mean return, so picking argmax_a Q(a) on the merge is the intended
# decision. (Root parallelization is only valid for the ROOT decision — it does not
# share sub-tree statistics, which is exactly what we want here: we only need the
# top-level action.)
#
# DETERMINISM: a fixed master seed must give a reproducible MERGED result regardless
# of which worker runs which chunk or in what order they finish. We achieve this by
# deriving a DISTINCT, WORKER-INDEXED substream seed from the master seed
# (`worker_seeds`) — chunk i always uses the same seed and the same iteration count,
# so its (Na, Qa) is fixed; and the Na-weighted merge is order-independent when
# combined pairwise from a fixed chunk ordering (we fold chunks in index order, not
# completion order). Result: same master seed + same chunking ⇒ identical merged Qa
# and identical decision, run to run.
#
# ⚠️ REQUIRES WORKERS THAT HAVE LOADED THIS CODE. The caller must have added
# Distributed workers and `@everywhere include(...)`d the project (so each worker
# has SpacecraftCAPOMDP + beliefMCTS + can build its own brahe). If no extra workers
# are attached, `plan_parallel` runs the whole budget locally (still correct, just
# not parallel) and says so. The once-per-plan Σ table is built ONCE on the
# coordinator and shipped to the workers (it is deterministic in the root, so this
# only saves recompute — it does not change results).
# =========================================================================

"""
    worker_seeds(master_seed, n_chunks) -> Vector{UInt}

Derive `n_chunks` distinct, reproducible per-chunk RNG seeds from a single
`master_seed`. Chunk `i` always gets the same seed for a given `master_seed`, so
the parallel search is reproducible regardless of worker assignment or completion
order. Uses a `MersenneTwister(master_seed)`-driven draw of independent `UInt`
seeds (a simple, adequate substream scheme for independent trees — each chunk then
seeds its own `MersenneTwister`).
"""
function worker_seeds(master_seed::Integer, n_chunks::Int)
    seed_rng = MersenneTwister(UInt(master_seed))
    return UInt[rand(seed_rng, UInt) for _ in 1:n_chunks]
end

"""
    split_iterations(n_iterations, n_chunks) -> Vector{Int}

Split `n_iterations` across `n_chunks` as evenly as possible (the first
`n_iterations % n_chunks` chunks get one extra). Deterministic in the inputs, so
the per-chunk budget — and hence each chunk's result — is fixed run to run.
"""
function split_iterations(n_iterations::Int, n_chunks::Int)
    n_chunks <= 1 && return [n_iterations]
    base = div(n_iterations, n_chunks)
    rem  = mod(n_iterations, n_chunks)
    return [base + (i <= rem ? 1 : 0) for i in 1:n_chunks]
end

"""
    merge_action_stats(stats) -> (Na_total, Qa_merged)

Merge a vector of per-chunk `(Na, Qa)` action-statistics into a single set, using
the Na-WEIGHTED running-average combine (standard root-parallel MCTS): for each
action `a`,
    Na_total(a) = Σ_i Na_i(a)
    Qa_merged(a) = ( Σ_i Na_i(a) · Qa_i(a) ) / Na_total(a)
i.e. the visit-count-weighted mean of the per-chunk Q's, which equals the mean
return a single serial run of the combined budget would have accumulated for that
action. Chunks are folded in the given (fixed, index) order so the result is
order-independent and reproducible. Actions with zero total visits are omitted
(they carry no value estimate), matching a serial run that never took them.
"""
function merge_action_stats(stats::AbstractVector)
    Na_total = Dict{CAAction,Int}()
    num      = Dict{CAAction,Float64}()   # Σ Na·Q accumulator per action
    for (Na, Qa) in stats
        for (a, na) in Na
            na == 0 && continue
            Na_total[a] = get(Na_total, a, 0) + na
            num[a]      = get(num, a, 0.0) + na * Qa[a]
        end
    end
    Qa_merged = Dict{CAAction,Float64}()
    for (a, na) in Na_total
        Qa_merged[a] = num[a] / na
    end
    return Na_total, Qa_merged
end

"""
    _run_chunk(planner, root, n_iter, seed, table) -> (Na, Qa)

Run one worker's share of the budget: `deepcopy` the root (so the worker mutates
its OWN independent tree — critical when this runs on a remote process, and
harmless locally), seed a fresh `MersenneTwister(seed)`, run `n_iter` sims via the
shared `run_sims!` kernel, and return only the root's per-action `(Na, Qa)` (the
merge needs nothing else, and small dicts serialize cheaply back to the
coordinator). The `table` is built once on the coordinator and passed in.
"""
function _run_chunk(planner::MCTSPlanner, root::BeliefNode, n_iter::Int,
                    seed::UInt, table::Union{SigmaTCATable,Nothing})
    local_root = deepcopy(root)
    rng = MersenneTwister(seed)
    run_sims!(planner, local_root, rng; n_iterations = n_iter, table = table)
    return (copy(local_root.Na), copy(local_root.Qa))
end

"""
    plan_parallel(planner, root, rng) -> (best_action, root)

Root-parallel multiprocess `plan` (see the ROOT-PARALLEL MCTS section header). Uses
`rng` only to draw a master seed, from which per-chunk substream seeds are derived
(`worker_seeds`) so the merged result is reproducible under a fixed input `rng`.
Splits `n_iterations` across the available Distributed workers (or `planner.
n_workers` if set, capped at the workers actually attached), builds the fast Σ
table once on the coordinator, dispatches one chunk per worker with `remotecall`,
merges the returned per-action `(Na, Qa)` with the Na-weighted combine, writes the
merged stats back onto `root`, and returns the best action by merged Q.

Falls back to running the whole budget locally (still correct) when no extra
workers are attached — so the flag can be flipped on without a Distributed cluster
and simply not parallelize.
"""
function plan_parallel(planner::MCTSPlanner, root::BeliefNode, rng::AbstractRNG)
    # :fast is invalid under probabilistic arrival (stochastic per-branch debris Σ);
    # guard here too since plan_parallel builds the table directly (see run_sims!).
    if planner.sigma_mode == :fast && planner.p_arrival < 1.0
        error("sigma_mode = :fast is invalid with p_arrival = $(planner.p_arrival) < 1.0 " *
              "(stochastic debris arrival breaks the branch-invariant Σ table). Use :exact.")
    end
    master_seed = rand(rng, UInt)

    # available worker processes (procs other than the coordinator). If the caller
    # capped n_workers, honor it but never exceed the workers actually attached.
    avail = workers()
    have_workers = !(length(avail) == 1 && avail[1] == myid())   # workers() is [1] when none added
    pool = have_workers ? avail : Int[]
    n_pool = length(pool)
    n_chunks = planner.n_workers === nothing ? max(n_pool, 1) :
               (n_pool == 0 ? 1 : min(planner.n_workers, n_pool))
    n_chunks = max(n_chunks, 1)

    seeds  = worker_seeds(master_seed, n_chunks)
    iters  = split_iterations(planner.n_iterations, n_chunks)

    # Build the deterministic Σ table ONCE on the coordinator and ship it to the
    # workers (it is a pure function of the root, so this only avoids recompute).
    table = planner.sigma_mode == :fast ?
            build_sigma_tca_table(planner.pomdp, root; dt = planner.dt,
                                  cadence_sc = planner.cadence_sc,
                                  cadence_debris = planner.cadence_debris,
                                  max_depth = planner.max_depth,
                                  grid = planner.grid) : nothing

    local stats::Vector{Any}
    if n_pool == 0
        # No cluster attached: run every chunk locally, in fixed index order. Still
        # reproducible and correct — just not parallel.
        stats = [_run_chunk(planner, root, iters[i], seeds[i], table)
                 for i in 1:n_chunks]
    else
        # One chunk per worker (round-robin if more chunks than workers). Dispatch
        # all, then collect in FIXED index order (not completion order) so the merge
        # is reproducible regardless of which worker finishes first.
        futures = Vector{Future}(undef, n_chunks)
        for i in 1:n_chunks
            w = pool[mod1(i, n_pool)]
            futures[i] = remotecall(_run_chunk, w, planner, root,
                                    iters[i], seeds[i], table)
        end
        stats = [fetch(futures[i]) for i in 1:n_chunks]
    end

    Na_total, Qa_merged = merge_action_stats(stats)
    # Write the merged decision statistics back onto the passed root so callers /
    # tests can inspect them exactly as with a serial plan. (The per-node tree
    # itself lives on the worker copies and is intentionally not merged — root
    # parallelization shares only the top-level action statistics.)
    root.Na = Na_total
    root.Qa = Qa_merged
    root.N  = sum(values(Na_total); init = 0)
    return best_action_by_q(POMDPs.actions(planner.pomdp), Qa_merged), root
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
