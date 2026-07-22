# =========================================================================
# beliefMCTS.jl — Phase 5: baseline belief-space MCTS skeleton
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
#   5–6. Evaluate the reward from the resulting (belief, true state). In Phase 5
#      this is MISS-DISTANCE-ONLY (no Pc, no chance constraint — that is Phase 6).
#   7. BACK UP the return as a running average per node/action.
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
# CONSTANTS (all in CONSTANTS.md, "Belief MCTS (Phase 5)" section): the UCB
# exploration constant c, widening k_o / α_o, simulation budget (tree_queries),
# and max_depth all take POMCPOW's published defaults as a starting point and
# are flagged TODO: needs source (they are tuning knobs, not physical values).
#
# dt: swappable (defaults pomdp.dt) — NEVER hard-coded (Phase 3 convention).
# Phase 5 is where the planner finally consumes dt; both the 1-hr (24-step) and
# 30-min (48-step) grids are meant to be run and compared (Grace's call).
#
# REWARD IS MISS-DISTANCE-ONLY here. Pc / the chance constraint (architecture §4
# step 6, §7 leaf value on Pc) is deliberately NOT wired in — that is Phase 6.
# §7's leaf/cutoff structure (true leaf at TCA vs. computational-budget cutoff)
# IS honored, but valued by miss distance for now.
# =========================================================================

using LinearAlgebra
using Random

# --- Phase 5 tuning constants (see CONSTANTS.md) -------------------------------
const MCTS_UCB_C        = 1.0      # POMCPOW MaxUCB default
const MCTS_K_OBS        = 10.0     # POMCPOW k_observation default
const MCTS_ALPHA_OBS    = 0.5      # POMCPOW alpha_observation default
const MCTS_TREE_QUERIES = 1000     # POMCPOW tree_queries default (sim budget)
const MCTS_MAX_DEPTH    = 24       # 24-hr window / 1 decision-per-step cap

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
end

function BeliefNode(belief::Belief, s_true::CAState, is_terminal::Bool)
    return BeliefNode(belief, s_true, 0,
                      Dict{CAAction,Int}(), Dict{CAAction,Float64}(),
                      Dict{CAAction,Vector{BeliefNode}}(), is_terminal)
end

# =========================================================================
# Reward — MISS-DISTANCE-ONLY (Phase 5). No Pc (that is Phase 6).
#
# At a terminal/leaf node (reached TCA, or a collision, or the budget cutoff),
# the value is driven by the miss distance at that node: farther apart is
# better. Every MANEUVER along the way costs `maneuver_cost`. This is the
# single risk metric for the mechanics checkpoint; Phase 6 replaces the
# miss-distance leaf value with Pc-at-TCA (architecture §7) and adds the
# per-step chance-constraint check (§4 step 6).
# =========================================================================

"""
    miss_distance(s::CAState) -> Float64

Relative position magnitude (m) between spacecraft and debris (RTN position
block, same convention as `get_x_rel`). This is the Phase-5 risk metric.
"""
miss_distance(s::CAState) = norm(get_x_rel(s)[1:3])

"""
    step_reward(pomdp, s, a, sp) -> Float64

Miss-distance-only step reward for the transition s --a--> sp. A MANEUVER costs
`maneuver_cost`. At a terminal `sp` (TCA reached or collision) the miss distance
is scored: a collision (miss < combined hard-body radius) is a large penalty;
otherwise the reward rewards larger miss distance (scaled to keep it O(cost)).
Non-terminal steps only carry the maneuver cost — the miss-distance payoff lands
at the leaf, matching architecture §7 (leaf/cutoff value).
"""
function step_reward(pomdp::SpacecraftCAPOMDP, s::CAState, a::CAAction, sp::CAState)
    r = 0.0
    if a == MANEUVER
        r -= pomdp.maneuver_cost
    end
    if isterminal(pomdp, sp)
        md = miss_distance(sp)
        R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
        if md < R_combined
            r -= 1000.0                     # collision — large penalty
        else
            r += md / 1000.0                # reward miss distance (km scale)
        end
    end
    return r
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
    expand_child(pomdp, node, a, rng; dt=pomdp.dt) -> (child::BeliefNode, r::Float64)

Take action `a` from `node`: propagate the sampled true state (Phase 0
`POMDPs.transition`), predict the belief (Phase 4 `predict`), sample a genuine
observation of the new true state (`sample_observation`), and correct the belief
(`correct_linear`, the runtime path). Returns the new observation-child node and
the miss-distance step reward. `dt` is swappable (defaults pomdp.dt).
"""
function expand_child(pomdp::SpacecraftCAPOMDP, node::BeliefNode, a::CAAction,
                      rng::AbstractRNG; dt::Real = pomdp.dt)
    # 1. propagate the sampled true state (Phase 0 dynamics)
    sp = rand(rng, POMDPs.transition(pomdp, node.s_true, a))

    # 2. predict the belief forward one dt step (Phase 4)
    b_pred = predict(pomdp, node.belief, a; dt = dt)

    # 3–4. sample a real observation of sp, then correct (runtime path)
    z = sample_observation(pomdp, a, sp, rng)
    b_post = correct_linear(pomdp, b_pred, z)

    child = BeliefNode(b_post, sp, isterminal(pomdp, sp))
    r = step_reward(pomdp, node.s_true, a, sp)
    return child, r
end

# =========================================================================
# leaf / cutoff value (architecture §7).
#   True leaf (reached TCA / collision): score by the node's own miss distance.
#   Cutoff (depth budget hit before TCA): reuse the SAME miss-distance metric on
#   the current node's sampled true state — no separate heuristic.
# (Phase 6 replaces both with Pc-at-TCA.)
# =========================================================================

"""
    leaf_value(pomdp, node) -> Float64

Value estimate for a node the rollout does not expand past — either a true leaf
(TCA/collision) or a computational-budget cutoff. Both use the miss-distance
metric on the node's sampled true state (architecture §7; Phase 6 swaps in Pc).
"""
function leaf_value(pomdp::SpacecraftCAPOMDP, node::BeliefNode)
    md = miss_distance(node.s_true)
    R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    return md < R_combined ? -1000.0 : md / 1000.0
end

# =========================================================================
# One MCTS simulation (recursive): selection → expansion (with widening) →
# recurse → backup as a running average.  Returns the return from this node.
# =========================================================================

"""
    simulate!(pomdp, node, depth, rng; dt=pomdp.dt, c=..., k=..., α=...) -> Float64

Run one MCTS simulation from `node` (architecture §4 steps 1–7). Selects an
action by UCB, adds or reuses an observation-child under progressive widening,
recurses to `depth-1`, and backs up a running-average value into the node's
per-action stats. Returns the (discounted) return seen from this node. Terminal
nodes and the depth==0 cutoff return `leaf_value`.
"""
function simulate!(pomdp::SpacecraftCAPOMDP, node::BeliefNode, depth::Int,
                   rng::AbstractRNG; dt::Real = pomdp.dt,
                   c::Real = MCTS_UCB_C, k::Real = MCTS_K_OBS, α::Real = MCTS_ALPHA_OBS)
    if node.is_terminal || depth <= 0
        return leaf_value(pomdp, node)
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
        child, r = expand_child(pomdp, node, a, rng; dt = dt)
        push!(kids, child)
    else
        child = rand(rng, kids)
        # step reward for the reused edge (miss-distance-only, deterministic in a)
        r = step_reward(pomdp, node.s_true, a, child.s_true)
    end

    q = r + POMDPs.discount(pomdp) * simulate!(pomdp, child, depth - 1, rng;
                                               dt = dt, c = c, k = k, α = α)

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
    MCTSPlanner(pomdp; n_iterations, max_depth, c, k, α, dt)

Configuration for the Phase-5 baseline belief-space MCTS planner. `dt` defaults
to `pomdp.dt` and is swappable (run both grids to compare). All search knobs
default to POMCPOW's published constants (see CONSTANTS.md).
"""
struct MCTSPlanner
    pomdp::SpacecraftCAPOMDP
    n_iterations::Int
    max_depth::Int
    c::Float64
    k::Float64
    α::Float64
    dt::Float64
end

function MCTSPlanner(pomdp::SpacecraftCAPOMDP;
                     n_iterations::Int = MCTS_TREE_QUERIES,
                     max_depth::Int = MCTS_MAX_DEPTH,
                     c::Real = MCTS_UCB_C,
                     k::Real = MCTS_K_OBS,
                     α::Real = MCTS_ALPHA_OBS,
                     dt::Real = pomdp.dt)
    return MCTSPlanner(pomdp, n_iterations, max_depth, Float64(c),
                       Float64(k), Float64(α), Float64(dt))
end

"""
    plan(planner, root::BeliefNode, rng) -> (best_action, root)

Run the planner's simulations from `root` and return the best action by
per-action value `Qa`. The mutated `root` (with visit/value stats and the built
tree) is returned too for inspection/testing.
"""
function plan(planner::MCTSPlanner, root::BeliefNode, rng::AbstractRNG)
    for _ in 1:planner.n_iterations
        simulate!(planner.pomdp, root, planner.max_depth, rng;
                  dt = planner.dt, c = planner.c, k = planner.k, α = planner.α)
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
