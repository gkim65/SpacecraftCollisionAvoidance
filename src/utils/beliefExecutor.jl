# =========================================================================
# beliefExecutor.jl — Phase 6.5: outer closed-loop / receding-horizon (MPC)
# episode driver.
#
# The planner (`plan`, beliefMCTS.jl) returns ONE action for the CURRENT step
# after running many now→TCA belief rollouts. It does NOT advance the real
# world. This driver is the outer loop that wraps it (architecture doc §4 — one
# simulated step becomes one EXECUTED step; §7 — the executor is the real-world
# loop around the per-decision plan):
#
#   repeat until TCA or collision (isterminal):
#     1. PLAN from the current belief + true state → get an action a.
#     2. EXECUTE a: advance the TRUE state one dt step via POMDPs.transition.
#     3. UPDATE the belief: predict → sample z → cadence-aware correct, via the
#        SHARED `step_belief` (beliefMCTS.jl) — the exact same per-step belief
#        update `expand_child` uses inside the planner, so the belief actually
#        tracked in execution matches the belief the planner assumed internally
#        (they cannot drift — the whole correctness point of MPC here). The
#        cadence timers (since_sc/since_debris) and correct_at_root phase carry
#        through identically.
#     4. RE-PLAN from the updated belief + advanced true state. Repeat.
#
# Each iteration logs a per-step trace row (step index, time-remaining, action,
# Δv spent that step, Pc-at-TCA at that step, miss distance). The full trace is
# returned so a single executed episode can be inspected / plotted.
#
# SCOPE (Phase 6.5): a SINGLE-episode driver on ONE fixture — NOT a multi-seed
# statistics harness (that is Phase 10). All planner knobs (sigma_mode, parallel,
# constraint_mode, dt, cadences, budget) pass straight through the MCTSPlanner,
# so the executor is agnostic to them.
#
# dt CONSISTENCY (load-bearing): POMDPs.transition advances the true state by
# `pomdp.dt` (it does not take a swappable dt). So the executed world steps by
# pomdp.dt, and the planner's dt MUST equal pomdp.dt or the planner would be
# optimizing over a step size the world doesn't actually take. `run_episode`
# asserts planner.dt == pomdp.dt.
#
# KNOWN GAP THIS EXPOSES (Phase 7, NOT built here): the reward is −Pc − fuel with
# no nominal-trajectory-deviation term. A closed-loop policy under that reward
# can over-maneuver / drift with nothing penalizing motion away from the nominal
# once Pc is already low. The trace makes that visible (watch the maneuver count
# after Pc is driven safe). Motivates Phase 7 — do not fix it here.
# =========================================================================

using Random

"""
    ExecStep

One executed decision step of a closed-loop episode. `step` is the 1-based step
index; `t_remaining` is the time-to-TCA (s) at the START of the step (i.e. the
belief the plan was made from); `action` is the executed action; `Δv` is the
delta-v spent THIS step (0.0 on WAIT, `pomdp.Δv` on MANEUVER); `pc` is the
planner's Pc-at-TCA evaluated from the post-step belief; `miss` is the true
relative-position miss distance (m) at the post-step true state.
"""
struct ExecStep
    step::Int
    t_remaining::Float64
    action::CAAction
    Δv::Float64
    pc::Float64
    miss::Float64
end

"""
    run_episode(planner, pomdp, s0, rng; correct_at_root=pomdp.correct_at_root,
                max_steps=planner.max_depth, verbose=false) -> Vector{ExecStep}

Run one closed-loop (receding-horizon / MPC) episode from true state `s0`:
plan → execute → advance the true state → update the belief → re-plan, until TCA
or collision (`isterminal`) or `max_steps` is hit. Returns the per-step trace.

The belief is tracked across steps with the SAME `step_belief` update the planner
uses internally (predict → sample z → cadence-aware correct), including the
per-object cadence timers and the `correct_at_root` phase, so the executed belief
is consistent with what each `plan` call assumed. A fresh root node is built each
step from the CURRENT belief + true state, and one `plan` call decides that step's
action. All planner knobs (sigma_mode / parallel / constraint_mode / dt /
cadences) are used as configured on `planner`.

`rng` seeds both the per-step planning and the real observation draws; pass a
seeded RNG for a reproducible episode. `pomdp.dt` and `planner.dt` must match
(asserted) — the true state advances by `pomdp.dt` via `POMDPs.transition`.
"""
function run_episode(planner::MCTSPlanner, pomdp::SpacecraftCAPOMDP, s0::CAState,
                     rng::AbstractRNG;
                     correct_at_root::Bool = pomdp.correct_at_root,
                     max_steps::Int = planner.max_depth,
                     grid_builder = nothing,
                     verbose::Bool = false)
    # dt-consistency: on the FIXED grid the true state advances by pomdp.dt, so the
    # planner must simulate that same step (asserted). On the ADAPTIVE grid the step
    # is the grid's first epoch gap and the truth is advanced by exactly that (via
    # transition_dt), so the fixed-dt equality does not apply — skip the assert.
    adaptive = planner.grid !== nothing || grid_builder !== nothing
    if !adaptive
        @assert planner.dt == pomdp.dt "planner.dt ($(planner.dt)) must equal pomdp.dt "*
            "($(pomdp.dt)): POMDPs.transition advances the true state by pomdp.dt, so the "*
            "planner must simulate the same step size the executed world takes."
    end

    trace = ExecStep[]

    # Current true state + tracked belief. The belief starts at P0 anchored on the
    # true state (root_from_pomdp / belief_from_pomdp), and the cadence timers start
    # per correct_at_root — exactly as the planner's root does.
    s_true       = s0
    belief       = belief_from_pomdp(pomdp, s0.sc_eci, s0.debris_eci, s0.t)
    since_sc     = correct_at_root ? 0.0 : pomdp.cadence_sc
    since_debris = correct_at_root ? 0.0 : pomdp.cadence_debris

    step = 0
    while !isterminal(pomdp, s_true) && step < max_steps
        step += 1
        t_remaining = s_true.t

        # ADAPTIVE GRID: rebuild the grid from the CURRENT time-to-TCA each step, so
        # it adapts as the episode marches down the horizon (receding-horizon /
        # per-step grid). `grid_builder(t)` returns a DecisionGrid for the remaining
        # horizon; else reuse the planner's fixed grid. The step actually EXECUTED
        # is the grid's FIRST epoch gap (its first decision), keeping the executed
        # world on the same timeline the planner optimized over.
        step_planner = planner
        if grid_builder !== nothing
            g = grid_builder(t_remaining)
            step_planner = MCTSPlanner(pomdp;
                n_iterations = planner.n_iterations, max_depth = planner.max_depth,
                c = planner.c, k = planner.k, α = planner.α, dt = planner.dt,
                cadence_sc = planner.cadence_sc, cadence_debris = planner.cadence_debris,
                constraint_mode = planner.constraint_mode, pc_weight = planner.pc_weight,
                pc_penalty = planner.pc_penalty, sigma_mode = planner.sigma_mode,
                parallel = planner.parallel, n_workers = planner.n_workers,
                reward_mode = planner.reward_mode, terminal_penalty = planner.terminal_penalty,
                grid = g)
        end
        exec_grid = step_planner.grid

        # 1. PLAN from the current belief + true state. Build a fresh root node
        #    carrying the current belief, true state, and cadence-timer phase, then
        #    run one plan. (The planner re-derives its own timers from the node.)
        root = BeliefNode(belief, s_true, isterminal(pomdp, s_true);
                          since_sc = since_sc, since_debris = since_debris)
        a, _ = plan(step_planner, root, rng)

        # 2. EXECUTE: advance the TRUE state. Fixed grid ⇒ pomdp.dt; adaptive grid
        #    ⇒ the grid's first epoch gap (the step the plan actually decided).
        exec_dt = exec_grid === nothing ? pomdp.dt : exec_grid.dts[1]
        sp = rand(rng, transition_dt(pomdp, s_true, a, exec_dt))

        # 3. UPDATE the belief with the SHARED per-step update (predict → sample z
        #    → cadence-aware correct) — identical to what the planner simulates. On
        #    the adaptive grid pass the grid + grid_depth=1 so the correction schedule
        #    matches the planner's first step exactly.
        belief, since_sc, since_debris = step_belief(pomdp, belief, a, sp, rng;
                                                     dt = exec_dt,
                                                     cadence_sc = pomdp.cadence_sc,
                                                     cadence_debris = pomdp.cadence_debris,
                                                     since_sc = since_sc,
                                                     since_debris = since_debris,
                                                     grid = exec_grid,
                                                     grid_depth = exec_grid === nothing ? nothing : 1)

        # 4. LOG the step. Pc-at-TCA is evaluated from the post-step belief via the
        #    planner's own node_pc (exact per-node — the trace is not the hot loop,
        #    so use the exact path regardless of sigma_mode for a clean number).
        post = BeliefNode(belief, sp, isterminal(pomdp, sp);
                          since_sc = since_sc, since_debris = since_debris)
        pc   = node_pc_at_tca(pomdp, post)
        Δv   = a == MANEUVER ? pomdp.Δv : 0.0
        miss = miss_distance(sp)
        push!(trace, ExecStep(step, t_remaining, a, Δv, pc, miss))

        if verbose
            @info "executed step" step=step t_h=round(t_remaining/3600, digits=2) action=a Δv=Δv pc=pc miss_km=round(miss/1000, digits=3)
        end

        # advance the true state; re-plan from here next iteration.
        s_true = sp
    end
    return trace
end

"""
    episode_summary(trace) -> NamedTuple

Roll a per-step `trace` up into episode-level numbers: total Δv spent, number of
MANEUVER steps, the final (closest-to-TCA) Pc and miss distance, and the peak Pc
seen over the episode. Convenience for inspecting / comparing single episodes;
per-episode metric aggregation across many seeds is Phase 10, not here.
"""
function episode_summary(trace::AbstractVector{ExecStep})
    isempty(trace) && return (; n_steps = 0, total_Δv = 0.0, n_maneuvers = 0,
                              final_pc = NaN, final_miss = NaN, peak_pc = NaN)
    return (; n_steps      = length(trace),
            total_Δv     = sum(s.Δv for s in trace),
            n_maneuvers  = count(s -> s.action == MANEUVER, trace),
            final_pc     = trace[end].pc,
            final_miss   = trace[end].miss,
            peak_pc      = maximum(s.pc for s in trace))
end
