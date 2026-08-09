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

# =========================================================================
# WANDB-LOGGABLE PER-EPISODE METRICS (config in → flat Dict out)
#
# A cluster sweep varies a CONFIG (case, cadence, sensor quality, seed, …), runs
# ONE receding-horizon episode, and logs ONE dict of results — `wandb.log(dict)`.
# `run_episode_metrics` is that config-in/metrics-out wrapper: it loads the CDM
# scenario, builds the planner + a per-step ADAPTIVE decision grid, runs
# `run_episode`, and rolls the trace into a single JSON-serializable `Dict{String}`
# whose values are all plain scalars / strings / flat vectors / list-of-dicts (no
# Julia structs, enums, or matrices) so `JSON.print` and a later `wandb.log` are
# trivial. Every knob is echoed back into the dict so a logged run is
# self-describing.  We do NOT wire the wandb client here (no network / no dep in
# the harness) — see the README note on how a cluster script consumes this dict.
# =========================================================================

"""
    episode_config(; case_path, cadence_secondary=nothing, coarse=8*3600,
                   fine=nothing, sensor_quality=:median, seed=20240809,
                   sigma_mode=:exact, parallel=false, n_iterations=12,
                   reward_mode=:terminal, constraint_mode=:penalize, k=2.0,
                   truncate_safe=false, sec_class_override=nothing,
                   dt=60*60, t_horizon=nothing, pc_threshold=nothing,
                   max_steps=nothing) -> Dict{String,Any}

Build a default episode CONFIG dict (the input a sweep varies). Every field maps
to a knob `run_episode_metrics` consumes; a sweep script copies this, overrides
what it sweeps, and passes it back in. `case_path` is required (path to a CDM).
`cadence_secondary`/`fine` default to the loaded case's tiered secondary cadence
when `nothing`.
"""
function episode_config(; case_path::AbstractString,
                        cadence_secondary::Union{Real,Nothing} = nothing,
                        coarse::Real = 8 * 3600,
                        fine::Union{Real,Nothing} = nothing,
                        sensor_quality::Symbol = :median,
                        seed::Integer = 20240809,
                        sigma_mode::Symbol = :exact,
                        parallel::Bool = false,
                        n_iterations::Integer = 12,
                        reward_mode::Symbol = :terminal,
                        constraint_mode::Symbol = :penalize,
                        k::Real = 2.0,
                        truncate_safe::Bool = false,
                        sec_class_override::Union{Symbol,Nothing} = nothing,
                        dt::Real = 60 * 60,
                        t_horizon::Union{Real,Nothing} = nothing,
                        pc_threshold::Union{Real,Nothing} = nothing,
                        max_steps::Union{Integer,Nothing} = nothing,
                        verbose::Bool = false)
    return Dict{String,Any}(
        "case_path"         => String(case_path),
        "cadence_secondary" => cadence_secondary,
        "coarse"            => Float64(coarse),
        "fine"              => fine,
        "sensor_quality"    => String(sensor_quality),
        "seed"              => Int(seed),
        "sigma_mode"        => String(sigma_mode),
        "parallel"          => parallel,
        "n_iterations"      => Int(n_iterations),
        "reward_mode"       => String(reward_mode),
        "constraint_mode"   => String(constraint_mode),
        "k"                 => Float64(k),
        "truncate_safe"     => truncate_safe,
        "sec_class_override" => sec_class_override === nothing ? nothing : String(sec_class_override),
        "dt"                => Float64(dt),
        "t_horizon"         => t_horizon,
        "pc_threshold"      => pc_threshold,
        "max_steps"         => max_steps,
        "verbose"           => verbose,
    )
end

# small helpers: read a config value with a default; symbolize a stored string.
_cfg(cfg, key, default) = (v = get(cfg, key, nothing); v === nothing ? default : v)
_sym(v) = v isa Symbol ? v : Symbol(v)

"""
    run_episode_metrics(cfg::AbstractDict) -> Dict{String,Any}

Run ONE clean receding-horizon episode described by `cfg` (see `episode_config`)
and return a flat, JSON-serializable metrics dict ready for `wandb.log`. Steps:

 1. Load the CDM scenario (`load_cdm_scenario`) with the config's sensor quality /
    class override / horizon / pc_threshold.
 2. Build a per-step ADAPTIVE grid builder (`adaptive_decision_grid` at the config
    cadence) — rebuilt each executed step from the current time-to-TCA (receding
    horizon), so the grid adapts as the episode marches down.
 3. Run `run_episode` at `:exact` (or the configured `sigma_mode`) with a seeded
    RNG.
 4. Roll the trace up into metrics AND compute the no-maneuver WAIT-spine
    feasibility curve (`wait_spine_pc` from the root belief) — the "was deferral
    feasible" ground truth the decision is scored against.

The returned dict carries: the echoed CONFIG (`config` sub-dict), scenario
PROVENANCE (names / ids / class / horizon / real HBR / CARA Pc / validity flag),
the core METRICS (resolved-without-maneuver, peak/integrated/at-TCA Pc, total Δv,
maneuver count + timings, final miss), the DECISION-VS-FEASIBILITY verdict, the
full per-step TRACE (list-of-dicts), and the WAIT-spine feasibility curve. All
values are scalars / strings / flat numeric vectors / list-of-flat-dicts.
"""
function run_episode_metrics(cfg::AbstractDict)
    t_wall = @elapsed begin
        case_path      = _cfg(cfg, "case_path", nothing)
        case_path === nothing && error("run_episode_metrics: config needs a \"case_path\".")
        sensor_quality = _sym(_cfg(cfg, "sensor_quality", :median))
        sec_override_s = _cfg(cfg, "sec_class_override", nothing)
        sec_override   = sec_override_s === nothing ? nothing : _sym(sec_override_s)
        dt             = Float64(_cfg(cfg, "dt", 60 * 60))
        t_horizon_cfg  = _cfg(cfg, "t_horizon", nothing)
        pc_threshold   = _cfg(cfg, "pc_threshold", nothing)
        seed           = Int(_cfg(cfg, "seed", 20240809))
        sigma_mode     = _sym(_cfg(cfg, "sigma_mode", :exact))
        parallel       = Bool(_cfg(cfg, "parallel", false))
        n_iterations   = Int(_cfg(cfg, "n_iterations", 12))
        reward_mode    = _sym(_cfg(cfg, "reward_mode", :terminal))
        constraint_mode = _sym(_cfg(cfg, "constraint_mode", :penalize))
        k              = Float64(_cfg(cfg, "k", 2.0))
        coarse         = Float64(_cfg(cfg, "coarse", 8 * 3600))
        truncate_safe  = Bool(_cfg(cfg, "truncate_safe", false))

        # 1. load the scenario. Only pass pc_threshold / t_horizon overrides when set.
        loader_kwargs = Dict{Symbol,Any}(:dt => dt, :sensor_quality => sensor_quality)
        sec_override !== nothing && (loader_kwargs[:sec_class_override] = sec_override)
        t_horizon_cfg !== nothing && (loader_kwargs[:t_horizon] = Float64(t_horizon_cfg))
        pc_threshold  !== nothing && (loader_kwargs[:pc_threshold] = Float64(pc_threshold))
        sc = load_cdm_scenario(case_path; loader_kwargs...)
        pomdp = sc.pomdp

        # secondary cadence: config override else the loaded tier value.
        cad_cfg = _cfg(cfg, "cadence_secondary", nothing)
        sec_cad = cad_cfg === nothing ? pomdp.cadence_debris : Float64(cad_cfg)
        fine_cfg = _cfg(cfg, "fine", nothing)
        fine = fine_cfg === nothing ? sec_cad : Float64(fine_cfg)

        # 2. per-step adaptive grid builder (receding horizon). Rebuilt from the
        #    CURRENT time-to-TCA each executed step by run_episode.
        grid_builder = t -> begin
            g, _ = adaptive_decision_grid(pomdp, sc.b0, sc.s_true, t;
                                          cadence_secondary = sec_cad,
                                          coarse = coarse, fine = fine,
                                          truncate_safe = truncate_safe)
            g
        end

        # depth cap = the root grid's step count (a rollout reaches TCA there); a
        # config `max_steps` can shorten it further.
        root_grid, crossing_t = adaptive_decision_grid(pomdp, sc.b0, sc.s_true, sc.t_horizon;
                                    cadence_secondary = sec_cad, coarse = coarse,
                                    fine = fine, truncate_safe = truncate_safe)
        grid_steps = grid_depth_count(root_grid)
        ms_cfg = _cfg(cfg, "max_steps", nothing)
        max_steps = ms_cfg === nothing ? grid_steps : min(Int(ms_cfg), grid_steps)

        # 3. run one episode. The planner carries the ROOT grid (its max_depth caps
        #    to grid steps); run_episode rebuilds the grid per step via grid_builder.
        planner = MCTSPlanner(pomdp; n_iterations = n_iterations, max_depth = grid_steps,
                              c = MCTS_UCB_C, k = k, dt = dt, sigma_mode = sigma_mode,
                              parallel = parallel, reward_mode = reward_mode,
                              constraint_mode = constraint_mode, grid = root_grid)
        # `verbose=true` (config key) streams a per-step @info line AS EACH STEP
        # EXECUTES (run_episode's own live trace) — intermediate progress for long
        # / cluster runs, not just the final rolled-up dict.
        verbose = Bool(_cfg(cfg, "verbose", false))
        rng = MersenneTwister(seed)
        trace = run_episode(planner, pomdp, sc.s_true, rng;
                            max_steps = max_steps, grid_builder = grid_builder,
                            verbose = verbose)

        # 4. WAIT-spine feasibility curve from the root belief (no-maneuver ground
        #    truth — "could deferral resolve Pc, and when"). Uses the root grid's epochs.
        spine_pc = wait_spine_pc(pomdp, sc.b0, sc.s_true, root_grid.t_epochs)

        metrics = _episode_metrics_dict(cfg, sc, pomdp, trace, root_grid, spine_pc,
                                        crossing_t, sec_cad, grid_steps, max_steps,
                                        n_iterations, seed)
    end
    metrics["wall_time_s"] = round(t_wall, digits=3)
    return metrics
end

# Roll one finished episode (trace + scenario + feasibility spine) into the flat
# JSON-serializable metrics dict. Split out so it is independently testable.
function _episode_metrics_dict(cfg, sc, pomdp, trace, root_grid, spine_pc,
                               crossing_t, sec_cad, grid_steps, max_steps,
                               n_iterations, seed)
    thr = pomdp.pc_threshold

    # --- per-step trace as a list of flat dicts (t descending; maneuver flag) ---
    trace_rows = [Dict{String,Any}(
        "step"          => s.step,
        "t_remaining_h" => s.t_remaining / 3600,
        "action"        => s.action == MANEUVER ? "MANEUVER" : "WAIT",
        "dv"            => s.Δv,
        "pc"            => s.pc,
        "miss_m"        => s.miss,
    ) for s in trace]

    n_steps      = length(trace)
    total_dv     = isempty(trace) ? 0.0 : sum(s.Δv for s in trace)
    n_maneuvers  = count(s -> s.action == MANEUVER, trace)
    resolved_no_maneuver = n_maneuvers == 0
    peak_pc      = isempty(trace) ? NaN : maximum(s.pc for s in trace)
    pc_at_tca    = isempty(trace) ? NaN : trace[end].pc
    final_miss   = isempty(trace) ? NaN : trace[end].miss
    maneuver_timings_h = [s.t_remaining / 3600 for s in trace if s.action == MANEUVER]
    first_maneuver_h   = isempty(maneuver_timings_h) ? nothing : first(maneuver_timings_h)

    # --- integrated Pc over the episode (trapezoid in time-remaining, s) ---
    # ∫ Pc dt across the executed epochs; a scalar "how much risk was carried".
    integrated_pc = 0.0
    for i in 1:(n_steps - 1)
        dtsec = trace[i].t_remaining - trace[i + 1].t_remaining
        integrated_pc += 0.5 * (trace[i].pc + trace[i + 1].pc) * dtsec
    end

    # --- feasibility: was WAIT (no maneuver) able to resolve Pc below threshold? ---
    # From the no-maneuver spine: does it EVER cross below threshold, and where.
    spine_below   = [p <= thr for p in spine_pc]
    wait_feasible = any(spine_below)          # deferral resolves at all
    # "durably safe": below threshold at the last trusted epoch before TCA (index
    # end-1; the very-last epoch is the forward-growth endpoint, see debris findings).
    durable_idx   = max(1, length(spine_pc) - 1)
    wait_durably_safe = spine_pc[durable_idx] <= thr
    crossing_h = isnan(crossing_t) ? nothing : crossing_t / 3600

    # --- decision-vs-feasibility verdict (the "did it do the right thing" check) ---
    # If WAIT was durably feasible → the right call is to DEFER (no maneuver);
    # else the right call is to MANEUVER. Compare to what the episode actually did.
    right_call = wait_durably_safe ? "defer" : "maneuver"
    actual     = resolved_no_maneuver ? "defer" : "maneuver"
    decision_matches_feasibility = right_call == actual

    # --- finiteness / well-formedness self-check (a sweep can filter on it) ---
    pc_in_range = all(0.0 <= s.pc <= 1.0 for s in trace)
    all_finite  = all(isfinite(s.pc) && isfinite(s.miss) && isfinite(s.Δv) for s in trace) &&
                  all(isfinite(p) for p in spine_pc)
    # Δv only on MANEUVER steps (the trace-hygiene invariant).
    dv_only_on_maneuver = all((s.action == MANEUVER) == (s.Δv > 0) for s in trace)
    well_formed = pc_in_range && all_finite && dv_only_on_maneuver

    return Dict{String,Any}(
        # ---- echoed config (self-describing logged run) ----
        "config" => Dict{String,Any}(
            "case_path"       => sc_case_id(sc),
            "cadence_secondary_h" => sec_cad / 3600,
            "sensor_quality"  => String(_sym(_cfg(cfg, "sensor_quality", :median))),
            "seed"            => seed,
            "sigma_mode"      => String(_sym(_cfg(cfg, "sigma_mode", :exact))),
            "reward_mode"     => String(_sym(_cfg(cfg, "reward_mode", :terminal))),
            "constraint_mode" => String(_sym(_cfg(cfg, "constraint_mode", :penalize))),
            "n_iterations"    => n_iterations,
            "grid_steps"      => grid_steps,
            "max_steps"       => max_steps,
            "coarse_h"        => Float64(_cfg(cfg, "coarse", 8 * 3600)) / 3600,
            "truncate_safe"   => Bool(_cfg(cfg, "truncate_safe", false)),
            "dt_h"            => pomdp.dt / 3600,
            "pc_threshold"    => thr,
            "delta_v_mps"     => pomdp.Δv,
        ),
        # ---- scenario provenance ----
        "name1"          => sc.name1,
        "name2"          => sc.name2,
        "id1"            => sc.id1,
        "id2"            => sc.id2,
        "sec_class"      => String(sc.sec_class),
        "horizon_h"      => sc.t_horizon / 3600,
        "hbr_m"          => sc.hbr,
        "miss_cdm_m"     => sc.miss_distance,
        "relative_speed_mps" => sc.relative_speed,
        "pc_cdm"         => sc.pc_cdm,
        "valid_2d"       => sc.valid,
        "tca"            => sc.tca,
        # ---- core metrics ----
        "resolved_without_maneuver" => resolved_no_maneuver,
        "peak_pc"        => peak_pc,
        "integrated_pc"  => integrated_pc,
        "pc_at_tca"      => pc_at_tca,
        "total_dv_mps"   => total_dv,
        "n_maneuvers"    => n_maneuvers,
        "maneuver_timings_h" => maneuver_timings_h,
        "first_maneuver_h"   => first_maneuver_h,
        "final_miss_m"   => final_miss,
        "n_steps"        => n_steps,
        # ---- decision-vs-feasibility ----
        "wait_feasible"          => wait_feasible,
        "wait_durably_safe"      => wait_durably_safe,
        "crossing_h"             => crossing_h,
        "right_call"             => right_call,
        "actual_decision"        => actual,
        "decision_matches_feasibility" => decision_matches_feasibility,
        # ---- well-formedness self-check ----
        "well_formed"            => well_formed,
        "pc_in_range"            => pc_in_range,
        "all_finite"             => all_finite,
        "dv_only_on_maneuver"    => dv_only_on_maneuver,
        # ---- full trace + feasibility curve (nested, for plotting later) ----
        "trace"          => trace_rows,
        "wait_spine_pc"  => collect(Float64.(spine_pc)),
        "wait_spine_t_h" => [t / 3600 for t in root_grid.t_epochs],
    )
end

# Case id for logging: reconstruct from the object ids (always present on a loaded
# CDMScenario). Untyped so beliefExecutor.jl need not be included after cdmScenario.jl.
sc_case_id(sc) = string(sc.id1, "_vs_", sc.id2)
