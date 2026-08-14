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
    # Full tracked belief covariance (6×6 ECI, mean+velocity) for BOTH objects at
    # THIS step's post-step belief — so a trace can show Σ growing (predict) and
    # shrinking (measurement) over the episode, and diagnose drift (the whole
    # tracking-fidelity story). The STARTING seed Σ is logged separately in the
    # metrics dict's provenance (`b0_sigma_*`), so both the beginning and every step
    # are recoverable. These are the tracked belief Σ at the node epoch (NOT grown to
    # TCA); `pc` above is Pc-at-TCA from these grown to TCA.
    sigma_sc::Matrix{Float64}
    sigma_debris::Matrix{Float64}
end

"""
    mcts_policy

The DEFAULT decision policy for `run_episode`: the belief-space MCTS planner. It
just runs one `plan(step_planner, root, rng)` and returns the best action, so a
`run_episode` call with no `policy` argument behaves byte-for-byte as before the
policy refactor. It is a plain function (not a closure) so it can be imported and
compared against the baseline policies (`src/utils/baselines.jl`) on the identical
`run_episode` loop — the whole point of the F3 comparison is that the DECISION rule
is the only thing that varies; the belief update, grid, metrics, and observation
draws are shared.

Policy contract (all policies, `mcts_policy` + the baselines): a policy is a
function

    policy(pomdp, root, rng; planner, t_remaining, grid, pc_threshold,
           delta_v, step) -> CAAction

called ONCE per executed step at the single decision point in `run_episode` (the
`plan` call it replaces). `root` is the fresh `BeliefNode` carrying the CURRENT
tracked belief + true state + cadence phase; `t_remaining` is time-to-TCA (s) at
the step; `planner` is the fully-configured `MCTSPlanner` for this step (the
default policy uses it; the gate baselines ignore it and read Pc off `root`).
Every gate reads Pc from the SAME (possibly noisy / drifted) belief the MCTS
planner sees — `node_pc_at_tca(pomdp, root)` — never the truth: that is the crux
of a fair comparison. `grid` / `pc_threshold` / `delta_v` / `step` are the
remaining decision context (the timing gate needs `t_remaining` + `pc_threshold`).
"""
function mcts_policy(pomdp::SpacecraftCAPOMDP, root::BeliefNode, rng::AbstractRNG;
                     planner::MCTSPlanner, t_remaining::Real = root.belief.t,
                     grid = nothing, pc_threshold::Real = pomdp.pc_threshold,
                     delta_v::Real = pomdp.Δv, step::Int = 0, verbose::Bool = false,
                     kwargs...)
    a, r = plan(planner, root, rng)
    if verbose
        # Per-decision chance-constraint trace: the ACTUAL per-root-action rollout
        # count, p_viol = P̂[Pc(TCA)>δ | a], and E[Pc|a] — so we can confirm ~n_iter/2
        # rollouts/action and see which actions the α gate masked. `α`/`rule` echo
        # the constraint that produced `a` (the committed action).
        stats = root_action_stats(r)
        for (act, s) in stats
            @info "root chance-constraint" step=step action=act n_rollouts=s.n p_viol=round(s.p_viol, digits=4) E_pc=s.epc α=planner.α_cc rule=planner.root_rule chosen=(act == a)
        end
    end
    return a
end

"""
    run_episode(planner, pomdp, s0, rng; policy=mcts_policy,
                correct_at_root=pomdp.correct_at_root,
                max_steps=planner.max_depth, verbose=false) -> Vector{ExecStep}

Run one closed-loop (receding-horizon / MPC) episode from true state `s0`:
DECIDE (via `policy`) → execute → advance the true state → update the belief →
re-decide, until TCA or collision (`isterminal`) or `max_steps` is hit. Returns
the per-step trace.

`policy` is the DECISION rule swapped in at the single decision point (see
`mcts_policy` for the contract). It defaults to `mcts_policy` (run the planner), so
the default call is unchanged from before the refactor. The F3 baseline policies
(`src/utils/baselines.jl`) slot in here to get the IDENTICAL belief update (incl.
the noisy drift), grid, metrics, and observation draws — the comparison isolates
the decision rule only.

`b0` is the STARTING tracked belief. For the CDM pipeline pass the loader's
back-propagated DETECTION seed (`sc.b0`); `nothing` (default) falls back to a fresh
`belief_from_pomdp` P0 (the synthetic-suite path, byte-identical). See the belief-
init comment below for why the seed choice is load-bearing (a fresh P0 grows the
raw CDM-TCA covariance forward and reads Pc ~100× below CARA).

The belief is tracked across steps with the SAME `step_belief` update the planner
uses internally (predict → sample z → cadence-aware correct), including the
per-object cadence timers and the `correct_at_root` phase, so the executed belief
is consistent with what each decision assumed. A fresh root node is built each
step from the CURRENT belief + true state, and one policy call decides that step's
action. All planner knobs (sigma_mode / parallel / constraint_mode / dt /
cadences) are used as configured on `planner`.

`rng` seeds both the per-step planning and the real observation draws; pass a
seeded RNG for a reproducible episode. `pomdp.dt` and `planner.dt` must match
(asserted) — the true state advances by `pomdp.dt` via `POMDPs.transition`.
"""
function run_episode(planner::MCTSPlanner, pomdp::SpacecraftCAPOMDP, s0::CAState,
                     rng::AbstractRNG;
                     policy = mcts_policy,
                     b0::Union{Belief,Nothing} = nothing,
                     correct_at_root::Bool = pomdp.correct_at_root,
                     max_steps::Int = planner.max_depth,
                     grid_builder = nothing,
                     final_state_ref::Union{Base.RefValue,Nothing} = nothing,
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

    # Current true state + tracked belief. When a `b0` is supplied (the CDM pipeline
    # path) the executor STARTS from that belief — this MUST be the loader's
    # back-propagated DETECTION seed (`sc.b0`, from `backprop_belief_to_detection`),
    # NOT a fresh `belief_from_pomdp` P0. `belief_from_pomdp` anchors `pomdp.P0_debris`
    # (= the raw CDM-TCA covariance) at the DETECTION epoch and then grows it FORWARD
    # to TCA, which smears the geometry and drives Pc-at-TCA ~100× BELOW CARA (the
    # exact "forward-Pc→0" failure the back-prop fix, beliefMCTS.jl
    # `backprop_belief_to_detection`, was built to prevent). So the closed-loop
    # executor must track the SAME back-propagated seed the feasibility spine
    # (`wait_spine_pc(pomdp, sc.b0, …)`) and the planner's rollouts assume — else the
    # executed belief and the scoring spine sit on DIFFERENT covariances. `b0 ===
    # nothing` (the synthetic-suite path) falls back to `belief_from_pomdp`, so those
    # runs are byte-identical.
    s_true       = s0
    belief       = b0 === nothing ? belief_from_pomdp(pomdp, s0.sc_eci, s0.debris_eci, s0.t) : b0
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
            # `with_grid` copies EVERY planner field reflectively and swaps only the
            # grid (re-capping max_depth). It replaced a hand-written field-by-field
            # copy that silently dropped any newly added field — see with_grid's
            # docstring (beliefMCTS.jl) for the leaf_only_pc regression that caused.
            step_planner = with_grid(planner, grid_builder(t_remaining))
        end
        exec_grid = step_planner.grid

        # 1. DECIDE from the current belief + true state. Build a fresh root node
        #    carrying the current belief, true state, and cadence-timer phase, then
        #    call the policy for this step's action. The DEFAULT policy (`mcts_policy`)
        #    runs one `plan(step_planner, root, rng)` (the planner re-derives its own
        #    timers from the node); a baseline policy (gate) reads Pc off THIS root's
        #    belief instead — the SAME (possibly noisy) belief the planner would see,
        #    so the comparison isolates the decision rule. `exec_grid` (built above) is
        #    passed so a timing gate can see the step's grid context if it wants.
        root = BeliefNode(belief, s_true, isterminal(pomdp, s_true);
                          since_sc = since_sc, since_debris = since_debris)
        a = policy(pomdp, root, rng; planner = step_planner, t_remaining = t_remaining,
                   grid = exec_grid, pc_threshold = pomdp.pc_threshold,
                   delta_v = pomdp.Δv, step = step, verbose = verbose)

        # 2. EXECUTE: advance the TRUE state. Fixed grid ⇒ pomdp.dt; adaptive grid
        #    ⇒ the grid's first epoch gap (the step the plan actually decided).
        exec_dt = exec_grid === nothing ? pomdp.dt : exec_grid.dts[1]
        sp = rand(rng, transition_dt(pomdp, s_true, a, exec_dt))

        # 3. UPDATE the belief with the SHARED per-step update (predict → sample z
        #    → cadence-aware correct) — identical to what the planner simulates. On
        #    the adaptive grid pass the grid + grid_depth=1 so the correction schedule
        #    matches the planner's first step exactly.
        # p_arrival comes from the planner (single source of truth): the executed
        # belief uses the SAME probabilistic debris-arrival model the rollouts do, so
        # the planner reasons about arrival on the exact distribution it then faces.
        belief, since_sc, since_debris = step_belief(pomdp, belief, a, sp, rng;
                                                     dt = exec_dt,
                                                     cadence_sc = pomdp.cadence_sc,
                                                     cadence_debris = pomdp.cadence_debris,
                                                     since_sc = since_sc,
                                                     since_debris = since_debris,
                                                     p_arrival = step_planner.p_arrival,
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
        # capture the FULL tracked belief Σ (6×6 ECI) for both objects at this step's
        # post-step belief — for the covariance trace (growth/shrink/drift diagnosis).
        push!(trace, ExecStep(step, t_remaining, a, Δv, pc, miss,
                              Matrix{Float64}(belief.sc.Σ), Matrix{Float64}(belief.debris.Σ)))

        if verbose
            @info "executed step" step=step t_h=round(t_remaining/3600, digits=2) action=a Δv=Δv pc=pc miss_km=round(miss/1000, digits=3)
        end

        # advance the true state; re-plan from here next iteration.
        s_true = sp
    end
    # Capture the final executed TRUE state (at TCA / collision) for the caller — the
    # `outcome_pc` common-yardstick score (baselines.jl) needs where the policy
    # actually steered the objects. Additive: the return contract stays the trace.
    final_state_ref !== nothing && (final_state_ref[] = s_true)
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
                   max_steps=nothing, grid_mode=:measurement) -> Dict{String,Any}

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
                        # Skip the to-TCA Pc propagation at INTERNAL tree nodes
                        # (:terminal mode only; see MCTS_LEAF_ONLY_PC). Behavior-
                        # neutral ~1.7-2x speedup. Default false so the figure probes
                        # that read per-node .pc / .violated keep working; sweeps
                        # opt in via the YAML so the choice is logged to wandb.
                        leaf_only_pc::Bool = MCTS_LEAF_ONLY_PC,
                        k::Real = 2.0,
                        truncate_safe::Bool = false,
                        sec_class_override::Union{Symbol,Nothing} = nothing,
                        dt::Real = 60 * 60,
                        t_horizon::Union{Real,Nothing} = nothing,
                        pc_threshold::Union{Real,Nothing} = nothing,
                        max_steps::Union{Integer,Nothing} = nothing,
                        grid_mode::Symbol = :measurement,
                        p_arrival::Real = 1.0,
                        α_cc::Real = MCTS_ALPHA,
                        root_rule::Symbol = MCTS_ROOT_RULE,
                        policy::AbstractString = "mcts",
                        policy_params::Union{AbstractDict,Nothing} = nothing,
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
        "leaf_only_pc"      => leaf_only_pc,
        "k"                 => Float64(k),
        "truncate_safe"     => truncate_safe,
        "sec_class_override" => sec_class_override === nothing ? nothing : String(sec_class_override),
        "dt"                => Float64(dt),
        "t_horizon"         => t_horizon,
        "pc_threshold"      => pc_threshold,
        "max_steps"         => max_steps,
        "grid_mode"         => String(grid_mode),
        # p_arrival = P(a SCHEDULED debris measurement actually arrives). 1.0 (default)
        # = the guaranteed-measurement model (byte-identical to pre-arrival runs); < 1
        # makes each due debris fix a Bernoulli arrival, else predict-only that step.
        "p_arrival"         => Float64(p_arrival),
        # α_cc = risk level of the ROOT chance constraint; root_rule selects the root
        # decision rule (:chance = mask on p_viol < α + argmax-Qa on the feasible set,
        # least-infeasible fallback; :legacy = plain argmax-Qa, the soft-penalty
        # planner, for reproducing pre-2026-08-11 results). See MCTS_ALPHA/MCTS_ROOT_RULE.
        "alpha_cc"          => Float64(α_cc),
        "root_rule"         => String(root_rule),
        # policy = the DECISION rule this episode runs (F3 baseline comparison).
        # "mcts" (default) = the planner; "pc_gate"/"timing_gate" = the baselines.
        # policy_params carries the swept parameter for a gate ("theta" for pc_gate,
        # "T_trigger_h" for timing_gate); nothing for mcts. See baselines.make_policy.
        "policy"            => String(policy),
        "policy_params"     => policy_params,
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
 2. Build a per-step decision-grid builder — rebuilt each executed step from the
    current time-to-TCA (receding horizon). Default `grid_mode=:measurement`: decision
    epochs = the SECONDARY's measurement schedule (its cadence) + TCA (`decision_grid`).
    `grid_mode=:adaptive` keeps the crossing-refined grid (`adaptive_decision_grid`) as
    a swappable diagnostic; measurement-schedule is the sweep default.
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
        leaf_only_pc   = Bool(_cfg(cfg, "leaf_only_pc", MCTS_LEAF_ONLY_PC))
        k              = Float64(_cfg(cfg, "k", 2.0))
        coarse         = Float64(_cfg(cfg, "coarse", 8 * 3600))
        truncate_safe  = Bool(_cfg(cfg, "truncate_safe", false))
        grid_mode      = _sym(_cfg(cfg, "grid_mode", :measurement))
        p_arrival      = Float64(_cfg(cfg, "p_arrival", 1.0))
        α_cc           = Float64(_cfg(cfg, "alpha_cc", MCTS_ALPHA))
        root_rule      = _sym(_cfg(cfg, "root_rule", MCTS_ROOT_RULE))

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

        # 2. per-step decision-grid builder (receding horizon). Rebuilt from the
        #    CURRENT time-to-TCA each executed step by run_episode.
        #    :measurement (default) — decision epochs = the SECONDARY's measurement
        #      schedule (its cadence) + TCA. Simple, principled (decide when info
        #      arrives), no adaptive/crossing logic. `decision_grid`.
        #    :adaptive — the crossing-refined grid (coarse everywhere, `fine` around
        #      the WAIT-becomes-safe crossing). Kept swappable for diagnostics; NOT
        #      the sweep default (untrusted adaptive logic, and the :fast path crashes
        #      on it). `adaptive_decision_grid`.
        if grid_mode === :adaptive
            grid_builder = t -> begin
                g, _ = adaptive_decision_grid(pomdp, sc.b0, sc.s_true, t;
                                              cadence_secondary = sec_cad,
                                              coarse = coarse, fine = fine,
                                              truncate_safe = truncate_safe)
                g
            end
            root_grid, _ = adaptive_decision_grid(pomdp, sc.b0, sc.s_true, sc.t_horizon;
                                        cadence_secondary = sec_cad, coarse = coarse,
                                        fine = fine, truncate_safe = truncate_safe)
        else
            grid_mode === :measurement ||
                error("run_episode_metrics: grid_mode must be :measurement or :adaptive (got $grid_mode)")
            grid_builder = t -> decision_grid(t; cadence_secondary = sec_cad)
            root_grid = decision_grid(sc.t_horizon; cadence_secondary = sec_cad)
        end

        # depth cap = the root grid's step count (a rollout reaches TCA there); a
        # config `max_steps` can shorten it further.
        grid_steps = grid_depth_count(root_grid)
        ms_cfg = _cfg(cfg, "max_steps", nothing)
        max_steps = ms_cfg === nothing ? grid_steps : min(Int(ms_cfg), grid_steps)

        # 3. run one episode. The planner carries the ROOT grid (its max_depth caps
        #    to grid steps); run_episode rebuilds the grid per step via grid_builder.
        planner = MCTSPlanner(pomdp; n_iterations = n_iterations, max_depth = grid_steps,
                              c = MCTS_UCB_C, k = k, dt = dt, sigma_mode = sigma_mode,
                              parallel = parallel, reward_mode = reward_mode,
                              constraint_mode = constraint_mode, p_arrival = p_arrival,
                              α_cc = α_cc, root_rule = root_rule,
                              leaf_only_pc = leaf_only_pc,
                              grid = root_grid)
        # `verbose=true` (config key) streams a per-step @info line AS EACH STEP
        # EXECUTES (run_episode's own live trace) — intermediate progress for long
        # / cluster runs, not just the final rolled-up dict.
        verbose = Bool(_cfg(cfg, "verbose", false))
        # DECISION POLICY (F3 baseline comparison). "mcts" (default) = the planner;
        # a gate baseline reads Pc off the same belief and decides without MCTS. The
        # policy slots into the SAME run_episode loop → identical belief update /
        # grid / metrics / observation draws → the comparison isolates the decision.
        policy_kind   = String(_cfg(cfg, "policy", "mcts"))
        policy_params = _cfg(cfg, "policy_params", nothing)
        policy_spec   = Dict{String,Any}("kind" => policy_kind)
        if policy_params !== nothing
            for (kk, vv) in policy_params; policy_spec[String(kk)] = vv; end
        end
        policy = make_policy(policy_spec)
        rng = MersenneTwister(seed)
        # START the executor from the loader's back-propagated DETECTION seed sc.b0
        # (NOT a fresh belief_from_pomdp P0) so the executed belief sits on the SAME
        # covariance as the feasibility spine + the planner rollouts (see the belief-
        # init comment in run_episode). This was the executor-seed fix (2026-08-10).
        trace = run_episode(planner, pomdp, sc.s_true, rng;
                            policy = policy, b0 = sc.b0,
                            max_steps = max_steps, grid_builder = grid_builder,
                            verbose = verbose)

        # 4. WAIT-spine feasibility curve from the root belief (no-maneuver ground
        #    truth — "could deferral resolve Pc, and when"). Uses the root grid's epochs.
        #    This clean, no-drift spine is ALSO the mitigation ground-truth for the
        #    per-maneuver `maneuver_mitigated` check (was the true geometry actually
        #    resolved, vs. a drift-driven precautionary burn).
        spine_pc = wait_spine_pc(pomdp, sc.b0, sc.s_true, root_grid.t_epochs)

        metrics = _episode_metrics_dict(cfg, sc, pomdp, trace, root_grid, spine_pc,
                                        sec_cad, grid_steps, max_steps,
                                        n_iterations, seed)
    end
    metrics["wall_time_s"] = round(t_wall, digits=3)
    return metrics
end

# Roll one finished episode (trace + scenario + feasibility spine) into the flat
# JSON-serializable metrics dict. Split out so it is independently testable.
function _episode_metrics_dict(cfg, sc, pomdp, trace, root_grid, spine_pc,
                               sec_cad, grid_steps, max_steps,
                               n_iterations, seed)
    thr = pomdp.pc_threshold
    epochs_h = [t / 3600 for t in root_grid.t_epochs]

    # --- per-step trace as a list of flat dicts (t descending; maneuver flag) ---
    # The full 6×6 belief Σ (both objects) is flattened ROW-MAJOR to a 36-vector
    # (`sigma_*_eci_flat`) so it stays JSON-/wandb-Table-serializable (the hand-rolled
    # JSON writers handle flat vectors, not matrices); `reshape(v,6,6)'` recovers the
    # matrix. `sigma_*_pos_m` is the convenience position σ = √ of the ECI position-
    # block diagonal (m), the quick read on Σ growth/shrink/drift over the episode.
    trace_rows = [Dict{String,Any}(
        "step"          => s.step,
        "t_remaining_h" => s.t_remaining / 3600,
        "action"        => s.action == MANEUVER ? "MANEUVER" : "WAIT",
        "dv"            => s.Δv,
        "pc"            => s.pc,
        "miss_m"        => s.miss,
        "sigma_sc_eci_flat"     => vec(permutedims(s.sigma_sc)),
        "sigma_debris_eci_flat" => vec(permutedims(s.sigma_debris)),
        "sigma_sc_pos_m"     => sqrt.(abs.(diag(s.sigma_sc)[1:3])),
        "sigma_debris_pos_m" => sqrt.(abs.(diag(s.sigma_debris)[1:3])),
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
    # crossing = first epoch (excluding the root) where the WAIT-spine Pc drops to/
    # below threshold — the WAIT-becomes-safe time. Derived from the spine directly
    # (the measurement-schedule grid has no separate crossing probe). `nothing` if it
    # never crosses (a never-safe debris case). Matches the adaptive grid's definition.
    xi = findfirst(i -> spine_pc[i] <= thr, 2:length(spine_pc))
    crossing_h = xi === nothing ? nothing : epochs_h[xi + 1]

    # --- decision-vs-feasibility verdict (the "did it do the right thing" check) ---
    # If WAIT was durably feasible → the right call is to DEFER (no maneuver);
    # else the right call is to MANEUVER. Compare to what the episode actually did.
    right_call = wait_durably_safe ? "defer" : "maneuver"
    actual     = resolved_no_maneuver ? "defer" : "maneuver"
    decision_matches_feasibility = right_call == actual

    # --- lead time at the first maneuver (did the planner genuinely WAIT?) ---
    # The time-to-TCA (h) at the FIRST burn. Large ⇒ burned early / immediately;
    # small ⇒ deferred and burned late (the "wait-and-measure" behavior). Same value
    # as `first_maneuver_h` (t_remaining is time-to-TCA), surfaced under the explicit
    # lead-time name the sweep tracks. `nothing` if the episode never maneuvered.
    lead_time_at_first_maneuver_h = first_maneuver_h

    # --- maneuver mitigation: did each burn actually drop Pc below threshold? ---
    # Guards the "waited too long, burned, too late to matter" failure mode (esp.
    # last-epoch burns — the maneuver study's latest-fixable-lead boundary). For each
    # MANEUVER step we ask: is Pc-at-TCA at/after this burn below threshold along the
    # EXECUTED path (trace `pc` IS Pc-at-TCA from the post-step belief, so it reflects
    # the burn's effect). A burn is "mitigated" if Pc-at-TCA is below threshold by the
    # END of the episode (at TCA) — a mid-episode burn that is later undone/insufficient
    # is NOT counted mitigated. The overall flag: at least one maneuver AND the episode
    # ends below threshold. When there are no maneuvers it is `nothing` (N/A — the
    # deferral case is scored by wait feasibility, not mitigation).
    maneuver_steps = [i for i in eachindex(trace) if trace[i].action == MANEUVER]
    per_maneuver_mitigated = Bool[]
    for i in maneuver_steps
        # Pc-at-TCA reached by end-of-episode following this burn (the last trace Pc is
        # the executed Pc-at-TCA at TCA; a burn "mitigates" if that final Pc is safe).
        push!(per_maneuver_mitigated, isempty(trace) ? false : (trace[end].pc <= thr))
    end
    maneuver_mitigated = isempty(maneuver_steps) ? nothing :
                         (!isempty(trace) && trace[end].pc <= thr && all(per_maneuver_mitigated))

    # --- finiteness / well-formedness self-check (a sweep can filter on it) ---
    pc_in_range = all(0.0 <= s.pc <= 1.0 for s in trace)
    all_finite  = all(isfinite(s.pc) && isfinite(s.miss) && isfinite(s.Δv) for s in trace) &&
                  all(isfinite(p) for p in spine_pc)
    # Δv only on MANEUVER steps (the trace-hygiene invariant).
    dv_only_on_maneuver = all((s.action == MANEUVER) == (s.Δv > 0) for s in trace)
    # Maneuver actually mitigated: if the episode burned, did Pc-at-TCA end below
    # threshold? A burn that leaves Pc-at-TCA above threshold is the too-late-burn
    # failure mode — a real, reportable outcome, NOT a malformed trace, so it is
    # surfaced as its OWN selfcheck field and does NOT gate `well_formed`. `true`
    # when no maneuver (vacuously — nothing to mitigate).
    maneuver_effective = maneuver_mitigated === nothing ? true : maneuver_mitigated
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
            # Which Pc path produced this run (provenance): true = internal-node to-TCA
            # propagation skipped (leaf-only). Behavior-neutral, but the pre-2026-08-14
            # sweeps ran with it false, so log it to keep the two distinguishable.
            "leaf_only_pc"    => Bool(_cfg(cfg, "leaf_only_pc", MCTS_LEAF_ONLY_PC)),
            "n_iterations"    => n_iterations,
            "grid_steps"      => grid_steps,
            "max_steps"       => max_steps,
            "coarse_h"        => Float64(_cfg(cfg, "coarse", 8 * 3600)) / 3600,
            "truncate_safe"   => Bool(_cfg(cfg, "truncate_safe", false)),
            "dt_h"            => pomdp.dt / 3600,
            "pc_threshold"    => thr,
            "delta_v_mps"     => pomdp.Δv,
            "p_arrival"       => Float64(_cfg(cfg, "p_arrival", 1.0)),
            "alpha_cc"        => Float64(_cfg(cfg, "alpha_cc", MCTS_ALPHA)),
            "root_rule"       => String(_sym(_cfg(cfg, "root_rule", MCTS_ROOT_RULE))),
            "policy"          => String(_cfg(cfg, "policy", "mcts")),
            "policy_params"   => _cfg(cfg, "policy_params", nothing),
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
        # ---- STARTING seed belief covariance (sc.b0, the back-propagated detection
        #      seed the executor now starts from) — the "beginning" Σ Grace wants
        #      tracked alongside the per-step trace Σ. Full 6×6 flattened row-major
        #      (reshape(v,6,6)') + the position σ (√ ECI position-block diag, m). ----
        "b0_sigma_sc_eci_flat"     => vec(permutedims(Matrix{Float64}(sc.b0.sc.Σ))),
        "b0_sigma_debris_eci_flat" => vec(permutedims(Matrix{Float64}(sc.b0.debris.Σ))),
        "b0_sigma_sc_pos_m"        => sqrt.(abs.(diag(sc.b0.sc.Σ)[1:3])),
        "b0_sigma_debris_pos_m"    => sqrt.(abs.(diag(sc.b0.debris.Σ)[1:3])),
        "b0_t_remaining_h"         => sc.b0.t / 3600,
        # ---- core metrics ----
        "resolved_without_maneuver" => resolved_no_maneuver,
        "peak_pc"        => peak_pc,
        "integrated_pc"  => integrated_pc,
        "pc_at_tca"      => pc_at_tca,
        "total_dv_mps"   => total_dv,
        "n_maneuvers"    => n_maneuvers,
        "maneuver_timings_h" => maneuver_timings_h,
        "first_maneuver_h"   => first_maneuver_h,
        "lead_time_at_first_maneuver_h" => lead_time_at_first_maneuver_h,
        "maneuver_mitigated"        => maneuver_mitigated,
        "per_maneuver_mitigated"    => per_maneuver_mitigated,
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
        "maneuver_effective"     => maneuver_effective,
        # ---- full trace + feasibility curve (nested, for plotting later) ----
        "trace"          => trace_rows,
        "wait_spine_pc"  => collect(Float64.(spine_pc)),
        "wait_spine_t_h" => [t / 3600 for t in root_grid.t_epochs],
    )
end

# Case id for logging: reconstruct from the object ids (always present on a loaded
# CDMScenario). Untyped so beliefExecutor.jl need not be included after cdmScenario.jl.
sc_case_id(sc) = string(sc.id1, "_vs_", sc.id2)
