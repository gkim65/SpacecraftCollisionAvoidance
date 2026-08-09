# full_rollout_run.jl — ONE clean full receding-horizon EPISODE end-to-end, on
# real CARA debris cases, at 8 h decision increments (paper_readiness_audit.md
# RESULTS-PHASE PLAN 2026-08-09: "ONE CLEAN FULL ROLLOUT").
#
# This is the FIRST time the planner runs as a full POLICY (plan → execute →
# advance truth → update belief → re-plan) rather than a single-decision probe.
# It uses the wandb-loggable config-in / metrics-out wrapper `run_episode_metrics`
# (beliefExecutor.jl): a cluster sweep just sets a CONFIG and gets back a flat dict
# ready for `wandb.log(dict)`.
#
# Two real DEBRIS-secondary cases (Grace), a WAIT vs MANEUVER contrast:
#   • 40115 vs 30660 (WORLDVIEW 3 vs FENGYUN 1C DEB) — miss 405 m, Pc 1.1e-4,
#     WAIT-spine crosses safe at ~11.7 h → EXPECT the episode to DEFER (no maneuver).
#   • 37849 vs 13512 (NPP vs THOR ABLESTAR DEB) — miss 99 m, Pc 1.0e-2,
#     WAIT-spine only safe at TCA (3e-5 still at 2.5 h) → EXPECT it to MANEUVER.
#
# Adaptive grid at 8 h coarse (Grace), :exact serial. Dumps each metrics dict to
# figureScripts/data/full_rollout_<id>.json (a cluster script logs the same dict).
#
# Usage: julia --project=. figureScripts/full_rollout_run.jl

using LinearAlgebra, Random, PyCall, POMDPs, POMDPTools, Distributions, Dates, Printf

include(joinpath(@__DIR__, "..", "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "sensorTiers.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "genConjunctions.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "computePc.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "covarianceTable.jl"))
include(joinpath(@__DIR__, "..", "src", "states.jl"))
include(joinpath(@__DIR__, "..", "src", "actions.jl"))
include(joinpath(@__DIR__, "..", "src", "rewards.jl"))
include(joinpath(@__DIR__, "..", "src", "observations.jl"))
include(joinpath(@__DIR__, "..", "src", "transitions.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefTracker.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefMCTS.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefExecutor.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "cdmScenario.jl"))

const CDMDIR  = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms"))
const DATADIR = joinpath(@__DIR__, "data")
isdir(DATADIR) || mkpath(DATADIR)

# label, CDM file, sensor_quality. The 40115 case is run at BOTH :best and :median
# to show the HEADLINE FLIP: with well-calibrated SSN (:best) the belief mean stays
# pinned to the ~405 m truth and WAIT durably resolves (defer, 0 Δv); at :median the
# ~469 m cross-range measurement error (≈ the miss) drifts the belief mean → Pc reads
# unsafe → the planner correctly maneuvers on that noisy belief. 37849 (miss 99 m,
# Pc 1e-2) is the must-MANEUVER control (WAIT-spine only crosses at TCA).
const F40115 = "000040115_conj_000030660_20230721_100115_20230720_061903.cdm"
const F37849 = "000037849_conj_000013512_20210612_084905_20210611_062043.cdm"
const CASES = [
    ("40115 vs 30660  :best   — well-tracked → expect DEFER (WAIT, 0 Δv)", F40115, :best),
    ("40115 vs 30660  :median — noisy track  → expect MANEUVER (drift)",   F40115, :median),
    ("37849 vs 13512  :median — high Pc/tiny miss → expect MANEUVER (control)", F37849, :median),
]

# --- hand-rolled JSON writer (project convention: JSON only a transitive dep) ---
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_j(x::Nothing) = "null"
_j(::Missing) = "null"
_j(x::Symbol) = _j(string(x))
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
_j(d::AbstractDict) = "{" * join([_j(string(k)) * ":" * _j(v) for (k, v) in d], ",") * "}"

function print_metrics(m)
    println("  ── scenario ──────────────────────────────────────────────")
    @printf("  %s vs %s   class=%s   horizon %.2f h   HBR %.1f m\n",
            m["name1"], m["name2"], m["sec_class"], m["horizon_h"], m["hbr_m"])
    @printf("  CARA Pc=%.3e   miss(CDM)=%.0f m   rel-speed=%.0f m/s   valid-2D=%s\n",
            m["pc_cdm"], m["miss_cdm_m"], m["relative_speed_mps"], m["valid_2d"])
    println("  ── WAIT-spine feasibility (no-maneuver ground truth) ──────")
    ts = m["wait_spine_t_h"]; ps = m["wait_spine_pc"]; thr = m["config"]["pc_threshold"]
    for i in eachindex(ts)
        @printf("    %6.2f h   Pc=%.3e%s\n", ts[i], ps[i],
                ps[i] <= thr ? "   <-- below threshold" : "")
    end
    cr = m["crossing_h"]
    println("    crossing (WAIT becomes safe): ",
            cr === nothing ? "none" : @sprintf("%.2f h", cr),
            "   durably-safe=", m["wait_durably_safe"])
    println("  ── executed episode trace ─────────────────────────────────")
    @printf("    %-5s %-10s %-9s %-8s %-11s %-9s\n",
            "step", "t_rem(h)", "action", "Δv", "Pc-at-TCA", "miss(km)")
    for r in m["trace"]
        @printf("    %-5d %-10.2f %-9s %-8.3f %-11.3e %-9.3f\n",
                r["step"], r["t_remaining_h"], r["action"], r["dv"],
                r["pc"], r["miss_m"]/1000)
    end
    println("  ── metrics ────────────────────────────────────────────────")
    @printf("    resolved without maneuver : %s\n", m["resolved_without_maneuver"])
    @printf("    peak Pc                   : %.3e\n", m["peak_pc"])
    @printf("    integrated Pc (Pc·s)      : %.3e\n", m["integrated_pc"])
    @printf("    Pc at TCA                 : %.3e\n", m["pc_at_tca"])
    @printf("    total Δv                  : %.3f m/s   (n_maneuvers=%d)\n",
            m["total_dv_mps"], m["n_maneuvers"])
    @printf("    maneuver timings (h)      : %s\n", string(m["maneuver_timings_h"]))
    @printf("    final miss                : %.3f km\n", m["final_miss_m"]/1000)
    @printf("    decision vs feasibility   : right=%s  actual=%s  match=%s\n",
            m["right_call"], m["actual_decision"], m["decision_matches_feasibility"])
    @printf("    well-formed trace         : %s  (pc∈[0,1]=%s finite=%s Δv-only-on-MANEUVER=%s)\n",
            m["well_formed"], m["pc_in_range"], m["all_finite"], m["dv_only_on_maneuver"])
    @printf("    wall time                 : %.1f s\n", m["wall_time_s"])
end

println("="^70)
println("ONE CLEAN FULL ROLLOUT — receding-horizon episode, 8 h increments, :exact")
println("="^70)

for (label, fname, quality) in CASES
    println("\n", "#"^70)
    println("# ", label)
    println("#"^70)
    # verbose=true → run_episode streams a per-step @info line AS each step executes
    # (intermediate progress for long runs), on top of the final rolled-up dict.
    cfg = episode_config(; case_path = joinpath(CDMDIR, fname),
                         coarse = 8*3600, sigma_mode = :exact, parallel = false,
                         n_iterations = 12, reward_mode = :terminal, seed = 20240809,
                         sensor_quality = quality, verbose = true)
    m = run_episode_metrics(cfg)
    print_metrics(m)

    out = joinpath(DATADIR, "full_rollout_$(m["id1"])_vs_$(m["id2"])_$(quality).json")
    open(out, "w") do io; write(io, _j(m)); end
    println("  wrote metrics dict → ", relpath(out, pwd()))
end

println("\n", "="^70)
println("done — two metrics dicts written to figureScripts/data/")
println("A cluster sweep would call run_episode_metrics(cfg) and wandb.log(dict).")
println("="^70)
