# baselines_validation.jl — LOCAL validation of the F3 baselines (paper_readiness_
# audit.md "SEQUENCE UPDATE 2026-08-09 night"; baseline set REVISED with Grace).
#
# Runs, on two real CARA debris cases, the MCTS planner vs the two baseline
# policies through the SAME run_episode_metrics loop — a side-by-side table. The
# point is the machinery + the qualitative contrast, NOT a statistical claim.
#
# Baseline set (Grace 2026-08-09):
#   (1) DELAY-CLOCK gate — fixed δ=1e-5, single-burn, only knob = act-time T_act:
#       force-defer until t_remaining<T_act, then MANEUVER iff belief-Pc>δ. Swept
#       over T_act (act-now … act-late). Naive threshold-on-a-clock foil.
#   (2) WAIT-FEASIBILITY oracle (IDEALIZED) — defers while the clean full-horizon
#       WAIT+measure spine (zero-innovation, Σ shrinks, no drift) is durably <δ;
#       else MANEUVER. The near-oracle "could WAIT have resolved this" reference.
#
# Expect (qualitative, 2 cases): on 40115 (defer case) the wait-feasibility oracle
# DEFERS (spine durably safe), the delay gate over-maneuvers at large T_act (burns
# early on the drifted belief) and MCTS sits between; on 37849 (must-maneuver) all
# maneuver. Cheap: :exact serial, n_iterations=12, 8 h cadence, :median sensors.
# Usage: julia --project=. figureScripts/baselines_validation.jl

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
include(joinpath(@__DIR__, "..", "src", "utils", "baselines.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "cdmScenario.jl"))

const CDMDIR = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms"))
const F40115 = "000040115_conj_000030660_20230721_100115_20230720_061903.cdm"  # defer case
const F37849 = "000037849_conj_000013512_20210612_084905_20210611_062043.cdm"  # must-maneuver

const CASES = [
    ("40115 vs 30660 (defer case)",       F40115, :best),   # low drift → MCTS should MATCH oracle
    ("40115 vs 30660 (defer case)",       F40115, :median), # drift → MCTS may DIVERGE from oracle
    ("37849 vs 13512 (must-maneuver)",    F37849, :median),
]

# policies to compare on each case: (label, policy kind, params dict).
const POLICIES = [
    ("MCTS (planner)",            "mcts",             nothing),
    ("wait-feas oracle",          "wait_feasibility", nothing),
    ("delay T_act=28h (now)",     "delay_gate",       Dict("T_act_h" => 28.0)),
    ("delay T_act=12h",           "delay_gate",       Dict("T_act_h" => 12.0)),
    ("delay T_act=6h",            "delay_gate",       Dict("T_act_h" => 6.0)),
    ("delay T_act=3h",            "delay_gate",       Dict("T_act_h" => 3.0)),
]

function run_one(fname, quality, kind, params)
    cfg = episode_config(; case_path = joinpath(CDMDIR, fname),
                         coarse = 8*3600, sigma_mode = :exact, parallel = false,
                         n_iterations = 12, reward_mode = :terminal, seed = 20240809,
                         sensor_quality = quality, policy = kind, policy_params = params,
                         verbose = false)
    return run_episode_metrics(cfg)
end

for (clabel, fname, quality) in CASES
    println("\n", "="^92)
    println("# ", clabel, "   (sensor_quality=", quality, ")")
    println("="^92)

    results = Tuple{String,Dict{String,Any}}[]
    for (plabel, kind, params) in POLICIES
        @printf("  running %-22s ...\n", plabel)
        push!(results, (plabel, run_one(fname, quality, kind, params)))
    end

    m0 = results[1][2]
    @printf("\n  scenario: %s vs %s  class=%s  horizon %.1f h  miss(CDM) %.0f m  CARA Pc %.2e\n",
            m0["name1"], m0["name2"], m0["sec_class"], m0["horizon_h"],
            m0["miss_cdm_m"], m0["pc_cdm"])
    thr = m0["config"]["pc_threshold"]
    @printf("  chance-constraint threshold δ = %.0e\n", thr)

    # the clean WAIT-spine (idealized feasibility) — the "true answer" the
    # wait-feasibility oracle reads. Same across policies; print from MCTS run.
    println("\n  ── clean WAIT+measure spine (zero-innovation; the feasibility truth) ──")
    ts = m0["wait_spine_t_h"]; ps = m0["wait_spine_pc"]
    for i in eachindex(ts)
        @printf("    %6.2f h   Pc=%.3e%s\n", ts[i], ps[i], ps[i] <= thr ? "   <-- <δ" : "")
    end
    cr = m0["crossing_h"]
    @printf("    durably-safe=%s   crossing(WAIT becomes safe)=%s\n",
            m0["wait_durably_safe"], cr === nothing ? "never" : @sprintf("%.2f h", cr))

    println("\n  ── policies (same loop, same belief/grid; DECISION differs) ──")
    @printf("    %-22s %-9s %-5s %-9s %-11s %-9s %-9s\n",
            "policy", "decision", "nMan", "Δv(m/s)", "beliefPcTCA", "mitigated?", "match?")
    for (plabel, m) in results
        mit = m["maneuver_mitigated"]
        @printf("    %-22s %-9s %-5d %-9.3f %-11.3e %-9s %-9s\n",
                plabel, m["actual_decision"], m["n_maneuvers"], m["total_dv_mps"],
                m["pc_at_tca"], mit === nothing ? "n/a" : string(mit),
                m["decision_matches_feasibility"])
    end
    println("    (beliefPcTCA = the NOISY realized belief Pc the policy acted on;")
    println("     match? = decision vs the clean-spine feasibility 'right call')")
end

println("\n", "="^92)
println("done — MCTS vs delay-gate vs wait-feasibility oracle on the two cases.")
println("="^92)
