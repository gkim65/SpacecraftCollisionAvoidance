# =========================================================================
# oscillation_pc_data.jl — READ-ONLY: the canonical "oscillation-aware Pc"
# characterization (experiment_ideas #1 / audit E3 / Figures.md "Figure C").
#
# THE RIPPLE (distinct from the maneuver-effectiveness burn-timing ripple): for a
# FIXED belief with NO burn, the node's Pc-at-TCA OSCILLATES once per orbit because
# the radial/cross-track covariance BREATHES (bounded CW sin/cos modes) as Σ is
# grown Φ Σ Φᵀ to TCA. The planner evaluates node_pc_at_tca only at discrete
# (hourly) decision points, so it can land on a lucky TROUGH or unlucky PEAK —
# adjacent hourly samples can differ by ~10× (Phase-3 finding). Basing a maneuver
# decision on one aliased sample is fragile. The fix: evaluate Pc over a WINDOW
# (±½ orbit, fine) around each decision point and drive the constraint off a window
# STATISTIC — max (worst-case) and/or mean (expected exposure).
#
# THIS SCRIPT (on the real SWIFT/JILIN case, tight payload Σ = the regime where the
# ripple SHOWS; it is washed out under a 1 km debris Σ):
#   • Walk a WAIT-only belief detection→TCA at a FINE sub-orbit dt (class cadence
#     corrections applied as the executor would), evaluating the REAL
#     node_pc_at_tca at every fine step → the resolved Pc(τ) ripple.
#   • Overlay the HOURLY point-samples the planner actually sees (the aliased grid).
#   • Compute, at each hourly decision point, the window-MAX and window-MEAN of Pc
#     over ±½ orbit → the oscillation-aware statistics.
#   • GRID-PHASE test: shift the hourly grid by fractions of an orbit and record how
#     much the point-sampled Pc at a fixed decision point moves (fragility) vs the
#     window-max (stable).
#
# Uses the REAL node_pc_at_tca (grows the node's accumulated Σ to TCA) — the exact
# quantity the reward/constraint see. NO MCTS, NO planner search; a pure belief
# walk + Pc evaluation.
#
# Usage:  julia --project=. figureScripts/oscillation_pc_data.jl
# =========================================================================

using LinearAlgebra, Random, PyCall, POMDPs, POMDPTools, Distributions, Printf

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
include(joinpath(@__DIR__, "..", "src", "utils", "cdmScenario.jl"))

const JILIN_CDM = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms",
    "000028485_conj_000044777_20220407_231108_20220406_140506.cdm"))

sc = load_cdm_scenario(JILIN_CDM)
pomdp = sc.pomdp
hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris

# Orbital period (ripple cadence), vis-viva from the primary state.
const MU = 3.986004418e14
r0 = norm(sc.b_tca.sc.μ[1:3]); v0 = norm(sc.b_tca.sc.μ[4:6])
a0 = 1.0/(2.0/r0 - v0^2/MU); period_s = 2π*sqrt(a0^3/MU)

@printf("case %s/%s  horizon %.1f h  period %.3f h  thr %.0e\n",
        sc.name1, sc.name2, sc.t_horizon/3600, period_s/3600, pomdp.pc_threshold)

# --- Fine WAIT-only belief walk, evaluating the REAL node_pc_at_tca -----------
# We grow the belief from the detection seed b0 toward TCA in FINE dt steps, apply
# class-cadence corrections (as the executor would — the ripple lives BETWEEN
# fixes), and evaluate node_pc_at_tca at each fine step. τ (time-to-TCA) decreases.
fine_dt = period_s / 24          # ~4 min → resolves the ~1.6 h ripple (24 pts/orbit)
function wait_walk(; dt = fine_dt)
    b = sc.b0
    since_sc = 0.0; since_db = 0.0
    τs = Float64[]; pcs = Float64[]
    # evaluate at the root first
    push!(τs, b.t); push!(pcs, node_pc_at_tca(pomdp, BeliefNode(b, sc.s_true, false)))
    while b.t - dt > 0.5
        b = predict(pomdp, b, WAIT; dt = dt)
        since_sc += dt; since_db += dt
        if since_sc >= pomdp.cadence_sc || since_db >= pomdp.cadence_debris
            z = vcat(b.sc.μ, b.debris.μ)     # zero-innovation (mean) fix
            since_sc >= pomdp.cadence_sc && (b = correct_linear_sc(pomdp, b, z); since_sc = 0.0)
            since_db >= pomdp.cadence_debris && (b = correct_linear_debris(pomdp, b, z); since_db = 0.0)
        end
        st = CAState(copy(b.sc.μ), copy(b.debris.μ), b.t)
        push!(τs, b.t); push!(pcs, node_pc_at_tca(pomdp, BeliefNode(b, st, false)))
    end
    return τs, pcs
end

println("walking WAIT belief at dt=", round(fine_dt/60, digits=1), " min …"); flush(stdout)
@time τs, pcs = wait_walk()
τh = τs ./ 3600
@printf("  %d fine samples; Pc range %.2e … %.2e (spans %.1f orders)\n",
        length(pcs), minimum(filter(>(0), pcs); init=1.0), maximum(pcs),
        log10(maximum(pcs) / max(minimum(filter(>(0), pcs); init=1e-300), 1e-300)))

# --- Hourly point-samples (what the planner sees) + window stats --------------
# Point grid: every 1 h (the planner's coarse grid). For each, window-max and
# window-mean over ±½ orbit, computed from the FINE walk (interpolate by nearest).
half_win_s = period_s / 2
hourly_τh = collect(1.0 : 1.0 : floor(sc.t_horizon/3600))
function nearest_pc(τh_query)
    i = argmin(abs.(τh .- τh_query)); return pcs[i]
end
function window_stat(τh_center)
    lo = τh_center - half_win_s/3600; hi = τh_center + half_win_s/3600
    idx = findall(t -> lo <= t <= hi, τh)
    isempty(idx) && return (NaN, NaN)
    w = pcs[idx]
    return (maximum(w), sum(w)/length(w))
end
hourly_pt   = [nearest_pc(t) for t in hourly_τh]
hourly_wmax = [window_stat(t)[1] for t in hourly_τh]
hourly_wmean= [window_stat(t)[2] for t in hourly_τh]

# --- Grid-phase fragility test ------------------------------------------------
# Shift the hourly grid phase across one orbit; at a fixed decision point (~12 h),
# record how much the POINT-sampled Pc moves vs the window-max. The point value
# should swing wildly with phase; the window-max should be stable.
decision_τh = 12.0
phase_shifts_h = collect(0.0 : period_s/3600/12 : period_s/3600)  # over one orbit
pt_vs_phase = Float64[]
for ph in phase_shifts_h
    push!(pt_vs_phase, nearest_pc(decision_τh + ph))
end
wmax_at_decision = window_stat(decision_τh)[1]

println("\nGrid-phase fragility @ ", decision_τh, " h decision point:")
@printf("  point-sample Pc across one orbit of grid phase: %.2e … %.2e (%.1f orders swing)\n",
        minimum(filter(>(0), pt_vs_phase); init=1.0), maximum(pt_vs_phase),
        log10(maximum(pt_vs_phase)/max(minimum(filter(>(0), pt_vs_phase); init=1e-300),1e-300)))
@printf("  window-max Pc (±½ orbit) at the same point: %.2e (single value, phase-independent)\n",
        wmax_at_decision)

# --- console: hourly point vs window-max, a few rows --------------------------
println("\nHourly decision points — point-sample vs window-max vs window-mean:")
@printf("%8s %12s %12s %12s\n", "τ(h)", "point", "win-max", "win-mean")
for (k, t) in enumerate(hourly_τh)
    (t in (2.0,4.0,6.0,8.0,12.0,16.0,24.0,32.0)) || continue
    @printf("%8.1f %12.2e %12.2e %12.2e\n", t, hourly_pt[k], hourly_wmax[k], hourly_wmean[k])
end

# --- JSON out -----------------------------------------------------------------
_j(x::Bool)=x ? "true" : "false"; _j(x::Real)=isfinite(x) ? string(x) : "null"
_j(x::AbstractString)="\""*replace(x,"\""=>"\\\"")*"\""; _j(x::Nothing)="null"
_j(v::AbstractVector)="["*join(_j.(v),",")*"]"
_j(d::AbstractDict)="{"*join(["\"$k\":"*_j(v) for (k,v) in d],",")*"}"

outdir = joinpath(@__DIR__, "data"); isdir(outdir) || mkpath(outdir)
payload = Dict(
    "case"=>Dict("name1"=>sc.name1, "name2"=>sc.name2, "sec_class"=>String(sc.sec_class),
                 "horizon_h"=>sc.t_horizon/3600, "period_h"=>period_s/3600,
                 "hbr_m"=>hbr, "pc_threshold"=>pomdp.pc_threshold,
                 "fine_dt_min"=>fine_dt/60, "half_window_orbit"=>0.5),
    "fine_tau_h"=>τh, "fine_pc"=>pcs,                       # the resolved ripple
    "hourly_tau_h"=>hourly_τh,
    "hourly_point_pc"=>hourly_pt,                           # aliased planner samples
    "hourly_winmax_pc"=>hourly_wmax, "hourly_winmean_pc"=>hourly_wmean,
    "phase_test"=>Dict("decision_tau_h"=>decision_τh,
                       "phase_shifts_h"=>phase_shifts_h,
                       "point_pc_vs_phase"=>pt_vs_phase,
                       "winmax_pc"=>wmax_at_decision),
    "note"=>"WAIT-only belief, real node_pc_at_tca; ripple = R/N covariance breathing under Φ Σ Φᵀ; window = ±½ orbit over the decision point")
open(joinpath(outdir, "oscillation_pc.json"), "w") do io; write(io, _j(payload)); end
println("\nwrote figureScripts/data/oscillation_pc.json")
