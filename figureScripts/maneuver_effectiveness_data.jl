# =========================================================================
# maneuver_effectiveness_data.jl — READ-ONLY physics characterization.
#
# QUESTION (prerequisite for the reward/constraint redesign): how does a SINGLE
# impulsive along-track Δv mitigate a real CARA conjunction, as a function of
# burn MAGNITUDE (M) and burn TIMING (time-to-TCA, T)?  No MCTS, no planner, no
# reward — pure "apply the burn, propagate the mean to TCA, compute Pc-at-TCA".
#
# METHOD (per (M, T) grid point):
#   1. Back-propagate the primary's REAL CDM TCA mean state to the burn epoch
#      (TCA − T) under the accurate force model.  (Debris mean too — for the
#      forward leg.)
#   2. Apply a signed along-track impulse Δv = M · v̂  (M > 0 speeds up / +v̂,
#      M < 0 slows down / −v̂) to the primary velocity at the burn epoch.
#   3. Forward-propagate the burned primary mean to TCA.  Debris propagated to
#      TCA unburned.
#   4. Evaluate elrod_pc at TCA on the two propagated ECI means + the REAL CDM
#      TCA covariances + the real combined HBR.
#
# WHY HOLD Σ AT THE REAL CDM TCA VALUE: a small impulsive burn does not
# materially change the position covariance; holding Σ fixed at the real NASA
# endpoint isolates the MISS-GEOMETRY effect of the burn, which is exactly what
# a maneuver-effectiveness surface should show.  The Pc endpoint (states + cov)
# is real NASA data; only the counterfactual burn is ours.
#
# CONVENTION: along-track = along the primary velocity unit vector v̂ (RTN "T"
# for a near-circular orbit).  We sweep BOTH signs (+v̂ / −v̂).  Radial /
# cross-track burns change the geometry differently and are NOT swept this
# session (noted in the findings) — an along-track burn moves the along-track
# separation, which for a crossing/overtaking geometry is the dominant lever.
#
# OUTPUT: figureScripts/data/maneuver_effectiveness.json  (consumed by the
# matplotlib plotter).  Also prints the baseline Pc + a compact summary.
#
# Usage:  julia --project=. figureScripts/maneuver_effectiveness_data.jl
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

# -------------------------------------------------------------------------
# Mean-only propagation helpers (no covariance — this is a pure-geometry sweep).
# propagate_mean(state, params, epochTCA, t_from, t_to): propagate a 6-ECI mean
# from time-to-TCA `t_from` to time-to-TCA `t_to` (both seconds; larger = earlier)
# under the POMDP force model.  Positive Δt = forward in real time.
# -------------------------------------------------------------------------
function propagate_mean(pomdp, μ::AbstractVector, params, t_from::Real, t_to::Real)
    bh = get_brahe()
    epoch_tca = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_from = epoch_tca - Float64(t_from)
    epoch_to   = epoch_tca - Float64(t_to)
    epoch_from_tuple = epoch_to_tuple(epoch_from)
    prop, _ = eci2orb_brahe(collect(float.(μ)), epoch_from_tuple, params, pomdp.forceModel)
    prop.propagate_to(epoch_to)
    return collect(prop.current_state()[1:6])
end

# Apply a signed along-track impulse of magnitude `m` (m/s; sign = direction)
# to a 6-ECI state's velocity (along +v̂).
function apply_along_track(μ::AbstractVector, m::Real)
    v = μ[4:6]
    v_hat = v ./ norm(v)
    return vcat(μ[1:3], v .+ Float64(m) .* v_hat)
end

# -------------------------------------------------------------------------
# Load the real case.  Use the CDM's own TCA states + covariances (b_tca is the
# untouched real endpoint) — this is the physics anchor, not the planner seed.
# -------------------------------------------------------------------------
sc = load_cdm_scenario(JILIN_CDM)
pomdp = sc.pomdp
hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris

# REAL TCA means (from b_tca, anchored at time-remaining 0.5 s ≈ TCA).
μ_sc_tca = copy(sc.b_tca.sc.μ)
μ_db_tca = copy(sc.b_tca.debris.μ)
Σ_sc_tca = Matrix(sc.b_tca.sc.Σ)
Σ_db_tca = Matrix(sc.b_tca.debris.Σ)

horizon = sc.t_horizon                       # s, creation→TCA lead time (~33 h)
Δv_current = pomdp.Δv                          # the planner's current impulse magnitude

# Baseline (no-burn) Pc at TCA on the real endpoint — should reproduce CARA.
pc_baseline = elrod_pc(μ_sc_tca, μ_db_tca, Σ_sc_tca, Σ_db_tca, hbr)

println("="^72)
println("MANEUVER EFFECTIVENESS SWEEP — ", sc.name1, " vs ", sc.name2,
        "  (sec_class=", sc.sec_class, ")")
println("="^72)
@printf("horizon (creation→TCA)   = %.2f h\n", horizon/3600)
@printf("combined HBR             = %.1f m\n", hbr)
@printf("miss distance @ TCA      = %.1f m   (CDM)\n", sc.miss_distance)
@printf("relative speed @ TCA     = %.1f m/s (CDM)\n", sc.relative_speed)
@printf("current planner Δv       = %.3f m/s  (single impulsive along-track)\n", Δv_current)
@printf("CARA operational Pc      = %.4e\n", sc.pc_cdm)
@printf("our elrod Pc @ TCA (no burn) = %.4e   (baseline)\n", pc_baseline)
println()

# -------------------------------------------------------------------------
# SWEEP GRID (finalized with Grace after the CA-Δv literature review —
# notes/ca_maneuver_deltav_litreview.md).
#   Magnitudes: SIGNED along-track, cm/s-FOCUSED (real LEO CA burns are ~1 cm/s
#   to a few tens of cm/s; our pomdp.Δv=0.1 is already at the operational high
#   end). Log-dense from ±1 mm/s through ±0.5 m/s (the literature upper end),
#   symmetric, includes 0 (no-burn baseline) and the current Δv (0.1).
#   Timings: ~0.5 h steps out to the full ~33 h horizon — oversamples the
#   once-per-orbit (~1.5 h) phasing ripple (CW bounded ΔD_R ≈ 2Δa term) so the
#   surface is smooth, not aliased.
# -------------------------------------------------------------------------
mags_pos = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07,
            0.1, 0.15, 0.2, 0.3, 0.5]                          # m/s, |M|
# DENSE in the 3–30 mm/s feasibility band (where the Pc=thr boundary lives);
# coarser above 0.05 (settled "more than enough").
mags = sort(unique(vcat(-reverse(mags_pos), mags_pos)))       # signed, symmetric, 0 once

# Timings (h before TCA): NON-UNIFORM — the once-per-orbit ripple (~1.5 h) and
# the along-track-only-breaks-down regime both live near TCA, so resolve it
# finely there. 5 min steps 0–5 h, 15 min 5–15 h, 1 h 15 h–horizon.
timings_h = sort(unique(vcat(
    collect(5/60 : 5/60 : 5.0),      # 5 min steps, 0–5 h (finest — the ripple)
    collect(5.25 : 0.25 : 15.0),     # 15 min steps, 5–15 h
    collect(15.5 : 0.5  : 33.0))))   # 30 min steps, 15–33 h
timings_s = timings_h .* 3600.0
# Clamp any timing beyond the available lead time.
timings_s = filter(t -> t <= horizon + 1.0, timings_s)
timings_h = timings_s ./ 3600.0

@printf("magnitude grid (m/s): %s\n", string(round.(mags, digits=3)))
@printf("timing grid (h):      %s\n", string(round.(timings_h, digits=1)))
@printf("grid: %d magnitudes × %d timings = %d Pc evaluations\n\n",
        length(mags), length(timings_s), length(mags)*length(timings_s))
flush(stdout)

# Debris mean at TCA does not depend on the burn — but it does not need
# re-propagation either (it IS the TCA mean).  We only ever propagate the
# PRIMARY (back to the burn epoch, then forward to TCA after the burn).

# pc_grid[i, j] = Pc for magnitude mags[i] at timing timings_s[j].
pc_grid = fill(NaN, length(mags), length(timings_s))
missdist_grid = fill(NaN, length(mags), length(timings_s))  # along the way: miss @ TCA (m)

for (j, T) in enumerate(timings_s)
    # Back-propagate the primary TCA mean to the burn epoch (TCA − T), ONCE per T.
    μ_sc_burn_epoch = propagate_mean(pomdp, μ_sc_tca, pomdp.satParams, 0.5, T)
    for (i, m) in enumerate(mags)
        μ_burned = m == 0.0 ? μ_sc_burn_epoch : apply_along_track(μ_sc_burn_epoch, m)
        μ_sc_at_tca = propagate_mean(pomdp, μ_burned, pomdp.satParams, T, 0.5)
        pc = elrod_pc(μ_sc_at_tca, μ_db_tca, Σ_sc_tca, Σ_db_tca, hbr)
        pc_grid[i, j] = pc
        missdist_grid[i, j] = norm(μ_sc_at_tca[1:3] .- μ_db_tca[1:3])
    end
    @printf("  T=%5.1f h done  (Pc range %.2e … %.2e)\n",
            T/3600, minimum(pc_grid[:, j]), maximum(pc_grid[:, j]))
    flush(stdout)
end
println()

# -------------------------------------------------------------------------
# Latest-time-you-can-still-fix-it, per magnitude: the LARGEST T (earliest is
# easiest; we want the smallest lead time) at which |M| still drives Pc below
# threshold.  Report as the smallest time-to-TCA that still clears threshold,
# i.e. how late you can leave a burn of that magnitude.
# -------------------------------------------------------------------------
thr = pomdp.pc_threshold
latest_fix_h = Dict{Float64,Any}()
for (i, m) in enumerate(mags)
    below = [pc_grid[i, j] < thr for j in eachindex(timings_s)]
    # timings are increasing h; a burn is "still fixable" at time-to-TCA T if
    # Pc<thr there.  The LATEST fixable = min T with Pc<thr.
    idxs = findall(below)
    latest_fix_h[m] = isempty(idxs) ? nothing : minimum(timings_h[idxs])
end

println("Latest fixable time-to-TCA (h) per |signed Δv| (min T with Pc<thr=$(thr)):")
for m in mags
    v = latest_fix_h[m]
    @printf("  Δv=%+6.3f m/s : %s\n", m, v === nothing ? "never (Pc≥thr at all T)" : @sprintf("%.1f h", v))
end
println()

# -------------------------------------------------------------------------
# ANALYTIC CHECK: the Clohessy–Wiltshire secular along-track drift law says a
# tangential impulse Δv produces along-track separation ΔD_T ≈ 3·Δv·ΔT (see
# notes/ca_maneuver_deltav_litreview.md). We record it so the plotter can
# overlay it on the measured MISS-DISTANCE shift and confirm our brahe
# propagation reproduces the textbook physics. Also record the primary orbital
# period (the ripple should be once-per-period).
# -------------------------------------------------------------------------
const MU_EARTH = 3.986004418e14   # m³/s² (WGS-84 / EGM); orbital period only
r_sc = norm(μ_sc_tca[1:3]); v_sc = norm(μ_sc_tca[4:6])
a_sc = 1.0 / (2.0 / r_sc - v_sc^2 / MU_EARTH)     # vis-viva semi-major axis (m)
period_s = 2π * sqrt(a_sc^3 / MU_EARTH)            # s
@printf("primary a = %.1f km, orbital period = %.2f h (ripple should be once/period)\n",
        a_sc/1000, period_s/3600)

# Predicted along-track drift ΔD_T = 3·|Δv|·T (m), per (|Δv|, T). Sign-agnostic
# magnitude of the induced separation; the plotter compares |measured miss −
# baseline miss| against this on the +v̂ branch.
dD_pred = [[3.0 * abs(mags[i]) * timings_s[j] for j in eachindex(timings_s)]
           for i in eachindex(mags)]

# -------------------------------------------------------------------------
# Write JSON.
# -------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "data")
isdir(outdir) || mkpath(outdir)
outpath = joinpath(outdir, "maneuver_effectiveness.json")

# Hand-rolled JSON writer (matches project convention — JSON is only a
# transitive dep under --project=., so figureScripts serialize by hand).
_j(x::Bool) = x ? "true" : "false"
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = "\"" * replace(x, "\"" => "\\\"") * "\""
_j(x::Nothing) = "null"
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
_j(d::AbstractDict) = "{" * join(["\"$k\":" * _j(v) for (k, v) in d], ",") * "}"

payload = Dict(
    "case" => Dict(
        "name1" => sc.name1, "name2" => sc.name2,
        "id1" => sc.id1, "id2" => sc.id2,
        "sec_class" => String(sc.sec_class),
        "horizon_h" => horizon/3600,
        "hbr_m" => hbr,
        "miss_distance_m" => sc.miss_distance,
        "relative_speed_ms" => sc.relative_speed,
        "pc_cara" => sc.pc_cdm,
        "pc_baseline" => pc_baseline,
        "dv_current_ms" => Δv_current,
        "pc_threshold" => thr,
        "tca" => sc.tca,
        "orbital_period_h" => period_s/3600,
        "sma_km" => a_sc/1000,
    ),
    "magnitudes_ms" => collect(mags),          # signed along-track Δv
    "timings_h" => collect(timings_h),         # time-to-TCA of the burn
    # pc[i][j] = Pc for magnitudes[i] at timings[j]  (row-major over magnitude)
    "pc" => [[pc_grid[i, j] for j in eachindex(timings_s)] for i in eachindex(mags)],
    "miss_distance_m_grid" => [[missdist_grid[i, j] for j in eachindex(timings_s)] for i in eachindex(mags)],
    # ΔD_T = 3·|Δv|·T predicted along-track separation (m), CW secular law.
    "dD_predicted_m_grid" => dD_pred,
    "latest_fixable_h" => Dict(string(m) => latest_fix_h[m] for m in mags),
    "convention" => "along-track burn along +v̂ (M>0 speeds up); Σ held at real CDM TCA value; means propagated under accurate force model; Pc via elrod_pc",
)

open(outpath, "w") do io
    write(io, _j(payload))
end
println("wrote ", outpath)
