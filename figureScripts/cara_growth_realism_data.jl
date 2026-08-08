# =========================================================================
# cara_growth_realism_data.jl — audit F3, CORE deliverable: is our Σ(τ)
# covariance GROWTH realistic vs the real primary-object growth curve?
#
# WHAT IS UNDER TEST: the Φ Σ Φᵀ STM propagation (brahe covariance_gcrf /
# covariance_rtn) used by beliefTracker.jl `predict` and beliefMCTS.jl
# `_grow_belief_to_tca`. build_covariance_table (covarianceTable.jl) is used
# ONLY as a convenient offline way to SWEEP that same Φ Σ Φᵀ growth over a τ
# grid — the table is the tool, not the thing validated (see the audit's FINAL
# Prompt-3 scoping block).
#
# DESIGN (locked with Grace 2026-08-08):
#   Our growth model is a LINEAR map Σ(τ) = Φ(τ) Σ₀ Φ(τ)ᵀ — it transforms a
#   seed, it does not create uncertainty. So a seed Σ₀ is unavoidable. The real
#   CDMs can't be the seed for OUR curve (each is a single TCA snapshot, no
#   time series), so we ANCHOR our propagator at the real primary curve's
#   short-lead (<1 d) covariance and let Φ Σ Φᵀ PREDICT the longer leads, then
#   check the prediction against the real measured covariances at those leads.
#
#   Real PRIMARY growth curve (measured by the deep-dive Claude, pooling all 53
#   well-tracked NASA-asset primaries across lead-time bins — R/T/N 1σ, m):
#     lead <1 d:  R 3.6   in-track 38     cross-track 1.8
#     lead 1-2 d: R 5.1   in-track 228    cross-track 3.2
#     lead 2-4 d: R 8.5   in-track 906    cross-track 4.9
#     lead >4 d:  R 22.5  in-track 5588   cross-track 5.3
#   i.e. in-track grows ~150×; radial ~6×; cross-track ~3× (nearly flat).
#
#   TWO SEEDS, both grown through the SAME Φ Σ Φᵀ:
#     (A) anchored  — R 3.6 / in-track 38 / cross-track 1.8 m (matches real <1 d).
#                     Divergence downstream is then PURE growth-model error.
#     (B) isotropic — 10 m/axis (shape-sensitivity contrast; what a round P0 does).
#   Plus the shipped production P0_sc (10 m isotropic) is the same family as (B).
#
# CAVEAT (state in the note): the real curve pools DIFFERENT events (confounded
# by OBS_USED / OD span), not one object's trajectory. Small confound for a
# homogeneous well-tracked class, but real. NOT a single-event growth validation.
#
# READ-ONLY. Changes no src/ production code. Output → JSON for matplotlib.
# Run from repo root:  julia --project=. figureScripts/cara_growth_realism_data.jl
# writes figureScripts/data/cara_growth_realism.json
# =========================================================================
using LinearAlgebra
using Random
using PyCall
using POMDPs
using POMDPTools
using Printf

# Minimal JSON writer (avoid adding JSON to the project's direct deps).
_json(x::Bool) = x ? "true" : "false"
_json(x::Real) = isfinite(x) ? string(x) : "null"
_json(x::AbstractString) = "\"$x\""
_json(v::AbstractVector) = "[" * join(_json.(v), ",") * "]"
_json(m::AbstractMatrix) = "[" * join([_json(collect(m[i, :])) for i in 1:size(m, 1)], ",") * "]"
_json(d::AbstractDict) = "{" * join(["\"$k\":" * _json(v) for (k, v) in d], ",") * "}"

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(REPO, "src", "utils", "genConjunctions.jl"))
include(joinpath(REPO, "src", "utils", "computePc.jl"))
include(joinpath(REPO, "src", "utils", "covarianceTable.jl"))

# ---------------------------------------------------------------------------
# Real primary growth curve (the ground truth we compare against). Bin midpoints
# used as the representative lead for each bin: <1 d → 0.5 d, 1-2 d → 1.5 d,
# 2-4 d → 3 d, >4 d → 5.5 d (median of the deep-dive's >4 d cases ≈ 5.5 d).
# ---------------------------------------------------------------------------
const REAL_BINS = [
    ("<1 d",  0.5, 3.6,  38.0,   1.8),
    ("1-2 d", 1.5, 5.1,  228.0,  3.2),
    ("2-4 d", 3.0, 8.5,  906.0,  4.9),
    (">4 d",  5.5, 22.5, 5588.0, 5.3),
]

# ---------------------------------------------------------------------------
# Build a POMDP with a chosen seed P0_sc, sweep Φ Σ Φᵀ out to `window` days on a
# `dt`-grid, and return τ (days) + the spacecraft RTN R/T/N 1σ curves.
# RTN axis 1 = radial (R), 2 = transverse / along-track (T), 3 = normal / cross (N).
# randAdd=false → deterministic 400 km / i=75° LEO, a well-tracked-asset-like orbit.
# The mean geometry only sets the STM reference; we sweep the SC object's Σ.
# ---------------------------------------------------------------------------
sig(tbl, fld, i) = [sqrt(getfield(tbl, fld)[k][i, i]) for k in 1:tbl.n_steps]

# RTN→ECI 6×6 block-diagonal rotation from an ECI state [r; v] (m, m/s).
#   R̂ = r̂,  N̂ = (r×v)̂,  T̂ = N̂ × R̂.   A = [R̂ T̂ N̂] maps RTN→ECI.
# Σ_eci = Q Σ_rtn Qᵀ with Q = blkdiag(A, A). This lets us seed P0 in RTN (so its
# R/T/N diagonal means what we intend) and hand brahe the equivalent ECI seed,
# so covariance_rtn read-back at τ=0 recovers the intended R/T/N (verified below).
function rtn_to_eci_rotation(sc_eci::AbstractVector)
    r = sc_eci[1:3]; v = sc_eci[4:6]
    R̂ = r / norm(r)
    N̂ = cross(r, v); N̂ = N̂ / norm(N̂)
    T̂ = cross(N̂, R̂)
    A = hcat(R̂, T̂, N̂)                      # columns = RTN axes in ECI
    Q = zeros(6, 6); Q[1:3, 1:3] = A; Q[4:6, 4:6] = A
    return Q
end

# Fidelity: 5-min sweep step so the once-per-orbit (~93 min) R/N ripple is sampled
# ~19×/cycle → smooth lines (not the choppy ~3×/cycle a 30-min step gives).
const DT_SWEEP = 300.0

# ANCHOR LEAD: the real <1 d bin is reported at its midpoint lead 0.5 d. The seed
# Σ₀ IS "the covariance at that bin," so we place it at lead 0.5 d and grow FORWARD
# from there. The returned tau_days is then anchor_lead + swept-time, so our curve
# LITERALLY STARTS at the real <1 d marker (same timestamp 0.5 d, same magnitude
# 38 m in-track) and only diverges afterward — the anchoring is visible, not implied.
const ANCHOR_LEAD_DAYS = 0.5

function sweep_growth(P0_sc_rtn; window_days = 7.0, dt = DT_SWEEP,
                      anchor_lead_days = ANCHOR_LEAD_DAYS)
    # Build the geometry first (needs a POMDP), get the SC ECI state at TCA,
    # rotate the RTN seed into ECI, then rebuild the POMDP with that ECI P0.
    pomdp0 = SpacecraftCAPOMDP(seed = 42, randAdd = false)
    sc_eci, debris_eci = generate_conjunction_geometry(pomdp0;
        geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)
    Q = rtn_to_eci_rotation(sc_eci)
    P0_sc_eci = Q * Matrix{Float64}(P0_sc_rtn) * transpose(Q)
    pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, P0_sc = P0_sc_eci)
    # Sweep the coast growth. The seed Σ₀ represents the covariance AT the anchor
    # lead, so the plotted lead time is anchor_lead + swept-coast: our curve begins
    # exactly at the anchor point (0.5 d, seed magnitude) and grows forward. We
    # prepend the τ=0 seed point so the line literally starts on the anchor marker.
    tbl = build_covariance_table(pomdp, sc_eci, debris_eci;
        dt = dt, tca_window = (window_days - anchor_lead_days) * 86400.0, verbose = false)
    seed_R = sqrt(Matrix{Float64}(P0_sc_rtn)[1, 1])
    seed_T = sqrt(Matrix{Float64}(P0_sc_rtn)[2, 2])
    seed_N = sqrt(Matrix{Float64}(P0_sc_rtn)[3, 3])
    return Dict(
        "tau_days" => vcat(anchor_lead_days, anchor_lead_days .+ tbl.τ_s ./ 86400.0),
        "R" => vcat(seed_R, sig(tbl, :Σ_sc_rtn, 1)),
        "T" => vcat(seed_T, sig(tbl, :Σ_sc_rtn, 2)),
        "N" => vcat(seed_N, sig(tbl, :Σ_sc_rtn, 3)),
    )
end

# Sample a swept curve at the real-bin lead midpoints (nearest τ grid point).
function at_leads(curve, lead_days_list)
    τ = curve["tau_days"]
    out = Dict{String,Vector{Float64}}("R" => Float64[], "T" => Float64[], "N" => Float64[])
    for ld in lead_days_list
        k = argmin(abs.(τ .- ld))
        for ax in ("R", "T", "N")
            push!(out[ax], curve[ax][k])
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# The two seeds. Position variances on the diagonal (m²); a small isotropic
# velocity term (0.01 m/s → 1e-4 m²/s², the production P0_sc velocity value) so
# the STM has something physical to propagate on the velocity block.
# ---------------------------------------------------------------------------
vel_var = 1e-4  # (m/s)² — production P0_sc velocity diagonal
P0_anchored  = diagm([3.6^2, 38.0^2, 1.8^2, vel_var, vel_var, vel_var])   # real <1 d RTN shape
P0_isotropic = diagm([10.0^2, 10.0^2, 10.0^2, vel_var, vel_var, vel_var]) # round 10 m/axis
P0_prod      = diagm([100.0, 100.0, 100.0, 1e-4, 1e-4, 1e-4])             # shipped default (10 m iso)

# P0 is specified in RTN here (R/T/N diagonal); sweep_growth rotates it into ECI
# via rtn_to_eci_rotation before handing it to the POMDP, so brahe's τ=0
# covariance_rtn read-back recovers the intended R/T/N diagonal. We still report
# the ACTUAL τ=0 RTN read-back alongside the intended values as an honesty check
# that the rotation round-trips (isotropic seeds are rotation-invariant either way).

println("Sweeping Φ Σ Φᵀ growth (anchored, isotropic, production seeds)...")
lead_mids = [b[2] for b in REAL_BINS]

curve_anchored  = sweep_growth(P0_anchored)
curve_isotropic = sweep_growth(P0_isotropic)
curve_prod      = sweep_growth(P0_prod)

sampled_anchored  = at_leads(curve_anchored,  lead_mids)
sampled_isotropic = at_leads(curve_isotropic, lead_mids)
sampled_prod      = at_leads(curve_prod,      lead_mids)

# τ=0 RTN read-back (frame-honesty check): sweep a tiny window with NO anchor shift
# and read the first grown point (index 2; index 1 is the prepended intended seed).
tiny = sweep_growth(P0_anchored; window_days = 0.02, dt = 60.0, anchor_lead_days = 0.0)
tau0_anchored = Dict("R" => tiny["R"][2], "T" => tiny["T"][2], "N" => tiny["N"][2])

# ---------------------------------------------------------------------------
# Growth-law diagnostic: fit log(σ_T) = a + p·log(τ) over the swept range for the
# in-track (T) axis of each seed, and the same on the REAL bin points. p≈1 →
# velocity-dominated (σ∝τ); p≈1.5 → SMA/energy-dominated (σ∝τ^1.5). This is the
# audit's "velocity-dominated (σ∝τ) vs SMA-dominated (σ∝τ^1.5)" question.
# Fit only where lead > 0.55 d (skip the prepended anchor seed point at 0.5 d).
# ---------------------------------------------------------------------------
function powerlaw_exponent(τ_days, σ; τmin = 0.55)
    m = τ_days .>= τmin
    x = log.(τ_days[m]); y = log.(σ[m])
    n = length(x); xm = sum(x)/n; ym = sum(y)/n
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm).^2)
end

p_anchored_T  = powerlaw_exponent(curve_anchored["tau_days"],  curve_anchored["T"])
p_isotropic_T = powerlaw_exponent(curve_isotropic["tau_days"], curve_isotropic["T"])
# Real bins: fit on the 4 (lead, in-track) points, all leads.
real_leads = [b[2] for b in REAL_BINS]
real_T     = [b[4] for b in REAL_BINS]
p_real_T   = powerlaw_exponent(real_leads, real_T; τmin = 0.0)

# In-track growth factor <1 d → >4 d (the headline ~150×).
gf_real      = REAL_BINS[4][4] / REAL_BINS[1][4]
gf_anchored  = sampled_anchored["T"][4]  / sampled_anchored["T"][1]
gf_isotropic = sampled_isotropic["T"][4] / sampled_isotropic["T"][1]

data = Dict(
    "real_bins" => Dict(
        "label"   => [b[1] for b in REAL_BINS],
        "lead_d"  => [b[2] for b in REAL_BINS],
        "R"       => [b[3] for b in REAL_BINS],
        "T"       => [b[4] for b in REAL_BINS],
        "N"       => [b[5] for b in REAL_BINS],
    ),
    "curve_anchored"  => curve_anchored,
    "curve_isotropic" => curve_isotropic,
    "curve_prod"      => curve_prod,
    "sampled_anchored"  => Dict(k => sampled_anchored[k]  for k in ("R","T","N")),
    "sampled_isotropic" => Dict(k => sampled_isotropic[k] for k in ("R","T","N")),
    "sampled_prod"      => Dict(k => sampled_prod[k]      for k in ("R","T","N")),
    "lead_mids" => lead_mids,
    "tau0_anchored_readback" => tau0_anchored,
    "tau0_anchored_intended" => Dict("R" => 3.6, "T" => 38.0, "N" => 1.8),
    "growth_law" => Dict(
        "p_real_T" => p_real_T,
        "p_anchored_T" => p_anchored_T,
        "p_isotropic_T" => p_isotropic_T,
    ),
    "intrack_growth_factor" => Dict(
        "real" => gf_real, "anchored" => gf_anchored, "isotropic" => gf_isotropic,
    ),
)

out = joinpath(@__DIR__, "data", "cara_growth_realism.json")
open(out, "w") do io; write(io, _json(data)); end
println("wrote $out")
println()
println("τ=0 RTN read-back (anchored seed): R=$(round(tau0_anchored["R"],digits=2)) ",
        "T=$(round(tau0_anchored["T"],digits=2)) N=$(round(tau0_anchored["N"],digits=2)) m ",
        "(intended 3.6 / 38 / 1.8)")
println()
@printf "In-track σ (m) at bin-midpoint leads — REAL vs OUR growth:\n"
@printf "%-7s %10s %12s %12s %12s\n" "bin" "lead(d)" "real_T" "anchored_T" "iso_T"
for (i, b) in enumerate(REAL_BINS)
    @printf "%-7s %10.1f %12.1f %12.1f %12.1f\n" b[1] b[2] b[4] sampled_anchored["T"][i] sampled_isotropic["T"][i]
end
println()
@printf "In-track growth factor <1d→>4d:  real %.0f×  anchored %.0f×  iso %.0f×\n" gf_real gf_anchored gf_isotropic
@printf "In-track power-law exponent p (σ_T∝τ^p):  real %.2f  anchored %.2f  iso %.2f\n" p_real_T p_anchored_T p_isotropic_T
