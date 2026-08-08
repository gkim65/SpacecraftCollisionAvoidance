# =========================================================================
# cara_missigma_transfer_data.jl — audit F3, SECOND deliverable: does the
# 2D-validity result TRANSFER to our system? (the "so what" of the growth study)
#
# Prompt 2 (notes/cara_2d3d_gap_findings.md) proved that Mahalanobis MISS-IN-σ is
# THE gate on 2D-vs-3D validity (Spearman −0.80): below ~2σ the exact 2D ≈ 3D,
# beyond it the gap opens. That result used the REAL CDM covariance. It only
# transfers to OUR planner if OUR grown Σ puts the miss at a similar tail depth.
# If our Σ is the wrong size/shape, our miss-in-σ differs → our Pc sits in a
# different validity regime than reality → "2D valid at 24 h" would NOT carry.
#
# WHAT THIS DOES: for each real case, hold the REAL relative geometry (r,v of both
# objects, from the CDM — this is the real encounter plane + real miss vector) but
# SUBSTITUTE our grown Σ for the combined covariance, then recompute miss-in-σ:
#     miss-in-σ = sqrt( muᵀ C⁻¹ mu ),  C = B (Σ_sc + Σ_debris) Bᵀ  (encounter plane)
# We do this for our TWO SC seeds (anchored to the real <1 d shape / isotropic
# 10 m) grown by Φ Σ Φᵀ to that case's REAL lead time, with the debris on the
# production P0_debris (1 km isotropic) grown the same way. Compared against the
# REAL miss-in-σ already in cara_2d3d_joined.json (miss_maha).
#
# CAVEAT: our grown Σ replaces the real Σ but the geometry is real — this isolates
# "if our covariance model held, where would the same encounters land in the
# tail?" It is the downstream consequence of the growth realism in
# cara_growth_realism_data.jl, not an independent result.
#
# READ-ONLY. Changes no src/ production code. Consumes the vendored CDMs +
# cara_2d3d_joined.json. Output → JSON for matplotlib.
# Run from repo root:  julia --project=. figureScripts/cara_missigma_transfer_data.jl
# writes figureScripts/data/cara_missigma_transfer.json
# =========================================================================
using LinearAlgebra
using Printf
using Statistics: median
using Random
using PyCall
using POMDPs
using POMDPTools

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "tests", "cdmParser.jl"))

# computePc.jl / covarianceTable need SpacecraftCAPOMDP; the growth sweep needs
# the full planner stack. Load it (same as phase3_covariance_data.jl).
include(joinpath(REPO, "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(REPO, "src", "utils", "genConjunctions.jl"))
include(joinpath(REPO, "src", "utils", "computePc.jl"))
include(joinpath(REPO, "src", "utils", "covarianceTable.jl"))

# ---------------------------------------------------------------------------
# Grow one seed Σ (RTN diagonal, m² / (m/s)²) by our Φ Σ Φᵀ to lead time `lead_s`,
# return the ECI covariance at that lead. Reuses build_covariance_table's exact
# growth path (the mechanism under test). We seed in RTN and rotate into ECI so
# the RTN diagonal means what we intend, then read back the ECI covariance at the
# grid point nearest `lead_s`.
# ---------------------------------------------------------------------------
function rtn_to_eci_rotation(sc_eci::AbstractVector)
    r = sc_eci[1:3]; v = sc_eci[4:6]
    R̂ = r / norm(r)
    N̂ = cross(r, v); N̂ = N̂ / norm(N̂)
    T̂ = cross(N̂, R̂)
    A = hcat(R̂, T̂, N̂)
    Q = zeros(6, 6); Q[1:3, 1:3] = A; Q[4:6, 4:6] = A
    return Q
end

# Precompute the two SC-seed grown ECI Σ curves + the debris grown ECI Σ curve
# ONCE on a fixed reference orbit (our nominal 400 km / i=75° LEO), then look up
# by lead time per case. Our growth is a pure function of τ (noiseless), so a
# single reference sweep gives Σ(τ) for any case's lead; the geometry substituted
# per-case is the REAL CDM geometry, not this reference orbit's.
const VEL_VAR = 1e-4
const P0_ANCHORED  = diagm([3.6^2, 38.0^2, 1.8^2, VEL_VAR, VEL_VAR, VEL_VAR])
const P0_ISOTROPIC = diagm([10.0^2, 10.0^2, 10.0^2, VEL_VAR, VEL_VAR, VEL_VAR])

function build_reference_curves(; window_days = 7.0, dt = 300.0)   # 5-min fidelity
    pomdp0 = SpacecraftCAPOMDP(seed = 42, randAdd = false)
    sc_eci, debris_eci = generate_conjunction_geometry(pomdp0;
        geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)
    Q = rtn_to_eci_rotation(sc_eci)

    # SC seeds rotated into ECI; debris uses the production P0_debris (1 km iso,
    # already ECI-isotropic → rotation-invariant).
    pomdp_anch = SpacecraftCAPOMDP(seed = 42, randAdd = false,
        P0_sc = Q * P0_ANCHORED * transpose(Q))
    pomdp_iso  = SpacecraftCAPOMDP(seed = 42, randAdd = false,
        P0_sc = Q * P0_ISOTROPIC * transpose(Q))

    win = window_days * 86400.0
    tbl_anch = build_covariance_table(pomdp_anch, sc_eci, debris_eci; dt = dt, tca_window = win, verbose = false)
    tbl_iso  = build_covariance_table(pomdp_iso,  sc_eci, debris_eci; dt = dt, tca_window = win, verbose = false)
    return (τ = tbl_anch.τ_s,
            sc_anch = tbl_anch.Σ_sc_eci,
            sc_iso  = tbl_iso.Σ_sc_eci,
            debris  = tbl_anch.Σ_debris_eci)   # debris P0 identical across the two
end

nearest_idx(τ_grid, lead_s) = argmin(abs.(τ_grid .- lead_s))

# Miss-in-σ from real geometry (states) + a supplied pair of ECI covariances.
# s1/s2 are full 6-vector ECI states (pos+vel) as parse_cdm returns them.
function missigma(s1, s2, C1, C2)
    mu2d, C2d = encounter_plane(s1[1:3], s1[4:6], s2[1:3], s2[4:6], C1, C2)
    Cm = Matrix(C2d)
    det(Cm) <= 0 && return NaN
    return sqrt(max(0.0, dot(mu2d, Cm \ mu2d)))
end

# ---------------------------------------------------------------------------
# Load the joined frame (real miss-in-σ per case + lead time) and the CDMs.
# ---------------------------------------------------------------------------
using Base: Filesystem
function read_json_rows()
    # tiny JSON array-of-objects reader for cara_2d3d_joined.json "rows"
    txt = read(joinpath(REPO, "figureScripts", "data", "cara_2d3d_joined.json"), String)
    # crude: rely on Python via a shell-out would be heavier; instead parse the
    # numbers we need per row with regex on the flat rows. Fields: id, miss_maha,
    # lead_hours, class, violation.
    rows = []
    for m in eachmatch(r"\{[^{}]*\}", txt)
        s = m.match
        id  = match(r"\"id\"\s*:\s*\"([^\"]+)\"", s)
        mm  = match(r"\"miss_maha\"\s*:\s*([-\d.eE+]+)", s)
        lh  = match(r"\"lead_hours\"\s*:\s*([-\d.eE+]+)", s)
        cl  = match(r"\"class\"\s*:\s*\"([^\"]*)\"", s)
        vi  = match(r"\"violation\"\s*:\s*([-\d.eE+]+)", s)
        (id === nothing || mm === nothing || lh === nothing) && continue
        push!(rows, (id = id.captures[1],
                     miss_maha = parse(Float64, mm.captures[1]),
                     lead_hours = parse(Float64, lh.captures[1]),
                     class = cl === nothing ? "" : cl.captures[1],
                     violation = vi === nothing ? NaN : parse(Float64, vi.captures[1])))
    end
    return rows
end

println("Building reference Σ(τ) growth curves (anchored / isotropic SC seed)...")
ref = build_reference_curves()

rows = read_json_rows()
@printf "loaded %d rows from cara_2d3d_joined.json\n" length(rows)

cdm_dir = joinpath(REPO, "data", "cara_cdms")
cdm_files = Dict(replace(basename(f), ".cdm" => "") => f
                 for f in readdir(cdm_dir; join = true) if endswith(f, ".cdm"))

records = Vector{Dict{String,Any}}()
for row in rows
    path = get(cdm_files, row.id, nothing)
    path === nothing && continue
    p = parse_cdm(path)
    lead_s = row.lead_hours * 3600.0
    idx = nearest_idx(ref.τ, lead_s)

    # real miss-in-σ straight from the joined frame (real Σ)
    real_ms = row.miss_maha

    # our grown Σ substituted for the combined covariance, real geometry kept
    C_sc_anch = ref.sc_anch[idx]
    C_sc_iso  = ref.sc_iso[idx]
    C_db      = ref.debris[idx]
    ms_anch = missigma(p.state1, p.state2, C_sc_anch, C_db)
    ms_iso  = missigma(p.state1, p.state2, C_sc_iso,  C_db)

    push!(records, Dict(
        "id" => row.id, "class" => row.class,
        "lead_hours" => row.lead_hours, "violation" => row.violation,
        "miss_sigma_real"      => real_ms,
        "miss_sigma_ours_anch" => ms_anch,
        "miss_sigma_ours_iso"  => ms_iso,
    ))
end

# ---------------------------------------------------------------------------
# Focus band: ~24 h (18–30 h) — the operational decision point, where prompt 2
# established 2D validity on the real data. That is exactly where transfer must
# hold. Report the near-24h band + the full set.
# ---------------------------------------------------------------------------
function band_stats(recs, key)
    vals = Float64[r[key] for r in recs if isfinite(r[key])]
    isempty(vals) && return (median = NaN, n = 0)
    return (median = median(vals), n = length(vals))
end

near24 = [r for r in records if 18.0 <= r["lead_hours"] <= 30.0]
@printf "\n=== miss-in-σ transfer ===\n"
@printf "%-22s %10s %10s\n" "" "n" "median σ"
for (lbl, recs) in (("ALL 53", records), ("~24 h band (18-30h)", near24))
    @printf "%s (n=%d):\n" lbl length(recs)
    for (name, key) in (("real (CDM Σ)", "miss_sigma_real"),
                        ("ours anchored", "miss_sigma_ours_anch"),
                        ("ours isotropic", "miss_sigma_ours_iso"))
        s = band_stats(recs, key)
        @printf "   %-20s %10d %10.2f\n" name s.n s.median
    end
end

# JSON out (hand-rolled).
_j(v::AbstractString) = "\"" * replace(v, "\"" => "\\\"") * "\""
_j(v::Real) = (v isa AbstractFloat && !isfinite(v)) ? "null" : string(v)
function _jrec(r)
    "{" * join(["$(_j(string(k))): $(_j(v))" for (k, v) in r], ", ") * "}"
end
outdir = joinpath(REPO, "figureScripts", "data")
outpath = joinpath(outdir, "cara_missigma_transfer.json")
open(outpath, "w") do io
    println(io, "{")
    println(io, "  \"gate_sigma\": 2.0,")
    println(io, "  \"rows\": [")
    for (i, r) in enumerate(records)
        print(io, "    ", _jrec(r)); println(io, i == length(records) ? "" : ",")
    end
    println(io, "  ]")
    println(io, "}")
end
@printf "\nwrote %s\n" outpath
