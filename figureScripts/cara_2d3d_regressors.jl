# cara_2d3d_regressors.jl — dump per-case elrod_pc + Mahalanobis miss-in-sigma
# for the 2D-vs-3D underestimation characterization (paper_readiness_audit B3).
#
# The deep-dive JSON (figureScripts/data/cara_deepdive.json) already carries the
# regressors we reuse: lead_hours, aniso_s2_over_s1, vrel_mps, vang_deg, miss_m,
# NASA Pc2D and Nc3D, per-object RTN sigmas, and object class. This script adds
# the two things it does NOT have:
#
#   - elrod_pc : OUR exact 2D Pc (Chebyshev-Gauss quadrature) run on the real CDM
#                covariances. This is the numerator of the response
#                log10(elrod_pc / NASA_Nc3D). We recompute it rather than reuse
#                NASA's Pc2D column so the characterization is literally of our
#                method (elrod_pc matches NASA Pc2D to 0.29%, so they are close,
#                but the task asks for ours).
#   - miss_maha_sigma : Mahalanobis distance of the projected miss vector under
#                the encounter-plane covariance = sqrt(mu' Cinv mu). The physically
#                meaningful "miss distance in sigma" regressor for Pc.
#
# READ-ONLY. Changes no production code. Reuses the SAME parse_cdm + encounter
# plane + elrod_pc as the validation harness, so numbers are consistent.
#
# Output: figureScripts/data/cara_2d3d_regressors.json (one record per case,
# joined downstream to cara_deepdive.json on "id").
#
# Run:  julia --project=. figureScripts/cara_2d3d_regressors.jl

using LinearAlgebra
using Printf
using Statistics: median

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "tests", "cdmParser.jl"))

# computePc.jl references SpacecraftCAPOMDP in its dispatcher signatures; load it
# in a bare module with a stub type so elrod_pc / encounter_plane come in without
# the planner stack (identical trick to src/tests/test_cara_validation.jl).
module PcOnly
    using LinearAlgebra, Distributions
    using SpecialFunctions: erfc
    struct SpacecraftCAPOMDP end
    include(joinpath(@__DIR__, "..", "src", "utils", "computePc.jl"))
end
using .PcOnly: elrod_pc, encounter_plane

# NASA truth CSV: HBR used in the reference computation, plus Nc3D / Pc2D so this
# script is self-checking against the deep-dive JSON.
function load_truth(csv_path)
    lines = readlines(csv_path)
    header = split(lines[1], ',')
    idx = Dict(h => i for (i, h) in enumerate(header))
    rows = Dict{String,Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        id = f[idx["Conjunction_ID"]]
        rows[id] = Dict(String(h) => String(f[idx[h]]) for h in header)
    end
    return rows
end

num(d, k) = (v = get(d, k, ""); isempty(v) ? NaN : parse(Float64, replace(v, "E" => "e")))

cdm_dir  = joinpath(REPO, "data", "cara_cdms")
csv_path = joinpath(REPO, "src", "tests", "cara_truth.csv")
truth    = load_truth(csv_path)

cdm_files = sort(filter(f -> endswith(f, ".cdm"), readdir(cdm_dir; join = true)))
@assert length(cdm_files) == 53 "expected 53 CDMs, found $(length(cdm_files))"

records = Vector{Dict{String,Any}}()

for path in cdm_files
    base = replace(basename(path), ".cdm" => "")
    p = parse_cdm(path)
    t = get(truth, base, nothing)

    # HBR exactly as CARA used it (truth column); fall back to the CDM's own.
    hbr = t === nothing ? NaN : num(t, "HBR_m")
    isnan(hbr) && (hbr = p.hbr === nothing ? NaN : p.hbr)

    # OUR exact 2D Pc on the real covariances.
    pc_elrod = try
        elrod_pc(p.state1, p.state2, p.cov1_eci, p.cov2_eci, hbr)
    catch err
        @warn "elrod_pc failed" base err
        NaN
    end

    # Mahalanobis miss-in-sigma in the encounter plane: sqrt(mu' Cinv mu).
    miss_maha = try
        mu2d, C2d = encounter_plane(p.state1[1:3], p.state1[4:6],
                                    p.state2[1:3], p.state2[4:6],
                                    p.cov1_eci, p.cov2_eci)
        Cm = Matrix(C2d)
        det(Cm) <= 0 ? NaN : sqrt(max(0.0, dot(mu2d, Cm \ mu2d)))
    catch err
        @warn "miss_maha failed" base err
        NaN
    end

    push!(records, Dict(
        "id"              => base,
        "hbr_m"           => hbr,
        "elrod_pc"        => pc_elrod,
        "miss_maha_sigma" => miss_maha,
        # carried for a self-check against the deep-dive JSON (should match)
        "nasa_pc2d"       => t === nothing ? NaN : num(t, "Pc2D"),
        "nasa_nc3d"       => t === nothing ? NaN : num(t, "Nc3D"),
    ))
end

# ---------------------------------------------------------------------------
# Write JSON (hand-rolled; matches cara_deepdive_analysis.jl's approach so we
# add no dependency). All values are scalars: string / float.
# ---------------------------------------------------------------------------
function json_scalar(v)
    v isa AbstractString && return "\"" * replace(v, "\"" => "\\\"") * "\""
    (v isa AbstractFloat && !isfinite(v)) && return "null"   # NaN/Inf -> null
    return string(v)
end
function json_record(r)
    parts = ["\"$(k)\": $(json_scalar(v))" for (k, v) in r]
    return "{" * join(parts, ", ") * "}"
end

outdir = joinpath(REPO, "figureScripts", "data")
isdir(outdir) || mkpath(outdir)
outpath = joinpath(outdir, "cara_2d3d_regressors.json")
open(outpath, "w") do io
    println(io, "[")
    for (i, r) in enumerate(records)
        print(io, "  ", json_record(r))
        println(io, i == length(records) ? "" : ",")
    end
    println(io, "]")
end

# ---------------------------------------------------------------------------
# Quick sanity print: elrod vs NASA Pc2D agreement, and the headline response.
# ---------------------------------------------------------------------------
println("="^72)
println("elrod_pc + miss-in-sigma dumped for $(length(records)) cases -> $outpath")
println("="^72)

# elrod vs NASA Pc2D (should agree to <1%, confirming the recompute is faithful)
rel = Float64[]
for r in records
    e, n = r["elrod_pc"], r["nasa_pc2d"]
    (e > 0 && n > 0) && push!(rel, abs(e - n) / n)
end
@printf("elrod_pc vs NASA Pc2D:  median |rel| = %.4f%%   max |rel| = %.4f%%   (n=%d)\n",
        100 * (isempty(rel) ? NaN : median(sort(rel))),
        100 * (isempty(rel) ? NaN : maximum(rel)), length(rel))

# response = log10(elrod_pc / NASA Nc3D)
resp = Float64[]
for r in records
    e, n = r["elrod_pc"], r["nasa_nc3d"]
    (e > 0 && n > 0) && push!(resp, log10(e / n))
end
sorted = sort(resp)
med = isempty(sorted) ? NaN : sorted[cld(length(sorted), 2)]
@printf("log10(elrod/Nc3D):      median = %+.4f   min = %+.4f   max = %+.4f   (n=%d)\n",
        med, isempty(sorted) ? NaN : sorted[1],
        isempty(sorted) ? NaN : sorted[end], length(sorted))
@printf("cases elrod < Nc3D (unsafe, log<0): %d / %d\n",
        count(<(0.0), resp), length(resp))
