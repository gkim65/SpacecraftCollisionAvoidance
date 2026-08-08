# test_usage_violation_curvilinear.jl — validate the FULL (rectilinear +
# curvilinear) 2D-Pc usage-violation detector against NASA CARA.
#
# Runs `usage_violation_pc2d_curvilinear` (src/utils/usageViolationCurvilinear.jl,
# which ports NASA's curvilinear PeakOverlapPos / EquinoctialMatrices machinery)
# on all 53 real CARA conjunctions and validates on TWO levels:
#
#   1. NUMERIC ANCHOR (the acceptance gate). Our LogPcCorrectionFactor gives
#      log10(2D-Nc / 2D-Pc); NASA's xlsx gives Nc2D and Pc2D directly. On the
#      non-degenerate cases these must agree tightly (the xlsx has no per-quantity
#      curvilinear diagnostics, so this ratio is the finest-grained checkable
#      number available — see the port's design note). Bar: median |err| < 0.05
#      dex. This checks the POP solver + refinement math NUMERICALLY, not just
#      whether the final flag lands on the right side of a cutoff.
#
#   2. LABEL AGREEMENT. The thresholded any-violation flag vs NASA's
#      ViolationsPc2D code (nonzero = violation). Bars: 0 false positives, and
#      the Inaccurate bit must be caught on the NASA-Inaccurate cases.
#
# NASA's ViolationsPc2D 3-digit code decodes as [Inaccurate Offset Extended]
# (see test_usage_violation.jl / usageViolation.jl headers).
#
# Usage:
#   julia --project=. src/tests/test_usage_violation_curvilinear.jl [path/to/CARA_Analysis_Tools]

using LinearAlgebra
using Printf
using Statistics
using Test

include(joinpath(@__DIR__, "cdmParser.jl"))
include(joinpath(@__DIR__, "exportCaraTruth.jl"))
include(joinpath(@__DIR__, "..", "utils", "usageViolationCurvilinear.jl"))

const VENDORED_CDM_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "cara_cdms"))
const DEFAULT_CARA_ROOT = joinpath(
    homedir(), "Documents", "School_Everything_and_LEARNING", "Stanford",
    "Githubs", "CARA_Analysis_Tools",
)

function resolve_cdm_dir(root::AbstractString)
    isdir(VENDORED_CDM_DIR) &&
        !isempty(filter(f -> endswith(f, ".cdm"), readdir(VENDORED_CDM_DIR))) &&
        return VENDORED_CDM_DIR
    sdk = joinpath(root, "DataFiles", "PcTestCaseCDMs")
    isdir(sdk) && return sdk
    error("No CDM data found (looked in $VENDORED_CDM_DIR and $sdk).")
end

function read_truth(path::AbstractString)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = Dict{String,Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        length(f) < length(hdr) && continue
        rows[String(f[1])] = Dict(String(hdr[i]) => String(f[i]) for i in eachindex(hdr))
    end
    return rows
end

num(d, k) = (v = get(d, k, ""); isempty(v) ? NaN : parse(Float64, v))

"Decode NASA's 3-digit ViolationsPc2D as (inaccurate, offset, extended)."
function decode_violation_code(code::Real)
    isnan(code) && return (false, false, false)
    c = Int(round(code))
    return ((c ÷ 100) % 10 != 0, (c ÷ 10) % 10 != 0, c % 10 != 0)
end

function run_validation(cara_root::AbstractString = DEFAULT_CARA_ROOT; verbose::Bool = true)
    cdm_dir = resolve_cdm_dir(cara_root)
    truth_path = joinpath(@__DIR__, "cara_truth.csv")
    if !isfile(truth_path)
        @info "Reference CSV not found; exporting from CARA workbook" truth_path
        export_truth(cara_root; outpath = truth_path)
    end
    truth = read_truth(truth_path)
    cdm_files = sort(filter(f -> endswith(f, ".cdm"), readdir(cdm_dir)))

    results = NamedTuple[]
    for fname in cdm_files
        conj_id = replace(fname, ".cdm" => "")
        haskey(truth, conj_id) || continue
        t = truth[conj_id]
        cdm = parse_cdm(joinpath(cdm_dir, fname))
        hbr = num(t, "HBR_m")
        isnan(hbr) && (hbr = cdm.hbr === nothing ? NaN : cdm.hbr)

        res = usage_violation_pc2d_curvilinear(
            cdm.state1[1:3], cdm.state1[4:6], cdm.cov1_eci,
            cdm.state2[1:3], cdm.state2[4:6], cdm.cov2_eci, hbr)

        code = num(t, "ViolationsPc2D")
        nina, noff, next = decode_violation_code(code)

        push!(results, (
            id = conj_id,
            primary = get(t, "PrimaryName", cdm.name1),
            secondary = get(t, "SecondaryName", cdm.name2),
            class = classify_secondary(cdm.name2),
            res = res,
            pc2d = num(t, "Pc2D"),
            nc2d = num(t, "Nc2D"),
            code = code,
            nasa_ext = next, nasa_off = noff, nasa_ina = nina,
            nasa_any = next || noff || nina,
        ))
    end

    verbose && report(results)
    return results
end

function report(results)
    println("\n", "="^100)
    println("Curvilinear usage-violation detector vs NASA CARA — $(length(results)) conjunctions")
    println("="^100)

    # --- 1. numeric anchor: LogPcCorrectionFactor vs NASA log10(Nc2D/Pc2D) ---
    anchor = NamedTuple[]
    for r in results
        (r.pc2d > 1e-12 && r.nc2d > 1e-12 && r.res.qconverged &&
         isfinite(r.res.log_pc_correction)) || continue
        ours = r.res.log_pc_correction / log(10)
        nasa = log10(r.nc2d / r.pc2d)
        push!(anchor, (id = r.id, ours = ours, nasa = nasa, err = ours - nasa))
    end
    errs = [abs(a.err) for a in anchor]
    println("\nNUMERIC ANCHOR — log10(2D-Nc/2D-Pc): ours (LogPcCorrectionFactor) vs NASA (Nc2D/Pc2D)")
    println("-"^70)
    @printf("  usable cases         : %d / %d\n", length(anchor), length(results))
    @printf("  median |err|         : %.4f dex\n", median(errs))
    @printf("  max    |err|         : %.4f dex\n", maximum(errs))
    @printf("  within 0.05 dex      : %d / %d\n", count(<(0.05), errs), length(errs))
    @printf("  within 0.10 dex      : %d / %d\n", count(<(0.10), errs), length(errs))
    if !isempty(errs)
        println("  worst anchor cases:")
        for a in first(sort(anchor, by = x -> -abs(x.err)), min(4, length(anchor)))
            @printf("    %-34s ours=%+.3f  NASA=%+.3f  err=%+.3f dex\n",
                    first(a.id, 34), a.ours, a.nasa, a.err)
        end
    end

    # --- 2. label confusion matrix ---
    tp = count(r ->  r.res.any_violation &&  r.nasa_any, results)
    tn = count(r -> !r.res.any_violation && !r.nasa_any, results)
    fp = count(r ->  r.res.any_violation && !r.nasa_any, results)
    fn = count(r -> !r.res.any_violation &&  r.nasa_any, results)
    ina_caught = count(r -> r.nasa_ina && r.res.inaccurate === true, results)
    ina_total = count(r -> r.nasa_ina, results)

    println("\nLABEL AGREEMENT — any-violation flag vs NASA ViolationsPc2D")
    println("-"^70)
    @printf("                     NASA: VIOLATION   NASA: clean\n")
    @printf("  ours: VIOLATION  %14d %14d\n", tp, fp)
    @printf("  ours: clean      %14d %14d\n", fn, tn)
    @printf("  agree            : %d / %d  (%.1f%%)\n", tp + tn, length(results),
            100 * (tp + tn) / length(results))
    @printf("  false positives  : %d\n", fp)
    @printf("  false negatives  : %d\n", fn)
    @printf("  Inaccurate bit caught: %d / %d NASA-Inaccurate cases\n", ina_caught, ina_total)

    # --- per-case detail ---
    println("\nPER-CASE (sorted by NASA code)")
    println("-"^100)
    @printf("  %-20s %-16s %8s %8s %8s  %4s %4s %3s %s\n",
            "PRIMARY", "SECONDARY", "extInd", "offInd", "inaInd", "our", "NASA", "cod", "conv")
    for r in sort(results, by = x -> -x.code)
        g(x) = x === missing ? NaN : Float64(x)
        mark = r.res.any_violation == r.nasa_any ? "" :
               (r.res.any_violation ? "  <-- FP" : "  <-- FN")
        @printf("  %-20s %-16s %8.4f %8.4f %8.4f  %4s %4s %3d %s%s\n",
                first(r.primary, 20), first(r.secondary, 16),
                g(r.res.extended_ind), g(r.res.offset_ind), g(r.res.inaccurate_ind),
                r.res.any_violation ? "YES" : "-", r.nasa_any ? "YES" : "-",
                Int(round(r.code)), r.res.qconverged, mark)
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CARA_ROOT
    results = run_validation(root)

    anchor_err = Float64[]
    for r in results
        (r.pc2d > 1e-12 && r.nc2d > 1e-12 && r.res.qconverged &&
         isfinite(r.res.log_pc_correction)) || continue
        push!(anchor_err, abs(r.res.log_pc_correction / log(10) - log10(r.nc2d / r.pc2d)))
    end
    fp = count(r ->  r.res.any_violation && !r.nasa_any, results)
    fn = count(r -> !r.res.any_violation &&  r.nasa_any, results)
    ina_caught = count(r -> r.nasa_ina && r.res.inaccurate === true, results)
    ina_total = count(r -> r.nasa_ina, results)

    @testset "curvilinear usage-violation vs NASA" begin
        @test !isempty(results)
        # 1. Numeric anchor: LogPcCorrectionFactor must reproduce NASA's
        #    Nc2D/Pc2D ratio tightly on the non-degenerate cases. This checks the
        #    POP solver + refinement NUMERICALLY (not just the final label).
        @test median(anchor_err) < 0.05
        # 2. Safety: no false positives.
        @test fp == 0
        # 3. Curvilinear tier must catch the Inaccurate-flagged cases the
        #    rectilinear tier could not (the whole reason for the port).
        @test ina_caught == ina_total
        # 4. No false negatives on this validated set.
        @test fn == 0
    end
end