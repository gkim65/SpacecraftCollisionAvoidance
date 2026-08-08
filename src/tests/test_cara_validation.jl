# test_cara_validation.jl — validate `chan_pc` against NASA CARA reference values
#
# Runs the project's Chan (1997) Pc implementation on the 53 real conjunctions
# in the NASA CARA Analysis Tools SDK, and compares against CARA's own computed
# values (Pc2D, Nc2D, Nc3D, PcSDMC) stored in
# `DataFiles/PcTestCaseCDMs/CARA_PcMethod_Test_Conjunctions.xlsx`.
#
# Why this matters: the existing Phase-1 cross-validation checks `chan_pc`
# against a Python port of the same algorithm — it confirms the port is
# faithful, but not that the algorithm agrees with operational NASA values on
# real conjunction geometry. This harness does the latter.
#
# What is being compared: `chan_pc` is a 2D encounter-plane method, so the
# like-for-like reference column is CARA's `Pc2D`. The `Nc2D` / `Nc3D` /
# `PcSDMC` columns are higher-fidelity methods; `chan_pc` is EXPECTED to
# diverge from them on the cases CARA flags as 2D usage violations. That
# divergence pattern is the interesting result, not a failure.
#
# Usage:
#   julia --project=. src/tests/test_cara_validation.jl [path/to/CARA_Analysis_Tools]

using LinearAlgebra
using Printf
using Statistics
using Test

include(joinpath(@__DIR__, "cdmParser.jl"))
include(joinpath(@__DIR__, "exportCaraTruth.jl"))

# `computePc.jl` pulls in the POMDP type for its dispatcher, but `chan_pc` and
# `_chan_series` are self-contained (states + covariances + HBR only). Load the
# file in a bare module so the Chan functions come in without requiring the
# whole POMDP stack.
module ChanOnly
    using LinearAlgebra, Distributions
    # Stub the type referenced by the dispatcher signatures we don't call.
    struct SpacecraftCAPOMDP end
    include(joinpath(@__DIR__, "..", "utils", "computePc.jl"))
end
using .ChanOnly: chan_pc, elrod_pc, numeric_pc, covariance_anisotropy

# The CDMs and NASA's reference workbook are vendored into this repo (see
# data/cara_cdms/README.md) so a clean clone reproduces without the SDK.
const VENDORED_CDM_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "cara_cdms"))

# Fallback: a full CARA Analysis Tools SDK checkout, if one is present.
const DEFAULT_CARA_ROOT = joinpath(
    homedir(), "Documents", "School_Everything_and_LEARNING", "Stanford",
    "Githubs", "CARA_Analysis_Tools",
)

"""
    resolve_cdm_dir(root) -> String

Locate the CDM directory. Prefers the vendored copy in this repo; falls back to
`<root>/DataFiles/PcTestCaseCDMs` in a CARA SDK checkout.
"""
function resolve_cdm_dir(root::AbstractString)
    isdir(VENDORED_CDM_DIR) &&
        !isempty(filter(f -> endswith(f, ".cdm"), readdir(VENDORED_CDM_DIR))) &&
        return VENDORED_CDM_DIR

    sdk = joinpath(root, "DataFiles", "PcTestCaseCDMs")
    isdir(sdk) && return sdk

    error("""
          No CDM data found. Looked in:
            $VENDORED_CDM_DIR   (vendored copy — expected)
            $sdk   (CARA SDK fallback)
          Pass a CARA_Analysis_Tools root as the first argument if it lives elsewhere.
          """)
end

# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------

"Read the CSV exported from CARA_PcMethod_Test_Conjunctions.xlsx."
function read_truth(path::AbstractString)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = Dict{String,Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        # No quoted commas in this export, but comments contain parentheses only.
        f = split(ln, ',')
        length(f) < length(hdr) && continue
        d = Dict(String(hdr[i]) => String(f[i]) for i in eachindex(hdr))
        rows[d["Conjunction_ID"]] = d
    end
    return rows
end

num(d, k) = (v = get(d, k, ""); isempty(v) ? NaN : parse(Float64, v))

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

"Relative difference between a computed and reference Pc, guarding zeros."
function reldiff(computed, reference)
    (isnan(computed) || isnan(reference)) && return NaN
    reference == 0.0 && return computed == 0.0 ? 0.0 : Inf
    return (computed - reference) / reference
end

"Ratio in orders of magnitude, for cases where relative error is uninformative."
function log10ratio(computed, reference)
    (computed <= 0 || reference <= 0 || isnan(computed) || isnan(reference)) && return NaN
    return log10(computed / reference)
end

function run_validation(cara_root::AbstractString = DEFAULT_CARA_ROOT;
                        truth_csv::Union{Nothing,String} = nothing,
                        verbose::Bool = true)

    cdm_dir = resolve_cdm_dir(cara_root)

    truth_path = truth_csv === nothing ?
        joinpath(@__DIR__, "cara_truth.csv") : truth_csv
    if !isfile(truth_path)
        @info "Reference CSV not found; exporting it from the CARA workbook" truth_path
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

        # Use the HBR from the truth table so the comparison is exactly
        # like-for-like with what CARA ran (the CDM's own HBR comment can
        # differ from the value used in the reference computation).
        hbr = num(t, "HBR_m")
        isnan(hbr) && (hbr = cdm.hbr === nothing ? NaN : cdm.hbr)

        # Positive-definiteness of the summed 3x3 position covariance, which is
        # what the Chan projection actually consumes.
        Cpos = cdm.cov1_eci[1:3, 1:3] + cdm.cov2_eci[1:3, 1:3]
        eigmin_pos = minimum(eigvals(Symmetric(Cpos)))
        npd = eigmin_pos <= 0

        safecall(f, label) = try
            f(cdm.state1, cdm.state2, cdm.cov1_eci, cdm.cov2_eci, hbr)
        catch err
            @warn "$label failed" conj_id err
            NaN
        end

        pc = safecall(chan_pc, "chan_pc")
        pc_elrod = safecall(elrod_pc, "elrod_pc")
        pc_numeric = safecall(numeric_pc, "numeric_pc")
        aniso = try
            covariance_anisotropy(cdm.state1, cdm.state2, cdm.cov1_eci, cdm.cov2_eci)
        catch
            NaN
        end

        push!(results, (
            id = conj_id,
            primary = get(t, "PrimaryName", cdm.name1),
            secondary = get(t, "SecondaryName", cdm.name2),
            class = classify_secondary(cdm.name2),
            comment = get(t, "Comment", ""),
            hbr = hbr,
            miss = num(t, "MissDist_m"),
            vrel = num(t, "Vrel_mps"),
            vang = num(t, "Vang_deg"),
            pc_chan = pc,
            pc_elrod = pc_elrod,
            pc_numeric = pc_numeric,
            aniso = aniso,
            pc2d = num(t, "Pc2D"),
            nc2d = num(t, "Nc2D"),
            nc3d = num(t, "Nc3D"),
            pc_sdmc = num(t, "PcSDMC"),
            viol_pc2d = num(t, "ViolationsPc2D"),
            eigmin_pos = eigmin_pos,
            npd = npd,
            pc_cdm = cdm.pc_cdm,
        ))
    end

    verbose && report(results)
    return results
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

function report(results)
    println("\n", "="^100)
    println("chan_pc vs. NASA CARA reference values — $(length(results)) real conjunctions")
    println("="^100)

    # --- per-case table ---
    @printf("\n%-22s %-18s %-6s %10s %10s %9s %8s %s\n",
            "PRIMARY", "SECONDARY", "CLASS", "chan_pc", "CARA Pc2D", "rel.err", "viol", "")
    println("-"^100)

    for r in sort(results, by = x -> -x.pc2d)
        rel = reldiff(r.pc_chan, r.pc2d)
        flag = r.viol_pc2d > 0 ? "YES" : "-"
        marker = abs(rel) > 0.01 ? "  <-- >1%" : ""
        @printf("%-22s %-18s %-6s %10.3e %10.3e %8.2f%% %8s%s\n",
                first(r.primary, 22), first(r.secondary, 18),
                string(r.class)[1:min(6, length(string(r.class)))],
                r.pc_chan, r.pc2d, 100 * rel, flag, marker)
    end

    # --- agreement vs. CARA's 2D-Pc ---
    ok = filter(r -> !isnan(r.pc_chan) && !isnan(r.pc2d) && r.pc2d > 0, results)
    rels = [abs(reldiff(r.pc_chan, r.pc2d)) for r in ok]

    println("\n", "-"^100)
    println("AGREEMENT WITH CARA Pc2D (like-for-like: both are 2D encounter-plane methods)")
    println("-"^100)
    @printf("  cases compared      : %d\n", length(ok))
    if !isempty(rels)
        @printf("  median |rel. error| : %.3g %%\n", 100 * median(rels))
        @printf("  mean   |rel. error| : %.3g %%\n", 100 * mean(rels))
        @printf("  max    |rel. error| : %.3g %%\n", 100 * maximum(rels))
        @printf("  within 1%%           : %d / %d\n", count(<(0.01), rels), length(rels))
        @printf("  within 10%%          : %d / %d\n", count(<(0.10), rels), length(rels))
    end

    # --- three-method comparison ---
    println("\n", "-"^100)
    println("METHOD COMPARISON vs CARA Pc2D  (chan = series, elrod = quadrature, numeric = oracle)")
    println("-"^100)
    for (mname, getter) in (("chan_pc", r -> r.pc_chan),
                            ("elrod_pc", r -> r.pc_elrod),
                            ("numeric_pc", r -> r.pc_numeric))
        sub = filter(r -> !isnan(getter(r)) && r.pc2d > 0, results)
        isempty(sub) && continue
        e = [abs(reldiff(getter(r), r.pc2d)) for r in sub]
        @printf("  %-11s n=%2d  median=%9.4f%%  max=%10.2f%%  within1%%=%2d  within10%%=%2d\n",
                mname, length(e), 100 * median(e), 100 * maximum(e),
                count(<(0.01), e), count(<(0.10), e))
    end

    # --- accuracy vs. covariance anisotropy: the failure mode ---
    println("\n", "-"^100)
    println("ERROR vs ENCOUNTER-PLANE ANISOTROPY (sigma2/sigma1) — why chan_pc fails")
    println("-"^100)
    bands = [(0.0, 10.0, "< 10"), (10.0, 50.0, "10-50"), (50.0, 100.0, "50-100"),
             (100.0, 300.0, "100-300"), (300.0, Inf, "> 300")]
    @printf("  %-10s %4s %14s %14s\n", "aniso", "n", "chan median", "elrod median")
    for (lo, hi, lbl) in bands
        sub = filter(r -> !isnan(r.aniso) && lo <= r.aniso < hi && r.pc2d > 0, results)
        isempty(sub) && continue
        ec = [abs(reldiff(r.pc_chan, r.pc2d)) for r in sub]
        ee = [abs(reldiff(r.pc_elrod, r.pc2d)) for r in sub]
        @printf("  %-10s %4d %13.3f%% %13.4f%%\n",
                lbl, length(sub), 100 * median(ec), 100 * median(ee))
    end

    # --- stratified by object class ---
    println("\n", "-"^100)
    println("STRATIFIED BY SECONDARY OBJECT CLASS")
    println("-"^100)
    for cls in (:debris, :rocket_body, :payload, :unknown)
        sub = filter(r -> r.class == cls, ok)
        isempty(sub) && continue
        subrels = [abs(reldiff(r.pc_chan, r.pc2d)) for r in sub]
        @printf("  %-12s n=%2d   median |rel.err| = %8.3g %%   max = %8.3g %%\n",
                string(cls), length(sub), 100 * median(subrels), 100 * maximum(subrels))
    end

    # --- divergence from the higher-fidelity methods, split by violation flag ---
    println("\n", "-"^100)
    println("DIVERGENCE FROM HIGHER-FIDELITY METHODS (log10 ratio chan_pc / reference)")
    println("  Expectation: agreement where CARA reports no 2D usage violation,")
    println("  divergence on the flagged cases. That pattern is the validity envelope.")
    println("-"^100)

    for (label, sel) in (("no violation", r -> r.viol_pc2d == 0),
                         ("VIOLATION",    r -> r.viol_pc2d > 0))
        sub = filter(sel, ok)
        isempty(sub) && continue
        println("\n  $label  (n = $(length(sub)))")
        for (mname, getter) in (("Nc2D", r -> r.nc2d), ("Nc3D", r -> r.nc3d),
                                ("PcSDMC", r -> r.pc_sdmc))
            lrs = filter(!isnan, [log10ratio(r.pc_chan, getter(r)) for r in sub])
            isempty(lrs) && continue
            @printf("    vs %-7s median log10 ratio = %+6.3f   max |log10| = %5.3f\n",
                    mname, median(lrs), maximum(abs.(lrs)))
        end
    end

    # --- non-positive-definite covariance rate ---
    npd_count = count(r -> r.npd, results)
    println("\n", "-"^100)
    println("NON-POSITIVE-DEFINITE COVARIANCE RATE")
    println("-"^100)
    @printf("  summed 3x3 position covariance not PD: %d / %d cases\n",
            npd_count, length(results))
    @printf("  smallest eigenvalue across all cases : %.4g m^2\n",
            minimum(r.eigmin_pos for r in results))

    # --- worst disagreements ---
    println("\n", "-"^100)
    println("LARGEST DISAGREEMENTS WITH CARA Pc2D")
    println("-"^100)
    worst = sort(ok, by = r -> -abs(reldiff(r.pc_chan, r.pc2d)))
    for r in first(worst, min(5, length(worst)))
        @printf("  %6.2f%%  %-20s vs %-20s  Vrel=%8.1f m/s  Vang=%6.2f deg\n",
                100 * reldiff(r.pc_chan, r.pc2d), first(r.primary, 20),
                first(r.secondary, 20), r.vrel, r.vang)
        println("           $(r.comment)")
    end
    println()
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CARA_ROOT
    results = run_validation(root)

    ok = filter(r -> !isnan(r.pc_chan) && !isnan(r.pc2d) && r.pc2d > 0, results)
    rels = [abs(reldiff(r.pc_chan, r.pc2d)) for r in ok]

    @testset "chan_pc vs CARA Pc2D" begin
        @test !isempty(ok)
        # Chan (1997) and CARA's PcCircle solve the same 2D integral by
        # different routes; agreement should be tight on non-violating cases.
        nonviol = filter(r -> r.viol_pc2d == 0, ok)
        nvrels = [abs(reldiff(r.pc_chan, r.pc2d)) for r in nonviol]
        @test median(nvrels) < 0.05
    end
end
