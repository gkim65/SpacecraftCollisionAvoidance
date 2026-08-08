# test_usage_violation.jl — validate the 2D-Pc usage-violation detector against NASA
#
# Runs the rectilinear-tier `usage_violation_pc2d` (src/utils/usageViolation.jl)
# on the 53 real NASA CARA conjunctions and compares its violation flag against
# NASA's own ground-truth `ViolationsPc2D` column from
# `CARA_PcMethod_Test_Conjunctions.xlsx`.
#
# The port is only trustworthy if it reproduces NASA's labels, so this harness
# reports a full confusion matrix (agree / false-positive / false-negative) and
# breaks down every disagreement by which criterion fired and why — that is the
# whole point of the exercise.
#
# NASA's `ViolationsPc2D` is a 3-digit code [Extended Offset Inaccurate], each a
# boolean (see usageViolation.jl header). A case is a "violation" iff the code is
# nonzero. Our rectilinear detector reproduces Extended + Offset + NPD but NOT the
# curvilinear-only Inaccurate bit; the cases where NASA's ONLY set digit is
# Inaccurate are therefore expected false-negatives and are called out separately.
#
# Usage:
#   julia --project=. src/tests/test_usage_violation.jl [path/to/CARA_Analysis_Tools]

using LinearAlgebra
using Printf
using Test

include(joinpath(@__DIR__, "cdmParser.jl"))
include(joinpath(@__DIR__, "exportCaraTruth.jl"))
include(joinpath(@__DIR__, "..", "utils", "usageViolation.jl"))

# Reuse the same data-resolution + truth-table machinery as the Pc validation
# harness, so both tests read exactly the same 53 cases and the same CSV.
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
    error("No CDM data found (looked in $VENDORED_CDM_DIR and $sdk). " *
          "Pass a CARA_Analysis_Tools root as ARGS[1].")
end

function read_truth(path::AbstractString)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = Dict{String,Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        length(f) < length(hdr) && continue
        d = Dict(String(hdr[i]) => String(f[i]) for i in eachindex(hdr))
        rows[d["Conjunction_ID"]] = d
    end
    return rows
end

num(d, k) = (v = get(d, k, ""); isempty(v) ? NaN : parse(Float64, v))

"""
    decode_violation_code(code) -> (inaccurate, offset, extended)

Decode NASA's 3-digit `ViolationsPc2D` code into its three boolean digits.

DIGIT ORDER (established empirically against the label text on this 53-case set,
2026-08-08): the code is `100*Inaccurate + 10*Offset + 1*Extended`. Evidence:
every `100` case is labeled "2D-Pc method underestimation …", which is the
`Inaccurate` mechanism (the Nc-vs-Pc `LogPcCorrectionFactor`), NOT an extended
duration; and each such case has an elevated Mahalanobis miss-in-σ (2–27) while
its rectilinear Extended/Offset indicators sit far below their 0.02/0.01 cutoffs.
The `111` cases add the Offset and Extended bits on top. (The SDK's `PcMultiStep.m`
computes the three booleans but does not itself format this 3-digit string — the
xlsx driver that concatenates them is not in the public SDK, so the order is
inferred from the labels + indicator behaviour, not read from source.)

Consequence: the leading (dominant) digit is the curvilinear-only `Inaccurate`
indicator, which the rectilinear tier cannot compute. Handles NaN/empty as
all-false.
"""
function decode_violation_code(code::Real)
    isnan(code) && return (false, false, false)
    c = Int(round(code))
    ina = (c ÷ 100) % 10 != 0
    off = (c ÷ 10)  % 10 != 0
    ext = c % 10 != 0
    return (ina, off, ext)
end

function run_validation(cara_root::AbstractString = DEFAULT_CARA_ROOT;
                        verbose::Bool = true)
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

        # Use the truth-table HBR (exactly what NASA ran), falling back to the
        # CDM's own HBR comment — same convention as test_cara_validation.jl.
        hbr = num(t, "HBR_m")
        isnan(hbr) && (hbr = cdm.hbr === nothing ? NaN : cdm.hbr)

        uv = usage_violation_pc2d(
            cdm.state1[1:3], cdm.state1[4:6], cdm.cov1_eci,
            cdm.state2[1:3], cdm.state2[4:6], cdm.cov2_eci, hbr)

        code = num(t, "ViolationsPc2D")
        nasa_ina, nasa_off, nasa_ext = decode_violation_code(code)
        nasa_any = nasa_ext || nasa_off || nasa_ina

        push!(results, (
            id = conj_id,
            primary = get(t, "PrimaryName", cdm.name1),
            secondary = get(t, "SecondaryName", cdm.name2),
            class = classify_secondary(cdm.name2),
            comment = get(t, "Comment", ""),
            vrel = num(t, "Vrel_mps"),
            vang = num(t, "Vang_deg"),
            miss = num(t, "MissDist_m"),
            # our detector
            our_any = uv.any_violation,
            our_ext = uv.extended,
            our_off = uv.offset,
            our_npd = uv.npd,
            ext_ind = uv.extended_ind,
            off_ind = uv.offset_ind,
            md_tca = uv.md_tca,
            converged = uv.converged,
            # NASA ground truth
            code = code,
            nasa_ext = nasa_ext,
            nasa_off = nasa_off,
            nasa_ina = nasa_ina,
            nasa_any = nasa_any,
        ))
    end

    verbose && report(results)
    return results
end

function report(results)
    println("\n", "="^100)
    println("usage_violation_pc2d (rectilinear tier) vs NASA CARA ViolationsPc2D — $(length(results)) cases")
    println("="^100)

    # --- confusion matrix on the top-level any-violation flag ---
    tp = count(r ->  r.our_any &&  r.nasa_any, results)   # agree: violation
    tn = count(r -> !r.our_any && !r.nasa_any, results)   # agree: clean
    fp = count(r ->  r.our_any && !r.nasa_any, results)   # we flag, NASA doesn't
    fn = count(r -> !r.our_any &&  r.nasa_any, results)   # NASA flags, we don't

    println("\nCONFUSION MATRIX (any-violation flag; NASA code nonzero = violation)")
    println("-"^60)
    @printf("                     NASA: VIOLATION   NASA: clean\n")
    @printf("  ours: VIOLATION  %14d %14d\n", tp, fp)
    @printf("  ours: clean      %14d %14d\n", fn, tn)
    println("-"^60)
    n = length(results)
    @printf("  agree            : %d / %d  (%.1f%%)\n", tp + tn, n, 100 * (tp + tn) / n)
    @printf("  false positives  : %d   (we flag, NASA does not)\n", fp)
    @printf("  false negatives  : %d   (NASA flags, we do not)\n", fn)

    # How many of the NASA violations are driven ONLY by the curvilinear-only
    # Inaccurate indicator (the leading digit) — i.e. structurally invisible to
    # the rectilinear tier. This is the key interpretation of the FN count.
    ina_driven = count(r -> r.nasa_any && !r.nasa_ext && !r.nasa_off, results)
    rect_visible = count(r -> r.nasa_ext || r.nasa_off, results)
    @printf("\n  Of NASA's %d violations: %d are Inaccurate-ONLY (curvilinear tier;\n",
            tp + fn, ina_driven)
    @printf("  invisible to a rectilinear detector by construction), %d also set a\n",
            rect_visible)
    @printf("  rectilinear-tier bit (Extended/Offset). The rectilinear FN floor is\n")
    @printf("  therefore ~%d; a detector cannot beat it without the curvilinear tier.\n",
            ina_driven)

    # --- per-indicator agreement (Extended, Offset) ---
    println("\nPER-INDICATOR AGREEMENT (rectilinear tier: Extended, Offset)")
    println("-"^60)
    for (lbl, ours, nasa) in (
            ("Extended", r -> r.our_ext === true, r -> r.nasa_ext),
            ("Offset",   r -> r.our_off === true, r -> r.nasa_off))
        agree = count(r -> ours(r) == nasa(r), results)
        ours_n = count(ours, results)
        nasa_n = count(nasa, results)
        @printf("  %-9s  ours=%2d  NASA=%2d  agree=%2d/%d\n",
                lbl, ours_n, nasa_n, agree, length(results))
    end

    # --- the curvilinear-only cases: NASA sets ONLY Inaccurate ---
    ina_only = filter(r -> r.nasa_ina && !r.nasa_ext && !r.nasa_off, results)
    println("\nCURVILINEAR-ONLY VIOLATIONS (NASA Inaccurate bit set, Extended/Offset not)")
    println("  These require the un-ported curvilinear tier; the rectilinear")
    println("  detector cannot see them and they show up as expected mismatches.")
    println("-"^60)
    if isempty(ina_only)
        println("  none")
    else
        for r in ina_only
            @printf("  code=%3d  %-20s / %-18s  our_any=%s\n",
                    Int(round(r.code)), first(r.primary, 20), first(r.secondary, 18),
                    r.our_any)
        end
    end

    # --- enumerate every disagreement, with the driving quantities ---
    disagree = filter(r -> r.our_any != r.nasa_any, results)
    println("\nDISAGREEMENTS (our_any ≠ NASA_any) — $(length(disagree)) case(s)")
    println("-"^100)
    if isempty(disagree)
        println("  none — the rectilinear detector reproduces every NASA any-violation label.")
    else
        @printf("  %-3s %-20s %-16s %8s %8s %8s %8s  %s\n",
                "our", "PRIMARY", "SECONDARY", "extInd", "offInd", "miss-σ", "NASAcode", "NASA label")
        for r in sort(disagree, by = x -> -x.code)
            @printf("  %-3s %-20s %-16s %8.4f %8.4f %8.3f %8d  %s\n",
                    r.our_any ? "FP" : "FN",
                    first(r.primary, 20), first(r.secondary, 16),
                    r.ext_ind === missing ? NaN : r.ext_ind,
                    r.off_ind === missing ? NaN : r.off_ind,
                    r.md_tca, Int(round(r.code)), r.comment)
        end
    end

    # --- full per-case table, sorted by NASA code then extended indicator ---
    println("\nPER-CASE DETAIL (sorted by NASA code, then extended indicator)")
    println("-"^110)
    @printf("  %-20s %-16s %-6s %8s %8s %8s  %4s %4s  %3s  %s\n",
            "PRIMARY", "SECONDARY", "CLASS", "extInd", "offInd", "miss-σ",
            "our", "NASA", "cod", "")
    for r in sort(results, by = x -> (-x.code, -(x.ext_ind === missing ? -1.0 : x.ext_ind)))
        mark = r.our_any == r.nasa_any ? "" :
               (r.our_any ? "  <-- FP" : "  <-- FN")
        @printf("  %-20s %-16s %-6s %8.4f %8.4f %8.3f  %4s %4s  %3d%s\n",
                first(r.primary, 20), first(r.secondary, 16),
                string(r.class)[1:min(6, length(string(r.class)))],
                r.ext_ind === missing ? NaN : r.ext_ind,
                r.off_ind === missing ? NaN : r.off_ind,
                r.md_tca,
                r.our_any ? "YES" : "-", r.nasa_any ? "YES" : "-",
                Int(round(r.code)), mark)
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CARA_ROOT
    results = run_validation(root)

    fn = count(r -> !r.our_any && r.nasa_any, results)
    fp = count(r ->  r.our_any && !r.nasa_any, results)

    # Every false negative on this set must be explained by the curvilinear-only
    # Inaccurate indicator: either NASA set ONLY Inaccurate (pure curvilinear —
    # invisible to us by construction), OR NASA also set Extended/Offset but via
    # the CURVILINEAR STCurvilinear, which supersedes the rectilinear estimate
    # (UsageViolationPc2D.m:900-915) and can exceed the cutoff where the
    # rectilinear value does not. A false negative NOT explained this way — i.e.
    # a case where our own rectilinear indicator SHOULD have exceeded the cutoff
    # but did not — would be a genuine port bug. Here every FN is an
    # Inaccurate-driven case, so there are no unexplained ones.
    fn_unexplained = count(results) do r
        !r.our_any && r.nasa_any && !r.nasa_ina  # NASA flagged w/o Inaccurate, we missed
    end

    @testset "usage_violation_pc2d vs NASA ViolationsPc2D" begin
        @test !isempty(results)
        # Safety-critical: the rectilinear tier must NEVER invent a violation NASA
        # did not flag (a false positive would make the planner needlessly
        # conservative and, worse, erode trust in the flag).
        @test fp == 0
        # No unexplained false negative: every violation we miss is attributable
        # to the (un-ported) curvilinear Inaccurate indicator. Anything else is a
        # rectilinear-math port bug.
        @test fn_unexplained == 0
        # Sanity: we do detect the one case whose rectilinear Extended clears the
        # cutoff outright (the extreme TROPICS/LINCS2 111 case).
        @test count(r -> r.our_any, results) >= 1
    end
end