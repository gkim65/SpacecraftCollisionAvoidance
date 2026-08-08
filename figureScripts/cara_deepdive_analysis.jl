# cara_deepdive_analysis.jl — read-only characterization of the 53 NASA CARA CDMs.
#
# Answers the CARA-CDM deep-dive block in notes/paper_readiness_audit.md:
#   (a) LEAD TIME     : CDM creation vs TCA (hours before TCA); how many near ~24h.
#   (b) COV MAG+SHAPE : per-object RTN R/T/N 1-sigma AND encounter-plane sigma1/sigma2
#                       + anisotropy, stratified by secondary object class.
#   (c) SLOW/FAST     : bin by relative velocity, cross-check NASA ViolationsPc2D.
#   (d) DEBRIS-SWAMPS : primary (NASA payload) vs secondary position-cov magnitude.
#
# READ-ONLY. Changes no production code. Reads:
#   - data/cara_cdms/*.cdm          (states + RTN covariances + names + TCA/creation)
#   - src/tests/cara_truth.csv       (NASA Vrel, Vang, ViolationsPc2D, HBR, orbit params)
#   - src/utils/computePc.jl         (encounter_plane / covariance_anisotropy only)
#   - src/tests/cdmParser.jl         (parse_cdm, classify_secondary, rtn rotation)
#
# CAREFUL ON OBJECT CLASS: the primary is ALWAYS the NASA/NOAA operational asset
# (an active payload). classify_secondary only labels the SECONDARY. Every record
# dumps raw OBJECT_NAME for both objects + inferred class so misclassification is
# auditable downstream.
#
# Output: writes a single JSON blob to figureScripts/data/cara_deepdive.json for
# the (later) matplotlib figure pass. Prints a human-readable summary to stdout.
#
# Run:  julia --project=. figureScripts/cara_deepdive_analysis.jl

using LinearAlgebra
using Dates
using Statistics
using Printf

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "tests", "cdmParser.jl"))

# We only need encounter_plane / covariance_anisotropy from computePc.jl, but that
# file references SpacecraftCAPOMDP in its dispatcher. Re-implement the two pure
# geometry helpers here so this script has NO dependency on the planner types.
# (Kept byte-consistent with computePc.jl's encounter_plane convention.)
"""
    encounter_plane_cov(r1, v1, r2, v2, C1, C2) -> (mu2d, C2d)

Project the summed relative position covariance onto the 2D encounter plane
(the plane perpendicular to relative velocity). Mirrors computePc.jl.
"""
function encounter_plane_cov(r1, v1, r2, v2, C1, C2)
    Csum = C1[1:3, 1:3] .+ C2[1:3, 1:3]          # summed position cov, ECI (m^2)
    vrel = v1 .- v2
    dr   = r1 .- r2

    y = vrel ./ norm(vrel)                        # out-of-plane = along rel velocity
    # Build an orthonormal in-plane basis (x, z) perpendicular to y.
    tmp = abs(y[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    x = tmp .- (tmp ⋅ y) .* y
    x ./= norm(x)
    z = cross(y, x)

    P = hcat(x, z)                                # 3x2 projection onto encounter plane
    C2d = P' * Csum * P
    mu2d = P' * dr
    return mu2d, Symmetric(C2d)
end

function anisotropy_2d(C2d)
    ev = eigvals(Matrix(C2d))
    (ev[1] <= 0 || ev[2] <= 0) && return (Inf, NaN, NaN)
    s1 = sqrt(minimum(ev))                        # short-axis 1-sigma (m)
    s2 = sqrt(maximum(ev))                        # long-axis  1-sigma (m)
    return (s2 / s1, s1, s2)
end

# --- filename / field time parsing -----------------------------------------
# Filename: <prim>_conj_<sec>_<YYYYMMDD>_<HHMMSS>_<YYYYMMDD>_<HHMMSS>.cdm
#           where the first date/time pair is TCA, the second is CDM creation.
function times_from_filename(path)
    base = basename(path)
    m = match(r"_(\d{8})_(\d{6})_(\d{8})_(\d{6})\.cdm$", base)
    m === nothing && return (nothing, nothing)
    tca  = DateTime(m.captures[1] * m.captures[2], dateformat"yyyymmddHHMMSS")
    crea = DateTime(m.captures[3] * m.captures[4], dateformat"yyyymmddHHMMSS")
    return (tca, crea)
end

# In-file fields are ISO with sub-seconds; more precise, use as ground truth.
parse_iso(s) = DateTime(split(strip(s), '.')[1], dateformat"yyyy-mm-ddTHH:MM:SS")

# --- load NASA truth CSV (Vrel, Vang, ViolationsPc2D, HBR, orbit params) ----
function load_truth(csv_path)
    lines = readlines(csv_path)
    header = split(lines[1], ',')
    idx = Dict(h => i for (i, h) in enumerate(header))
    rows = Dict{String,Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        id = f[idx["Conjunction_ID"]]
        rows[id] = Dict(h => f[idx[h]] for h in header)
    end
    return rows, idx
end

# ---------------------------------------------------------------------------
cdm_dir   = joinpath(REPO, "data", "cara_cdms")
csv_path  = joinpath(REPO, "src", "tests", "cara_truth.csv")
truth, _  = load_truth(csv_path)

cdm_files = sort(filter(f -> endswith(f, ".cdm"), readdir(cdm_dir; join = true)))
@assert length(cdm_files) == 53 "expected 53 CDMs, found $(length(cdm_files))"

records = Vector{Dict{String,Any}}()

for path in cdm_files
    base = replace(basename(path), ".cdm" => "")
    p = parse_cdm(path)

    # --- times / lead time ---
    tca_fn, crea_fn = times_from_filename(path)
    tca_field  = parse_iso(p.tca)
    crea_field = parse_iso(p.creation_date)
    # lead = TCA - creation, in hours. Use in-file fields (sub-second precise).
    lead_h = (tca_field - crea_field).value / 3.6e6   # ms -> hours
    # sanity: filename vs field agreement (hours)
    fn_field_gap_h = tca_fn === nothing ? NaN :
        abs((tca_fn - tca_field).value) / 3.6e6

    # --- classes: primary is ALWAYS the operational NASA asset (payload). ---
    sec_class = classify_secondary(p.name2)
    # We still run the classifier on the primary name for AUDIT visibility, but
    # the primary is operationally known to be an active payload.
    prim_class_raw = classify_secondary(p.name1)

    # --- per-object RTN position 1-sigma (R,T,N), from the RTN cov block (m) ---
    rtn_sigmas(C_rtn) = (sqrt(max(C_rtn[1,1], 0.0)),   # R radial
                         sqrt(max(C_rtn[2,2], 0.0)),   # T in-track
                         sqrt(max(C_rtn[3,3], 0.0)))   # N cross-track
    R1, T1, N1 = rtn_sigmas(p.cov1_rtn)
    R2, T2, N2 = rtn_sigmas(p.cov2_rtn)

    # --- overall position-cov magnitude per object: trace of 3x3 pos block ---
    pos_rms(C_rtn) = sqrt((C_rtn[1,1] + C_rtn[2,2] + C_rtn[3,3]))  # m, 3D 1-sigma
    prim_pos_rms = pos_rms(p.cov1_rtn)
    sec_pos_rms  = pos_rms(p.cov2_rtn)

    # --- encounter-plane shape (uses ECI covs, per computePc convention) ---
    mu2d, C2d = encounter_plane_cov(p.state1[1:3], p.state1[4:6],
                                    p.state2[1:3], p.state2[4:6],
                                    p.cov1_eci, p.cov2_eci)
    aniso, s1, s2 = anisotropy_2d(C2d)

    # --- NASA truth fields for this conjunction (may be missing for a few) ---
    t = get(truth, base, nothing)
    getf(k) = t === nothing || !haskey(t, k) || isempty(t[k]) ? NaN :
        parse(Float64, replace(t[k], "E" => "e"))
    gets(k) = t === nothing ? "" : get(t, k, "")

    push!(records, Dict(
        "id"              => base,
        "name1"           => p.name1, "id1" => p.id1,
        "name2"           => p.name2, "id2" => p.id2,
        "primary_class_inferred"   => String(prim_class_raw),  # audit only
        "primary_class_operational"=> "payload",               # ground truth
        "secondary_class" => String(sec_class),
        # times / lead
        "tca"             => string(tca_field),
        "creation"        => string(crea_field),
        "lead_hours"      => lead_h,
        "lead_days"       => lead_h / 24,
        "fn_field_gap_h"  => fn_field_gap_h,
        # RTN 1-sigma per object (m)
        "prim_sigR" => R1, "prim_sigT" => T1, "prim_sigN" => N1,
        "sec_sigR"  => R2, "sec_sigT"  => T2, "sec_sigN"  => N2,
        # overall 3D position 1-sigma per object (m)
        "prim_pos_rms" => prim_pos_rms,
        "sec_pos_rms"  => sec_pos_rms,
        "sec_over_prim_rms" => sec_pos_rms / prim_pos_rms,
        # encounter-plane shape
        "aniso_s2_over_s1" => aniso,
        "enc_sig_short_m"  => s1,
        "enc_sig_long_m"   => s2,
        # NASA truth
        "vrel_mps"    => getf("Vrel_mps"),
        "vang_deg"    => getf("Vang_deg"),
        "hbr_m"       => getf("HBR_m"),
        "miss_m"      => getf("MissDist_m"),
        "pc2d"        => getf("Pc2D"),
        "nc3d"        => getf("Nc3D"),
        "violation_label" => gets("Comment"),
        # ViolationsPc2D is a CODE, not 0/1: 0 = no violation; non-zero (100, 111)
        # = 2D-Pc usage violation. 24 zeros / 29 non-zero, matching the findings note.
        "violation_flag"  => getf("ViolationsPc2D"),
        "prim_country" => gets("PrimaryCountry"),
        "sec_country"  => gets("SecondaryCountry"),
    ))
end

# ---------------------------------------------------------------------------
# SUMMARY (printed)
# ---------------------------------------------------------------------------
println("="^78)
println("CARA CDM deep-dive — 53 real NASA conjunctions")
println("="^78)

# Filename/field cross-check
gaps = [r["fn_field_gap_h"] for r in records if !isnan(r["fn_field_gap_h"])]
@printf("\nFilename-vs-field TCA agreement: max gap = %.4f h (%d files checked)\n",
        maximum(gaps), length(gaps))

# --- (a) LEAD TIME ---
println("\n" * "-"^78)
println("(a) LEAD TIME — CDM creation before TCA")
println("-"^78)
leads = [r["lead_hours"] for r in records]
@printf("min=%.1f h  median=%.1f h  mean=%.1f h  max=%.1f h  (%.1f–%.1f days)\n",
        minimum(leads), median(leads), mean(leads), maximum(leads),
        minimum(leads)/24, maximum(leads)/24)
# Bins around the ~24h operational decision point.
near24 = count(r -> 18 <= r["lead_hours"] <= 30, records)
under18 = count(r -> r["lead_hours"] < 18, records)
d1to3 = count(r -> 30 < r["lead_hours"] <= 72, records)
over3d = count(r -> r["lead_hours"] > 72, records)
@printf("  <18h: %d   ~24h (18–30h): %d   1–3d (30–72h): %d   >3d: %d\n",
        under18, near24, d1to3, over3d)

# --- (b) COV MAGNITUDE + SHAPE by SECONDARY class ---
println("\n" * "-"^78)
println("(b) COVARIANCE MAGNITUDE + SHAPE by secondary class")
println("-"^78)
for cls in ["debris", "rocket_body", "payload", "unknown"]
    sub = filter(r -> r["secondary_class"] == cls, records)
    isempty(sub) && continue
    sT = [r["sec_sigT"] for r in sub]
    sR = [r["sec_sigR"] for r in sub]
    sN = [r["sec_sigN"] for r in sub]
    an = filter(isfinite, [r["aniso_s2_over_s1"] for r in sub])
    @printf("\n  %-12s n=%d\n", cls, length(sub))
    @printf("    secondary RTN 1σ (m): R med=%.1f  T med=%.1f  N med=%.1f\n",
            median(sR), median(sT), median(sN))
    @printf("    secondary T/R ratio (in-track dominance): median=%.1f×\n",
            median(sT ./ sR))
    isempty(an) || @printf("    encounter-plane anisotropy σ2/σ1: median=%.0f×  max=%.0f×  >100×: %d/%d\n",
            median(an), maximum(an), count(>(100), an), length(an))
end
# vs our synthetic 1 km ISOTROPIC debris Σ
all_an = filter(isfinite, [r["aniso_s2_over_s1"] for r in records])
@printf("\n  ALL 53: anisotropy median=%.0f×  >100×: %d/53  >300×: %d/53\n",
        median(all_an), count(>(100), all_an), count(>(300), all_an))
println("  (our synthetic debris Σ is 1 km ISOTROPIC → anisotropy = 1×; real data above)")

# --- (c) SLOW vs FAST, vs NASA violation flag ---
println("\n" * "-"^78)
println("(c) RELATIVE VELOCITY regime vs NASA 2D-Pc usage violation")
println("-"^78)
isviol(r) = isfinite(r["violation_flag"]) && r["violation_flag"] != 0
vr = [r["vrel_mps"] for r in records]
# Three physical regimes: prox-ops (<~100 m/s, near co-orbiting), slow-tail
# (~0.1–2 km/s), and fast (≥2 km/s, the typical crossing conjunction).
prox = filter(r -> r["vrel_mps"] < 100, records)
slow = filter(r -> 100 <= r["vrel_mps"] < 2000, records)
fast = filter(r -> r["vrel_mps"] >= 2000, records)
@printf("Vrel: min=%.1f  median=%.0f  max=%.0f m/s\n",
        minimum(vr), median(vr), maximum(vr))
@printf("  PROX-OPS (<100 m/s): %d   SLOW (0.1–2 km/s): %d   FAST (≥2 km/s): %d\n",
        length(prox), length(slow), length(fast))
for (lbl, grp) in [("PROX", prox), ("SLOW", slow), ("FAST", fast)]
    viol = count(isviol, grp)
    @printf("    %-4s: %d/%d flagged as 2D-Pc usage violation\n",
            lbl, viol, length(grp))
end
nviol = count(isviol, records)
@printf("  TOTAL flagged: %d/53  (NB: CURATED set — validity map, NOT prevalence)\n", nviol)

# --- (d) DEBRIS-DOMINATES-UNCERTAINTY ---
println("\n" * "-"^78)
println("(d) DEBRIS/secondary vs PRIMARY (NASA payload) covariance magnitude")
println("-"^78)
ratios = [r["sec_over_prim_rms"] for r in records]
@printf("secondary/primary 3D position 1σ ratio: median=%.1f×  min=%.2f×  max=%.1f×\n",
        median(ratios), minimum(ratios), maximum(ratios))
sec_bigger = count(>(1.0), ratios)
@printf("  secondary uncertainty LARGER than primary in %d/53 cases\n", sec_bigger)
for cls in ["debris", "rocket_body", "payload"]
    sub = filter(r -> r["secondary_class"] == cls, records)
    isempty(sub) && continue
    rr = [r["sec_over_prim_rms"] for r in sub]
    @printf("    %-12s (n=%d): median sec/prim = %.1f×\n", cls, length(sub), median(rr))
end

# --- (e) LEAD-TIME × CLASS × VIOLATION cross-tabs -------------------------
# The operationally-important cut: what lives near the ~24h decision point, and
# does 2D-Pc validity correlate with how far out the CDM was issued?
println("\n" * "-"^78)
println("(e) LEAD-TIME REGION × secondary class × violation")
println("-"^78)
region(h) = h < 30 ? "~24h (<30h)" : (h <= 72 ? "1-3d (30-72h)" : ">3d")
regions = ["~24h (<30h)", "1-3d (30-72h)", ">3d"]
classes = ["debris", "rocket_body", "payload", "unknown"]

println("\n  TIME × CLASS (secondary):")
@printf("    %-16s%10s%13s%10s%10s%8s\n", "region", "debris", "rocket_body", "payload", "unknown", "total")
for reg in regions
    sub = filter(r -> region(r["lead_hours"]) == reg, records)
    cnts = [count(r -> r["secondary_class"] == c, sub) for c in classes]
    @printf("    %-16s%10d%13d%10d%10d%8d\n", reg, cnts..., length(sub))
end

println("\n  TIME × VIOLATION (2D-Pc usage validity):")
@printf("    %-16s%12s%10s%8s%8s\n", "region", "violation", "no-viol", "total", "viol%")
for reg in regions
    sub = filter(r -> region(r["lead_hours"]) == reg, records)
    v = count(isviol, sub)
    @printf("    %-16s%12d%10d%8d%7.0f%%\n", reg, v, length(sub) - v, length(sub),
            100v / length(sub))
end

println("\n  Median encounter-plane anisotropy + along-track σ by region:")
for reg in regions
    sub = filter(r -> region(r["lead_hours"]) == reg, records)
    an = filter(isfinite, [r["aniso_s2_over_s1"] for r in sub])
    secT = [r["sec_sigT"] for r in sub]
    @printf("    %-16s aniso med=%6.0f×   secondary along-track 1σ med=%7.1f km\n",
            reg, median(an), median(secT) / 1000)
end

# --- object-class audit table (so classes are fully traceable) ---
println("\n" * "-"^78)
println("OBJECT-CLASS AUDIT (raw names → inferred class; primary is always payload)")
println("-"^78)
for r in records
    @printf("  %-20s | sec: %-28s → %-11s | prim: %-18s\n",
            r["id"][1:min(end,20)], r["name2"], r["secondary_class"], r["name1"])
end

# ---------------------------------------------------------------------------
# JSON dump (hand-rolled; no JSON dep needed — records are flat Dicts)
# ---------------------------------------------------------------------------
function json_val(v)
    v isa AbstractString && return "\"" * replace(v, "\"" => "\\\"") * "\""
    v isa Bool && return v ? "true" : "false"
    (v isa AbstractFloat && !isfinite(v)) && return v == Inf ? "\"Infinity\"" :
        (v == -Inf ? "\"-Infinity\"" : "\"NaN\"")
    return string(v)
end
function json_obj(d)
    ks = sort(collect(keys(d)))
    "{" * join(["\"$k\":$(json_val(d[k]))" for k in ks], ",") * "}"
end

outdir = joinpath(@__DIR__, "data")
isdir(outdir) || mkpath(outdir)
outpath = joinpath(outdir, "cara_deepdive.json")
open(outpath, "w") do io
    println(io, "[")
    for (i, r) in enumerate(records)
        print(io, "  ", json_obj(r))
        println(io, i < length(records) ? "," : "")
    end
    println(io, "]")
end
println("\nWrote per-conjunction records → $outpath")
