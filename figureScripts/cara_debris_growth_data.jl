# =========================================================================
# cara_debris_growth_data.jl — the DEBRIS analogue of the primary growth study
# (audit F3 follow-up, Grace 2026-08-08). Measures the real SECONDARY-object
# covariance growth vs lead time from the 53 CDMs, STRATIFIED, because debris is
# heterogeneous. This is the TARGET the subsequent process-noise Q / anisotropic
# P0 fix will be tuned against (the primary study — notes/
# cara_covariance_growth_realism_findings.md — validated the model against the
# PRIMARY; the fix is applied to the DEBRIS, so we must measure the debris target).
#
# CROSS-SECTIONAL, NOT A TRAJECTORY (carried from the primary study §1): each CDM
# is a single TCA snapshot of a DIFFERENT event/object, so a "growth curve" here is
# same-class pooling across lead-time bins, confounded by per-object OD span / obs /
# altitude — NOT one object's real Σ(τ). Debris is MORE heterogeneous than the
# well-tracked primary class, so EXPECT MORE SCATTER; we report the spread, we do
# NOT force a clean line.
#
# STRATIFICATION (Grace, marginal cuts — one driver at a time, NOT a cross-product;
# 53 CDMs across 4 lead bins can't fill a full cross-product):
#   (1) object class    — debris (n=35) / payload-secondary (n=13) fitted;
#                         rocket_body (n=4) + unknown (n=1) reported as points only.
#   (2) altitude regime — debris split at perigee 600 km (17 low / 18 high);
#                         low-perigee = more drag ⇒ hypothesis: steeper p.
#   (3) tracking quality— debris split at ACTUAL_OD_SPAN 3 d (12 short / 23 long)
#                         and OBS_USED 50 (well- vs sparsely-tracked).
#
# We CONSUME figureScripts/data/cara_deepdive.json for the secondary RTN sigmas
# (sec_sigR/T/N), lead_hours, anisotropy, class — already extracted by the deep-dive,
# NOT re-extracted here. We add ONLY the per-secondary stratification fields the JSON
# lacks (perigee/apogee altitude, ACTUAL_OD_SPAN, OBS_USED), read from the raw CDMs.
#
# READ-ONLY. No src/ production code changed. Output → JSON for matplotlib.
# Run:  julia --project=. figureScripts/cara_debris_growth_data.jl
# writes figureScripts/data/cara_debris_growth.json
# =========================================================================
using Statistics
using Printf

const REPO = normpath(joinpath(@__DIR__, ".."))

# --- read the deep-dive JSON (secondary sigmas + class + lead already there) -----
# Minimal JSON array-of-objects reader (regex on flat records — same trick as
# cara_missigma_transfer_data.jl; the deepdive records are flat scalar dicts).
function read_deepdive()
    txt = read(joinpath(REPO, "figureScripts", "data", "cara_deepdive.json"), String)
    recs = Dict{String,Any}[]
    for m in eachmatch(r"\{[^{}]*\}", txt)
        s = m.match
        rec = Dict{String,Any}()
        gs(k) = (mm = match(Regex("\"$k\"\\s*:\\s*\"([^\"]*)\""), s); mm === nothing ? nothing : mm.captures[1])
        gn(k) = (mm = match(Regex("\"$k\"\\s*:\\s*([-\\d.eE+]+)"), s); mm === nothing ? NaN : parse(Float64, mm.captures[1]))
        id = gs("id"); id === nothing && continue
        rec["id"] = id
        rec["class"] = gs("secondary_class")
        rec["name2"] = gs("name2")
        rec["lead_hours"] = gn("lead_hours")
        rec["sigR"] = gn("sec_sigR"); rec["sigT"] = gn("sec_sigT"); rec["sigN"] = gn("sec_sigN")
        rec["aniso"] = gn("aniso_s2_over_s1")
        push!(recs, rec)
    end
    return recs
end

# --- per-SECONDARY (OBJECT2) stratification fields from the raw CDM --------------
# OBJECT2 is the 2nd occurrence of each repeated key/comment.
function secondary_strat_fields(path)
    peris = Float64[]; apos = Float64[]; odspan = Float64[]; obsused = Float64[]; wrms = Float64[]
    for ln in eachline(path)
        s = strip(ln)
        m = match(r"Perigee Altitude\s*=\s*([-\d.eE+]+)", s); m !== nothing && push!(peris, parse(Float64, m.captures[1]))
        m = match(r"Apogee Altitude\s*=\s*([-\d.eE+]+)", s);  m !== nothing && push!(apos, parse(Float64, m.captures[1]))
        m = match(r"^ACTUAL_OD_SPAN\s*=\s*([-\d.eE+]+)", s);  m !== nothing && push!(odspan, parse(Float64, m.captures[1]))
        m = match(r"^OBS_USED\s*=\s*([-\d.eE+]+)", s);        m !== nothing && push!(obsused, parse(Float64, m.captures[1]))
        m = match(r"^WEIGHTED_RMS\s*=\s*([-\d.eE+]+)", s);    m !== nothing && push!(wrms, parse(Float64, m.captures[1]))
    end
    g(v) = length(v) >= 2 ? v[2] : (length(v) == 1 ? v[1] : NaN)   # OBJECT2
    return (perigee = g(peris), apogee = g(apos), od_span = g(odspan),
            obs_used = g(obsused), wrms = g(wrms))
end

# map deepdive id -> raw CDM path (id starts with the filename stem's first token
# pattern; match by the id2/id1 designators embedded in the filename).
function build_id_to_path()
    d = Dict{String,String}()
    for p in readdir(joinpath(REPO, "data", "cara_cdms"); join = true)
        endswith(p, ".cdm") || continue
        d[replace(basename(p), ".cdm" => "")] = p
    end
    return d
end

# --- lead-time bins (same as the primary study: <1d / 1-2d / 2-4d / >4d) ---------
# Bin midpoints used as representative lead for plotting/fitting (days).
const BINS = [("<1 d", 0.0, 1.0, 0.5), ("1-2 d", 1.0, 2.0, 1.5),
              ("2-4 d", 2.0, 4.0, 3.0), (">4 d", 4.0, Inf, 5.5)]
bin_of(lead_days) = findfirst(b -> b[2] <= lead_days < b[3], BINS)

# --- log-log power-law fit σ_T ∝ τ^p across bin points (need ≥2 non-empty bins) --
function fit_p(lead_days::Vector{Float64}, sigT::Vector{Float64})
    m = (lead_days .> 0) .& (sigT .> 0) .& isfinite.(sigT)
    sum(m) < 2 && return (p = NaN, n = sum(m))
    x = log.(lead_days[m]); y = log.(sigT[m])
    xm = mean(x); ym = mean(y)
    p = sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
    return (p = p, n = sum(m))
end

# --- per-stratum bin summary: median + min/max band of along-track σ per lead bin
function bin_summary(recs)
    out = []
    for (label, lo, hi, mid) in BINS
        vals = Float64[r["sigT"] for r in recs if lo <= r["lead_hours"]/24 < hi && isfinite(r["sigT"])]
        aniso = Float64[r["aniso"] for r in recs if lo <= r["lead_hours"]/24 < hi && isfinite(r["aniso"])]
        push!(out, Dict("label" => label, "mid_d" => mid, "n" => length(vals),
            "med_T" => isempty(vals) ? NaN : median(vals),
            "min_T" => isempty(vals) ? NaN : minimum(vals),
            "max_T" => isempty(vals) ? NaN : maximum(vals),
            "med_aniso" => isempty(aniso) ? NaN : median(aniso)))
    end
    return out
end

# stratum → (median σ_T per bin midpoint) + fitted p, for the JSON.
function stratum(recs)
    lead = Float64[r["lead_hours"]/24 for r in recs]
    sigT = Float64[r["sigT"] for r in recs]
    return Dict("n" => length(recs),
                "fit" => let f = fit_p(lead, sigT); Dict("p" => f.p, "n_fit" => f.n) end,
                "bins" => bin_summary(recs),
                # raw scatter points (lead_d, sigT) for the figure
                "pts_lead_d" => lead, "pts_sigT" => sigT,
                "pts_class" => String[r["class"] for r in recs])
end

# =========================================================================
recs = read_deepdive()
id2path = build_id_to_path()
# attach stratification fields
for r in recs
    p = get(id2path, r["id"], nothing)
    if p === nothing
        r["perigee"] = NaN; r["od_span"] = NaN; r["obs_used"] = NaN
    else
        sf = secondary_strat_fields(p)
        r["perigee"] = sf.perigee; r["apogee"] = sf.apogee
        r["od_span"] = sf.od_span; r["obs_used"] = sf.obs_used; r["wrms"] = sf.wrms
    end
end

debris  = [r for r in recs if r["class"] == "debris"]
payload = [r for r in recs if r["class"] == "payload"]
rb      = [r for r in recs if r["class"] == "rocket_body"]
unk     = [r for r in recs if r["class"] == "unknown"]

# altitude + tracking cuts WITHIN debris (the strata Grace chose)
const PERIGEE_CUT = 600.0    # km  (17 low / 18 high)
const ODSPAN_CUT  = 3.0      # days (12 short / 23 long)
const OBS_CUT     = 50.0     # OBS_USED
deb_lowalt  = [r for r in debris if r["perigee"] < PERIGEE_CUT]
deb_highalt = [r for r in debris if r["perigee"] >= PERIGEE_CUT]
deb_shortod = [r for r in debris if r["od_span"] < ODSPAN_CUT]
deb_longod  = [r for r in debris if r["od_span"] >= ODSPAN_CUT]
deb_sparse  = [r for r in debris if r["obs_used"] < OBS_CUT]
deb_dense   = [r for r in debris if r["obs_used"] >= OBS_CUT]

strata = Dict(
    "debris_all"   => stratum(debris),
    "payload_sec"  => stratum(payload),
    "rocket_body"  => stratum(rb),
    "unknown"      => stratum(unk),
    "debris_lowalt_lt600"  => stratum(deb_lowalt),
    "debris_highalt_ge600" => stratum(deb_highalt),
    "debris_shortod_lt3d"  => stratum(deb_shortod),
    "debris_longod_ge3d"   => stratum(deb_longod),
    "debris_sparse_lt50obs"=> stratum(deb_sparse),
    "debris_dense_ge50obs" => stratum(deb_dense),
)

# real PRIMARY curve (from the primary study) for overlay + p comparison.
primary = Dict("bins" => [Dict("label"=>"<1 d","mid_d"=>0.5,"T"=>38.0),
                          Dict("label"=>"1-2 d","mid_d"=>1.5,"T"=>228.0),
                          Dict("label"=>"2-4 d","mid_d"=>3.0,"T"=>906.0),
                          Dict("label"=>">4 d","mid_d"=>5.5,"T"=>5588.0)],
               "p" => 1.93, "our_stm_p" => 1.4)

# --- JSON writer (hand-rolled, matches project convention) -----------------------
_j(x::Bool) = x ? "true" : "false"
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = "\"" * replace(x, "\"" => "\\\"") * "\""
_j(x::Nothing) = "null"
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
_j(d::AbstractDict) = "{" * join(["\"$k\":" * _j(v) for (k, v) in d], ",") * "}"

data = Dict("cuts" => Dict("perigee_km" => PERIGEE_CUT, "od_span_d" => ODSPAN_CUT, "obs_used" => OBS_CUT),
            "strata" => strata, "primary" => primary,
            "bin_edges_d" => [b[2] for b in BINS])
open(joinpath(REPO, "figureScripts", "data", "cara_debris_growth.json"), "w") do io
    write(io, _j(data))
end

# --- console report --------------------------------------------------------------
println("="^74)
@printf "Real DEBRIS covariance growth, stratified (n: debris %d, payload-sec %d, RB %d, unk %d)\n" length(debris) length(payload) length(rb) length(unk)
println("="^74)
println("\nAlong-track σ (m) median per lead bin, and fitted exponent p (σ_T ∝ τ^p):")
@printf "%-26s %6s | %8s %8s %8s %8s | %6s\n" "stratum" "n" "<1d" "1-2d" "2-4d" ">4d" "p"
println("-"^74)
function report(name, key)
    s = strata[key]
    b = s["bins"]
    med(i) = b[i]["med_T"]
    @printf "%-26s %6d | %8s %8s %8s %8s | %6s\n" name s["n"] (
        [isnan(med(i)) ? "  –" : @sprintf("%7.0f", med(i)) for i in 1:4]...) (
        isnan(s["fit"]["p"]) ? "  –" : @sprintf("%.2f", s["fit"]["p"]))
end
report("debris (all)",            "debris_all")
report("  debris perigee <600km", "debris_lowalt_lt600")
report("  debris perigee ≥600km", "debris_highalt_ge600")
report("  debris OD span <3 d",   "debris_shortod_lt3d")
report("  debris OD span ≥3 d",   "debris_longod_ge3d")
report("  debris OBS <50",        "debris_sparse_lt50obs")
report("  debris OBS ≥50",        "debris_dense_ge50obs")
report("payload secondary",       "payload_sec")
report("rocket body",             "rocket_body")
report("unknown",                 "unknown")
@printf "\nprimary (reference):        %6s | %8.0f %8.0f %8.0f %8.0f | %6.2f\n" "53" 38 228 906 5588 primary["p"]
@printf "our STM model:                                                            | %6.2f\n" primary["our_stm_p"]
println("\nwrote figureScripts/data/cara_debris_growth.json")
