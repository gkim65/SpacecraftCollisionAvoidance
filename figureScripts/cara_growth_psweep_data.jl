# =========================================================================
# cara_growth_psweep_data.jl — phenomenological p-sweep: which along-track
# growth EXPONENT best matches the real CARA primary 4-point curve? (audit F3
# follow-up, Grace 2026-08-08). NUMBERS ONLY — no figure (Grace's call).
#
# Context: `cara_growth_realism_data.jl` showed our Φ Σ Φᵀ STM growth fits an
# apparent along-track exponent ~1.4 (concave, flattening), while the real CARA
# well-tracked-primary in-track 1σ grows ~τ² (accelerating). Folcik AMOS 2011
# (Eq. 12) shows the true law is a QUADRATIC POLYNOMIAL in τ (C_xx = C_xx|0 +
# 2τ C_xv|0 + τ² C_vv|0), so there is no single exponent — the apparent slope
# slides 1→2 with C_vv/C_xx. This script asks the phenomenological question:
# treating σ_T(τ) = σ0·(τ/τ0)^p as a pure power law anchored at the real <1 d
# point, which p best threads the 4 real bin points? This DEFINES THE GOALPOST
# for a future process-noise (Q) fix: "we need a Q that produces ~this p."
#
# This is a KNOB ON THE COVARIANCE SHAPE (scale along-track var by τ^p directly),
# NOT a physical process-noise tuning — it is the cheap phenomenological
# sensitivity target, deliberately separate from Q (see the findings note §6e /
# next-steps TODO #1 vs #2).
#
# READ-ONLY, no src/ code, no brahe — closed-form. Prints a table; writes JSON.
# Run:  julia figureScripts/cara_growth_psweep_data.jl
# writes figureScripts/data/cara_growth_psweep.json
# =========================================================================
using Printf
using Statistics: mean

# Real CARA primary in-track 1σ at bin-midpoint leads (from cara_cdm_deepdive_findings.md).
const REAL = [(0.5, 38.0), (1.5, 228.0), (3.0, 906.0), (5.5, 5588.0)]

# Anchor: every p-line passes through the real <1 d point (0.5 d, 38 m) by construction.
const ANCHOR_LEAD = 0.5
const ANCHOR_SIG  = 38.0
pline(p, τ) = ANCHOR_SIG * (τ / ANCHOR_LEAD)^p

const PS = [1.0, 1.4, 1.5, 2.0, 3.0]     # 1.4 ≈ our STM fit; others are candidates
const OURS_P = 1.4                        # measured this session (cara_growth_realism_data.jl)

# --- table: in-track σ at the 4 real leads for each candidate p -----------------
println("In-track σ (m) at the 4 CARA lead-bin midpoints (all anchored at 0.5 d, 38 m)")
@printf "%8s %9s | %s\n" "lead(d)" "REAL" join([@sprintf("p=%-5.1f", p) for p in PS], " ")
println("-"^70)
for (ld, tr) in REAL
    @printf "%8.1f %9.0f | %s\n" ld tr join([@sprintf("%7.0f", pline(p, ld)) for p in PS], " ")
end

# --- fit quality: RMS of log10(ours/real) across the 4 points -------------------
println("\nFit vs the 4 real points (RMS log10 ratio, lower = better):")
fit = Dict{Float64,Float64}()
for p in PS
    res = [log10(pline(p, ld) / tr) for (ld, tr) in REAL]
    rms = sqrt(mean(res .^ 2))
    fit[p] = rms
    ratios = [pline(p, ld) / tr for (ld, tr) in REAL]
    @printf "  p=%-4.1f  RMS log10 = %.3f   ratios ours/real = %s\n" p rms join([@sprintf("%5.2f", x) for x in ratios], " ")
end

# --- best single anchored exponent to the 4 real points (log-log LSQ) -----------
xs = [log(ld / ANCHOR_LEAD) for (ld, _) in REAL]
ys = [log(tr / ANCHOR_SIG)  for (_, tr) in REAL]
p_best = sum(xs .* ys) / sum(xs .^ 2)     # anchored through the origin (x=0 at anchor)

@printf "\nBest-fit anchored exponent to the 4 real points: p = %.2f\n" p_best
@printf "Our STM model gives p ≈ %.1f (too shallow); p=3 overshoots mid-range ~9×.\n" OURS_P
@printf "Takeaway: p ≈ 2 threads the 4 points (RMS %.2f); this is the target shape a\n" fit[2.0]
println("future process-noise Q must reproduce (findings note §6e, TODO #2).")

# --- JSON out -------------------------------------------------------------------
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
rows = "[" * join([@sprintf("{\"lead_d\":%g,\"real_T\":%g,%s}", ld, tr,
    join([@sprintf("\"p_%g\":%g", p, pline(p, ld)) for p in PS], ","))
    for (ld, tr) in REAL], ",") * "]"
open(joinpath(@__DIR__, "data", "cara_growth_psweep.json"), "w") do io
    write(io, "{\"anchor_lead_d\":$(ANCHOR_LEAD),\"anchor_sigma_m\":$(ANCHOR_SIG)," *
              "\"ps\":$(_j(PS)),\"ours_p\":$(OURS_P),\"p_best_fit\":$(_j(p_best))," *
              "\"rms_log10\":{" * join([@sprintf("\"%g\":%g", p, fit[p]) for p in PS], ",") * "}," *
              "\"points\":$(rows)}")
end
println("\nwrote figureScripts/data/cara_growth_psweep.json")
