#!/usr/bin/env python3
"""cara_2d3d_analysis.py — characterize the exact-2D-vs-3D Pc gap over the 53 CARA cases.

Response:  y = log10(elrod_pc / NASA_Nc3D)   (negative = our 2D is too LOW = unsafe)
Regressors: relative velocity, approach angle, encounter-plane anisotropy (sigma2/sigma1),
            Mahalanobis miss-in-sigma, and lead-time-at-CDM-creation.

Two data sources, joined on the conjunction "id":
  - figureScripts/data/cara_deepdive.json     (lead, aniso, vrel, vang, miss, Nc3D, class, viol)
  - figureScripts/data/cara_2d3d_regressors.json (OUR elrod_pc + Mahalanobis miss-in-sigma)

Key questions (per paper_readiness_audit.md B3 + deep-dive finding (e)):
  1. How big is the exact-2D-vs-3D gap actually? (median, spread)  -- vs the "5x" Chan figure.
  2. Is the gap driven by anisotropy (mechanism) with lead-time as a downstream proxy,
     or does lead-time add independent signal? (partial correlations / nested regression)
  3. HYPOTHESIS: is the needed correction ~=1 at operational ~24h lead times, with large
     corrections confined to the >3-day regime we would NOT decide on?

Prints numbers only; writes the joined analysis frame to
figureScripts/data/cara_2d3d_joined.json for the figure script. No figures here.

Run:  uv run --with numpy figureScripts/cara_2d3d_analysis.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def spearman(x, y):
    """Spearman rank correlation (no scipy dependency)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), 0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / denom) if denom else float("nan")), len(x)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), 0
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt((xc * xc).sum() * (yc * yc).sum())
    return (float((xc * yc).sum() / denom) if denom else float("nan")), len(x)


def partial_spearman(x, y, z):
    """Spearman correlation of x,y controlling for z (rank-residual method)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 4:
        return float("nan"), 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)

    def resid(a, b):
        # residual of a after linear regression on b (b augmented with intercept)
        B = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    denom = math.sqrt((ex * ex).sum() * (ey * ey).sum())
    return (float((ex * ey).sum() / denom) if denom else float("nan")), len(x)


def ols(y, X, names):
    """Plain OLS with intercept. Returns coefs, R^2. X columns already selected."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[m], X[m]
    A = np.hstack([np.ones((len(y), 1)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return coef, r2, len(y)


def pct(a, q):
    a = np.asarray([v for v in a if np.isfinite(v)], float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ---------------------------------------------------------------------------
# 1. Load + join
# ---------------------------------------------------------------------------
deep = {r["id"]: r for r in load("cara_deepdive.json")}
reg = {r["id"]: r for r in load("cara_2d3d_regressors.json")}
assert set(deep) == set(reg), "id mismatch between the two JSONs"

rows = []
for cid, d in deep.items():
    r = reg[cid]
    elrod = r["elrod_pc"]
    nc3d = d["nc3d"]
    y = math.log10(elrod / nc3d) if (elrod and nc3d and elrod > 0 and nc3d > 0) else float("nan")
    rows.append({
        "id": cid,
        "name1": d["name1"], "name2": d["name2"],
        "class": d["secondary_class"],
        "elrod_pc": elrod,
        "nc3d": nc3d,
        "pc2d_nasa": d["pc2d"],
        "log_ratio": y,                       # response
        "vrel_mps": d["vrel_mps"],
        "vang_deg": d["vang_deg"],
        "aniso": d["aniso_s2_over_s1"],
        "miss_maha": r["miss_maha_sigma"],
        "miss_m": d["miss_m"],
        "lead_hours": d["lead_hours"],
        "lead_days": d["lead_days"],
        "violation": 0 if d["violation_flag"] == 0 else 1,
        "violation_label": d.get("violation_label", ""),
    })

N = len(rows)
print("=" * 78)
print(f"Exact-2D (elrod_pc) vs 3D (NASA Nc3D) — {N} real CARA conjunctions")
print("=" * 78)

# ---------------------------------------------------------------------------
# 2. The degenerate-tail guard. Pc ~ 1e-168 cases make log-ratios meaningless.
#    Exclude any case with Nc3D or elrod below a sane operational floor (1e-12).
# ---------------------------------------------------------------------------
FLOOR = 1e-12
usable = [r for r in rows
          if np.isfinite(r["log_ratio"]) and r["nc3d"] > FLOOR and r["elrod_pc"] > FLOOR]
degenerate = [r for r in rows if r["elrod_pc"] <= FLOOR or r["nc3d"] <= FLOOR]
print(f"\nusable cases (Nc3D & elrod_pc > {FLOOR:g}): {len(usable)} / {N}")
if degenerate:
    print(f"excluded {len(degenerate)} degenerate deep-tail case(s) "
          f"(Pc below {FLOOR:g}, log-ratio numerically meaningless):")
    for r in degenerate:
        print(f"    {r['name1']} / {r['name2']}  elrod={r['elrod_pc']:.2e}  Nc3D={r['nc3d']:.2e}")

y = np.array([r["log_ratio"] for r in usable])

# ---------------------------------------------------------------------------
# 3. Headline: how big is the exact-2D-vs-3D gap, really?
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("HEADLINE — size of the exact-2D-vs-3D gap  (log10(elrod_pc / Nc3D))")
print("-" * 78)
print(f"  median   = {np.median(y):+.4f}   ->  correction factor {10**(-np.median(y)):.3f}x")
print(f"  mean     = {y.mean():+.4f}")
print(f"  10th pct = {pct(y,10):+.4f}   90th pct = {pct(y,90):+.4f}")
print(f"  min      = {y.min():+.4f}   max = {y.max():+.4f}")
print(f"  cases where 2D too LOW (log<0, unsafe): {int((y<0).sum())} / {len(y)}")
print(f"  cases beyond 2x too low (log<-0.30):    {int((y<-0.30).sum())} / {len(y)}")
print(f"  cases beyond 5x too low (log<-0.70):    {int((y<-0.70).sum())} / {len(y)}")
print("  NOTE: the validation note's '5x too low median' was chan_pc-vs-Nc3D;")
print("        chan's anisotropy bug inflated it. This is exact-2D-vs-3D.")

# ---------------------------------------------------------------------------
# 4. Univariate correlations of the response with each regressor
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("UNIVARIATE correlation of log-ratio with each regressor  (usable cases)")
print("-" * 78)
regressors = {
    "log10 vrel":        [math.log10(r["vrel_mps"]) if r["vrel_mps"] > 0 else float("nan") for r in usable],
    "approach angle deg":[r["vang_deg"] for r in usable],
    "log10 anisotropy":  [math.log10(r["aniso"]) if r["aniso"] > 0 and np.isfinite(r["aniso"]) else float("nan") for r in usable],
    "miss-in-sigma (Maha)":[r["miss_maha"] for r in usable],
    "lead hours":        [r["lead_hours"] for r in usable],
}
print(f"  {'regressor':22s} {'Spearman':>10s} {'Pearson':>10s} {'n':>4s}")
for name, xv in regressors.items():
    sp, ns = spearman(xv, y)
    pe, _ = pearson(xv, y)
    print(f"  {name:22s} {sp:>10.3f} {pe:>10.3f} {ns:>4d}")

# ---------------------------------------------------------------------------
# 5. Mechanism test: is lead-time just a downstream proxy for anisotropy?
#    (a) how tightly are lead-time and anisotropy themselves coupled?
#    (b) partial correlation of response with lead-time CONTROLLING for aniso,
#        and vice versa. If lead-time's signal vanishes once aniso is held
#        fixed but aniso survives, aniso is the mechanism.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("MECHANISM — anisotropy vs lead-time  (does lead-time add independent signal?)")
print("-" * 78)
lead = [r["lead_hours"] for r in usable]
laniso = [math.log10(r["aniso"]) if r["aniso"] > 0 and np.isfinite(r["aniso"]) else float("nan") for r in usable]
sp_la, _ = spearman(lead, laniso)
print(f"  lead-time  vs  log-anisotropy      Spearman = {sp_la:+.3f}   "
      f"(strong + => aniso shrinks as TCA nears)")

pr_lead, n1 = partial_spearman(lead, y, laniso)
pr_aniso, n2 = partial_spearman(laniso, y, lead)
print(f"  response ~ lead   | anisotropy held : partial Spearman = {pr_lead:+.3f}  (n={n1})")
print(f"  response ~ aniso  | lead-time held  : partial Spearman = {pr_aniso:+.3f}  (n={n2})")
print("  Reading: the regressor whose partial correlation SURVIVES is the mechanism;")
print("           the one that collapses toward 0 is the downstream proxy.")

# nested OLS: does lead-time improve R^2 once anisotropy is in the model?
Xa = np.array([[la] for la in laniso])
Xal = np.array([[la, lh] for la, lh in zip(laniso, lead)])
ca, r2a, na = ols(y, Xa, ["log_aniso"])
cal, r2al, nal = ols(y, Xal, ["log_aniso", "lead_h"])
print(f"\n  OLS  y ~ log_aniso             : R^2 = {r2a:.3f}  (n={na})")
print(f"  OLS  y ~ log_aniso + lead_h    : R^2 = {r2al:.3f}  (n={nal})")
print(f"  delta R^2 from adding lead-time: {r2al - r2a:+.3f}  "
      f"(near 0 => lead-time is redundant given anisotropy)")

# ---------------------------------------------------------------------------
# 6. THE HYPOTHESIS: correction ~1 at operational ~24h, big corrections only far out.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("HYPOTHESIS — is the correction ~1 at ~24h, large only in the >3-day regime?")
print("-" * 78)
bands = [("~24 h (<=30 h)", lambda h: h <= 30),
         ("1-3 d (30-72 h)", lambda h: 30 < h <= 72),
         (">3 d (>72 h)",    lambda h: h > 72)]
print(f"  {'band':18s} {'n':>3s} {'median log':>11s} {'corr factor':>12s} "
      f"{'worst log':>10s} {'#unsafe':>8s} {'#viol':>6s}")
for label, sel in bands:
    sub = [r for r in usable if sel(r["lead_hours"])]
    if not sub:
        print(f"  {label:18s}   0")
        continue
    ys = np.array([r["log_ratio"] for r in sub])
    worst = ys.min()
    corr = 10 ** (-np.median(ys))
    nunsafe = int((ys < -0.30).sum())      # >2x too low
    nviol = sum(r["violation"] for r in sub)
    print(f"  {label:18s} {len(sub):>3d} {np.median(ys):>+11.4f} {corr:>11.3f}x "
          f"{worst:>+10.4f} {nunsafe:>8d} {nviol:>6d}")

# Also report the raw (unbanded) monotone relationship for completeness.
sp_lead_y, _ = spearman(lead, y)
print(f"\n  overall Spearman(lead-time, log-ratio) = {sp_lead_y:+.3f}  "
      f"(+ => farther out = more negative = more underestimate)")

# ---------------------------------------------------------------------------
# 7. Is a simple correction well-defined? Spread of the gap within the
#    operational band vs far out. Cross-check against NASA violation labels.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("IS A SIMPLE CORRECTION WELL-DEFINED?  (spread of the gap by band + by violation)")
print("-" * 78)
for label, sel in bands:
    sub = [r["log_ratio"] for r in usable if sel(r["lead_hours"])]
    if len(sub) >= 2:
        print(f"  {label:18s} IQR of log-ratio = [{pct(sub,25):+.3f}, {pct(sub,75):+.3f}]  "
              f"spread = {pct(sub,75)-pct(sub,25):.3f} dex")
print()
for lab, sel in [("no violation (2D valid)", lambda r: r["violation"] == 0),
                 ("VIOLATION (2D flagged)",  lambda r: r["violation"] == 1)]:
    sub = [r["log_ratio"] for r in usable if sel(r)]
    if sub:
        print(f"  {lab:26s} n={len(sub):>2d}  median log = {np.median(sub):+.4f}  "
              f"corr {10**(-np.median(sub)):.3f}x  worst {min(sub):+.4f}")

# ---------------------------------------------------------------------------
# 7b. Per-case correction table (easy to read), sorted by lead-time.
#     Correction factor = Nc3D / elrod_pc (how much to MULTIPLY our 2D by to
#     reach the 3D reference). >1 means our 2D is too low by that factor.
#     Written both to stdout and to a Markdown file for the findings note.
# ---------------------------------------------------------------------------
def corr_factor(r):
    if r["elrod_pc"] and r["elrod_pc"] > 0 and r["nc3d"] and r["nc3d"] > 0:
        return r["nc3d"] / r["elrod_pc"]
    return float("nan")

CLS = {"debris": "debris", "rocket_body": "R/B", "payload": "payload", "unknown": "unk"}
tbl = sorted(rows, key=lambda r: (r["lead_hours"]))
hdr = (f"| {'primary / secondary':34s} | {'class':7s} | {'lead h':>7s} | "
       f"{'miss-σ':>7s} | {'aniso':>8s} | {'elrod Pc':>10s} | {'Nc3D':>10s} | "
       f"{'corr ×':>10s} | viol |")
sep = "|" + "|".join(["-" * w for w in (36, 9, 9, 9, 10, 12, 12, 12, 6)]) + "|"
lines_md = ["| primary / secondary | class | lead h | miss-σ | aniso σ₂/σ₁ | "
            "elrod Pc | NASA Nc3D | correction × | viol |",
            "|---|---|---|---|---|---|---|---|---|"]
print("\n" + "-" * 78)
print("PER-CASE CORRECTION TABLE (sorted by lead-time; correction × = Nc3D / elrod_pc)")
print("-" * 78)
print(hdr)
print(sep)
for r in tbl:
    cf = corr_factor(r)
    deg = " *" if (r in degenerate) else ""      # mark degenerate deep-tail
    name = f"{r['name1']} / {r['name2']}"
    aniso = r["aniso"]
    aniso_s = f"{aniso:8.0f}" if np.isfinite(aniso) else "     inf"
    cf_s = "  degen*" if r in degenerate else (f"{cf:10.2f}" if np.isfinite(cf) else "       nan")
    miss_s = f"{r['miss_maha']:7.2f}" if np.isfinite(r["miss_maha"]) else "    nan"
    print(f"| {name[:34]:34s} | {CLS.get(r['class'],r['class']):7s} | "
          f"{r['lead_hours']:7.1f} | {miss_s} | {aniso_s} | "
          f"{r['elrod_pc']:10.3e} | {r['nc3d']:10.3e} | {cf_s} | "
          f"{'YES' if r['violation'] else '-':>4s} |{deg}")
    cf_md = "degenerate\\*" if r in degenerate else (f"{cf:.2f}" if np.isfinite(cf) else "nan")
    aniso_md = f"{aniso:.0f}" if np.isfinite(aniso) else "inf"
    miss_md = f"{r['miss_maha']:.2f}" if np.isfinite(r["miss_maha"]) else "nan"
    lines_md.append(f"| {name} | {CLS.get(r['class'],r['class'])} | {r['lead_hours']:.1f} | "
                    f"{miss_md} | {aniso_md} | {r['elrod_pc']:.3e} | {r['nc3d']:.3e} | "
                    f"{cf_md} | {'YES' if r['violation'] else '—'} |")
tbl_path = os.path.join(DATA, "cara_2d3d_table.md")
with open(tbl_path, "w") as f:
    f.write("Correction factor = NASA Nc3D / our elrod_pc. >1 = our 2D too low.\n")
    f.write("\\* = degenerate deep-tail case (Pc below 1e-12), log-ratio meaningless.\n\n")
    f.write("\n".join(lines_md) + "\n")
print(f"\n(markdown table -> {tbl_path})")

# ---------------------------------------------------------------------------
# 8. Dump joined frame for the figure script.
# ---------------------------------------------------------------------------
out = {
    "meta": {
        "response": "log10(elrod_pc / NASA_Nc3D)",
        "n_total": N,
        "n_usable": len(usable),
        "floor": FLOOR,
        "degenerate_ids": [r["id"] for r in degenerate],
        "median_log_ratio_usable": float(np.median(y)),
        "median_correction_factor": float(10 ** (-np.median(y))),
        "spearman_lead_aniso": sp_la,
        "partial_spearman_lead_given_aniso": pr_lead,
        "partial_spearman_aniso_given_lead": pr_aniso,
        "r2_aniso": r2a,
        "r2_aniso_plus_lead": r2al,
    },
    "rows": rows,
    "usable_ids": [r["id"] for r in usable],
}
outpath = os.path.join(DATA, "cara_2d3d_joined.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print("\n" + "=" * 78)
print(f"joined analysis frame -> {outpath}")
print("=" * 78)
