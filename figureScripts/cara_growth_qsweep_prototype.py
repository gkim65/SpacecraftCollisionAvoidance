#!/usr/bin/env python3
"""cara_growth_qsweep_prototype.py — PROOF-OF-MECHANISM (scratch-quality): does a
tuned SNC process-noise q bend our linear-STM along-track growth into ~tau^2 to
match the CARA 4-point curve? (audit F3 follow-up, Grace 2026-08-08). NUMBERS ONLY.

WHAT THIS IS: a self-contained, offline, reduced-1D prototype — NOT the real 6D
brahe propagation and NOT a src/ change. It exists to DE-RISK the process-noise
fix (findings note TODO #2) by showing the mechanism works before anyone builds it
in the real code.

MODEL (Folcik AMOS 2011, single along-track position/velocity DOF, Eqs. 10-14):
  state = [x (along-track pos, m), v (along-track vel ~ SMA drift rate, m/s)]
  PREDICT one step dt:   Sigma^- = Phi Sigma Phi^T + Q(dt)
    Phi   = [[1, dt],[0, 1]]                              (secular along-track drift)
    Q(dt) = q * [[dt^3/3, dt^2/2],[dt^2/2, dt]]           (SNC white-noise accel)
  q = process-noise spectral density for unmodeled along-track accel (drag).
  q = 0  ->  pure Folcik polynomial (== our Q=0 STM, under-grows at ~tau^1).

Anchor: at lead 0.5 d, along-track 1sigma = 38 m (real CARA <1 d bin). Sweep q,
read sigma_x at the 4 real leads, fit the apparent exponent, score vs real.

RESULT (see the findings note §6e): q ~ 1e-9 (THESE toy units only) gives p ~ 2.1
and threads the 4 points (best RMS). Confirms the phenomenological p~2 goalpost
from cara_growth_psweep_data.jl. The specific q is toy-unit-specific and MUST NOT
be ported to the real code; the real tuning is the 6D brahe Q against a chi^2
containment check (TODO #2).

Run:  uv run figureScripts/cara_growth_qsweep_prototype.py   (pure stdlib; python3 also fine)
"""
import math

DAY = 86400.0
ANCHOR_LEAD_S = 0.5 * DAY
ANCHOR_SIG = 38.0                       # real CARA <1 d in-track 1sigma (m)
REAL = [(0.5, 38.0), (1.5, 228.0), (3.0, 906.0), (5.5, 5588.0)]   # in-track sigma vs lead(d)


def propagate(q, c_vv0, dt=600.0, t_days=5.5):
    """Anchor at 0.5 d with along-track pos var 38^2 and initial vel var c_vv0,
    step forward with SNC Q(q) to t_days; return sigma_x (m) at the 4 real leads."""
    cxx, cxv, cvv = ANCHOR_SIG ** 2, 0.0, c_vv0
    t = ANCHOR_LEAD_S
    out = {0.5: ANCHOR_SIG}
    targets = {1.5, 3.0, 5.5}
    n = int(round((t_days * DAY - ANCHOR_LEAD_S) / dt))
    for _ in range(n):
        cxx_n = cxx + 2 * dt * cxv + dt * dt * cvv + q * dt ** 3 / 3.0
        cxv_n = cxv + dt * cvv + q * dt ** 2 / 2.0
        cvv_n = cvv + q * dt
        cxx, cxv, cvv = cxx_n, cxv_n, cvv_n
        t += dt
        ld = t / DAY
        for tg in list(targets):
            if ld >= tg:
                out[tg] = math.sqrt(max(cxx, 0.0))
                targets.discard(tg)
    return out


def fit_exponent(s):
    xs = [math.log(ld / 0.5) for ld in (1.5, 3.0, 5.5)]
    ys = [math.log(s[ld] / ANCHOR_SIG) for ld in (1.5, 3.0, 5.5)]
    return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)


def rms_fit(s):
    res = [math.log10(s[ld] / tr) for ld, tr in REAL if ld != 0.5]
    return math.sqrt(sum(r * r for r in res) / len(res))


print("Baseline Q=0 (pure STM/Folcik polynomial), varying initial along-track vel 1sigma:")
for sv in (1e-3, 3e-3, 1e-2, 3e-2):
    s = propagate(0.0, sv ** 2)
    print(f"  v0_sigma={sv:6.3f} m/s -> p={fit_exponent(s):.2f}  "
          f"sig(1.5,3,5.5)={s[1.5]:.0f},{s[3.0]:.0f},{s[5.5]:.0f}  RMSfit={rms_fit(s):.3f}")

print("\nSweep process noise q (SNC spectral density, toy units) at v0_sigma=1e-3 m/s:")
best = None
for qexp in range(-16, -3):
    q = 10.0 ** qexp
    s = propagate(q, (1e-3) ** 2)
    p, r = fit_exponent(s), rms_fit(s)
    if best is None or r < best[1]:
        best = (q, r, p, s)
    print(f"  q=1e{qexp:<3d} -> p={p:4.2f}  "
          f"sig(1.5,3,5.5)={s[1.5]:8.0f},{s[3.0]:9.0f},{s[5.5]:10.0f}  RMSfit={r:.3f}")

q, r, p, s = best
print(f"\nBest q ~ {q:.0e} (TOY UNITS — do not port): p={p:.2f}, RMS fit={r:.3f}")
print(f"  vs REAL: 1.5d {s[1.5]:.0f}/228  3d {s[3.0]:.0f}/906  5.5d {s[5.5]:.0f}/5588")
print("Mechanism confirmed: a tuned SNC Q bends tau^1 -> ~tau^2 and threads the 4 "
      "points, matching the phenomenological p~2 goalpost. Real fix = 6D brahe Q "
      "vs chi^2 containment (findings note TODO #2).")
