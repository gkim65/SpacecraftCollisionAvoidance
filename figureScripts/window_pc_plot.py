#!/usr/bin/env python3
"""window_pc_plot.py — READ-ONLY post-processing of the maneuver-effectiveness
surface: does WINDOW-MAX Pc (worst Pc over a band of burn timings, i.e. across
the once-per-orbit ripple) differ from single-instant Pc, and does it push the
feasibility boundary earlier?  (Audit E3, correctly posed for a fast crossing —
the "window" is over BURN TIMING, not a TCA time-window; see
notes/maneuver_effectiveness_findings.md "WINDOW-Pc".)

Consumes figureScripts/data/maneuver_effectiveness.json directly (pure re-
reduction, no brahe).  Two panels:
  (A) Pc vs nominal burn-timing at a near-threshold |Δv|, single-instant vs
      window-max over ±0.5 / ±1 / ±2 orbits — shows the instant curve dipping
      into the ripple while the window-max rides the envelope TOP.
  (B) latest-fixable lead time vs |Δv|, single-instant vs window-max — shows how
      much EARLIER you must commit once you demand robustness to timing/phase.

White-bg (paper) + black-bg (slides), CMU Serif, PDF+SVG+PNG. Per CLAUDE.md.
Run:  uv run --with matplotlib --with numpy figureScripts/window_pc_plot.py
"""
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, "data", "maneuver_effectiveness.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 9.5})

case = D["case"]
mags = np.array(D["magnitudes_ms"], float)
tim = np.array(D["timings_h"], float)
PC = np.array(D["pc"], float)                 # [mag][timing]
thr = case["pc_threshold"]
period_h = case["orbital_period_h"]
dv_cur = case["dv_current_ms"]
pc_base = case["pc_baseline"]

# window-max over BURN TIMING: for each timing, max Pc over timings within ±W_h.
def winmax_curve(pc_row, W_h):
    out = np.empty_like(pc_row)
    for j, T0 in enumerate(tim):
        out[j] = pc_row[np.abs(tim - T0) <= W_h + 1e-9].max()
    return out

def latest_fixable(pc_row, W_h):
    wm = winmax_curve(pc_row, W_h)
    below = tim[wm < thr]
    return below.min() if below.size else None

Ws_orbit = [0.0, 0.5, 1.0, 2.0]
Ws_h = [w * period_h for w in Ws_orbit]

# ---- console summary (the numbers) ---------------------------------------
print(f"period {period_h:.3f} h; thr {thr:.0e}; no-burn Pc {pc_base:.3e}")
print("\nlatest-fixable lead (h): single-instant (W=0) vs window-max")
print(f"{'|dv|':>7} | " + " ".join(f"{f'W={w}orb':>9}" for w in Ws_orbit))
report_mags = [0.003, 0.005, 0.007, 0.01, 0.02, 0.05, 0.1]
for mm in report_mags:
    i = int(np.argmin(np.abs(mags - mm)))
    vals = [latest_fixable(PC[i], W) for W in Ws_h]
    print(f"{mags[i]:7.3f} | " +
          " ".join(f"{('never' if v is None else f'{v:.1f}'):>9}" for v in vals))


def theme_axes(ax, fg, grid):
    ax.grid(True, which="both", color=grid, lw=0.4, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(fg)
    ax.tick_params(colors=fg, which="both")
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(fg)


def save(fig, base, dark):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{base}.{ext}"), transparent=True,
                    dpi=200, facecolor="none", edgecolor="none")
    fig.savefig(os.path.join(OUT, f"{base}.png"), transparent=False,
                dpi=200, facecolor=("black" if dark else "white"), edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def make(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.8, 8.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.08, hspace=0.32)

    # ---- Panel A: instant vs window-max Pc vs burn timing, near-threshold |Δv|
    imag = int(np.argmin(np.abs(mags - 0.01)))    # 1 cm/s
    cols = plt.get_cmap("plasma")(np.linspace(0.15, 0.8, len(Ws_orbit)))
    axA.semilogy(tim, np.clip(PC[imag], 1e-16, 1.0), color=fg, lw=1.3,
                 label="single instant (W=0)")
    for c, wo, wh in zip(cols, Ws_orbit, Ws_h):
        if wo == 0.0:
            continue
        axA.semilogy(tim, np.clip(winmax_curve(PC[imag], wh), 1e-16, 1.0),
                     color=c, lw=1.7, label=rf"window-max $\pm{wo:g}$ orbit")
    axA.axhline(thr, color=fg, lw=1.1, ls="--", alpha=0.8)
    axA.text(0.5, thr * 1.5, r"threshold $10^{-5}$", transform=axA.get_yaxis_transform(),
             color=fg, fontsize=8.5, va="bottom", ha="center")
    axA.set_xlabel("Nominal burn timing (h before TCA)")
    axA.set_ylabel(r"$P_c$ at TCA")
    axA.set_title(rf"Single-instant vs window-max $P_c$ "
                  rf"($|\Delta v|={mags[imag]:.2f}$ m/s, ripple once per {period_h:.1f} h)")
    axA.set_ylim(1e-13, 1e-1)
    axA.invert_xaxis()
    theme_axes(axA, fg, grid)
    leg = axA.legend(loc="lower left", framealpha=0.0, ncol=2)
    for t in leg.get_texts():
        t.set_color(fg)

    # ---- Panel B: latest-fixable lead vs |Δv|, per window ------------------
    pos = mags[mags > 0]
    for c, wo, wh in zip(
            [fg] + list(plt.get_cmap("plasma")(np.linspace(0.15, 0.8, len(Ws_orbit) - 1))),
            Ws_orbit, Ws_h):
        lead = []
        for m in pos:
            i = int(np.argmin(np.abs(mags - m)))
            v = latest_fixable(PC[i], wh)
            lead.append(np.nan if v is None else v)
        lab = "single instant (W=0)" if wo == 0.0 else rf"window-max $\pm{wo:g}$ orbit"
        axB.semilogx(pos, lead, color=c, lw=1.7, marker="o", ms=3.0, label=lab)
    axB.axvline(dv_cur, color="#2166ac" if not dark else "#6baed6", lw=1.2, ls="--")
    axB.text(dv_cur, axB.get_ylim()[1], r" current $\Delta v$", color=(
        "#2166ac" if not dark else "#6baed6"), fontsize=8.5, va="top", ha="left")
    axB.set_xlabel(r"Along-track burn magnitude $|\Delta v|$ (m/s, $+\hat v$)")
    axB.set_ylabel("Latest fixable lead time (h)")
    axB.set_title("Feasibility boundary: how much earlier robustness costs")
    theme_axes(axB, fg, grid)
    leg = axB.legend(loc="upper right", framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(fg)

    save(fig, f"window_pc_{theme}", dark)


make("light")
make("dark")
print("done")
