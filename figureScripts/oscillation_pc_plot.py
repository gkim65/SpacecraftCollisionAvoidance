#!/usr/bin/env python3
"""oscillation_pc_plot.py — oscillation-aware Pc (experiment_ideas #1 / E3).

The node's Pc-at-TCA breathes once per orbit (R/N covariance CW modes) as Σ is
grown to TCA. A planner sampling at discrete hourly decision points aliases it and
can land on a lucky trough / unlucky peak. A window statistic (max / mean over
±½ orbit) removes the aliasing.

Panels:
  (A) resolved Pc(τ) ripple (fine line) + the hourly point-samples the planner
      sees (dots) + window-max and window-mean overlays. Shows the single-instant
      fragility and that the window rides the envelope.
  (B) grid-phase fragility: at a fixed decision point, the point-sampled Pc swings
      orders of magnitude as the grid phase slides over one orbit, while the
      window-max is a single stable value.

White-bg (paper) + black-bg (slides), CMU Serif, PDF+SVG+PNG. Per CLAUDE.md.
Run:  uv run --with matplotlib --with numpy figureScripts/oscillation_pc_plot.py
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

with open(os.path.join(HERE, "data", "oscillation_pc.json")) as f:
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
thr = case["pc_threshold"]
period_h = case["period_h"]

ftau = np.array(D["fine_tau_h"], float)
fpc = np.clip(np.array(D["fine_pc"], float), 1e-30, 1.0)
htau = np.array(D["hourly_tau_h"], float)
hpt = np.clip(np.array(D["hourly_point_pc"], float), 1e-30, 1.0)
hmax = np.clip(np.array(D["hourly_winmax_pc"], float), 1e-30, 1.0)
hmean = np.clip(np.array(D["hourly_winmean_pc"], float), 1e-30, 1.0)

ph = D["phase_test"]
dtau = ph["decision_tau_h"]
shifts = np.array(ph["phase_shifts_h"], float)
pt_phase = np.clip(np.array(ph["point_pc_vs_phase"], float), 1e-30, 1.0)
wmax_dec = max(ph["winmax_pc"], 1e-30)

FLOOR = 1e-13


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
    c_ripple = "#999999" if not dark else "#bbbbbb"
    c_pt = "#d6604d"       # point samples (red-ish)
    c_max = "#2166ac" if not dark else "#6baed6"  # window-max (blue)
    c_mean = "#1a9850"     # window-mean (green)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.9, 8.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.08, hspace=0.34)

    # ---- Panel A: resolved ripple + hourly samples + window stats ----------
    axA.semilogy(ftau, np.clip(fpc, FLOOR, 1.0), color=c_ripple, lw=1.0,
                 label="resolved $P_c(\\tau)$ (fine)")
    axA.semilogy(htau, np.clip(hpt, FLOOR, 1.0), color=c_pt, lw=0, marker="o",
                 ms=4.0, label="hourly point-samples (planner grid)")
    axA.semilogy(htau, np.clip(hmax, FLOOR, 1.0), color=c_max, lw=1.7,
                 label=r"window-max ($\pm\frac12$ orbit)")
    axA.semilogy(htau, np.clip(hmean, FLOOR, 1.0), color=c_mean, lw=1.7, ls="--",
                 label=r"window-mean ($\pm\frac12$ orbit)")
    axA.axhline(thr, color=fg, lw=1.1, ls=":", alpha=0.8)
    axA.text(0.5, thr * 1.6, r"threshold $10^{-5}$", transform=axA.get_yaxis_transform(),
             color=fg, fontsize=8.5, va="bottom", ha="center")
    axA.set_xlabel(r"Time remaining to TCA, $\tau$ (h)")
    axA.set_ylabel(r"$P_c$ at TCA (node)")
    axA.set_title(f"Once-per-orbit $P_c$ ripple, aliased by the hourly grid "
                  f"({case['name1']}/{case['name2']})")
    axA.set_ylim(FLOOR, 1e-1)
    axA.invert_xaxis()
    theme_axes(axA, fg, grid)
    leg = axA.legend(loc="lower center", framealpha=0.0, ncol=2)
    for t in leg.get_texts():
        t.set_color(fg)

    # ---- Panel B: grid-phase fragility -------------------------------------
    axB.semilogy(shifts / period_h, np.clip(pt_phase, FLOOR, 1.0), color=c_pt,
                 lw=1.7, marker="o", ms=3.5,
                 label=f"point-sample $P_c$ (decision @ {dtau:.0f} h)")
    axB.axhline(wmax_dec, color=c_max, lw=1.8,
                label=r"window-max ($\pm\frac12$ orbit) — phase-independent")
    axB.axhline(thr, color=fg, lw=1.1, ls=":", alpha=0.8)
    axB.text(0.5, thr * 1.6, r"threshold $10^{-5}$", transform=axB.get_yaxis_transform(),
             color=fg, fontsize=8.5, va="bottom", ha="center")
    axB.set_xlabel("Hourly-grid phase shift (orbits)")
    axB.set_ylabel(r"$P_c$ at the decision point")
    axB.set_title("Grid-phase fragility: point-sample swings, window-max is stable")
    axB.set_ylim(FLOOR, 1e-1)
    theme_axes(axB, fg, grid)
    leg = axB.legend(loc="lower center", framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(fg)

    save(fig, f"oscillation_pc_{theme}", dark)


make("light")
make("dark")
print("done")
