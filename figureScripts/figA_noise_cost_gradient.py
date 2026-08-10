#!/usr/bin/env python3
"""FIG A [HEADLINE] — the cost of imperfect state knowledge (belief planner).

Quality x cadence heatmap of the MCTS DEFERRAL rate = % episodes the belief planner
resolved WITHOUT a maneuver (resolved_without_maneuver). The single truth here is
operational: how much can a belief planner safely WAIT as tracking fidelity varies?

  - Good, fresh tracking (best quality, short cadence): the belief contracts fast,
    the planner can defer most encounters (~71% at best / 2 h).
  - Poor, stale tracking (worst quality, long cadence): the belief stays uncertain,
    so the planner is forced to act far more often (deferral falls to ~22%).

That clean, monotone gradient in BOTH axes IS the cost of imperfect state knowledge:
every step of degraded tracking buys less room to wait-and-measure. (The precautionary
over-maneuver breakdown -- 15/270 burns, concentrated at poor tracking -- is a separate,
sparser finding reported in analyze_results_simple.py, not this grid.)

Single MCTS panel (the belief planner). Reads from sweep_all.csv so it re-renders in
one command after re-export; the grid is printed to console so the values are
checkable without the figure.

Render from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/figA_noise_cost_gradient.py
"""
import csv
import os
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "sweep_all.csv")
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))

POLICY = "mcts"
QUALITIES = ["best", "median", "worst"]        # rows (top -> bottom = degrading)
CADENCES = ["2", "4", "8", "24"]               # cols (left -> right = staler)
QLABEL = {"best": "Best", "median": "Median", "worst": "Worst"}


def load_grid():
    """Return (pct, cnt) arrays shaped (len(QUALITIES), len(CADENCES)) for MCTS deferral."""
    acc = defaultdict(lambda: [0, 0])          # (quality, cadence) -> [defer, n]
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["state"] != "finished" or r["policy_variant"] != POLICY:
                continue
            key = (r["sensor_quality"], r["cadence_h"])
            acc[key][1] += 1
            if r["resolved_without_maneuver"].lower() in ("true", "1"):
                acc[key][0] += 1
    pct = np.full((len(QUALITIES), len(CADENCES)), np.nan)
    cnt = np.zeros((len(QUALITIES), len(CADENCES)), dtype=int)
    for i, q in enumerate(QUALITIES):
        for j, c in enumerate(CADENCES):
            d, n = acc[(q, c)]
            cnt[i, j] = n
            if n > 0:
                pct[i, j] = 100 * d / n
    return pct, cnt


def print_grid(pct, cnt):
    print("\nFIG A — MCTS resolved-without-maneuver (deferral) %  (n)")
    print("quality \\ cadence_h " + "".join(f"{c:>12}" for c in CADENCES))
    for i, q in enumerate(QUALITIES):
        cells = []
        for j in range(len(CADENCES)):
            if cnt[i, j] == 0:
                cells.append(f"{'--':>12}")
            else:
                cells.append(f"{pct[i, j]:>6.1f} ({cnt[i, j]:>2})".rjust(12))
        print(f"{q:<20}" + "".join(cells))


def setup_fonts():
    if shutil.which("latex"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["CMU Serif", "Computer Modern Roman"]})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["CMU Serif", "DejaVu Serif"],
                             "mathtext.fontset": "cm"})
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 11})


def pctsym():
    return r"\%" if plt.rcParams.get("text.usetex") else "%"


def save(fig, base, dark):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{base}.{ext}", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(f"{base}.png", transparent=False, dpi=200,
                facecolor="black" if dark else "white", edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def make_fig(pct, cnt, dark, vmax):
    fg = "white" if dark else "black"
    # viridis (matches fig2_quality_cadence_heatmap): a full, bright gradient -- the
    # deferral rate is non-zero in every cell, so the grid reads as a smooth surface.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#333333" if dark else "#dddddd")   # empty (no-data) cells

    # figure aspect chosen so the 4-col x 3-row grid sits with SQUARE cells (aspect
    # "equal") and minimal surrounding whitespace.
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(left=0.14, right=0.82, top=0.84, bottom=0.16)

    masked = np.ma.masked_invalid(pct)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="equal", origin="upper")

    for i in range(len(QUALITIES)):
        for j in range(len(CADENCES)):
            if cnt[i, j] == 0:
                ax.text(j, i, "--", ha="center", va="center", color=fg, alpha=0.6)
                continue
            val = pct[i, j]
            tc = "white" if val < 0.55 * vmax else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=15, color=tc, fontweight="bold")

    ax.set_xticks(range(len(CADENCES)))
    ax.set_xticklabels(CADENCES)
    ax.set_yticks(range(len(QUALITIES)))
    ax.set_yticklabels([QLABEL[q] for q in QUALITIES])
    ax.set_xlabel("Measurement cadence (h)")
    ax.set_ylabel("Sensor quality")
    ax.set_title("Cost of imperfect state knowledge", color=fg, pad=26, fontsize=15.5)
    ax.text(0.5, 1.045, "how much the belief planner can safely defer, by tracking fidelity",
            transform=ax.transAxes, ha="center", va="bottom", color=fg,
            fontsize=11.5, alpha=0.85, style="italic")

    # (direction reads off the axes themselves: best->worst top-to-bottom,
    # 2 h -> 24 h left-to-right = degrading tracking toward the lower-right.)

    ax.set_xticks(np.arange(-0.5, len(CADENCES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(QUALITIES), 1), minor=True)
    ax.grid(which="minor", color=fg, lw=0.5, alpha=0.3)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=fg, which="major")
    for s in ax.spines.values():
        s.set_color(fg)
    ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)

    # colorbar tracks the (equal-aspect-shrunk) axes height so it stays aligned.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.18)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"Deferral rate ({pctsym()} resolved without a maneuver)", color=fg)
    cb.ax.yaxis.set_tick_params(color=fg)
    cb.outline.set_edgecolor(fg)
    for t in cb.ax.get_yticklabels():
        t.set_color(fg)

    save(fig, os.path.join(OUT, f"figA_noise_cost_gradient_{'dark' if dark else 'light'}"), dark)


def main():
    os.makedirs(OUT, exist_ok=True)
    pct, cnt = load_grid()
    print_grid(pct, cnt)
    vmax = float(np.nanmax(pct)) if np.isfinite(pct).any() else 100.0
    vmax = 10 * np.ceil(vmax / 10)              # clean ceiling
    print(f"colorbar vmax = {vmax:.0f}%")
    setup_fonts()
    for dark in (False, True):
        make_fig(pct, cnt, dark, vmax)


if __name__ == "__main__":
    main()
