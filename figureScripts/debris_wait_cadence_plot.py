#!/usr/bin/env python3
"""debris_wait_cadence_plot.py — the debris WAIT-feasibility × cadence headline.

For each real debris-secondary CARA CDM in the operational window (lead <= 30 h), a
WAIT-only belief walk (predict -> cadence-due measurement correct -> repeat) is run to
~1 h before TCA and Pc-at-TCA (elrod, exact) recorded at every step, swept over the
measurement CADENCE {2,4,8,24} h and SSN quality grade {best,median,worst}. NO MCTS
-- this isolates the feasibility physics from any planner decision.

The question: as measurements come in, does Pc drop DURABLY below the 1e-5 threshold
so no maneuver is needed -- and how fresh (cadence) / how good (quality) must tracking
be for that to happen? (See figureScripts/debris_wait_cadence_data.jl and
notes/debris_wait_cadence_findings.md.)

Figures (white paper + black slides, CMU Serif, PDF+SVG+PNG):
  - debris_wait_cadence_<quality>_<theme> : small-multiples, one panel per case,
    Pc vs time-to-TCA, one curve per cadence, threshold line, durable-safe crossing
    marked. One such figure per quality grade (headline = median, the sourced default).
  - debris_wait_feasibility_<theme> : summary grid -- for each (case, cadence) at the
    median quality, how many hours of lead WAIT+measure buys (x = never resolves).

Run:  uv run --with matplotlib --with numpy figureScripts/debris_wait_cadence_plot.py
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

with open(os.path.join(HERE, "data", "debris_wait_cadence.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 11, "axes.titlesize": 10.5, "legend.fontsize": 9})

CASES = D["cases"]
META = D["meta"]
THR = META["pc_threshold"]
CADENCES = [int(c) for c in META["cadences_h"]]
QUALITIES = META["qualities"]
PC_FLOOR = 1e-14   # floor Pc for the log axis (underflow -> this)

# One color per cadence -- fresh (2h) cool/blue, stale (24h) warm/red.
CAD_COLORS = {2: "#2166ac", 4: "#4daf4a", 8: "#ff7f00", 24: "#e41a1c"}


def theme_axes(ax, fg, grid):
    ax.grid(True, which="both", color=grid, lw=0.35, alpha=0.55)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, which="both", labelsize=8.5)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(fg)


def save(fig, base, dark):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{base}.{ext}"), transparent=True,
                    dpi=200, facecolor="none", edgecolor="none")
    bg = "black" if dark else "white"
    fig.savefig(os.path.join(OUT, f"{base}.png"), transparent=False,
                dpi=200, facecolor=bg, edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def clean_name(fname):
    # 000040059_conj_000035921_... -> "NORAD 40059 vs 35921"
    parts = fname.split("_")
    try:
        return f"NORAD {int(parts[0])} vs {int(parts[2])}"
    except Exception:
        return fname[:18]


def curves_grid(quality, theme):
    """Small-multiples: one panel per case, Pc vs time-to-TCA, curve per cadence."""
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"

    n = len(CASES)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    # Reserve a fixed top strip (suptitle + legend) in INCHES, independent of nrow, so the
    # top row's 2-line panel titles never collide with the legend (CLAUDE.md: no overlaps).
    # The strip must clear: suptitle (~0.3") + legend (~0.5") + the top-row panel title that
    # renders ABOVE its axes (~0.55"). Panel area gets ~2.05 in/row below the strip.
    top_strip_in = 1.75
    fig_h = 2.05 * nrow + top_strip_in + 0.55
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.0, fig_h), squeeze=False)
    top_frac = 1.0 - top_strip_in / fig_h
    fig.subplots_adjust(left=0.10, right=0.985, top=top_frac, bottom=0.085,
                        hspace=0.62, wspace=0.22)

    for k, case in enumerate(CASES):
        ax = axes[k // ncol][k % ncol]
        for cad in CADENCES:
            cv = case["curves"].get(f"{quality}_{cad}h")
            if cv is None:
                continue
            t = np.array(cv["t_h"], dtype=float)
            pc = np.clip(np.array(cv["pc"], dtype=float), PC_FLOOR, 1.0)
            col = CAD_COLORS.get(cad, "#888888")
            safe = cv.get("safe", False)
            ax.semilogy(t, pc, color=col, lw=1.5, solid_capstyle="round",
                        label=f"{cad} h")
            xt = cv.get("cross_time_h")
            if safe and xt is not None:
                ax.plot([xt], [THR], marker="v", color=col, ms=6.5,
                        mec=fg, mew=0.5, zorder=5, clip_on=False)
        ax.axhline(THR, color=fg, lw=1.1, ls="--", alpha=0.85)
        ax.set_title(f"{clean_name(case['file'])}\n"
                     rf"lead {case['lead_h']:.0f} h, miss {case['miss_m']:.0f} m, "
                     rf"CARA $P_c$ {case['pc_cara']:.1e}", fontsize=8.8)
        ax.set_ylim(PC_FLOOR, 1e0)
        ax.set_xlim(0, max(1.0, case["lead_h"]) * 1.02)
        ax.invert_xaxis()   # time-to-TCA counts DOWN toward the conjunction
        theme_axes(ax, fg, grid)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # bottom-of-strip fractions computed from the reserved inches (robust to nrow)
    sup_y = 1.0 - 0.30 / fig_h
    leg_y = 1.0 - 0.72 / fig_h
    fig.text(0.5, 0.030, "Time to TCA (h)", ha="center", color=fg, fontsize=11)
    fig.text(0.018, 0.5, r"$P_c$ at TCA (WAIT + measure)", va="center",
             rotation="vertical", color=fg, fontsize=11)
    fig.suptitle("Does waiting and measuring drive a debris conjunction safe? "
                 rf"(SSN quality: {quality})", color=fg, fontsize=12.5, y=sup_y)

    handles = [plt.Line2D([], [], color=CAD_COLORS[c], lw=2.2,
                          label=f"{c} h cadence") for c in CADENCES]
    handles.append(plt.Line2D([], [], color=fg, ls="--", lw=1.1,
                              label=r"threshold $10^{-5}$"))
    handles.append(plt.Line2D([], [], color=fg, marker="v", ls="none", ms=7,
                              label="durably safe from here"))
    leg = fig.legend(handles=handles, loc="upper center", ncol=3,
                     bbox_to_anchor=(0.5, leg_y), framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(fg)

    save(fig, f"debris_wait_cadence_{quality}_{theme}", dark)


def feasibility_heatmap(quality, theme):
    """Summary grid: for each (case, cadence) at `quality`, hours of lead WAIT+measure
    buys before TCA it becomes durably safe; x = never resolves."""
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"

    ncase = len(CASES)
    M = np.full((ncase, len(CADENCES)), np.nan)   # cross-time h; nan = never safe
    for i, case in enumerate(CASES):
        for j, cad in enumerate(CADENCES):
            cv = case["curves"].get(f"{quality}_{cad}h")
            if cv and cv.get("safe") and cv.get("cross_time_h") is not None:
                M[i, j] = cv["cross_time_h"]

    fig, ax = plt.subplots(figsize=(6.8, 0.52 * ncase + 1.9))
    fig.subplots_adjust(left=0.34, right=0.9, top=0.84, bottom=0.14)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#3a3a3a" if dark else "#dddddd")   # never-safe cells
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="upper", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(CADENCES)))
    ax.set_xticklabels([f"{c} h" for c in CADENCES])
    ax.set_yticks(range(ncase))
    ax.set_yticklabels([f"{clean_name(c['file'])}  ({c['lead_h']:.0f} h)"
                        for c in CASES], fontsize=8)
    ax.set_xlabel("Measurement cadence")
    ax.set_title(r"Lead bought by WAIT + measure (h before TCA it becomes safe)"
                 "\n" rf"$\times$ = never reaches safe; SSN quality {quality}",
                 fontsize=10)
    for i in range(ncase):
        for j in range(len(CADENCES)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "×", ha="center", va="center",
                        color=fg, fontsize=11)
            else:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        color="white" if M[i, j] < vmax * 0.6 else "black",
                        fontsize=9)
    theme_axes(ax, fg, grid)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Hours before TCA safe (higher = resolves earlier)", color=fg)
    cbar.ax.yaxis.set_tick_params(color=fg)
    cbar.outline.set_edgecolor(fg)
    for t in cbar.ax.get_yticklabels():
        t.set_color(fg)
    save(fig, f"debris_wait_feasibility_{quality}_{theme}", dark)


for theme in ("light", "dark"):
    for q in QUALITIES:
        curves_grid(q, theme)
    if "median" in QUALITIES:
        feasibility_heatmap("median", theme)
print("done")
