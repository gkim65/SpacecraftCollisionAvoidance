#!/usr/bin/env python3
"""FIG C companion [SKETCH] -- episode Pc trace vs the chance constraint, by SENSOR QUALITY.

Per conjunction, how the collision probability evolves over the encounter if the
spacecraft NEVER MANEUVERS (just keeps measuring) -- one line PER SENSOR QUALITY
(best / median / worst) -- riding against the 1e-5 chance constraint, plus where MCTS
ends up choosing to burn vs defer, and (when it burns) the executed post-manoeuvre Pc.

  1. NORAD 29108 vs 34995, 2 h cadence -- best/median defer, worst must burn.
  2. NORAD 28654 vs 41835, 2 h cadence -- unavoidable: all qualities burn.
  3. NORAD 38771 vs 30802, 8 h cadence -- best defers, median/worst burn.

COLOR = sensor quality (consistent across panels, explained in the legend).
Style: solid = never-manoeuvre baseline; dotted-to-floor = executed after a burn.
Matches the tree-zoom (CMU serif, red 1e-5 threshold).

Data: figureScripts/data/traces/spacecraftCA-mcts-clean/*.json (committed).
Render: uv run --with matplotlib --with numpy python figureScripts/figC_trace.py
"""
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "data", "traces", "spacecraftCA-mcts-clean")
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
DELTA = 1e-5

# color = sensor quality (colour-blind-safe-ish, distinct on white/black)
QCOLOR = {"best": "#1b7837", "median": "#2166ac", "worst": "#b2182b"}
QLABEL = {"best": "Best sensors", "median": "Median sensors", "worst": "Worst sensors"}

# Pc axis: real Pc spans ~1e-3 down to ~1e-16 when waiting works; a BURN drives Pc to
# exactly 0. Plot 0 (and anything below ZERO_AT) on a dedicated "= 0 (safe)" band at the
# very bottom, visually separated by a break, so it reads as "driven to zero", not clipped.
Y_LO, Y_HI = 1e-20, 3e-1      # top reaches ~1e-1 so the ~1e-2 burn-case lines aren't clipped
ZERO_AT = 1e-19          # where an exact-zero (post-burn) Pc is drawn


def setup_fonts():
    if shutil.which("latex"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["CMU Serif", "Computer Modern Roman"]})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["CMU Serif", "DejaVu Serif"],
                             "mathtext.fontset": "cm"})
    plt.rcParams.update({"font.size": 12})


def _clip(pc):
    return ZERO_AT if pc <= 0 else max(pc, ZERO_AT)


def load(run):
    d = json.load(open(os.path.join(TRACES, f"{run}.json")))
    sp = d["wait_spine"]
    spi = {c: i for i, c in enumerate(sp["columns"])}
    t = [r[spi["t_remaining_h"]] for r in sp["rows"]]
    pc = [_clip(r[spi["wait_spine_pc"]]) for r in sp["rows"]]
    tc = {c: i for i, c in enumerate(d["trace"]["columns"])}
    rows = d["trace"]["rows"]
    ex_t = [r[tc["t_remaining_h"]] for r in rows]
    ex_pc = [_clip(r[tc["pc"]]) for r in rows]
    # The X marks the last COMPUTED-Pc DOT on the solid trajectory before the maneuver
    # (i.e. the belief the planner saw when it decided to burn) -- snapped to a real data
    # point, not an interpolated value. For a t=0 burn, use the root belief-Pc (spine[0]).
    burn_t = burn_pc = None
    for i, r in enumerate(rows):
        if r[tc["action"]] == "MANEUVER":
            if i > 0:
                burn_t = ex_t[i - 1]
                burn_pc = ex_pc[i - 1]
            else:
                burn_t = ex_t[0]
                burn_pc = pc[0] if pc else DELTA
            break
    return {"t": t, "pc": pc, "ex_t": ex_t, "ex_pc": ex_pc,
            "burn_t": burn_t, "burn_pc": burn_pc,
            "quality": d["sensor_quality"], "meta": d}


def norad(case_id):
    a, b = case_id.split("_conj_")
    return f"{int(a)} vs {int(b)}"


def panel(ax, runs, title, fg):
    edge = fg                              # marker edge / X edge = theme foreground
    band = "#1c1c1c" if fg == "white" else "0.94"      # "= 0 (safe)" band
    grid = "0.28" if fg == "white" else "0.9"
    ax.axhline(DELTA, color="#e8483a" if fg == "white" else "#d7301f", lw=1.6, ls="--",
               zorder=2)
    QMARK = {"best": "o", "median": "s", "worst": "^"}
    for run in runs:
        d = load(run)
        col = QCOLOR[d["quality"]]
        mk = QMARK.get(d["quality"], "o")
        if d["burn_t"] is not None:
            ax.plot(d["t"], d["pc"], color=col, lw=1.4, ls=(0, (4, 2)), alpha=0.55,
                    zorder=3)
            waited = d["burn_t"] != d["ex_t"][0]
            if waited:
                ax.plot(d["ex_t"], d["ex_pc"], color=col, lw=2.8, marker=mk, ms=5,
                        zorder=5)
            else:
                line_t = [d["burn_t"]] + list(d["ex_t"][1:])
                line_pc = [d["burn_pc"]] + list(d["ex_pc"][1:])
                ax.plot(line_t, line_pc, color=col, lw=2.8, marker=mk, ms=5, zorder=5)
            ax.scatter([d["burn_t"]], [d["burn_pc"]], marker="X", s=190, color=col,
                       edgecolor=edge, linewidths=0.9, zorder=6)
        else:
            ax.plot(d["t"], d["pc"], color=col, lw=2.8, marker=mk, ms=5, zorder=5)
            ax.scatter([d["t"][-1]], [d["pc"][-1]], marker="o", s=150,
                       facecolor="none", edgecolor=col, linewidths=2.4, zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(Y_LO, Y_HI)
    ax.invert_xaxis()
    ax.set_title(title, fontsize=11.5, color=fg)
    ax.grid(True, which="major", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhspan(Y_LO, ZERO_AT * 3, color=band, zorder=0)
    # theme the axes furniture
    ax.patch.set_alpha(0.0)                                  # dark-PNG: show fig bg through
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)


def save(fig, base, dark):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{base}.{ext}", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(f"{base}.png", dpi=200, transparent=False,
                facecolor="black" if dark else "white", edgecolor="none")
    print(f"wrote {base}.{{pdf,svg,png}}")


P = {   # runs (best,median,worst); ordered: defer-all / unavoidable / quality-gradient
    "40059": (["qleag6wa", "a2j4fnlm", "ruhz63po"], "Every sensor can wait it out"),
    "28654": (["i4pgu7e5", "j1cdyshb", "1vmov3ll"], "Unavoidable: waiting never clears"),
    "38771": (["q2vo7e21", "gye8c5qp", "s62nlmom"], "Quality sets how long you can wait"),
}
IDS = {"40059": "000040059_conj_000035921", "28654": "000028654_conj_000041835",
       "38771": "000038771_conj_000030802"}


def build(dark):
    fg = "white" if dark else "black"
    thr = "#e8483a" if dark else "#d7301f"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, tag in zip(axes, ["40059", "28654", "38771"]):
        runs, sub = P[tag]   # `sub` = intended message, NOT drawn (goes in the caption)
        panel(ax, runs, f"NORAD {norad(IDS[tag])}", fg)

    axes[0].set_ylabel(r"Collision probability $P_c$ at TCA")
    for ax in axes:
        ax.set_xlabel("Time to TCA (h)")
        ax.text(ax.get_xlim()[0], ZERO_AT, r"$=0$ (safe)", fontsize=8,
                color="0.7" if dark else "0.4", va="center", ha="left")
    axes[0].text(axes[0].get_xlim()[0], DELTA * 3.0, r"chance constraint $10^{-5}$",
                 color=thr, fontsize=9, va="bottom", ha="left")

    _qm = {"best": "o", "median": "s", "worst": "^"}
    qhandles = [Line2D([0], [0], color=QCOLOR[q], lw=2.6, marker=_qm[q], ms=7,
                       label=QLABEL[q]) for q in ("best", "median", "worst")]
    ng = "0.75" if dark else "0.35"        # neutral gray for the style legend swatches
    style = [
        Line2D([0], [0], color=ng, lw=2.8, label="Solid: what MCTS actually did"),
        Line2D([0], [0], color=ng, lw=1.4, ls=(0, (4, 2)), alpha=0.7,
               label="Dashed: $P_c$ if it had waited instead"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=ng, markeredgewidth=2, ms=11,
               label="MCTS defers (never maneuvers)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=ng,
               markeredgecolor=fg, ms=12, label=r"MCTS maneuvers ($P_c \to 0$)"),
    ]
    leg1 = fig.legend(handles=qhandles, loc="lower center", ncol=3, frameon=False,
                      fontsize=10, bbox_to_anchor=(0.5, 0.10),
                      title="Line colour = sensor quality")
    leg1.get_title().set_color(fg)
    for t in leg1.get_texts():
        t.set_color(fg)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=style, loc="lower center", ncol=4, frameon=False, fontsize=9,
                      bbox_to_anchor=(0.5, 0.005))
    for t in leg2.get_texts():
        t.set_color(fg)
    fig.tight_layout(rect=(0, 0.20, 1, 0.99))
    save(fig, os.path.join(OUT, f"figC_trace_{'dark' if dark else 'light'}"), dark)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    setup_fonts()
    for dark in (False, True):
        build(dark)


if __name__ == "__main__":
    main()
