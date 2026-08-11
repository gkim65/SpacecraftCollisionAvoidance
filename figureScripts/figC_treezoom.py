#!/usr/bin/env python3
"""FIG C [SKETCH] -- cleaned tree v3, Grace's 3rd round of notes.

Panels: n=50, seed 3 -- best sensor quality (-> WAIT / defer) vs worst (-> MANEUVER /
burn). The CLEAN chance-constraint contrast (best can wait until Pc clears delta; worst
cannot), with the fuller n=50 trees. (seed 3 avoids the seed-1 stochastic flip.)

Changes this round:
  - ROOT drawn as a BUBBLE (not a star); nodes in the first few layers made bigger.
  - PATH thickness = cumulative visit count with a CONSISTENT scale, so a full pathway
    that many rollouts traversed (incl. +3 -> TCA) is clearly thick end-to-end.
  - LEFT axis = real hours-to-TCA at each measurement epoch; "TCA" at the bottom.
  - OUTCOMES at TCA shown via a 90-deg brace opening DOWNWARD off each leaf, fanning to
    the rollout-outcome markers (max 4 + "...").
  - "selected action" bubble tightened (was cut off at top).
  - Figure style per CLAUDE.md: CMU serif (mathtext-cm fallback), sentence-case labels.

Data: figureScripts/data/treezoom/{best,worst}_n50_s3.json  (from treezoom_probe.jl).
Render: uv run --with matplotlib --with numpy python figureScripts/figC_treezoom.py
"""
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, PathPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "treezoom")
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
DELTA = 1e-5
GREEN, AMBER, RED = "#2ca25f", "#e6a000", "#d7301f"
CW, CM = "#2b8cbe", "#e6550d"


def setup_fonts():
    """CMU serif via LaTeX, mathtext-cm fallback when latex isn't on PATH (CLAUDE.md)."""
    if shutil.which("latex"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["CMU Serif", "Computer Modern Roman"]})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["CMU Serif", "DejaVu Serif"],
                             "mathtext.fontset": "cm"})
    plt.rcParams.update({"font.size": 12})


def is_leaf(n):
    return not any(n["children"].get(a) for a in ("WAIT", "MANEUVER"))


def nsize(n, depth):
    """Node area ~ visit count, with a floor that is LARGER for the first few layers."""
    if depth == 0:
        return 300                       # root bubble a touch smaller
    floor = 300 if depth <= 2 else 170
    return floor + 150 * np.sqrt(max(n.get("N", 0), 0))


def ewidth(n):
    """Edge width = rollouts THROUGH this edge (child visit count N), CONSISTENT linear
    scale so a busy full pathway stays thick end-to-end (root->TCA), and a 1-visit twig
    stays thin. Linear (not sqrt) so heavily-traversed spines read clearly heavier."""
    return 1.2 + 0.9 * max(n.get("N", 0), 0)


def curly_brace(ax, xc, y_top, half, y_bot, color="0.5", lw=1.0):
    """A downward-opening curly brace centered at xc: stem from (xc, y_top), splitting
    to two curls whose tips reach x = xc +- half at y_bot. Spans wider than the cluster
    it groups (pass `half` a bit larger than the dots' half-width)."""
    ymid = 0.5 * (y_top + y_bot)
    for s in (-1, +1):
        tip = xc + s * half
        p = Path(
            [(xc, y_top), (xc, ymid), (xc + s * half * 0.5, ymid),  # inner curl to centre
             (xc + s * half * 0.5, ymid), (tip, ymid), (tip, y_bot)],  # out to the tip, down
            [Path.MOVETO, Path.CURVE3, Path.CURVE3,
             Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(PathPatch(p, fill=False, edgecolor=color, lw=lw, zorder=2,
                               capstyle="round"))


def node_face(n):
    p = n["pc"]
    viol = bool(n["violated"]) if p is not None else False
    if not viol:
        return ("o", GREEN, "k", False)
    if is_leaf(n):
        return ("X", RED, RED, True)
    return ("o", "none", AMBER, False)


def epoch_hours(t_horizon, cadence_h, n_epochs):
    """Reconstruct decision-grid epoch times (h to TCA): H, H-cad, ..., 0 (TCA last).
    Mirrors src decision_grid: walk down by cadence, final epoch = 0."""
    ep = [t_horizon]
    t = t_horizon
    while t > 1e-3 and len(ep) < n_epochs:
        t = max(0.0, t - cadence_h)
        ep.append(t)
    if ep[-1] > 1e-3:
        ep.append(0.0)
    # pad/truncate to exactly n_epochs
    ep = ep[:n_epochs]
    while len(ep) < n_epochs:
        ep.append(0.0)
    ep[-1] = 0.0
    return ep


def draw(ax, d, fg):
    root = d["tree"]
    maxd = d["max_depth"]
    chosen = d["chosen_action"]
    hours = epoch_hours(d["t_horizon_h"], d["cadence_h"], maxd + 1)
    dark = (fg == "white")
    wash_col = "#242424" if dark else "0.93"        # background wash tint
    scaffold = "0.30" if dark else "0.85"           # faint full-tree scaffold
    node_edge = fg                                  # green-node outline

    # Background wash: LIGHT theme dims the NOT-CHOSEN half (de-emphasis); DARK theme
    # HIGHLIGHTS the CHOSEN half instead (Grace). WAIT=left [.., .5], MANEUVER=right [.5, ..].
    chosen_left = (chosen == "WAIT")
    if dark:
        span = (-0.04, 0.5) if chosen_left else (0.5, 1.04)   # behind the chosen side
    else:
        span = (0.5, 1.04) if chosen_left else (-0.04, 0.5)   # behind the not-chosen side
    wash_bot = -maxd - 0.82
    ax.add_patch(Rectangle((span[0], wash_bot), span[1] - span[0], 0.35 - wash_bot,
                           facecolor=wash_col, edgecolor="none", zorder=0))
    ax.patch.set_alpha(0.0)                          # dark-PNG: show fig bg through

    # faint full binary tree -- MUST use the SAME span split as overlay() (incl. the
    # depth-0 push-apart gap) so the scaffold sits under the real nodes, not offset.
    def child_spans(x0, x1, depth):
        if depth == 0:
            return [(x0, 0.46), (0.54, x1)]     # WAIT-left / MANEUVER-right with a gap
        xm = 0.5 * (x0 + x1)
        return [(x0, xm), (xm, x1)]

    def full(x0, x1, depth):
        if depth >= maxd:
            return
        xm = 0.5 * (x0 + x1)
        for (cx0, cx1) in child_spans(x0, x1, depth):
            cxm = 0.5 * (cx0 + cx1)
            ax.plot([xm, cxm], [-depth, -(depth + 1)], color=scaffold, lw=0.4, zorder=1)
            full(cx0, cx1, depth + 1)
    full(0.0, 1.0, 0)

    leaf_slots = []

    def overlay(node, x0, x1, depth):
        xm = 0.5 * (x0 + x1)
        if is_leaf(node):
            leaf_slots.append((xm, node))
            return
        mk, fc, ec, isx = node_face(node) if depth > 0 else ("o", "none", fg, False)
        ec = node_edge if ec == "k" else ec          # theme the black node outline
        ax.scatter([xm], [-depth], marker=mk, s=nsize(node, depth), facecolor=fc,
                   edgecolor=ec, linewidths=2.2 if depth == 0 else
                   (2.0 if (fc == "none" or isx) else 0.8), zorder=7)
        for a in ("WAIT", "MANEUVER"):
            for c in node["children"].get(a, []):
                if depth == 0:
                    # push the two first-action subtrees APART with a gap around the
                    # midline so nodes/outcome dots never spill across into the other side.
                    cx0, cx1 = (x0, 0.46) if a == "WAIT" else (0.54, x1)
                else:
                    cx0, cx1 = (x0, xm) if a == "WAIT" else (xm, x1)
                cxm = 0.5 * (cx0 + cx1)
                ax.plot([xm, cxm], [-depth, -(depth + 1)],
                        color=(CW if a == "WAIT" else CM), lw=ewidth(c),
                        alpha=0.9, solid_capstyle="round", zorder=5)
                overlay(c, cx0, cx1, depth + 1)
    overlay(root, 0.0, 1.0, 0)

    # ---- outcomes at TCA via a downward brace off each leaf ----
    from collections import defaultdict
    groups = defaultdict(list)
    for (x, n) in leaf_slots:
        groups[round(x, 6)].append(n)

    MAXSHOW, dot_dx = 3, 0.022      # max 3 dots + "..." -> narrower clusters, no spillover
    y_leaf = -maxd
    y_out = -maxd - 0.56          # dots sit a bit LOWER -> more gap below the bracket
    y_brace_bot = -maxd - 0.30    # bracket tips end well above the dots
    for xleaf, nodes in sorted(groups.items()):
        fails = [n for n in nodes if (n["pc"] is not None and n["pc"] >= DELTA)]
        feas = [n for n in nodes if n not in fails]
        shown = fails + feas[:max(MAXSHOW - len(fails), 0)]
        overflow = len(nodes) > len(shown)
        k = len(shown) + (1 if overflow else 0)
        offs = (np.arange(k) - (k - 1) / 2) * dot_dx
        # bracket spans SLIGHTLY WIDER than the cluster; for a single marker keep it narrow.
        span_half = abs(offs[0]) if k > 1 else 0.0
        half = span_half + (0.012 if k > 1 else 0.010)
        curly_brace(ax, xleaf, y_leaf - 0.04, half, y_brace_bot,
                    color="0.6" if dark else "0.5", lw=1.0)
        for i, n in enumerate(shown):
            fail = n["pc"] is not None and n["pc"] >= DELTA
            ax.scatter([xleaf + offs[i]], [y_out], marker="X" if fail else "o",
                       s=230 if fail else 130, facecolor=RED if fail else GREEN,
                       edgecolor=node_edge, linewidths=1.1 if fail else 0.6, zorder=7)
        if overflow:
            ax.text(xleaf + offs[-1], y_out, r"$\cdots$", ha="center", va="center",
                    fontsize=15, color=GREEN, zorder=7)

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(y_out - 0.35, 0.5)
    ax.axis("off")

    # left axis: real hours-to-TCA per epoch (bigger font). The last epoch IS TCA, and
    # the rollout outcomes live at that epoch -> a single "TCA" label at the outcome row.
    for dep in range(maxd):
        ax.text(-0.05, -dep, f"{hours[dep]:.0f} h", ha="right", va="center",
                fontsize=12, color=fg)
    ax.text(-0.05, y_out, "TCA", ha="right", va="center", fontsize=12, color=fg)

    # SHORT, snug "selected action" pill on the chosen side (WAIT pill nudged right so it
    # sits at a comparable inset to the MANEUVER pill; action names UPPERCASE).
    sel_col = CW if chosen == "WAIT" else CM
    txt = f"Selected: {chosen.upper()}"
    bx = 0.12 if chosen == "WAIT" else 0.72
    # lowered slightly from 0.93 so the pill sits inside the gray highlight box without
    # poking out the top (0.86 was too low for Grace; 0.895 tucks it just under the edge).
    ax.text(bx, 0.895, txt, transform=ax.transAxes, ha="left", va="center",
            fontsize=12.5, color="white", fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=sel_col, edgecolor="none"))
    q = root["Qa"]
    ax.set_title(f"{d['sensor_quality'].capitalize()} sensor quality"
                 f"      $Q_\\mathrm{{WAIT}}$ = {q.get('WAIT',0):,.0f},   "
                 f"$Q_\\mathrm{{MANEUVER}}$ = {q.get('MANEUVER',0):,.0f}",
                 fontsize=13, color=fg, loc="right")


def load(quality, niter=50, seed=3):
    base = quality if niter == 12 else f"{quality}_n{niter}"
    fn = f"{base}.json" if seed == 1 else f"{base}_s{seed}.json"
    p = os.path.join(DATA, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, base, dark):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{base}.{ext}", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(f"{base}.png", dpi=200, transparent=False,
                facecolor="black" if dark else "white", edgecolor="none")
    print(f"wrote {base}.{{pdf,svg,png}}")


def build(best, worst, case_disp, dark):
    fg = "white" if dark else "black"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6))   # stubbier
    fig.suptitle(f"Chance-constrained first MCTS decision for conjunction {case_disp}\n"
                 r"(feasible if belief $P_c < 10^{-5}$ at TCA)", fontsize=14, y=0.99,
                 color=fg)
    draw(axes[0], best, fg)
    draw(axes[1], worst, fg)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN,
               markeredgecolor=fg, label=r"Feasible ($P_c < 10^{-5}$)", markersize=12),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=AMBER, markeredgewidth=1.6,
               label="Transient warning", markersize=12),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=RED,
               markeredgecolor=RED, label="Terminal fail at TCA", markersize=14),
        Line2D([0], [0], color=CW, lw=3.5, label="WAIT branch"),
        Line2D([0], [0], color=CM, lw=3.5, label="MANEUVER branch"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                     fontsize=11, bbox_to_anchor=(0.5, 0.045))
    for t in leg.get_texts():
        t.set_color(fg)
    fig.text(0.5, 0.012, "Node size and edge width scale with MCTS visit count; deeper "
             "levels are imagined look-ahead rollouts, not committed actions.",
             ha="center", fontsize=11, color="0.7" if dark else "0.35", style="italic")
    fig.tight_layout(rect=(0, 0.08, 1, 0.99))
    save(fig, os.path.join(OUT, f"figC_treezoom_{'dark' if dark else 'light'}"), dark)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    setup_fonts()
    best, worst = load("best"), load("worst")
    if best is None or worst is None:
        print("missing n=50 seed-3 data; run treezoom_probe.jl 50 with TREEZOOM_SEEDS incl 3")
        return
    a_id, b_id = best["case_id"].split("_conj_")   # NORAD ids (strip zero-padding)
    case_disp = f"NORAD {int(a_id)} vs {int(b_id)}"
    for dark in (False, True):
        build(best, worst, case_disp, dark)


if __name__ == "__main__":
    main()
