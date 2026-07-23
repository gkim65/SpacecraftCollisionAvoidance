#!/usr/bin/env python3
"""Phase 6 constraint-mode ablation as ACTUAL MCTS trees (Fig 6c redesign).

Renders phase6_tree_data.json (written by phase6_tree_data.jl; Julia is the
source of truth). Draws the belief-MCTS tree under each constraint mode
(:off / :penalize / :terminate) as a node-link diagram so the constraint's
effect on the SEARCH is visible: nodes are colored by Pc-at-TCA, violating
nodes (Pc > threshold) get a bold X, terminal nodes a square outline — so
:terminate is seen amputating the violating WAIT subtree that :off / :penalize
keep expanding.

The Julia data goes to depth 10; the figure subsets to MAX_DRAW_DEPTH for
legibility (change it and re-run the plotter only — no Julia re-run needed).

Three trees, laid out one per row (separate placement, Grace's call), white-bg
(paper) + black-bg (slides), CMU Serif, PDF + SVG + PNG.

Run from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/phase6_tree_plot.py
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "figures"))
with open(os.path.join(HERE, "phase6_tree_data.json")) as f:
    d = json.load(f)

MAX_DRAW_DEPTH = 3     # subset the depth-10 data to this for a legible figure
                       # (depth 4 fans out to ~45 leaves — too wide to draw cleanly)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "legend.fontsize": 9.5})

MODES = ("off", "penalize", "terminate")
MODE_LBL = {"off": ":off (no constraint)",
            "penalize": ":penalize (default)",
            "terminate": ":terminate (amputate)"}

# Pc colormap: low Pc (safe) -> cool, high Pc (risky) -> warm.
CMAP = plt.get_cmap("plasma")
PC_FLOOR, PC_CEIL = 1e-12, 1e-1     # clamp Pc for coloring (Pc spans many orders)


def theme_axes(ax, fg):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=fg, length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.title.set_color(fg)


def save(fig, base):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", transparent=True, dpi=200,
                    facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def subset(nodes, max_depth):
    """Keep nodes up to max_depth; return id->node and children adjacency."""
    keep = {n["id"]: n for n in nodes if n["depth"] <= max_depth}
    children = {i: [] for i in keep}
    for n in nodes:
        if n["id"] in keep and n["parent"] in keep:
            children[n["parent"]].append(n["id"])
    root = next(n["id"] for n in nodes if n["parent"] == -1)
    return keep, children, root


def layout(keep, children, root):
    """Simple layered layout: y = -depth, x assigned by a left-to-right DFS over
    leaves so siblings don't overlap. Returns id->(x, y)."""
    pos = {}
    leaf_x = [0.0]

    def place(nid):
        kids = children[nid]
        depth = keep[nid]["depth"]
        if not kids:
            x = leaf_x[0]; leaf_x[0] += 1.0
        else:
            xs = [place(k) for k in kids]
            x = sum(xs) / len(xs)
        pos[nid] = (x, -depth)
        return x

    place(root)
    return pos


def pc_color(pc):
    if pc is None or (isinstance(pc, float) and not np.isfinite(pc)):
        return "#888888"
    pcc = min(max(pc, PC_FLOOR), PC_CEIL)
    frac = (np.log10(pcc) - np.log10(PC_FLOOR)) / (np.log10(PC_CEIL) - np.log10(PC_FLOOR))
    return CMAP(frac)


def draw_tree(ax, md, fg, threshold):
    keep, children, root = subset(md["nodes"], MAX_DRAW_DEPTH)
    pos = layout(keep, children, root)

    # edges, styled by the action on the edge into the child
    for nid, (x, y) in pos.items():
        p = keep[nid]["parent"]
        if p in pos:
            px, py = pos[p]
            act = keep[nid]["action"]
            ls = "-" if act == "MANEUVER" else (0, (4, 2))   # MANEUVER solid, WAIT dashed
            lw = 1.3 if act == "MANEUVER" else 1.0
            ax.plot([px, x], [py, y], ls=ls, color=fg, lw=lw, alpha=0.55, zorder=1)

    # nodes
    for nid, (x, y) in pos.items():
        n = keep[nid]
        col = pc_color(n["pc"])
        # terminal nodes: square; else circle
        marker = "s" if n["terminal"] else "o"
        ax.scatter([x], [y], s=190, marker=marker, color=col,
                   edgecolors=fg, linewidths=0.7, zorder=3)
        # violating nodes: overlay a bold X
        if n["violated"]:
            ax.scatter([x], [y], s=95, marker="X", color=fg, linewidths=0.0, zorder=4)

    ax.set_ylim(-MAX_DRAW_DEPTH - 0.6, 0.6)
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)

    # depth / time-remaining axis labels down the left
    for dep in range(0, MAX_DRAW_DEPTH + 1):
        tau = d["depth"] - dep      # root is at t=DEPTH hr remaining, each step -1 hr
        ax.text(min(xs) - 0.5, -dep, f"{tau} h", color=fg, fontsize=8,
                ha="right", va="center", alpha=0.8)

    ttl = (f"{MODE_LBL[md['mode']]}  $\\rightarrow$ {md['best_action']}   "
           f"[{md['n_violating']} violating, {md['n_terminal_violating']} of them "
           f"terminated; {md['n_nodes']} nodes total to depth {d['depth']}]")
    ax.set_title(ttl, fontsize=11, loc="left")


def fig_trees(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.2))
    fig.subplots_adjust(left=0.06, right=0.86, top=0.93, bottom=0.055, hspace=0.22)

    for ax, m in zip(axes, MODES):
        draw_tree(ax, d[m], fg, d["pc_threshold"])
        theme_axes(ax, fg)

    # shared Pc colorbar
    sm = ScalarMappable(norm=LogNorm(vmin=PC_FLOOR, vmax=PC_CEIL), cmap=CMAP)
    cax = fig.add_axes([0.88, 0.30, 0.02, 0.4])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Node $P_c$ at TCA", color=fg, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=fg)
    cb.outline.set_edgecolor(fg)
    for t in cb.ax.get_yticklabels():
        t.set_color(fg)

    # legend (markers/edges) placed clear of the trees, upper-right margin
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#bbbbbb",
               markeredgecolor=fg, markersize=10, label="node (color = $P_c$)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#bbbbbb",
               markeredgecolor=fg, markersize=10, label="terminal node"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=fg,
               markeredgecolor="none", markersize=10, label="violating ($P_c>$ thresh)"),
        Line2D([0], [0], ls="-", color=fg, lw=1.5, label="MANEUVER edge"),
        Line2D([0], [0], ls=(0, (4, 2)), color=fg, lw=1.2, label="WAIT edge"),
    ]
    fig.legend(handles=handles, loc="upper right", framealpha=0.0,
               bbox_to_anchor=(0.99, 0.95), labelcolor=fg, fontsize=9)

    fig.suptitle(
        f"Phase 6 chance-constraint ablation — MCTS trees "
        f"(cross-track miss {d['miss_m']:.0f} m, $\\Delta v$ {d['dv_ms']:.0f} m/s, "
        f"threshold {d['pc_threshold']:.0e}, drawn to depth {MAX_DRAW_DEPTH})",
        color=fg, fontsize=12.5, y=0.975)
    save(fig, os.path.join(OUT, f"phase6_tree_{theme}"))


os.makedirs(OUT, exist_ok=True)
for th in ("light", "dark"):
    fig_trees(th)
