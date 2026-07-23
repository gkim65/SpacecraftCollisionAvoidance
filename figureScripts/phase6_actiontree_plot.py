#!/usr/bin/env python3
"""Phase 6 ablation as an ACTION tree (Fig 6c, action-collapsed view).

Same data as phase6_tree_plot.py (phase6_tree_data.json) but collapsed to the
ACTION structure instead of drawing every observation-child. Motivation (Grace,
2026-07-22): under the linear-Gaussian belief update the observation moves the
mean but NOT the covariance, so all the observation-children of a given action
have ~the same Pc — the raw obs progressive-widening fan-out (~45 wide) is
near-redundant for the decision and hard to read. Grouping the obs-children by
the ACTION that produced them gives the tree that actually matters: root ->
{WAIT, MANEUVER} -> {WAIT, MANEUVER} -> ...

Each ACTION node aggregates all observation-children reached by that action
sequence: it shows how many obs-samples it holds, the median Pc (color), and
whether ANY of them violated / were terminated. This is where the constraint's
effect is legible: :terminate cuts the WAIT-spine action node that :off /
:penalize keep expanding.

Three modes, one per row, white/black bg, CMU Serif, PDF+SVG+PNG.

Run from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/phase6_actiontree_plot.py
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

MAX_DEPTH = 4     # action depth to show (root=0). Action tree is only 2^depth wide.

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
CMAP = plt.get_cmap("plasma")
PC_FLOOR, PC_CEIL = 1e-12, 1e-1


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


def pc_color(pc):
    if pc is None or not np.isfinite(pc) or pc <= 0:
        return "#888888"
    pcc = min(max(pc, PC_FLOOR), PC_CEIL)
    frac = (np.log10(pcc) - np.log10(PC_FLOOR)) / (np.log10(PC_CEIL) - np.log10(PC_FLOOR))
    return CMAP(frac)


def build_action_tree(nodes):
    """Collapse observation-children into ACTION groups.

    Key each node by its action PATH from the root (tuple of 'W'/'M'). All obs-
    children sharing a path collapse into one action-group. Returns
    path -> aggregate dict.
    """
    by_id = {n["id"]: n for n in nodes}
    # path for each node id
    path = {}
    root_id = next(n["id"] for n in nodes if n["parent"] == -1)
    path[root_id] = ()

    # BFS by depth so parents get paths first
    for n in sorted(nodes, key=lambda x: x["depth"]):
        if n["id"] == root_id:
            continue
        p = n["parent"]
        if p not in path:
            continue
        step = "W" if n["action"] == "WAIT" else "M"
        path[n["id"]] = path[p] + (step,)

    groups = {}
    for n in nodes:
        pth = path.get(n["id"])
        if pth is None or len(pth) > MAX_DEPTH:
            continue
        g = groups.setdefault(pth, {"pcs": [], "n_obs": 0, "any_viol": False,
                                    "any_term": False, "depth": len(pth)})
        g["n_obs"] += 1
        if n["pc"] is not None and np.isfinite(n["pc"]):
            g["pcs"].append(n["pc"])
        g["any_viol"] |= bool(n["violated"])
        g["any_term"] |= bool(n["terminal"])
    for pth, g in groups.items():
        g["med_pc"] = float(np.median(g["pcs"])) if g["pcs"] else None
    return groups


def draw_action_tree(ax, md, fg):
    groups = build_action_tree(md["nodes"])
    # x-position: place each path by its binary address so W=left, M=right,
    # evenly spread within its depth.
    def xpos(pth):
        if not pth:
            return 0.0
        # map W->0, M->1 as a binary fraction in [-1, 1]
        val = 0.0
        for i, s in enumerate(pth):
            bit = 0 if s == "W" else 1
            val += (bit - 0.5) * 2 * (0.5 ** i)
        return val

    pos = {pth: (xpos(pth), -g["depth"]) for pth, g in groups.items()}

    # edges parent->child
    for pth, (x, y) in pos.items():
        if not pth:
            continue
        parent = pth[:-1]
        if parent in pos:
            px, py = pos[parent]
            act = pth[-1]
            ls = "-" if act == "M" else (0, (4, 2))
            ax.plot([px, x], [py, y], ls=ls, color=fg, lw=1.4, alpha=0.6, zorder=1)

    # nodes
    for pth, (x, y) in pos.items():
        g = groups[pth]
        col = pc_color(g["med_pc"]) if pth else "#999999"
        marker = "s" if g["any_term"] else "o"
        ax.scatter([x], [y], s=430, marker=marker, color=col,
                   edgecolors=fg, linewidths=1.0, zorder=3)
        if g["any_viol"]:
            ax.scatter([x], [y], s=150, marker="X", color=fg, zorder=4)
        # obs-count badge under each node
        if pth:
            ax.text(x, y - 0.28, f"{g['n_obs']}", color=fg, fontsize=7.5,
                    ha="center", va="top", alpha=0.8)

    ax.set_ylim(-MAX_DEPTH - 0.7, 0.7)
    ax.set_xlim(-1.25, 1.25)
    # depth / time-remaining down the left
    for dep in range(0, MAX_DEPTH + 1):
        tau = d["depth"] - dep
        ax.text(-1.2, -dep, f"{tau} h", color=fg, fontsize=8, ha="left",
                va="center", alpha=0.8)

    # compact two-line title so the three panels don't collide
    ttl = (f":{md['mode']} $\\rightarrow$ {md['best_action']}\n"
           f"{md['n_violating']} violating, {md['n_terminal_violating']} terminated")
    ax.set_title(ttl, fontsize=10.5, loc="center")


def fig_action_trees(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.2))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.79, bottom=0.20, wspace=0.10)

    for ax, m in zip(axes, MODES):
        draw_action_tree(ax, d[m], fg)
        theme_axes(ax, fg)

    # colorbar along the bottom (above the legend row)
    sm = ScalarMappable(norm=LogNorm(vmin=PC_FLOOR, vmax=PC_CEIL), cmap=CMAP)
    cax = fig.add_axes([0.32, 0.115, 0.38, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Action-node median $P_c$ at TCA", color=fg, fontsize=9.5)
    cb.ax.xaxis.set_tick_params(color=fg)
    cb.outline.set_edgecolor(fg)
    for t in cb.ax.get_xticklabels():
        t.set_color(fg)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#bbbbbb",
               markeredgecolor=fg, markersize=11, label="action node (color=$P_c$)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#bbbbbb",
               markeredgecolor=fg, markersize=11, label="terminal"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=fg,
               markeredgecolor="none", markersize=11, label="violating"),
        Line2D([0], [0], ls="-", color=fg, lw=1.6, label="MANEUVER edge"),
        Line2D([0], [0], ls=(0, (4, 2)), color=fg, lw=1.3, label="WAIT edge"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, framealpha=0.0,
               bbox_to_anchor=(0.5, 0.005), labelcolor=fg, fontsize=9)

    fig.suptitle(
        "Phase 6 chance-constraint ablation — action tree "
        "(observation-children grouped by action)\n"
        f"cross-track miss {d['miss_m']:.0f} m, $\\Delta v$ {d['dv_ms']:.0f} m/s, "
        f"threshold {d['pc_threshold']:.0e}; number under each node = obs-samples grouped",
        color=fg, fontsize=11.5, y=0.975)
    save(fig, os.path.join(OUT, f"phase6_actiontree_{theme}"))


os.makedirs(OUT, exist_ok=True)
for th in ("light", "dark"):
    fig_action_trees(th)
