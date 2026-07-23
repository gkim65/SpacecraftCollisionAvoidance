#!/usr/bin/env python3
"""Phase 6 constraint-mode ablation figure (Fig 6c).

Renders phase6_ablation_data.json (written by phase6_ablation_data.jl; Julia is
the source of truth). One figure, white-bg (paper) + black-bg (slides), CMU
Serif, as PDF + SVG + PNG:

  phase6_ablation_{light,dark}   Two panels:
    (i)  per-node Pc vs. time-remaining, violating nodes highlighted, across the
         three constraint modes (:off / :penalize / :terminate) — shows WHICH
         branches the constraint touches and which :terminate amputates.
    (ii) the decision: chosen action + root Q-margin Q(WAIT) vs Q(MANEUVER), and
         the prune counts (violating / terminal-violating nodes) per mode — the
         quantitative "how much does the constraint change the decision / prune."

Run from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/phase6_ablation_plot.py
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "figures"))
with open(os.path.join(HERE, "phase6_ablation_data.json")) as f:
    d = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10})

MODES = ("off", "penalize", "terminate")
MODE_LBL = {"off": ":off (no constraint)",
            "penalize": ":penalize (default)",
            "terminate": ":terminate (amputate)"}
MODE_COL = {"off": "#636363", "penalize": "#2b8cbe", "terminate": "#d62728"}


def theme_axes(axes, fg, grid):
    for ax in np.atleast_1d(axes).ravel():
        ax.grid(True, which="both", color=grid, lw=0.4, alpha=0.6)
        for s in ax.spines.values():
            s.set_color(fg)
        ax.tick_params(colors=fg, which="both")
        ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(fg)


def save(fig, base):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", transparent=True, dpi=200,
                    facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def fig_ablation(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    box = dict(boxstyle="round,pad=0.4", fc="black" if dark else "white",
               ec=fg, lw=0.6, alpha=0.78)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.14, wspace=0.26)

    # ---- (i) all node Pc vs. time-remaining; violating nodes (Pc>thresh) = X ----
    # These are the WAIT-descendant nodes whose Pc-at-TCA exceeds the threshold —
    # the branches the constraint acts on. (MANEUVER branches clear the threshold
    # by moving the mean, so they don't appear as violations.)
    ax = axes[0]
    for m in MODES:
        md = d[m]
        tau = np.array(md["node_tau_hr"]); pc = np.array(md["node_pc"])
        viol = np.array(md["node_violated"], dtype=bool)
        ok = np.isfinite(pc) & (pc > 0) & (~viol)
        jit = {"off": -0.06, "penalize": 0.0, "terminate": 0.06}[m]
        # non-violating nodes as faint dots (context)
        ax.scatter(tau[ok] + jit, pc[ok], s=10, color=MODE_COL[m],
                   alpha=0.20, edgecolors="none")
        # violating nodes as bold X's (the ones the constraint touches)
        ax.scatter(tau[viol] + jit, pc[viol], s=70, color=MODE_COL[m],
                   marker="X", edgecolors=fg, linewidths=0.6, zorder=4,
                   label=f"{MODE_LBL[m]}: {int(viol.sum())} violating")
    ax.axhline(d["pc_threshold"], color=fg, ls="--", lw=1.1, alpha=0.8)
    ax.text(0.98, d["pc_threshold"] * 1.25,
            f"$P_c$ threshold = {d['pc_threshold']:.0e}",
            transform=ax.get_yaxis_transform(), color=fg, fontsize=9,
            va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("Time remaining to TCA (hr)")
    ax.set_ylabel("Node $P_c$ at TCA")
    ax.set_title("(6c-i) Violating (WAIT) nodes; $:$terminate keeps fewer")
    ax.invert_xaxis()
    ax.legend(loc="lower left", framealpha=0.0, fontsize=9, title="X $= P_c > $ threshold")
    ax.get_legend().get_title().set_color(fg)
    ax.get_legend().get_title().set_fontsize(8.5)

    # ---- (ii) the decision: Q(WAIT) (mode-dependent) vs Q(MANEUVER) (~const) ----
    # Q(WAIT) is hugely negative (it carries the high WAIT Pc) and DIFFERS by mode;
    # Q(MANEUVER) is ~-10 (fuel only — the burn clears the risk) under every mode.
    # The scale gap is the point, so annotate MANEUVER as a reference near zero
    # rather than a bar that vanishes.
    ax = axes[1]
    x = np.arange(len(MODES))
    q_wait = [d[m]["q_wait"] for m in MODES]
    q_mvr  = [d[m]["q_maneuver"] for m in MODES]
    w = 0.5
    bars = ax.bar(x, q_wait, w, color="#969696", label="Q(WAIT)")
    # Q(MANEUVER) reference line near zero (all modes ~ -10)
    qm = np.mean(q_mvr)
    ax.axhline(qm, color="#31a354", lw=2.0,
               label=f"Q(MANEUVER) $\\approx$ {qm:.0f} (chosen, all modes)")
    ax.set_xticks(x)
    ax.set_xticklabels([f":{m}" for m in MODES])
    ax.set_xlabel("Constraint mode")
    ax.set_ylabel("Root per-action value $Q$")
    ax.set_title("(6c-ii) Constraint changes Q(WAIT) \\& pruning, not the choice")
    ax.legend(loc="lower center", framealpha=0.0, fontsize=9)

    ymin = min(q_wait); span = abs(ymin) or 1.0
    # value labels on each WAIT bar + prune counts
    for i, m in enumerate(MODES):
        md = d[m]
        ax.text(i, q_wait[i] - 0.02 * span, f"{q_wait[i]:,.0f}",
                color=fg, fontsize=9, ha="center", va="top")
        ax.text(i, 0.06 * span,
                f"viol {md['n_violating']}\nterm {md['n_terminal_violating']}\n"
                f"nodes {md['n_nodes']}",
                color=fg, fontsize=8.5, ha="center", va="bottom", bbox=box)
    # a note that all three still choose MANEUVER
    ax.text(0.5, 0.98,
            "All three modes still choose MANEUVER (it clears the risk).\n"
            "The constraint sharpens how strongly WAIT is discouraged\n"
            "(:penalize $<$ :off) and prunes its subtree (:terminate).",
            transform=ax.transAxes, fontsize=8.6, color=fg, ha="center", va="top",
            bbox=box)
    ax.set_ylim(ymin - 0.16 * span, 0.55 * span)

    theme_axes(axes, fg, grid)
    fig.suptitle(
        f"Phase 6 — chance-constraint ablation (cross-track, miss {d['miss_m']:.0f} m, "
        f"$\\Delta v$ {d['dv_ms']:.0f} m/s, {d['n_iterations']} sims/mode)",
        color=fg, fontsize=13.5, y=0.965)
    save(fig, os.path.join(OUT, f"phase6_ablation_{theme}"))


os.makedirs(OUT, exist_ok=True)
for th in ("light", "dark"):
    fig_ablation(th)
