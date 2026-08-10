#!/usr/bin/env python3
"""FIG B — operator-facing residual risk vs deferral, split by tracking fidelity.

Three panels (best / median / worst sensor quality), one per column so the reader
watches the tradeoff shift as tracking degrades. In each panel:
  x = deferral rate = % episodes resolved WITHOUT a maneuver (resolved_without_maneuver)
  y = residual-risk rate = % episodes whose OPERATOR-FACING risk estimate at TCA
      exceeds the chance-constraint delta (pc_at_tca > delta, delta = 1e-5).

WHY belief-Pc (not true geometry) here, and why this is fair:
  pc_at_tca is the risk the OPERATOR actually ends up believing, computed from the
  information available to them (the belief), NOT the true geometry. Every policy is
  judged on the SAME operator-facing quantity: "at TCA, did the operator still think
  the encounter was unsafe?" The fixed delay-gates read Pc off the same drifting
  belief and, by acting on a clock rather than on that belief, leave real residual
  risk on the table (violation climbs as they wait longer). MCTS acts on the belief,
  so it drives operator-facing residual risk to ~0. This is a decision-relevant
  comparison on a shared, operator-available metric -- not MCTS graded on its own rule.
  (True-geometry safety is reported separately; see analyze_results_simple.py.)

THE COMPARATORS (labels chosen to be precise, not flattering):
  - Delay gates (28/12/6/3 h): fixed-clock single-burn baselines; connected into a
    frontier per quality.
  - Belief planner (MCTS): acts on the running belief.
  - Perfect-obs offline plan (`wait_feasibility` in code): a wait schedule precomputed
    assuming PERFECT, no-drift (zero-innovation) observations -- it defers while the
    CLEAN spine stays < delta, else burns now. Because it is tuned to a noise-free
    world, its residual risk in the REAL noisy world is non-monotone in tracking
    quality (worse at median than at worst) -- an informative mismatch, not an oracle.

Data: figureScripts/data/sweep_all.csv (finished rows), per (policy_variant,
sensor_quality). Everything is computed FROM the CSV; the per-quality table prints to
console. delta is a documented constant (pc_threshold column is empty in the export);
a per-row fallback reads the column if a later export populates it.

Render from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/figB_safety_contrast.py
"""
import csv
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "sweep_all.csv")
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))

DELTA = 1e-5                                    # chance-constraint threshold
QUALITIES = ["best", "median", "worst"]         # one PANEL per quality (degrading L->R)
QTITLE = {"best": "Best tracking", "median": "Median tracking", "worst": "Worst tracking"}
GATES = ["delay_28h", "delay_12h", "delay_6h", "delay_3h"]   # frontier, ordered by wait
GLABEL = {"delay_28h": "28 h", "delay_12h": "12 h", "delay_6h": "6 h", "delay_3h": "3 h"}
MCTS = "mcts"
PLAN = "wait_feasibility"                        # perfect-obs offline plan

GATE_C = "#3182bd"                               # delay-gate frontier (blue)
MCTS_C = "#e6550d"                               # belief planner (orange)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    with open(CSV) as f:
        return [r for r in csv.DictReader(f) if r["state"] == "finished"]


def cell(rows, policy, quality):
    sub = [r for r in rows if r["policy_variant"] == policy
           and r["sensor_quality"] == quality]
    if not sub:
        return None
    n = len(sub)
    defer = 100 * sum(1 for r in sub
                      if r["resolved_without_maneuver"].lower() in ("true", "1")) / n
    # per-row delta: use the column if a future export populates it, else the constant
    viol = 0
    seen = 0
    for r in sub:
        pc = num(r.get("pc_at_tca"))
        if pc is None:
            continue
        seen += 1
        thr = num(r.get("pc_threshold")) or DELTA
        if pc > thr:
            viol += 1
    vrate = 100 * viol / seen if seen else float("nan")
    return dict(n=n, defer=defer, vrate=vrate)


def setup_fonts():
    if shutil.which("latex"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["CMU Serif", "Computer Modern Roman"]})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["CMU Serif", "DejaVu Serif"],
                             "mathtext.fontset": "cm"})
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10})


def save(fig, base, dark):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{base}.{ext}", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(f"{base}.png", transparent=False, dpi=200,
                facecolor="black" if dark else "white", edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def make_fig(S, dark):
    fg = "white" if dark else "black"
    plan_c = "#7a7a7a" if not dark else "#bdbdbd"
    tex = plt.rcParams.get("text.usetex")
    pct = r"\%" if tex else "%"

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.155, wspace=0.10)

    # global y ceiling with headroom
    vmax = max(s["vrate"] for s in S.values() if np.isfinite(s["vrate"]))
    ytop = 2 * np.ceil((vmax + 1) / 2)

    for k, q in enumerate(QUALITIES):
        ax = axes[k]
        ax.patch.set_alpha(0.0)

        gx = [S[(g, q)]["defer"] for g in GATES if S.get((g, q))]
        gy = [S[(g, q)]["vrate"] for g in GATES if S.get((g, q))]
        ax.plot(gx, gy, "-", color=GATE_C, lw=2.0, zorder=3, solid_capstyle="round")
        ax.scatter(gx, gy, s=70, facecolors=GATE_C, edgecolors=fg, linewidths=1.0, zorder=4)
        for g in GATES:
            s = S.get((g, q))
            if s:
                ax.annotate(GLABEL[g], (s["defer"], s["vrate"]),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom", color=fg, fontsize=9, alpha=0.9)

        sm = S.get((MCTS, q))
        if sm:
            ax.scatter(sm["defer"], sm["vrate"], marker="D", s=170, facecolors=MCTS_C,
                       edgecolors=fg, linewidths=1.4, zorder=6,
                       label="Belief planner (MCTS)")
        sp = S.get((PLAN, q))
        if sp:
            ax.scatter(sp["defer"], sp["vrate"], marker="s", s=150, facecolors="none",
                       edgecolors=plan_c, linewidths=2.0, zorder=5,
                       label="Perfect-obs offline plan")

        ax.set_title(QTITLE[q], color=fg, pad=6)
        ax.set_xlabel(f"Deferral rate ({pct})")
        if k == 0:
            ax.set_ylabel(f"Residual-risk rate ({pct} of episodes,\n"
                          f"operator Pc at TCA $>$ threshold)")
        ax.set_xlim(0, 82)
        ax.set_ylim(-0.6, ytop)
        ax.grid(True, color=fg, lw=0.4, alpha=0.18)
        ax.tick_params(colors=fg)
        for sp_ in ax.spines.values():
            sp_.set_color(fg)
        ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)

    # shared legend for the two non-gate markers + a gate proxy
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=GATE_C, markerfacecolor=GATE_C,
               markeredgecolor=fg, lw=2.0, markersize=8, label="Delay gates (fixed clock)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=MCTS_C,
               markeredgecolor=fg, markersize=9, label="Belief planner (MCTS)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor=plan_c, markeredgewidth=2.0, markersize=9,
               label="Perfect-obs offline plan"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
                     framealpha=0.85, edgecolor=fg, bbox_to_anchor=(0.5, 0.985))
    for t in leg.get_texts():
        t.set_color(fg)
    leg.get_frame().set_facecolor("black" if dark else "white")

    fig.suptitle("Operator-facing residual risk: fixed clocks leave it on the table, "
                 "the belief planner drives it to zero",
                 color=fg, fontsize=12.5, y=0.905)

    save(fig, os.path.join(OUT, f"figB_safety_contrast_{'dark' if dark else 'light'}"), dark)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    S = {}
    print(f"\n{'quality':8}{'policy':18}{'defer%':>9}{'resid-risk%':>13}{'n':>6}")
    for q in QUALITIES:
        for p in GATES + [MCTS, PLAN]:
            s = cell(rows, p, q)
            if s:
                S[(p, q)] = s
                print(f"{q:8}{p:18}{s['defer']:>8.1f}%{s['vrate']:>12.1f}%{s['n']:>6}")
        print()
    setup_fonts()
    for dark in (False, True):
        make_fig(S, dark)


if __name__ == "__main__":
    main()
