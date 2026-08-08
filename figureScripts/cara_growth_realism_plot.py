#!/usr/bin/env python3
"""cara_growth_realism_plot.py — Fig F3-a (HEADLINE): our Phi Sigma Phi^T growth
vs the real primary-object growth curve.

X = lead time (days, log). Y = RTN 1-sigma (m, log). The REAL primary growth
curve (pooled well-tracked NASA-asset primaries, 4 lead bins) is plotted as
markers for R / in-track / cross-track. OUR Phi Sigma Phi^T growth (swept offline
via build_covariance_table = the mechanism under test) is plotted as lines for
two seeds: anchored to the real <1 d shape, and isotropic 10 m.

The story: real in-track grows ~150x across the window (power law ~tau^2.0) while
our model grows only ~11x (~tau^1.0, velocity-dominated) EVEN when seeded from the
correct real <1 d shape. Radial and cross-track are nearly flat in both. So the
mismatch is a growth-LAW error, not a seed error (both seeds give the same slope).

Reads figureScripts/data/cara_growth_realism.json (cara_growth_realism_data.jl).
White-bg (paper) + black-bg (slides), CMU Serif, PDF + SVG + PNG. Per CLAUDE.md.

Run:  uv run --with matplotlib --with numpy figureScripts/cara_growth_realism_plot.py
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

with open(os.path.join(HERE, "data", "cara_growth_realism.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 9.5})

# One color per RTN axis (distinct on both backgrounds).
AX_COLOR = {"R": "#2b8cbe", "T": "#e6550d", "N": "#31a354"}
AX_LABEL = {"R": "Radial", "T": "In-track", "N": "Cross-track"}

real = D["real_bins"]
gl = D["growth_law"]
gf = D["intrack_growth_factor"]


def theme_axes(ax, fg, grid):
    ax.grid(True, which="both", color=grid, lw=0.4, alpha=0.6)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, which="both")
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(fg)


def save(fig, base):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"{base}.{ext}"), transparent=True,
                    dpi=200, facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


def make(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.16)

    tau_a = np.array(D["curve_anchored"]["tau_days"])
    tau_i = np.array(D["curve_isotropic"]["tau_days"])
    ma = tau_a >= 0.3
    mi = tau_i >= 0.3

    # Radial (R) and cross-track (N) carry the real once-per-orbit (~93 min)
    # breathing ripple (the Phase-3 oscillation). We plot it RAW — the honest
    # curve, no smoothing — so the real STM behavior is visible; the ripple is
    # dense but that is what the model actually does. In-track (T) is monotone.
    for ax_key in ("R", "T", "N"):
        c = AX_COLOR[ax_key]
        ya = np.array(D["curve_anchored"][ax_key])
        yi = np.array(D["curve_isotropic"][ax_key])
        # OUR growth — anchored seed (solid), isotropic seed (dashed)
        ax.plot(tau_a[ma], ya[ma], color=c, ls="-", lw=2.0, alpha=0.95, zorder=3)
        ax.plot(tau_i[mi], yi[mi], color=c, ls="--", lw=1.4, alpha=0.7, zorder=2)
        # REAL primary curve — markers + thin connecting line
        ax.plot(real["lead_d"], real[ax_key], color=c, ls=":", lw=1.0,
                marker="o", ms=9, mfc=c, mec=fg, mew=0.8, alpha=0.95, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lead time before TCA (days)")
    ax.set_ylabel(r"Position 1$\sigma$ in RTN (m)")
    ax.set_title("Our in-track growth over-predicts early, then real "
                 r"accelerates ($\sim\!\tau^{3}$) past us by $>\!4$ d")

    # Growth-law annotation box (upper-left, clear of the data which trends up-right).
    # Both curves anchored at the real <1 d point (0.5 d, 38 m in-track), so the
    # slope exponent is the honest same-anchor comparison.
    txt = (r"In-track slope (both anchored at $0.5$\,d):" "\n"
           rf"  real  $\sigma\propto\tau^{{{gl['p_real_T']:.1f}}}$  (accelerating)" "\n"
           rf"  ours  $\sigma\propto\tau^{{{gl['p_anchored_T']:.1f}}}$  (over early, flattens)")
    ax.text(0.025, 0.975, txt, transform=ax.transAxes, color=fg, fontsize=10,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="black" if dark else "white",
                      ec=fg, lw=0.6, alpha=0.75))

    # Legend: axis colors + line-style meaning. Lower-right clear corner.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=AX_COLOR[k], lw=2.5, label=AX_LABEL[k])
               for k in ("R", "T", "N")]
    handles += [
        Line2D([0], [0], color=fg, ls=":", marker="o", mfc=fg, mec=fg, ms=8,
               lw=1.0, label="Real primary (CDM)"),
        Line2D([0], [0], color=fg, ls="-", lw=2.0, label=r"Ours, anchored seed"),
        Line2D([0], [0], color=fg, ls="--", lw=1.4, label=r"Ours, isotropic seed"),
    ]
    # Legend in the clear mid band between the in-track lines (top) and the R/N
    # ripple (bottom) so it never overlaps data. A faint background keeps it
    # readable over the grid.
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.015, 0.42),
                    framealpha=0.75, facecolor="black" if dark else "white",
                    edgecolor=fg, handletextpad=0.6, borderpad=0.6, ncol=1)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_color(fg)

    ax.text(0.5, -0.185, r"Radial \& cross-track carry the real once-per-orbit "
            r"($\sim$93 min) breathing ripple (plotted raw)",
            transform=ax.transAxes, color=fg, fontsize=8, ha="center",
            va="top", alpha=0.7)

    theme_axes(ax, fg, grid)
    save(fig, f"cara_growth_realism_{theme}")


for th in ("light", "dark"):
    make(th)
print(f"\nreal in-track {real['T'][0]:.0f} m (<1 d) -> {real['T'][-1]:.0f} m (>4 d), "
      f"{gf['real']:.0f}x;  ours anchored {gf['anchored']:.0f}x, iso {gf['isotropic']:.0f}x")
