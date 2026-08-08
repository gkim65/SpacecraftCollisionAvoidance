#!/usr/bin/env python3
"""cara_2d3d_plot.py — mechanism figure for the exact-2D-vs-3D Pc gap.

Fig 2D3D-b: Mahalanobis miss-in-sigma (x) vs log10(elrod_pc / NASA Nc3D) (y).
Miss-in-sigma is the single strongest regressor of the gap (Spearman -0.80): the
deeper in the Gaussian tail the miss sits, the more our 2D underestimates vs the
3D reference. Color = secondary object class, marker = NASA 2D-Pc violation flag.

Reads figureScripts/data/cara_2d3d_joined.json (written by cara_2d3d_analysis.py).
Degenerate deep-tail cases (Pc < 1e-12) are dropped — their log-ratios are
numerically meaningless.

White-bg (paper) + black-bg (slides), CMU Serif, PDF + SVG + PNG. Per CLAUDE.md.

Run:  uv run --with matplotlib --with numpy figureScripts/cara_2d3d_plot.py
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

with open(os.path.join(HERE, "data", "cara_2d3d_joined.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10})

# Class colors (colorblind-safe-ish, distinct on both backgrounds).
CLS_COLOR = {"debris": "#e6550d", "rocket_body": "#756bb1",
             "payload": "#2b8cbe", "unknown": "#31a354"}
CLS_LABEL = {"debris": "Debris", "rocket_body": "Rocket body",
             "payload": "Payload", "unknown": "Unknown"}

usable = set(D["usable_ids"])
rows = [r for r in D["rows"] if r["id"] in usable
        and np.isfinite(r["miss_maha"]) and np.isfinite(r["log_ratio"])]


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

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.13)

    # Reference line: perfect agreement (2D == 3D).
    ax.axhline(0.0, color=fg, ls="-", lw=1.0, alpha=0.55)
    # 2x-too-low guide (the operationally-relevant "unsafe" line).
    ax.axhline(np.log10(0.5), color=fg, ls=":", lw=1.0, alpha=0.5)

    # Plot: filled circle = violation, open circle = no violation.
    for r in rows:
        c = CLS_COLOR.get(r["class"], "#999999")
        viol = r["violation"] == 1
        ax.scatter(r["miss_maha"], r["log_ratio"],
                   s=70, marker="o",
                   facecolor=c if viol else "none",
                   edgecolor=c, linewidths=1.6, alpha=0.9, zorder=4)

    # Annotate the agreement / unsafe regions with floating labels. Kept in the
    # empty upper-right so nothing overlaps the data or the lower-left legend.
    ax.text(0.97, 0.955, r"$0=$ exact 2D$\,=\,$3D;  more negative $=$ our 2D too"
            r" low (unsafe)", transform=ax.transAxes, color=fg, fontsize=9.5,
            ha="right", va="top", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.35", fc="black" if dark else "white",
                      ec=fg, lw=0.5, alpha=0.7))

    ax.set_ylim(top=2.7)   # headroom so the top annotation clears the data
    ax.set_xlabel(r"Miss distance at TCA (Mahalanobis $\sigma$, encounter plane)")
    ax.set_ylabel(r"$\log_{10}(\mathrm{elrod\ Pc}\ /\ \mathrm{NASA\ Nc3D})$")
    ax.set_title("Exact-2D vs 3D Pc gap grows with miss distance into the tail")

    # Legend: class colors + fill = violation. Placed in a clear corner.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="none", mfc=CLS_COLOR[k],
                      mec=CLS_COLOR[k], ms=8, label=CLS_LABEL[k])
               for k in ("debris", "rocket_body", "payload", "unknown")]
    handles += [
        Line2D([0], [0], marker="o", ls="none", mfc="none", mec=fg, ms=8,
               mew=1.6, label="2D valid (no violation)"),
        Line2D([0], [0], marker="o", ls="none", mfc=fg, mec=fg, ms=8,
               label="NASA 2D-Pc violation"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", framealpha=0.0,
                    handletextpad=0.4, borderpad=0.6)
    for t in leg.get_texts():
        t.set_color(fg)

    theme_axes(ax, fg, grid)
    save(fig, f"cara_2d3d_missigma_{theme}")


for th in ("light", "dark"):
    make(th)
print(f"\n{len(rows)} usable cases plotted (of {D['meta']['n_total']}).")
