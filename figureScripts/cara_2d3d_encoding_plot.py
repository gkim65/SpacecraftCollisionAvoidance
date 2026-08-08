#!/usr/bin/env python3
"""cara_2d3d_encoding_plot.py — miss-in-sigma vs the 2D/3D gap, colored by anisotropy.

Companion to cara_2d3d_plot.py. Same axes (Mahalanobis miss-in-sigma vs
log10(elrod_pc / NASA Nc3D)) but every point is colored by encounter-plane
anisotropy sigma2/sigma1 (log). Shows the mechanism in two moves:

  - X-axis (miss-sigma) is the GATE: below ~2 sigma every case sits on the
    exact 2D = 3D line. Spearman(gap, miss-sigma) = -0.80 (dominant).
  - Color (anisotropy) is the AMPLIFIER: once deep in the tail, the highly
    elongated (sliver) covariances are the ones that underestimate most.
    Spearman(gap, log-anisotropy) = -0.48 (the real 2nd driver; velocity, by
    contrast, is only -0.14 and does NOT organize the gap).

Reads figureScripts/data/cara_2d3d_joined.json. Degenerate deep-tail cases
(Pc < 1e-12) dropped. White-bg + black-bg, CMU Serif, PDF + SVG + PNG.

Run:  uv run --with matplotlib --with numpy figureScripts/cara_2d3d_encoding_plot.py
"""
import json
import math
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

usable = set(D["usable_ids"])
rows = [r for r in D["rows"] if r["id"] in usable
        and np.isfinite(r["miss_maha"]) and np.isfinite(r["log_ratio"])]

x = np.array([r["miss_maha"] for r in rows])
y = np.array([r["log_ratio"] for r in rows])
aniso = np.array([r["aniso"] for r in rows])


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


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def make(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.13)

    ax.axhline(0.0, color=fg, ls="-", lw=1.0, alpha=0.55)
    ax.axhline(np.log10(0.5), color=fg, ls=":", lw=1.0, alpha=0.5)

    sc = ax.scatter(x, y, c=np.log10(aniso), cmap="plasma",
                    s=95, marker="o", edgecolor=fg, linewidths=0.6, zorder=4)

    ax.set_ylim(top=2.7)
    ax.text(0.97, 0.955,
            r"$0=$ exact 2D$\,=\,$3D;  more negative $=$ our 2D too low (unsafe)",
            transform=ax.transAxes, color=fg, fontsize=9.5, ha="right", va="top",
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.35", fc="black" if dark else "white",
                      ec=fg, lw=0.5, alpha=0.7))

    ax.set_xlabel(r"Miss distance at TCA (Mahalanobis $\sigma$, encounter plane)")
    ax.set_ylabel(r"$\log_{10}(\mathrm{elrod\ Pc}\ /\ \mathrm{NASA\ Nc3D})$")
    sp_a = spearman(np.log10(aniso), y)
    sp_m = spearman(x, y)
    ax.set_title(rf"Tail depth gates the gap (Spearman ${sp_m:+.2f}$), "
                 rf"anisotropy amplifies it (${sp_a:+.2f}$)")

    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"$\log_{10}$ encounter-plane anisotropy $\sigma_2/\sigma_1$",
                 color=fg)
    cb.ax.yaxis.set_tick_params(color=fg)
    cb.outline.set_edgecolor(fg)
    for t in cb.ax.get_yticklabels():
        t.set_color(fg)

    theme_axes(ax, fg, grid)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"cara_2d3d_encoding_{theme}.{ext}"),
                    transparent=True, dpi=200, facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"wrote cara_2d3d_encoding_{theme}.{{pdf,svg,png}}")


for th in ("light", "dark"):
    make(th)
print(f"\n{len(rows)} usable cases plotted.")
print(f"Spearman(gap, miss-sigma)= {spearman(x, y):+.3f}   (dominant gate)")
print(f"Spearman(gap, log-aniso) = {spearman(np.log10(aniso), y):+.3f}   (amplifier)")
