#!/usr/bin/env python3
"""cara_missigma_transfer_plot.py — Fig F3-b: does 2D-validity transfer to us?

Prompt 2 proved Mahalanobis miss-in-sigma is THE gate on 2D-vs-3D validity
(below ~2 sigma exact 2D ~ 3D). That result used the REAL CDM covariance. This
figure asks whether it transfers to OUR planner: per real case, real miss-in-sigma
(real Sigma) vs miss-in-sigma recomputed with OUR grown Sigma substituted (real
geometry kept), at each case's real lead time.

X = lead time (h). Y = miss-in-sigma (encounter plane). Real = filled markers,
ours (anchored seed) = open markers, connected per case by a thin line so the
shift is visible. 2-sigma gate line drawn. The ~24 h decision band is shaded.

Story: at ~24 h our miss-in-sigma (~0.1 sigma) is an order of magnitude SHALLOWER
than real (~1.3 sigma) because our large round debris Sigma swamps the miss. Both
sit below the 2-sigma gate, so "2D valid at 24 h" transfers -- but our system is
valid for a DIFFERENT reason (Sigma too large) than reality (Sigma genuinely
tight). That distinction is the point.

Reads figureScripts/data/cara_missigma_transfer.json.
White-bg (paper) + black-bg (slides), CMU Serif, PDF + SVG + PNG. Per CLAUDE.md.

Run:  uv run --with matplotlib --with numpy figureScripts/cara_missigma_transfer_plot.py
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

with open(os.path.join(HERE, "data", "cara_missigma_transfer.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 9.5})

GATE = D["gate_sigma"]
rows = [r for r in D["rows"]
        if np.isfinite(r["miss_sigma_real"]) and np.isfinite(r["miss_sigma_ours_anch"])]

C_REAL = "#2b8cbe"
C_OURS = "#e6550d"


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

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.13)

    lead = np.array([r["lead_hours"] for r in rows])
    real = np.array([r["miss_sigma_real"] for r in rows])
    ours = np.array([r["miss_sigma_ours_anch"] for r in rows])

    # ~24 h decision band shading.
    ax.axvspan(18, 30, color=fg, alpha=0.06, zorder=0)

    # 2-sigma gate line (the validity threshold from prompt 2).
    ax.axhline(GATE, color=fg, ls="--", lw=1.1, alpha=0.6, zorder=1)

    # Per-case connector (real -> ours), thin, to show the systematic shift.
    for lo, rr, ou in zip(lead, real, ours):
        ax.plot([lo, lo], [rr, ou], color=fg, lw=0.5, alpha=0.35, zorder=2)

    ax.scatter(lead, real, s=55, marker="o", facecolor=C_REAL, edgecolor=fg,
               linewidths=0.6, alpha=0.9, zorder=4, label="Real (CDM $\\Sigma$)")
    ax.scatter(lead, ours, s=55, marker="o", facecolor="none", edgecolor=C_OURS,
               linewidths=1.6, alpha=0.95, zorder=5,
               label=r"Ours ($\Phi\Sigma\Phi^{\!\top}$-grown $\Sigma$)")

    ax.set_yscale("log")
    ax.set_xlabel("Lead time before TCA (hours)")
    ax.set_ylabel(r"Miss distance at TCA (Mahalanobis $\sigma$)")
    ax.set_title(r"Our miss-in-$\sigma$ is far shallower than real "
                 r"$-$ 2D-validity transfers, but for the wrong reason")

    # Gate + band labels, placed clear of the data.
    ymin, ymax = ax.get_ylim()
    ax.text(0.985, GATE * 1.08, r"$2\sigma$ validity gate (prompt 2)", color=fg,
            fontsize=9, ha="right", va="bottom", alpha=0.85,
            transform=ax.get_yaxis_transform())
    ax.text(24, ymax * 0.78, "24 h\ndecision", color=fg, fontsize=8.5,
            ha="center", va="top", alpha=0.7)

    leg = ax.legend(loc="lower right", framealpha=0.0, handletextpad=0.4,
                    borderpad=0.6)
    for t in leg.get_texts():
        t.set_color(fg)

    theme_axes(ax, fg, grid)
    save(fig, f"cara_missigma_transfer_{theme}")


for th in ("light", "dark"):
    make(th)

n24 = [r for r in rows if 18 <= r["lead_hours"] <= 30]
mr = np.median([r["miss_sigma_real"] for r in n24])
mo = np.median([r["miss_sigma_ours_anch"] for r in n24])
print(f"\n~24 h band (n={len(n24)}): real median {mr:.2f} sigma, ours {mo:.2f} sigma "
      f"(both < {GATE} gate).")
