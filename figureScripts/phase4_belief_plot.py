#!/usr/bin/env python3
"""Phase 4 belief-tracker validation figures, light + dark backgrounds.

Renders phase4_data.json (written by phase4_belief_data.jl; Julia is the source
of truth). Produces two figures, each in white-bg (paper) + black-bg (slides),
CMU Serif, as PDF + SVG + PNG:

  phase4_belief_validation_{light,dark}   3 panels: (a) correction shrinks Σ with
      measurement points marked, (b) Σ⁺ independent of z, (c) sc vs debris.
  phase4_measurement_cadence_{light,dark} how Σ evolves under different
      measurement cadences: every hour, every 6 h, and predict-only (never).

Run from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/phase4_belief_plot.py
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "figures"))
with open(os.path.join(HERE, "phase4_data.json")) as f:
    d = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10.5})

C_SC, C_DB = "#2b8cbe", "#e6550d"          # spacecraft blue, debris orange


def theme_axes(axes, fg, grid):
    for ax in np.atleast_1d(axes).ravel():
        ax.grid(True, which="both", color=grid, lw=0.4, alpha=0.6)
        for s in ax.spines.values():
            s.set_color(fg)
        ax.tick_params(colors=fg)
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


def ellipse_from_cov(cov, nsig, **kw):
    cov = np.array(cov)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals[::-1], 0))
    return Ellipse((0, 0), w, h, angle=ang, **kw)


# order series 24h -> TCA (descending tau) for left-to-right reading
def ordered(series, key):
    tau = np.array(series["tau_hr"])
    o = np.argsort(-tau)
    return tau[o], np.array(series[key])[o], np.array(series["measured"])[o]


# =====================================================================
# FIGURE 1 — 3-panel validation
# =====================================================================
def fig_validation(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.16, wspace=0.32)

    # ---- (a) correction shrinks uncertainty, measurement points marked ----
    ax = axes[0]
    for key, col, lbl in (("sc_pos", C_SC, "spacecraft"), ("db_pos", C_DB, "debris")):
        tau, y, meas = ordered(d["every1"], key)
        ax.plot(tau, y, "-", color=col, lw=1.4, label=lbl)
        ax.plot(tau[meas], y[meas], "o", color=col, ms=4.5, mec=fg, mew=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Time remaining to TCA (hr)")
    ax.set_ylabel("Position 1$\\sigma$ (m)")
    ax.set_title("(a) Correction shrinks uncertainty")
    ax.invert_xaxis()
    ax.legend(loc="upper right", framealpha=0.0)
    ax.text(0.03, 0.05, "$\\bullet$ = measurement applied\n(here: every 1 hr step)",
            transform=ax.transAxes, fontsize=9, color=fg, va="bottom")

    # ---- (b) Σ⁺ independent of z ----
    ax = axes[1]
    ax.scatter(d["muB_x"], d["muB_y"], s=8, color="#31a354", alpha=0.45,
               label="$\\mu^+$ per random $z$")
    for ns in (1, 2):
        ax.add_patch(ellipse_from_cov(d["Sigma_plus_xy"], ns, fill=False,
                     edgecolor=fg, lw=1.6 if ns == 1 else 1.0,
                     ls="-" if ns == 1 else ":"))
    ax.plot(*d["ab_mu_a"], "P", color="#756bb1", ms=11, mec=fg, mew=0.6, label="hand-rolled (a)")
    ax.plot(*d["ab_mu_b"], "x", color="#d62728", ms=9, mew=2.0, label="brahe EKF (b)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$\\mu^+_x - \\mu^-_x$ (m)")
    ax.set_ylabel("$\\mu^+_y - \\mu^-_y$ (m)")
    ax.set_title("(b) $\\Sigma^+$ fixed; only $\\mu^+$ moves with $z$")
    ax.legend(loc="upper left", framealpha=0.0, fontsize=9)
    ax.text(0.97, 0.03, "ellipses: $\\Sigma^+$ 1$\\sigma$/2$\\sigma$\n(identical for all $z$)",
            transform=ax.transAxes, fontsize=9, color=fg, ha="right", va="bottom")

    # ---- (c) sc vs debris ----
    ax = axes[2]
    tau, sc, _ = ordered(d["every1"], "sc_pos")
    _, db, _ = ordered(d["every1"], "db_pos")
    ax.plot(tau, sc, "-o", color=C_SC, ms=3, lw=1.4, label="sc position")
    ax.plot(tau, db, "-o", color=C_DB, ms=3, lw=1.4, label="debris position")
    ax.set_yscale("log")
    ax.set_xlabel("Time remaining to TCA (hr)")
    ax.set_ylabel("Position 1$\\sigma$ (m)")
    ax.set_title("(c) Spacecraft vs. debris uncertainty")
    ax.invert_xaxis()
    ratio = d["P0_db_pos"] / d["P0_sc_pos"]
    ax.legend(loc="upper right", framealpha=0.0)
    ax.text(0.03, 0.05,
            f"debris/sc $P_0$ pos 1$\\sigma$ = {ratio:.0f}$\\times$\n"
            f"($\\sigma_0$: sc {d['P0_sc_pos']:.0f} m, debris {d['P0_db_pos']:.0f} m)",
            transform=ax.transAxes, fontsize=9, color=fg, va="bottom")

    theme_axes(axes, fg, grid)
    fig.suptitle("Phase 4 — Kalman predict/correct belief tracker",
                 color=fg, fontsize=15, y=0.97)
    save(fig, os.path.join(OUT, f"phase4_belief_validation_{theme}"))


# =====================================================================
# FIGURE 2 — measurement cadence (Grace's "what if we don't measure?")
# =====================================================================
def fig_cadence(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.12)

    # debris (dominant uncertainty) under three cadences
    styles = (("every1",   "#2b8cbe", "-",  "measurement every 1 hr"),
              ("every6",   "#31a354", "-",  "measurement every 6 hr"),
              ("predonly", "#d62728", "--", "no measurements (predict-only)"))
    for key, col, ls, lbl in styles:
        tau, y, meas = ordered(d[key], "db_pos")
        ax.plot(tau, y, ls, color=col, lw=1.7, label=lbl)
        if meas.any():
            ax.plot(tau[meas], y[meas], "o", color=col, ms=6, mec=fg, mew=0.5, zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("Time remaining to TCA (hr)")
    ax.set_ylabel("Debris position 1$\\sigma$ (m)")
    ax.set_title("Uncertainty vs. measurement cadence (debris)")
    ax.invert_xaxis()
    ax.legend(loc="upper right", framealpha=0.0)
    box = dict(boxstyle="round,pad=0.4", fc="black" if dark else "white",
               ec=fg, lw=0.6, alpha=0.75)
    ax.text(0.02, 0.62,
            "$\\bullet$ = measurement applied.\n"
            "Between measurements $\\Sigma$ grows via the\n"
            "predict step $\\Sigma^-=\\Phi\\,\\Sigma\\,\\Phi^\\top$; each\n"
            "correction snaps it back down. Predict-only\n"
            "never corrects, so uncertainty only grows.",
            transform=ax.transAxes, fontsize=9.5, color=fg, va="top", bbox=box)

    theme_axes(ax, fg, grid)
    save(fig, os.path.join(OUT, f"phase4_measurement_cadence_{theme}"))


os.makedirs(OUT, exist_ok=True)
for th in ("light", "dark"):
    fig_validation(th)
    fig_cadence(th)
