#!/usr/bin/env python3
"""Phase 3 Σ(τ)-table validation figures, light + dark backgrounds.

Renders phase3_data.json (written by phase3_covariance_data.jl; Julia is the
source of truth). Produces two figures, each in white-bg (paper) + black-bg
(slides), CMU Serif, as PDF + SVG + PNG:

  phase3_sigma_growth_{light,dark}  RTN 1σ growth vs τ for sc + debris (log-log).
      Fine 6-min lines resolve the once-per-orbit radial/cross-track oscillation;
      1-hr planner-grid dots overlaid; dashed slope-1 (σ ∝ τ) reference shows the
      velocity-dominated along-track growth.
  phase3_pc_vs_tau_{light,dark}     Chan Pc through Σ(τ) vs τ (24 h → TCA). Fine
      6-min truth line + 1-hr planner samples; annotates that the oscillation
      period equals the orbital period (radial/cross-track Σ breathes per orbit).

Run from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/phase3_covariance_plot.py
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "figures"))
with open(os.path.join(HERE, "phase3_data.json")) as f:
    d = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10.5})

# RTN component colors, Pc colors.
C_T, C_R, C_N = "#e41a1c", "#4daf4a", "#984ea3"   # along-track, radial, cross-track
C_REF = "#888888"
C_PC, C_DOT = "#d62728", "#2b4fd6"                # Pc truth line, 1-hr sample dots


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


def arr(grid, key):
    """τ (ascending) and the matching series for a grid ('coarse' or 'fine')."""
    tau = np.array(d[grid]["tau_hr"])
    o = np.argsort(tau)
    return tau[o], np.array(d[grid][key])[o]


# =====================================================================
# FIGURE 1 — Σ(τ) growth: RTN 1σ vs time-to-TCA, fine + coarse overlay
# =====================================================================
def fig_growth(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3), sharex=True, sharey=True)
    # reserve headroom for the suptitle and a footer band for the legend so the
    # copied Phase-4 save() (no bbox_inches='tight') never clips them
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.24, wspace=0.08)

    for ax, obj, ttl in ((axes[0], "sc", "Spacecraft"),
                         (axes[1], "db", "Debris")):
        for comp, col, lab in (("T", C_T, "Along-track (T)"),
                               ("R", C_R, "Radial (R)"),
                               ("N", C_N, "Cross-track (N)")):
            tf, yf = arr("fine", f"{obj}_{comp}")
            tc, yc = arr("coarse", f"{obj}_{comp}")
            ax.loglog(tf, yf, color=col, lw=1.3, alpha=0.9, label=lab)
            ax.loglog(tc, yc, color=col, ls="none", marker="o", ms=4.5,
                      mec=fg, mew=0.4)
        # reference slope: sigma ∝ tau (exponent 1), anchored at the fine T[0]
        tf, yf = arr("fine", f"{obj}_T")
        ref = yf[0] * (tf / tf[0])
        ax.loglog(tf, ref, color=C_REF, ls="--", lw=1.1,
                  label=r"$\sigma \propto \tau$ (slope 1)")
        ax.set_title(ttl)
        ax.set_xlabel("Time remaining to TCA (hr)")

    axes[0].set_ylabel(r"Position 1$\sigma$ (m)")
    theme_axes(axes, fg, grid)

    # legend in its own clear band below the panels — never over data
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, 0.03), framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(fg)
    fig.suptitle("Covariance growth vs time remaining to TCA "
                 r"(dots = 1-hr planner grid; lines = 6-min truth)",
                 color=fg, y=0.95, fontsize=12)
    save(fig, os.path.join(OUT, f"phase3_sigma_growth_{theme}"))


# =====================================================================
# FIGURE 2 — Pc through Σ(τ) vs time-to-TCA, fine + coarse overlay
# =====================================================================
def fig_pc(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.12)

    tf, pf = arr("fine", "pc")
    tc, pc = arr("coarse", "pc")
    ax.semilogy(tf, pf, color=C_PC, lw=1.3, alpha=0.85,
                label="6-min grid (physical truth)")
    ax.semilogy(tc, pc, color=C_DOT, ls="none", marker="o", ms=5,
                mec=fg, mew=0.4, label="1-hr grid (planner samples)")
    ax.invert_xaxis()
    ax.set_xlabel(r"Time remaining to TCA (hr)  —  approaching TCA $\rightarrow$")
    ax.set_ylabel(r"Chan $P_c$ (each $\Sigma$ propagated to TCA)")
    ax.set_title("Collision probability vs time remaining\n"
                 f"(cross-track, miss {d['miss_m']:.0f} m, "
                 f"$v_{{rel}}$ {d['v_rel']:.0f} m/s)")

    theme_axes(ax, fg, grid)
    leg = ax.legend(loc="upper left", framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(fg)
    ax.text(0.97, 0.05,
            f"oscillation period $\\approx$ {d['Torb_h']*60:.0f} min = orbital period\n"
            "(radial/cross-track $\\Sigma$ breathes once per orbit)",
            transform=ax.transAxes, ha="right", va="bottom", color=fg, fontsize=9.5)
    save(fig, os.path.join(OUT, f"phase3_pc_vs_tau_{theme}"))


os.makedirs(OUT, exist_ok=True)
for th in ("light", "dark"):
    fig_growth(th)
    fig_pc(th)
