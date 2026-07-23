#!/usr/bin/env python3
"""Phase 6 validation figures (built on the CORRECTED accumulated-Σ Pc).

Renders phase6_validation_data.json (written by phase6_validation_data.jl; Julia
is the source of truth). Produces two figures, each white-bg (paper) + black-bg
(slides), CMU Serif, as PDF + SVG + PNG:

  phase6_pc_vs_tau_{light,dark}        Fig 6a — the planner's node_pc_at_tca vs.
      time-remaining: coarse 1-hr grid (dots) the planner samples + a fine
      sub-orbit grid (line) resolving the once-per-orbit R/N ripple. Shows the
      corrected Pc is finite/sensible and that a single-instant Pc is fragile
      (~10× hour-to-hour swing from orbital-phase aliasing).

  phase6_sigma_invariance_{light,dark} Fig 6b — Σ-at-TCA is ~branch-invariant
      across WAIT vs MANEUVER (identical to many digits) while the mean
      separation at TCA moves with the burn: the constraint acts through the mean.

Run from the repo root:
  uv run --with matplotlib --with numpy python figureScripts/phase6_pc_vs_tau_plot.py
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "figures"))
with open(os.path.join(HERE, "phase6_validation_data.json")) as f:
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
C_PC, C_FINE = "#756bb1", "#31a354"        # coarse-Pc purple, fine-Pc green


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


# =====================================================================
# FIGURE 6a — Pc vs. time-remaining (corrected node Pc)
# =====================================================================
def fig_pc_vs_tau(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    box = dict(boxstyle="round,pad=0.4", fc="black" if dark else "white",
               ec=fg, lw=0.6, alpha=0.78)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.13)

    tau_f = np.array(d["tau_fine_hr"]); pc_f = np.array(d["pc_fine"])
    tau_c = np.array(d["tau_coarse_hr"]); pc_c = np.array(d["pc_coarse"])

    # fine grid (line) resolves the once-per-orbit ripple
    ax.plot(tau_f, pc_f, "-", color=C_FINE, lw=1.3, alpha=0.9,
            label=f"fine grid ($\\sim${d['orbit_period_min']/8:.0f} min steps)")
    # coarse 1-hr grid (dots) — what the planner actually samples
    ax.plot(tau_c, pc_c, "o", color=C_PC, ms=6, mec=fg, mew=0.5, zorder=4,
            label="1-hr grid (planner samples)")

    # Pc threshold line
    ax.axhline(d["pc_threshold"], color=fg, ls="--", lw=1.0, alpha=0.7)
    ax.text(23.5, d["pc_threshold"] * 1.4, f"$P_c$ threshold = {d['pc_threshold']:g}",
            color=fg, fontsize=9.5, va="bottom")

    ax.set_yscale("log")
    ax.set_xlabel("Time remaining to TCA (hr)")
    ax.set_ylabel("$P_c$ at TCA (coast from $\\tau$)")
    ax.set_title("(6a) Corrected $P_c$ vs. time-remaining, and its fragility")
    ax.invert_xaxis()
    ax.legend(loc="upper left", framealpha=0.0, bbox_to_anchor=(0.0, 0.88))

    # quantify the fragility: how many orders of magnitude the 1-hr samples span
    pc_pos = pc_c[np.isfinite(pc_c) & (pc_c > 0)]
    orders = np.log10(np.nanmax(pc_pos) / np.nanmin(pc_pos))
    ax.text(0.97, 0.05,
            "Node $P_c$ grows the node's OWN accumulated\n"
            "belief $\\Sigma$ to TCA (corrected model), not a\n"
            "fresh $P_0$. Radial/cross-track $\\Sigma$ breathes\n"
            f"once per orbit ($\\sim${d['orbit_period_min']:.0f} min): the fine curve\n"
            "dives to a covariance null each orbit, and the\n"
            f"1-hr samples alias it $-$ spanning $>${orders:.0f} orders\n"
            "of magnitude $\\Rightarrow$ a single-instant $P_c$ is\n"
            "fragile (motivates window-$P_c$).",
            transform=ax.transAxes, fontsize=9, color=fg, ha="right", va="bottom",
            bbox=box)

    theme_axes(ax, fg, grid)
    fig.suptitle(f"Phase 6 validation — cross-track conjunction "
                 f"(miss {d['miss_m']:.0f} m, $v_{{rel}}$ {d['v_rel']:.0f} m/s)",
                 color=fg, fontsize=13.5, y=0.975)
    save(fig, os.path.join(OUT, f"phase6_pc_vs_tau_{theme}"))


# =====================================================================
# FIGURE 6b — Σ-at-TCA branch-invariance
# =====================================================================
def fig_sigma_invariance(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    box = dict(boxstyle="round,pad=0.4", fc="black" if dark else "white",
               ec=fg, lw=0.6, alpha=0.78)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.14, wspace=0.28)

    depths = np.array(d["branch_depths_hr"])
    x = np.arange(len(depths))
    w = 0.34

    # ---- (left) RELATIVE Σ-at-TCA difference WAIT vs MANEUVER, per object ----
    # Plotting the raw σ bars hides the signal (debris σ ≫ sc σ). What matters is
    # how much Σ-at-TCA DIFFERS between the two branches: debris (untouched by the
    # burn) is bitwise 0; sc differs only through the maneuver perturbing its STM.
    ax = axes[0]
    rel_sc = np.array(d["relSigma_sc"])
    rel_db = np.array(d["relSigma_db"])
    # floor exact-zero debris at a tiny value so it's visible as "≈0" on a bar
    FLOOR = 1e-16
    ax.bar(x - w/2, np.maximum(rel_db * 100, FLOOR), w, color=C_DB, alpha=0.9,
           label="debris (untouched)")
    ax.bar(x + w/2, np.maximum(rel_sc * 100, FLOOR), w, color=C_SC, alpha=0.9,
           label="spacecraft (maneuvered)")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 3, 30)
    ax.set_xticks(x); ax.set_xticklabels([f"{h:.0f} h" for h in depths])
    ax.set_xlabel("Branch depth (time remaining when branched)")
    ax.set_ylabel("$\\Sigma$-at-TCA diff, WAIT vs MANEUVER (\\%)")
    ax.set_title("(6b-i) $\\Sigma$ at TCA barely moves with the burn")
    ax.set_ylim(FLOOR / 3, 1e2)   # bars top out near a few %, leave a clear mid-band
    ax.legend(loc="upper right", framealpha=0.0, fontsize=9)
    # annotate the exact-zero debris bars
    for xi in x:
        ax.text(xi - w/2, FLOOR * 1.5, "$0$", color=fg, fontsize=8.5,
                ha="center", va="bottom", rotation=90)
    # place the note in the empty mid-band between the floored debris bars and
    # the few-% spacecraft bars (nothing plotted there)
    ax.text(0.5, 0.42,
            "Debris $\\Sigma$-at-TCA is bitwise identical across branches\n"
            "(the burn never touches it). The spacecraft's differs by\n"
            "only a few \\% $-$ the maneuver perturbs its STM slightly,\n"
            "shrinking as TCA nears. $\\Sigma$ is essentially branch-\n"
            "invariant (exactly so for the un-maneuvered object);\n"
            "$P_c$ moves across branches via the MEAN.",
            transform=ax.transAxes, fontsize=8.3, color=fg, ha="center", va="center",
            bbox=box)

    # ---- (right) mean separation at TCA DOES move with the maneuver ----------
    ax = axes[1]
    ax.bar(x - w/2, np.array(d["sep_wait"]) / 1000, w, color="#636363", alpha=0.9,
           label="WAIT")
    ax.bar(x + w/2, np.array(d["sep_mvr"]) / 1000, w, color="#31a354", alpha=0.9,
           label=f"MANEUVER ($\\Delta v$ {d['dv_ms']:.0f} m/s)")
    ax.set_xticks(x); ax.set_xticklabels([f"{h:.0f} h" for h in depths])
    ax.set_xlabel("Branch depth (time remaining when branched)")
    ax.set_ylabel("Mean separation at TCA (km)")
    ax.set_title("(6b-ii) Mean separation: the burn moves it")
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.0)
    ax.text(0.03, 0.05,
            "The maneuver changes $P_c$ through the\n"
            "MEAN (separation grows), not through\n"
            "$\\Sigma$ $\\Rightarrow$ the chance constraint acts on\n"
            "\"did the burn move the mean far enough.\"",
            transform=ax.transAxes, fontsize=9, color=fg, va="bottom", bbox=box)

    theme_axes(axes, fg, grid)
    fig.suptitle("Phase 6 validation — $\\Sigma$-at-TCA is branch-invariant; "
                 "the constraint acts through the mean",
                 color=fg, fontsize=13.5, y=0.965)
    save(fig, os.path.join(OUT, f"phase6_sigma_invariance_{theme}"))


os.makedirs(OUT, exist_ok=True)
for th in ("light", "dark"):
    fig_pc_vs_tau(th)
    fig_sigma_invariance(th)
