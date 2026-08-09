#!/usr/bin/env python3
"""maneuver_effectiveness_plot.py — the maneuver-effectiveness surface.

Headline: Pc-at-TCA as a function of a SINGLE impulsive along-track Δv
(magnitude M, signed +/- along +v̂) applied at time-to-TCA T, on the real
SWIFT/JILIN CARA conjunction. Pure physics — no MCTS, no planner (see
figureScripts/maneuver_effectiveness_data.jl and
notes/maneuver_effectiveness_findings.md).

Three panels:
  (A) HEADLINE surface: heatmap of log10(Pc) over (T, signed Δv), with the
      Pc = threshold contour (the feasibility boundary) and the operational
      Δv magnitude marked. This is the Pc-reduction surface.
  (B) Pc vs time-to-TCA at a few fixed |Δv| — reads off "does timing matter",
      and shows the once-per-orbit phasing ripple.
  (C) Pc vs |Δv| at a few fixed timings — reads off "does magnitude matter"
      and how the whole curve shifts earlier→later.

White-bg (paper) + black-bg (slides), CMU Serif, PDF + SVG + PNG. Per CLAUDE.md.

Run:  uv run --with matplotlib --with numpy figureScripts/maneuver_effectiveness_plot.py
"""
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, "data", "maneuver_effectiveness.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 9.5})

case = D["case"]
mags = np.array(D["magnitudes_ms"], dtype=float)         # signed Δv (m/s)
timings = np.array(D["timings_h"], dtype=float)          # time-to-TCA (h)
PC = np.array(D["pc"], dtype=float)                       # [mag][timing]
MISS = np.array(D["miss_distance_m_grid"], dtype=float)  # [mag][timing] miss @ TCA (m)
DPRED = np.array(D["dD_predicted_m_grid"], dtype=float)  # [mag][timing] 3·|Δv|·T (m)
thr = case["pc_threshold"]
pc_base = case["pc_baseline"]
dv_cur = case["dv_current_ms"]
miss_base = case["miss_distance_m"]
period_h = case.get("orbital_period_h", None)

# log10(Pc), floored so the huge dynamic range (down to ~1e-300 / 0) stays
# readable. Floor a bit below the threshold so "cleared" cells all read as the
# floor color rather than smearing the colormap over 300 dead decades.
LOG_FLOOR = -12.0
with np.errstate(divide="ignore"):
    L = np.log10(np.clip(PC, 10 ** LOG_FLOOR, 1.0))
L = np.clip(L, LOG_FLOOR, 0.0)

# --- theming helpers -------------------------------------------------------
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


def save(fig, base, dark):
    # Vector copies stay transparent (drop onto any slide/paper background).
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{base}.{ext}"), transparent=True,
                    dpi=200, facecolor="none", edgecolor="none")
    # PNG gets an OPAQUE background matching the theme so the white-on-dark text
    # is independently legible (a transparent dark PNG renders white text
    # invisibly against a white viewer). Per the global figure guidance.
    bg = "black" if dark else "white"
    fig.savefig(os.path.join(OUT, f"{base}.png"), transparent=False,
                dpi=200, facecolor=bg, edgecolor="none")
    plt.close(fig)
    print(f"wrote {base}.{{pdf,svg,png}}")


# Representative cuts. |Δv| line-cuts: pick a spread bracketing the operational
# scale. Timing line-cuts: early / mid / late.
mag_cuts = [0.005, 0.01, 0.03, 0.1, 0.3]      # |Δv| (m/s) for panel B
time_cuts_h = [3.0, 8.0, 16.0, 24.0, 33.0]    # time-to-TCA (h) for panel C


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def make(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"
    # Sequential white/(dark base)→red: LOW Pc (safe) is pale, HIGH Pc (unsafe)
    # is deep red. Unambiguous "lit red = danger".
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_under("#08306b" if not dark else "#0b1a3a")  # deep-safe below floor
    accent = "#2166ac" if not dark else "#6baed6"          # blue current-Δv line

    fig = plt.figure(figsize=(7.9, 12.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.55, 1.0, 1.0, 1.0], hspace=0.46,
                          left=0.135, right=0.9, top=0.965, bottom=0.055)

    # ---- Panel A: headline surface -----------------------------------------
    # y-axis is SYMLOG so the mm/s–cm/s band (where all the action is) expands
    # and the linear ±0.5 range does not swamp it. pcolormesh over the signed
    # magnitude centers directly (non-uniform), symlog transform on the axis.
    axA = fig.add_subplot(gs[0])
    def edges(c):
        c = np.asarray(c, dtype=float)
        e = np.empty(len(c) + 1)
        e[1:-1] = 0.5 * (c[:-1] + c[1:])
        e[0] = c[0] - (c[1] - c[0]) / 2
        e[-1] = c[-1] + (c[-1] - c[-2]) / 2
        return e
    Tx, My = np.meshgrid(edges(timings), edges(mags))
    norm = Normalize(vmin=LOG_FLOOR, vmax=np.log10(pc_base) + 0.3)
    pcm = axA.pcolormesh(Tx, My, L, cmap=cmap, norm=norm, shading="flat")
    # Feasibility boundary: Pc = threshold contour (dark, reads on white→red).
    Tc, Mc = np.meshgrid(timings, mags)
    cline = "black" if not dark else "white"
    try:
        axA.contour(Tc, Mc, np.log10(np.clip(PC, 1e-300, 1.0)),
                    levels=[np.log10(thr)], colors=[cline], linewidths=1.6,
                    linestyles="solid")
    except Exception:
        pass
    axA.axhline(0.0, color=fg, lw=0.8, ls=":", alpha=0.6)
    for dv in (dv_cur, -dv_cur):
        axA.axhline(dv, color=accent, lw=1.2, ls="--", alpha=0.95)
    axA.set_yscale("symlog", linthresh=1e-3)
    axA.set_ylim(-0.5, 0.5)
    axA.set_xlabel("Time to TCA of the burn (h)")
    axA.set_ylabel(r"Along-track $\Delta v$ (m/s, $\pm\hat v$)")
    axA.set_title("Collision probability after a single along-track burn "
                  "(SWIFT/JILIN)")
    axA.set_xlim(timings.min(), timings.max())
    axA.invert_xaxis()   # time-to-TCA counts DOWN toward the conjunction
    theme_axes(axA, fg, grid)
    # Legend-style annotations OUTSIDE the data (top-left), no overlap.
    axA.text(0.015, 0.965,
             r"black line: $P_c=10^{-5}$ (feasibility boundary)" + "\n"
             r"blue dashed: current $|\Delta v|=%.2f$ m/s" % dv_cur + "\n"
             "red = unsafe,  pale = safe",
             transform=axA.transAxes, color=fg, fontsize=8.5, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc=("#222222" if dark else "white"),
                       ec=fg, lw=0.6, alpha=0.8))
    cbar = fig.colorbar(pcm, ax=axA, pad=0.015)
    cbar.set_label(r"$\log_{10} P_c$ at TCA", color=fg)
    cbar.ax.yaxis.set_tick_params(color=fg)
    cbar.outline.set_edgecolor(fg)
    for t in cbar.ax.get_yticklabels():
        t.set_color(fg)

    # ---- Panel B: Pc vs time-to-TCA at fixed |Δv| --------------------------
    axB = fig.add_subplot(gs[1])
    colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(mag_cuts)))
    for c, m in zip(colors, mag_cuts):
        i = nearest_idx(mags, m)      # use the +v̂ branch
        axB.semilogy(timings, np.clip(PC[i], 1e-9, 1.0), color=c, lw=1.6,
                     label=rf"$|\Delta v|={mags[i]:.3f}$")
    axB.axhline(thr, color=fg, lw=1.2, ls="--", alpha=0.8)
    axB.text(timings.min() * 1.02, thr, r"threshold $10^{-5}$ ", color=fg,
             fontsize=8.5, va="bottom", ha="left")
    axB.axhline(pc_base, color=grid, lw=1.0, ls=":", alpha=0.9)
    axB.text(timings.min() * 1.02, pc_base, r"no-burn $P_c$ ", color=fg,
             fontsize=8.5, va="bottom", ha="left")
    axB.set_xlabel("Time to TCA of the burn (h)")
    axB.set_ylabel(r"$P_c$ at TCA")
    axB.set_title(r"Timing dependence at fixed magnitude (burn along $+\hat v$)")
    axB.set_ylim(1e-9, 1e-1)
    axB.invert_xaxis()
    theme_axes(axB, fg, grid)
    leg = axB.legend(loc="lower left", framealpha=0.0, ncol=2)
    for t in leg.get_texts():
        t.set_color(fg)

    # ---- Panel C: Pc vs |Δv| at fixed timings ------------------------------
    axC = fig.add_subplot(gs[2])
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(time_cuts_h)))
    pos = mags >= 0
    mags_pos = mags[pos]
    for c, th_ in zip(colors, time_cuts_h):
        j = nearest_idx(timings, th_)
        axC.loglog(np.clip(mags_pos, 1e-4, None), np.clip(PC[pos, j], 1e-16, 1.0),
                   color=c, lw=1.6, marker="o", ms=2.5,
                   label=rf"$T={timings[j]:.0f}$ h")
    axC.axhline(thr, color=fg, lw=1.2, ls="--", alpha=0.8)
    axC.axvline(dv_cur, color=accent, lw=1.2, ls="--", alpha=0.95)
    axC.text(dv_cur, 3e-2, r" current $\Delta v$",
             color=accent, fontsize=8.5, va="top", ha="left")
    axC.set_xlabel(r"Along-track burn magnitude $|\Delta v|$ (m/s, $+\hat v$)")
    axC.set_ylabel(r"$P_c$ at TCA")
    axC.set_title("Magnitude dependence at fixed timing")
    axC.set_ylim(1e-12, 1e-1)
    theme_axes(axC, fg, grid)
    leg = axC.legend(loc="lower left", framealpha=0.0, ncol=1)
    for t in leg.get_texts():
        t.set_color(fg)

    # ---- Panel D: physics check — measured miss-shift vs 3·|Δv|·T ----------
    # If our brahe propagation reproduces the Clohessy–Wiltshire secular drift,
    # the induced along-track miss shift |miss(Δv) − miss(0)| tracks 3·|Δv|·T.
    # Plot measured (y) vs predicted (x) at fixed timings; the dashed y=x is the
    # law. Deviation upward near TCA is the bounded radial/phasing term.
    axD = fig.add_subplot(gs[3])
    j0 = nearest_idx(mags, 0.0)
    miss0 = MISS[j0]                     # no-burn miss vs timing
    for c, th_ in zip(colors, time_cuts_h):
        j = nearest_idx(timings, th_)
        meas = np.abs(MISS[pos, j] - miss0[j])       # induced shift (m)
        pred = DPRED[pos, j]                          # 3·|Δv|·T (m)
        ok = (pred > 1.0)
        axD.loglog(pred[ok], np.clip(meas[ok], 1.0, None), color=c, lw=0,
                   marker="o", ms=3.0, label=rf"$T={timings[j]:.0f}$ h")
    lim = [1e1, 1e6]
    axD.plot(lim, lim, color=fg, lw=1.1, ls="--", alpha=0.8,
             label=r"$\Delta D_T = 3\,|\Delta v|\,T$ (CW law)")
    axD.set_xlim(*lim); axD.set_ylim(*lim)
    axD.set_xlabel(r"Predicted along-track shift $3\,|\Delta v|\,T$ (m)")
    axD.set_ylabel(r"Measured miss shift (m)")
    axD.set_title("Physics check: induced miss shift vs CW secular-drift law")
    theme_axes(axD, fg, grid)
    leg = axD.legend(loc="upper left", framealpha=0.0, ncol=2)
    for t in leg.get_texts():
        t.set_color(fg)

    save(fig, f"maneuver_effectiveness_{theme}", dark)


make("light")
make("dark")
print("done")
