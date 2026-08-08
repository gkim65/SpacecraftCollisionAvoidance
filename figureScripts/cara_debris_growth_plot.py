#!/usr/bin/env python3
"""cara_debris_growth_plot.py — Fig F3-c: real DEBRIS covariance-growth envelope
(stratified), the target the debris process-noise Q / anisotropic P0 fix must
reproduce. The DEBRIS analogue of the primary growth figure (F3-a).

X = lead time (days, log). Y = along-track (in-track) 1sigma (m, log). Shows:
  - the debris scatter cloud (all secondaries, colored by object class) -- honest
    about the heterogeneity/spread; cross-sectional single snapshots, NOT a
    trajectory, so it is noisy and even non-monotonic bin-to-bin;
  - the debris all-class per-lead-bin median with a min-max band (the envelope);
  - the real PRIMARY curve overlaid (F3-a) for comparison;
  - the altitude split (perigee <600 km vs >=600 km) as the physically-meaningful
    stratum -- low-perigee (more drag) debris grows fastest.

Headline: real debris grows STEEPER than the primary (fitted p~3.2 all-debris,
dragged by the far-out bins; primary p~1.9; our STM p~1.4). So a debris Q must be
larger than a primary Q, likely regime-dependent by altitude. The band + scatter
are the deliverable, NOT the single fitted line.

Reads figureScripts/data/cara_debris_growth.json.
White-bg (paper) + black-bg (slides), CMU Serif, PDF + SVG + PNG. Per CLAUDE.md.

Run:  uv run --with matplotlib --with numpy figureScripts/cara_debris_growth_plot.py
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

with open(os.path.join(HERE, "data", "cara_debris_growth.json")) as f:
    D = json.load(f)

if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["CMU Serif", "Computer Modern Roman"]})
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 9})

CLS_COLOR = {"debris": "#e6550d", "rocket_body": "#756bb1",
             "payload": "#2b8cbe", "unknown": "#31a354"}
CLS_LABEL = {"debris": "Debris", "rocket_body": "Rocket body",
             "payload": "Payload", "unknown": "Unknown"}

strata = D["strata"]
primary = D["primary"]
cuts = D["cuts"]


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


def bin_series(stratum_key):
    """Return (mid_d[], med_T[], min_T[], max_T[]) over non-empty bins."""
    b = strata[stratum_key]["bins"]
    mid, med, lo, hi = [], [], [], []
    for e in b:
        if e["med_T"] is not None and np.isfinite(e["med_T"]):
            mid.append(e["mid_d"]); med.append(e["med_T"])
            lo.append(e["min_T"]); hi.append(e["max_T"])
    return np.array(mid), np.array(med), np.array(lo), np.array(hi)


def make(theme):
    dark = theme == "dark"
    fg = "white" if dark else "black"
    grid = "#555555" if dark else "#cccccc"

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.16)

    # (i) debris scatter cloud, colored by class (all secondaries) -----------------
    # use the debris_all stratum's raw points + payload/RB/unknown for the cloud
    for key in ("debris_all", "payload_sec", "rocket_body", "unknown"):
        s = strata[key]
        lead = np.array(s["pts_lead_d"]); sigT = np.array(s["pts_sigT"])
        cls = s["pts_class"]
        for lo_, st_, c_ in zip(lead, sigT, cls):
            if not np.isfinite(st_) or st_ <= 0:
                continue
            ax.scatter(lo_, st_, s=26, marker="o",
                       facecolor="none", edgecolor=CLS_COLOR.get(c_, "#999"),
                       linewidths=1.0, alpha=0.55, zorder=2)

    # (ii) debris all-class median + min-max band ---------------------------------
    mid, med, lo, hi = bin_series("debris_all")
    ax.fill_between(mid, lo, hi, color=CLS_COLOR["debris"], alpha=0.12, zorder=1)
    ax.plot(mid, med, color=CLS_COLOR["debris"], lw=2.6, marker="s", ms=8,
            mec=fg, mew=0.7, zorder=5, label="Debris median (all)")

    # (iv) altitude split medians (the physically-meaningful stratum) -------------
    for key, ls, lbl in (("debris_lowalt_lt600", "--", r"Debris perigee $<600$ km"),
                         ("debris_highalt_ge600", ":", r"Debris perigee $\geq600$ km")):
        m2, md2, _, _ = bin_series(key)
        ax.plot(m2, md2, color=CLS_COLOR["debris"], ls=ls, lw=1.7, alpha=0.9,
                marker="^", ms=6, zorder=4, label=lbl)

    # (iii) primary reference curve -----------------------------------------------
    pmid = [b["mid_d"] for b in primary["bins"]]
    pT = [b["T"] for b in primary["bins"]]
    ax.plot(pmid, pT, color="#2b8cbe", lw=2.2, ls="-", marker="o", ms=8,
            mec=fg, mew=0.7, zorder=6, label="Primary (well-tracked)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Lead time before TCA (days)")
    ax.set_ylabel(r"Along-track (in-track) 1$\sigma$ (m)")
    ax.set_title("Real debris uncertainty grows steeper than the primary "
                 r"and splits by altitude")

    # exponent annotation box (upper-left, clear of the up-right data trend).
    p_all = strata["debris_all"]["fit"]["p"]
    # altitude split shows up in MAGNITUDE, not slope: quote the 2-4 d medians.
    b_lo = strata["debris_lowalt_lt600"]["bins"][2]["med_T"]   # 2-4 d bin
    b_hi = strata["debris_highalt_ge600"]["bins"][2]["med_T"]
    txt = (r"Along-track exponent $\sigma_T\!\propto\!\tau^p$:" "\n"
           rf"  debris (all)  $p\!\approx\!{p_all:.1f}$" "\n"
           rf"  primary  $p\!\approx\!{primary['p']:.1f}$;  our STM $p\!\approx\!{primary['our_stm_p']:.1f}$" "\n"
           r"Altitude splits the MAGNITUDE (2--4 d):" "\n"
           rf"  perigee $<600$ km $\approx${b_lo/1000:.0f} km vs $\geq600$ km $\approx${b_hi/1000:.1f} km")
    ax.text(0.025, 0.975, txt, transform=ax.transAxes, color=fg, fontsize=9,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="black" if dark else "white",
                      ec=fg, lw=0.6, alpha=0.8))

    # class-color legend + curve legend, lower-right (clear corner).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=CLS_COLOR["debris"], lw=2.6, marker="s", ms=7,
               label="Debris median (all)"),
        Line2D([0], [0], color=CLS_COLOR["debris"], ls="--", lw=1.7, marker="^",
               ms=6, label=r"Debris perigee $<600$ km"),
        Line2D([0], [0], color=CLS_COLOR["debris"], ls=":", lw=1.7, marker="^",
               ms=6, label=r"Debris perigee $\geq600$ km"),
        Line2D([0], [0], color="#2b8cbe", lw=2.2, marker="o", ms=7,
               label="Primary (well-tracked)"),
        Line2D([0], [0], marker="o", ls="none", mfc="none",
               mec=CLS_COLOR["payload"], ms=7, label="Payload secondary (pts)"),
        Line2D([0], [0], marker="o", ls="none", mfc="none",
               mec=CLS_COLOR["rocket_body"], ms=7, label="Rocket body (pts)"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", framealpha=0.75,
                    facecolor="black" if dark else "white", edgecolor=fg,
                    handletextpad=0.5, borderpad=0.6)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_color(fg)

    ax.text(0.5, -0.155, r"Cross-sectional pooling of heterogeneous single snapshots "
            r"$-$ the band \& scatter are the target, not the fitted line",
            transform=ax.transAxes, color=fg, fontsize=8, ha="center", va="top",
            alpha=0.7)

    theme_axes(ax, fg, grid)
    save(fig, f"cara_debris_growth_{theme}")


for th in ("light", "dark"):
    make(th)

p_all = strata["debris_all"]["fit"]["p"]
print(f"\ndebris p~={p_all:.2f} (primary {primary['p']}, STM {primary['our_stm_p']}); "
      f"low-perigee p~={strata['debris_lowalt_lt600']['fit']['p']:.2f}, "
      f"high-perigee p~={strata['debris_highalt_ge600']['fit']['p']:.2f}")
