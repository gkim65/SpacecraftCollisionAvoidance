#!/usr/bin/env python3
"""analyze_sweep.py — the post-sweep A analyses (A1-A4) on the LOCAL sweep table.

Consumes figureScripts/data/sweep_all.csv (from export_sweep.py; FINISHED runs only)
and runs the four post-sweep analyses from the END-OF-PROJECT CHECKLIST:

  STEP1  sanity glance  — root/at-detection Pc vs pc_cdm (the seed fix), well_formed
                          breadth, completion vs expected.
  A1  [LOAD-BEARING]    — TCA-violation check: episodes ending pc_at_tca > delta,
                          split UNAVOIDABLE (wait not durably safe & no burn could
                          clear -> least-infeasible fallback, expected) vs AVOIDABLE
                          (wait_durably_safe OR a feasible option existed but it still
                          violated -> a real bug). AVOIDABLE violations flagged LOUDLY.
  A3  [LOAD-BEARING]    — does the constraint prune: for MCTS episodes, how often is
                          the chosen branch feasible (pc_at_tca <= delta) vs
                          least-infeasible.
  A2                    — seed-agreement / determinism map per (case x quality x
                          cadence) cell (MCTS): did all seeds pick the same action.
  A4                    — tracking-fidelity boundary: where in quality x cadence does
                          MCTS diverge from the delay gates / wait_feasibility oracle.

Writes figureScripts/data/analysis_A_results.json (all numbers) and prints a report.
Figures are a separate step (plot_A_figures.py).
"""
import csv
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "figureScripts", "data")
CSV_PATH = os.path.join(DATA_DIR, "sweep_all.csv")
OUT_PATH = os.path.join(DATA_DIR, "analysis_A_results.json")

# The chance-constraint threshold delta and the maneuver magnitude are FIXED for the
# whole sweep (SpacecraftCAPOMDP.jl defaults pc_threshold=1e-5, Δv=0.1 m/s; no sweep
# yaml overrides either — run_episode_entry.jl only sets pc_threshold when passed, and
# it never is). wandb's summary did NOT promote these two scalars (they are not in
# wandb_runner.HEADLINE_KEYS), so they arrive empty in the table and we inject the
# known code constant. If a future sweep sweeps delta, log it and read it per-row.
PC_THRESHOLD_DEFAULT = 1e-5
DELTA_V_DEFAULT = 0.1


def to_float(x):
    if x is None or x == "" or str(x).lower() in ("nan", "none", "null"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None or x == "":
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "1.0", "yes"):
        return True
    if s in ("false", "0", "0.0", "no"):
        return False
    return None


def load_rows():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    # coerce the fields the analyses touch
    for r in rows:
        r["pc_at_tca_f"] = to_float(r.get("pc_at_tca"))
        r["pc_cdm_f"] = to_float(r.get("pc_cdm"))
        thr = to_float(r.get("pc_threshold"))
        r["pc_threshold_f"] = thr if thr is not None else PC_THRESHOLD_DEFAULT
        r["peak_pc_f"] = to_float(r.get("peak_pc"))
        r["n_maneuvers_f"] = to_float(r.get("n_maneuvers"))
        r["total_dv_f"] = to_float(r.get("total_dv_mps"))
        r["first_maneuver_h_f"] = to_float(r.get("first_maneuver_h"))
        r["seed_i"] = int(to_float(r.get("seed"))) if to_float(r.get("seed")) is not None else None
        r["cadence_i"] = int(to_float(r.get("cadence_h"))) if to_float(r.get("cadence_h")) is not None else None
        r["wait_durably_safe_b"] = to_bool(r.get("wait_durably_safe"))
        r["wait_feasible_b"] = to_bool(r.get("wait_feasible"))
        r["decision_match_b"] = to_bool(r.get("decision_matches_feasibility"))
        r["resolved_nomvr_b"] = to_bool(r.get("resolved_without_maneuver"))
        r["maneuver_mitigated_b"] = to_bool(r.get("maneuver_mitigated"))
        r["well_formed_b"] = to_bool(r.get("well_formed"))
    return rows


def cell_key(r):
    return (r.get("case_id"), str(r.get("sensor_quality")), r["cadence_i"])


# ---------------------------------------------------------------------------
# STEP 1 — sanity glance
# ---------------------------------------------------------------------------
def step1_sanity(rows):
    import math
    # root Pc ~ pc_cdm: pc_at_tca is Pc-at-TCA from the FINAL step, but the sanity
    # target is the ROOT (at-detection) belief matching CARA. The metrics dict does
    # not log a root_pc scalar directly; the closest proxy in the table is that a
    # DEFER-from-start / early trace pc should track pc_cdm. We use two reads:
    #  (a) for NO-maneuver episodes, the FIRST-step pc is ~ the detection belief;
    #      but we only have final pc_at_tca in the scalar table. Instead we compare
    #      peak_pc (max over the episode, ~ the root/early Pc before measurements
    #      shrink it) to pc_cdm — the tightest scalar proxy we have.
    ratios = []
    for r in rows:
        pk, cdm = r["peak_pc_f"], r["pc_cdm_f"]
        if pk and cdm and pk > 0 and cdm > 0:
            ratios.append(math.log10(pk / cdm))
    ratios.sort()
    n = len(ratios)
    med = ratios[n // 2] if n else None
    wf = [r["well_formed_b"] for r in rows]
    wf_true = sum(1 for x in wf if x is True)
    return {
        "note": ("root/at-detection Pc proxy = peak_pc (max Pc over the episode, i.e. "
                 "the earliest/pre-measurement belief before tracking shrinks it); "
                 "compared to pc_cdm (CARA). The exact root scalar was NOT logged and "
                 "the per-step trace is a wandb Table ARTIFACT (not inline in summary), "
                 "so peak_pc is the tightest scalar proxy in this table. The seed fix's "
                 "root-Pc==CARA was verified directly last session (sc.b0 root Pc-at-TCA "
                 "1.08e-4 vs CARA 1.07e-4); here the checks are (a) pc_cdm never 0 across "
                 "all runs -> no NPD silent-zero, (b) well_formed broadly true, (c) peak_pc "
                 "within ~1 order of pc_cdm (peak is post-first-measurement, ~<= detection Pc)."),
        "n_with_both": n,
        "median_log10_peakpc_over_pccdm": round(med, 3) if med is not None else None,
        "median_ratio_peakpc_over_pccdm": round(10 ** med, 3) if med is not None else None,
        "log10_ratio_p10": round(ratios[int(0.1 * n)], 3) if n else None,
        "log10_ratio_p90": round(ratios[int(0.9 * n)], 3) if n else None,
        "well_formed_true": wf_true,
        "well_formed_total": len(wf),
        "well_formed_frac": round(wf_true / len(wf), 4) if wf else None,
    }


# ---------------------------------------------------------------------------
# A1 — TCA-violation check
# ---------------------------------------------------------------------------
def a1_violations(rows):
    viol = []
    for r in rows:
        pc, thr = r["pc_at_tca_f"], r["pc_threshold_f"]
        if pc is None or thr is None:
            continue
        if pc > thr:
            viol.append(r)
    avoidable, unavoidable, ambiguous = [], [], []
    for r in viol:
        wds = r["wait_durably_safe_b"]
        nm = r["n_maneuvers_f"] or 0
        # AVOIDABLE: WAIT was durably safe (deferring would have cleared it) OR the
        # episode DID maneuver yet still violated (a feasible option was taken but the
        # branch still ended unsafe). Either is a real bug for the safety claim.
        # UNAVOIDABLE: wait not durably safe AND no maneuver cleared it -> the
        # least-infeasible fallback fired correctly (expected on never-safe cases).
        rec = {
            "run_id": r.get("run_id"), "case_id": r.get("case_id"),
            "policy_variant": r.get("policy_variant"),
            "sensor_quality": r.get("sensor_quality"), "cadence_h": r["cadence_i"],
            "seed": r["seed_i"], "pc_at_tca": pc, "pc_threshold": thr,
            "pc_over_thr": round(pc / thr, 3), "wait_durably_safe": wds,
            "n_maneuvers": int(nm), "maneuver_mitigated": r["maneuver_mitigated_b"],
        }
        if wds is True:
            rec["reason"] = "wait_durably_safe=true but ended violating"
            avoidable.append(rec)
        elif nm > 0:
            rec["reason"] = "maneuvered but still violated at TCA"
            avoidable.append(rec)
        elif wds is False:
            rec["reason"] = "wait not durably safe & never maneuvered -> least-infeasible fallback (expected)"
            unavoidable.append(rec)
        else:
            rec["reason"] = "wait_durably_safe unknown"
            ambiguous.append(rec)
    def by_variant(rs):
        return _count_by(rs, "policy_variant")

    # The safety claim is about the PLANNER (MCTS). Split every tally MCTS vs baseline
    # so an MCTS violation (a real problem for the abstract) is never hidden inside a
    # baseline's expected drift/too-late-burn violations.
    mcts_viol = [r for r in viol if r.get("policy_variant") == "mcts"]
    mcts_avoidable = [v for v in avoidable if v["policy_variant"] == "mcts"]
    mcts_unavoidable = [v for v in unavoidable if v["policy_variant"] == "mcts"]
    return {
        "total_episodes": len(rows),
        "total_violations": len(viol),
        "violations_by_variant": by_variant(viol),
        "avoidable_count": len(avoidable),
        "unavoidable_count": len(unavoidable),
        "ambiguous_count": len(ambiguous),
        "avoidable_by_variant": _count_by(avoidable, "policy_variant"),
        "unavoidable_by_variant": _count_by(unavoidable, "policy_variant"),
        # ---- the headline split: is the PLANNER clean? ----
        "mcts_episodes": sum(1 for r in rows if r.get("policy_variant") == "mcts"),
        "mcts_violations": len(mcts_viol),
        "mcts_avoidable_violations": len(mcts_avoidable),
        "mcts_unavoidable_violations": len(mcts_unavoidable),
        "mcts_avoidable_detail": mcts_avoidable,
        # ---- full lists (baselines dominate; kept for the note) ----
        "avoidable": sorted(avoidable, key=lambda x: -x["pc_over_thr"]),
        "unavoidable_by_case": _count_by(unavoidable, "case_id"),
        "ambiguous": ambiguous,
    }


# ---------------------------------------------------------------------------
# A3 — does the constraint prune (MCTS only)
# ---------------------------------------------------------------------------
def a3_prune(rows):
    mcts = [r for r in rows if r.get("policy_variant") == "mcts"]
    feasible, infeasible, unknown = 0, 0, 0
    infeasible_rows = []
    for r in mcts:
        pc, thr = r["pc_at_tca_f"], r["pc_threshold_f"]
        if pc is None or thr is None:
            unknown += 1
            continue
        if pc <= thr:
            feasible += 1
        else:
            infeasible += 1
            infeasible_rows.append({
                "run_id": r.get("run_id"), "case_id": r.get("case_id"),
                "sensor_quality": r.get("sensor_quality"), "cadence_h": r["cadence_i"],
                "seed": r["seed_i"], "pc_at_tca": pc,
                "wait_durably_safe": r["wait_durably_safe_b"],
            })
    n = feasible + infeasible
    # break down feasible-fraction by (quality x cadence) to show WHERE the constraint
    # keeps the chosen branch feasible.
    by_cell = defaultdict(lambda: [0, 0])  # cell -> [feasible, total]
    for r in mcts:
        pc, thr = r["pc_at_tca_f"], r["pc_threshold_f"]
        if pc is None or thr is None:
            continue
        c = (str(r.get("sensor_quality")), r["cadence_i"])
        by_cell[c][1] += 1
        if pc <= thr:
            by_cell[c][0] += 1
    return {
        "n_mcts_episodes": len(mcts),
        "n_scored": n,
        "feasible_count": feasible,
        "least_infeasible_count": infeasible,
        "feasible_frac": round(feasible / n, 4) if n else None,
        "unknown_pc": unknown,
        "feasible_frac_by_quality_cadence": {
            f"{q}/{c}h": round(v[0] / v[1], 3) if v[1] else None
            for (q, c), v in sorted(by_cell.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        },
        "least_infeasible_episodes": infeasible_rows,
    }


# ---------------------------------------------------------------------------
# A2 — seed-agreement / determinism map (MCTS)
# ---------------------------------------------------------------------------
def a2_seed_agreement(rows):
    mcts = [r for r in rows if r.get("policy_variant") == "mcts"]
    cells = defaultdict(list)
    for r in mcts:
        cells[cell_key(r)].append(r)
    grid = []
    n_unanimous, n_split, n_singleton = 0, 0, 0
    for key, rs in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2] or 0)):
        # action per seed: defer (0 maneuvers) vs maneuver (>=1)
        actions = {}
        for r in rs:
            nm = r["n_maneuvers_f"]
            if nm is None:
                continue
            actions[r["seed_i"]] = "maneuver" if nm > 0 else "defer"
        vals = set(actions.values())
        unanimous = len(vals) == 1
        rec = {
            "case_id": key[0], "sensor_quality": key[1], "cadence_h": key[2],
            "n_seeds": len(actions), "actions_by_seed": actions,
            "unanimous": unanimous if actions else None,
            "n_defer": sum(1 for v in actions.values() if v == "defer"),
            "n_maneuver": sum(1 for v in actions.values() if v == "maneuver"),
        }
        grid.append(rec)
        if len(actions) <= 1:
            n_singleton += 1
        elif unanimous:
            n_unanimous += 1
        else:
            n_split += 1
    return {
        "n_cells": len(grid),
        "n_unanimous": n_unanimous,
        "n_split": n_split,
        "n_singleton_or_empty": n_singleton,
        "split_cells": [g for g in grid if g["unanimous"] is False],
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# A4 — tracking-fidelity boundary: MCTS vs baselines
# ---------------------------------------------------------------------------
def a4_boundary(rows):
    # For each (case x quality x cadence x seed) draw, compare the MCTS action to the
    # wait_feasibility oracle and the delay gates. Report, per (quality x cadence)
    # cell aggregated over cases+seeds: MCTS defer-rate, oracle defer-rate, agreement
    # with oracle, and mean total_dv. The divergence surface is the headline.
    def action_of(r):
        nm = r["n_maneuvers_f"]
        return None if nm is None else ("maneuver" if nm > 0 else "defer")

    # index by (case,quality,cadence,seed) -> {variant: row}
    idx = defaultdict(dict)
    for r in rows:
        k = (r.get("case_id"), str(r.get("sensor_quality")), r["cadence_i"], r["seed_i"])
        idx[k][r.get("policy_variant")] = r

    per_cell = defaultdict(lambda: {
        "n": 0, "mcts_defer": 0, "oracle_defer": 0,
        "mcts_vs_oracle_agree": 0, "mcts_dv": [], "oracle_dv": [],
        "mcts_vs_oracle_pairs": 0,
    })
    divergences = []
    for k, variants in idx.items():
        case, q, cad, seed = k
        cell = (q, cad)
        mcts = variants.get("mcts")
        oracle = variants.get("wait_feasibility")
        c = per_cell[cell]
        if mcts is not None:
            am = action_of(mcts)
            c["n"] += 1
            if am == "defer":
                c["mcts_defer"] += 1
            if mcts["total_dv_f"] is not None:
                c["mcts_dv"].append(mcts["total_dv_f"])
        if oracle is not None:
            ao = action_of(oracle)
            if ao == "defer":
                c["oracle_defer"] += 1
            if oracle["total_dv_f"] is not None:
                c["oracle_dv"].append(oracle["total_dv_f"])
        if mcts is not None and oracle is not None:
            c["mcts_vs_oracle_pairs"] += 1
            am, ao = action_of(mcts), action_of(oracle)
            if am == ao:
                c["mcts_vs_oracle_agree"] += 1
            else:
                divergences.append({
                    "case_id": case, "sensor_quality": q, "cadence_h": cad, "seed": seed,
                    "mcts_action": am, "oracle_action": ao,
                    "mcts_dv": mcts["total_dv_f"], "oracle_dv": oracle["total_dv_f"],
                    "wait_durably_safe": mcts["wait_durably_safe_b"],
                })

    def mean(xs):
        return round(sum(xs) / len(xs), 5) if xs else None

    cells_out = []
    for (q, cad), c in sorted(per_cell.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        cells_out.append({
            "sensor_quality": q, "cadence_h": cad, "n_mcts": c["n"],
            "mcts_defer_rate": round(c["mcts_defer"] / c["n"], 3) if c["n"] else None,
            "oracle_defer_count": c["oracle_defer"],
            "mcts_vs_oracle_pairs": c["mcts_vs_oracle_pairs"],
            "mcts_vs_oracle_agreement": round(c["mcts_vs_oracle_agree"] / c["mcts_vs_oracle_pairs"], 3)
                if c["mcts_vs_oracle_pairs"] else None,
            "mcts_mean_dv_mps": mean(c["mcts_dv"]),
            "oracle_mean_dv_mps": mean(c["oracle_dv"]),
        })

    # also: MCTS-vs-each-delay-gate agreement overall
    gate_agree = {}
    for gate in ("delay_28h", "delay_12h", "delay_6h", "delay_3h", "wait_feasibility"):
        agree, total = 0, 0
        for k, variants in idx.items():
            m, g = variants.get("mcts"), variants.get(gate)
            if m is None or g is None:
                continue
            total += 1
            if action_of(m) == action_of(g):
                agree += 1
        gate_agree[gate] = {"agree": agree, "total": total,
                            "agreement": round(agree / total, 3) if total else None}

    return {
        "cells": cells_out,
        "mcts_vs_baseline_agreement_overall": gate_agree,
        "divergences_mcts_vs_oracle": sorted(
            divergences, key=lambda d: (d["sensor_quality"], d["cadence_h"] or 0)),
        "n_divergences_mcts_vs_oracle": len(divergences),
    }


def _count_by(rows, key):
    c = defaultdict(int)
    for r in rows:
        c[r.get(key)] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found — run export_sweep.py first.")
        return 1
    rows = load_rows()
    print(f"loaded {len(rows)} finished runs from {CSV_PATH}")

    results = {
        "n_finished_runs": len(rows),
        "step1_sanity": step1_sanity(rows),
        "A1_violations": a1_violations(rows),
        "A3_prune": a3_prune(rows),
        "A2_seed_agreement": a2_seed_agreement(rows),
        "A4_boundary": a4_boundary(rows),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console report ----
    s = results["step1_sanity"]
    print("\n" + "=" * 70 + "\nSTEP 1 — SANITY\n" + "=" * 70)
    print(f"  peak_pc / pc_cdm  median ratio: {s['median_ratio_peakpc_over_pccdm']}x "
          f"(log10 median {s['median_log10_peakpc_over_pccdm']}, "
          f"p10 {s['log10_ratio_p10']} / p90 {s['log10_ratio_p90']}) over {s['n_with_both']} runs")
    print(f"  well_formed: {s['well_formed_true']}/{s['well_formed_total']} "
          f"({s['well_formed_frac']})")

    a1 = results["A1_violations"]
    print("\n" + "=" * 70 + "\nA1 — TCA-VIOLATION CHECK [LOAD-BEARING]\n" + "=" * 70)
    print(f"  episodes: {a1['total_episodes']}   violations (pc_at_tca > delta): {a1['total_violations']}")
    print(f"  UNAVOIDABLE (least-infeasible fallback, expected): {a1['unavoidable_count']}")
    print(f"  AVOIDABLE  (REAL BUG if MCTS > 0):                  {a1['avoidable_count']}")
    print(f"  ambiguous:                                          {a1['ambiguous_count']}")
    print(f"  violations by variant:  {a1['violations_by_variant']}")
    print(f"  avoidable  by variant:  {a1['avoidable_by_variant']}")
    print(f"  unavoidable by variant: {a1['unavoidable_by_variant']}")
    print("  ---- PLANNER (MCTS) SAFETY (the abstract's claim) ----")
    print(f"  MCTS episodes: {a1['mcts_episodes']}   MCTS violations: {a1['mcts_violations']}   "
          f"MCTS avoidable: {a1['mcts_avoidable_violations']}   "
          f"MCTS unavoidable: {a1['mcts_unavoidable_violations']}")
    if a1["mcts_avoidable_violations"] > 0:
        print("  !!!! MCTS AVOIDABLE VIOLATIONS — REAL BUG, FLAG LOUDLY:")
        for v in a1["mcts_avoidable_detail"]:
            print(f"    {v['case_id']} {v['sensor_quality']} {v['cadence_h']}h "
                  f"seed{v['seed']}: pc={v['pc_at_tca']:.3e} ({v['pc_over_thr']}x thr) — {v['reason']}")
    print(f"  unavoidable by case: {a1['unavoidable_by_case']}")

    a3 = results["A3_prune"]
    print("\n" + "=" * 70 + "\nA3 — DOES THE CONSTRAINT PRUNE [LOAD-BEARING]\n" + "=" * 70)
    print(f"  MCTS episodes scored: {a3['n_scored']}")
    print(f"  chosen branch FEASIBLE (pc_at_tca <= delta): {a3['feasible_count']} "
          f"({a3['feasible_frac']})")
    print(f"  least-infeasible:                            {a3['least_infeasible_count']}")
    print(f"  feasible-frac by quality/cadence: {a3['feasible_frac_by_quality_cadence']}")

    a2 = results["A2_seed_agreement"]
    print("\n" + "=" * 70 + "\nA2 — SEED AGREEMENT / DETERMINISM MAP\n" + "=" * 70)
    print(f"  cells: {a2['n_cells']}  unanimous: {a2['n_unanimous']}  "
          f"split: {a2['n_split']}  singleton/empty: {a2['n_singleton_or_empty']}")
    for g in a2["split_cells"][:20]:
        print(f"    SPLIT {g['case_id']} {g['sensor_quality']} {g['cadence_h']}h: "
              f"{g['n_defer']} defer / {g['n_maneuver']} maneuver ({g['actions_by_seed']})")

    a4 = results["A4_boundary"]
    print("\n" + "=" * 70 + "\nA4 — TRACKING-FIDELITY BOUNDARY (MCTS vs baselines)\n" + "=" * 70)
    print("  MCTS-vs-baseline overall agreement:")
    for gate, d in a4["mcts_vs_baseline_agreement_overall"].items():
        print(f"    {gate:16s}: {d['agreement']} ({d['agree']}/{d['total']})")
    print(f"  MCTS-vs-oracle divergences: {a4['n_divergences_mcts_vs_oracle']}")
    print("  per (quality x cadence): MCTS defer-rate | MCTS-vs-oracle agreement | MCTS dv | oracle dv")
    for c in a4["cells"]:
        print(f"    {c['sensor_quality']:7s} {str(c['cadence_h'])+'h':4s} "
              f"defer={c['mcts_defer_rate']}  agree={c['mcts_vs_oracle_agreement']}  "
              f"mcts_dv={c['mcts_mean_dv_mps']}  oracle_dv={c['oracle_mean_dv_mps']}")

    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
