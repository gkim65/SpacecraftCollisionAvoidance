#!/usr/bin/env python3
"""export_sweep.py — pull the clean cluster sweep from wandb to a LOCAL table.

Two wandb projects hold the seed-fixed all-policies sweep:

  MCTS runs       : kmeans_gsopt/spacecraftCA-mcts-clean   (policy_variant = mcts)
  Baselines etc.  : kmeans_gsopt/spacecraftCA-belief-mcts  (wait_feasibility oracle
                    + delay_{28,12,6,3}h gates — the F3/A3/A4 comparison policies)

This exporter combines BOTH into ONE light LOCAL scalar table so the post-sweep A
analyses (A1-A4) run offline. It keeps the layout un-clunky by SPLITTING BY WEIGHT
(Grace's call): a small per-run SCALAR table for analysis, and heavy per-step
traces / Sigma-curves / wait-spine curves parked in per-run SIDECAR files pulled
only for the handful of episodes a figure needs.

OUTPUT (figureScripts/data/):
  sweep_all.csv    — one row per FINISHED run, all scalars + config/provenance,
                     tagged `source_project` + `policy_variant`. The analysis input.
  sweep_all.json   — the same rows as a JSON list (structured; nested config kept).
  sweep_coverage.json — per (policy x quality x cadence) cell: finished count,
                     crashed/failed count, expected count, seed set; + the holes.
  traces/<project>/<run_id>.json  — (only with --traces / --traces-only-mcts) the
                     heavy per-run data: the full per-step TRACE (each step's action,
                     Pc, miss, and the real 6x6 ECI Sigma for both objects) + the
                     WAIT-spine feasibility curve, DOWNLOADED from their wandb Table
                     artifacts and stored INLINE ({columns, rows}). wandb logs these as
                     table-file artifacts (the run summary holds only a reference), so
                     --traces fetches the actual table file per run. Not needed by
                     A1-A4 (those are scalar-only); pulled on demand for figures (F4).

FILTERING (CRITICAL): only runs with wandb state == "finished" enter the table.
state in {crashed, failed, killed, running, ...} is EXCLUDED and TALLIED so a
crashed run (no valid pc_at_tca) can never leak into a violation count.

Usage:
  uv run python scripts/export_sweep.py                 # scalar table + coverage
  uv run python scripts/export_sweep.py --traces        # + heavy trace sidecars
  uv run python scripts/export_sweep.py --traces-only-mcts   # sidecars for mcts only
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "figureScripts", "data")

PROJECTS = [
    "kmeans_gsopt/spacecraftCA-mcts-clean",
    "kmeans_gsopt/spacecraftCA-belief-mcts",
]

# The expected sweep grid (scripts/sweep.yaml): 8 debris cases x 3 quality x
# 4 cadence x 5 seed x 6 policy_variant. Coverage is checked against this.
EXPECTED_CASES = [
    "000040059_conj_000035921", "000040115_conj_000030660",
    "000025994_conj_000026132", "000038771_conj_000030802",
    "000029108_conj_000034995", "000037849_conj_000013512",
    "000033591_conj_000042216", "000028654_conj_000041835",
]
EXPECTED_QUALITY = ["best", "median", "worst"]
EXPECTED_CADENCE = [24, 8, 4, 2]
EXPECTED_SEEDS = [1, 2, 3, 4, 5]
EXPECTED_VARIANTS = ["mcts", "wait_feasibility",
                     "delay_28h", "delay_12h", "delay_6h", "delay_3h"]

# Heavy keys that live in the per-run summary but are stripped OUT of the scalar
# table and into the trace sidecars (the weight; not needed by A1-A4).
HEAVY_KEYS = {
    "trace", "wait_spine", "wait_spine_pc", "wait_spine_t_h",
    "b0_sigma_sc_eci_flat", "b0_sigma_debris_eci_flat",
    "b0_sigma_sc_pos_m", "b0_sigma_debris_pos_m",
    "per_maneuver_mitigated", "maneuver_timings_h",
}

# The scalar columns we surface in the CSV (order = readable). Anything else scalar
# in the summary/config is still kept in the JSON rows; the CSV is the analysis view.
CSV_COLUMNS = [
    # provenance / identity
    "run_id", "run_name", "source_project", "state",
    "policy_variant", "case_id", "sensor_quality", "cadence_h", "seed",
    # scenario
    "id1", "id2", "name1", "name2", "sec_class", "horizon_h", "hbr_m",
    "miss_cdm_m", "relative_speed_mps", "pc_cdm", "valid_2d", "tca",
    # config
    "pc_threshold", "delta_v_mps", "reward_mode", "constraint_mode",
    "sigma_mode", "n_iterations", "grid_mode", "grid_steps", "max_steps",
    # core metrics (the A-analysis inputs)
    "pc_at_tca", "peak_pc", "integrated_pc", "resolved_without_maneuver",
    "n_maneuvers", "total_dv_mps", "first_maneuver_h",
    "lead_time_at_first_maneuver_h", "maneuver_mitigated", "maneuver_effective",
    "final_miss_m", "n_steps",
    # decision-vs-feasibility
    "wait_feasible", "wait_durably_safe", "crossing_h",
    "right_call", "actual_decision", "decision_matches_feasibility",
    # well-formedness
    "well_formed", "pc_in_range", "all_finite", "dv_only_on_maneuver",
    # root-Pc sanity (b0 seed): position sigma at detection, and wall time
    "b0_t_remaining_h", "wall_time_s",
]


def case_id_from_path(p):
    """data/.../000040059_conj_000035921_2022..._2022....cdm -> 000040059_conj_000035921"""
    if not p:
        return None
    base = os.path.basename(str(p))
    parts = base.split("_")
    # id1 _conj_ id2 then date fields; the case id is the first 3 underscore tokens.
    if len(parts) >= 3 and parts[1] == "conj":
        return "_".join(parts[:3])
    return base


def get_variant(config, summary):
    v = config.get("policy_variant")
    if v:
        return str(v)
    # fall back to the echoed config sub-dict / policy field
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    return str(config.get("policy") or cfg.get("policy") or "unknown")


def flatten_row(run_id, run_name, project, state, config, summary):
    """Build ONE flat scalar row from a run's config + summary (heavy keys stripped)."""
    cfg_echo = summary.get("config") if isinstance(summary.get("config"), dict) else {}

    def pick(*keys, default=None):
        for k in keys:
            if k in summary and summary[k] is not None:
                return summary[k]
            if k in cfg_echo and cfg_echo[k] is not None:
                return cfg_echo[k]
            if k in config and config[k] is not None:
                return config[k]
        return default

    case_id = case_id_from_path(config.get("case_path") or cfg_echo.get("case_path"))
    # case_path in the echoed config was already reduced to id1_vs_id2 by the metrics
    # dict; prefer the raw config case_path for the canonical id.
    if case_id and "_vs_" in str(case_id):
        case_id = str(case_id).replace("_vs_", "_conj_")

    row = {
        "run_id": run_id,
        "run_name": run_name,
        "source_project": project.split("/")[-1],
        "state": state,
        "policy_variant": get_variant(config, summary),
        "case_id": case_id,
        "sensor_quality": pick("sensor_quality"),
        "cadence_h": config.get("cadence_h",
                                cfg_echo.get("cadence_secondary_h")),
        "seed": pick("seed"),
        "id1": pick("id1"), "id2": pick("id2"),
        "name1": pick("name1"), "name2": pick("name2"),
        "sec_class": pick("sec_class"),
        "horizon_h": pick("horizon_h"),
        "hbr_m": pick("hbr_m"),
        "miss_cdm_m": pick("miss_cdm_m"),
        "relative_speed_mps": pick("relative_speed_mps"),
        "pc_cdm": pick("pc_cdm"),
        "valid_2d": pick("valid_2d"),
        "tca": pick("tca"),
        "pc_threshold": pick("pc_threshold"),
        "delta_v_mps": pick("delta_v_mps"),
        "reward_mode": pick("reward_mode"),
        "constraint_mode": pick("constraint_mode"),
        "sigma_mode": pick("sigma_mode"),
        "n_iterations": pick("n_iterations"),
        "grid_mode": pick("grid_mode"),
        "grid_steps": pick("grid_steps"),
        "max_steps": pick("max_steps"),
        "pc_at_tca": pick("pc_at_tca"),
        "peak_pc": pick("peak_pc"),
        "integrated_pc": pick("integrated_pc"),
        "resolved_without_maneuver": pick("resolved_without_maneuver"),
        "n_maneuvers": pick("n_maneuvers"),
        "total_dv_mps": pick("total_dv_mps"),
        "first_maneuver_h": pick("first_maneuver_h"),
        "lead_time_at_first_maneuver_h": pick("lead_time_at_first_maneuver_h"),
        "maneuver_mitigated": pick("maneuver_mitigated"),
        "maneuver_effective": pick("maneuver_effective"),
        "final_miss_m": pick("final_miss_m"),
        "n_steps": pick("n_steps"),
        "wait_feasible": pick("wait_feasible"),
        "wait_durably_safe": pick("wait_durably_safe"),
        "crossing_h": pick("crossing_h"),
        "right_call": pick("right_call"),
        "actual_decision": pick("actual_decision"),
        "decision_matches_feasibility": pick("decision_matches_feasibility"),
        "well_formed": pick("well_formed"),
        "pc_in_range": pick("pc_in_range"),
        "all_finite": pick("all_finite"),
        "dv_only_on_maneuver": pick("dv_only_on_maneuver"),
        "b0_t_remaining_h": pick("b0_t_remaining_h"),
        "wall_time_s": pick("wall_time_s"),
    }
    return row


def _load_wandb_table(run, summary, key, dl_root):
    """The heavy trace / wait-spine are logged as wandb TABLE ARTIFACTS (the summary
    only holds a {path, _type:'table-file'} reference, not the data). Download the
    referenced table file and return it as {columns, rows} inline JSON. Returns None
    if the key is absent or is not a table reference (older inline runs)."""
    import json as _json
    ref = summary.get(key)
    if ref is None:
        return None
    if isinstance(ref, dict) and ref.get("_type") == "table-file" and ref.get("path"):
        try:
            f = run.file(ref["path"])
            local = f.download(root=dl_root, replace=True)
            data = _json.load(open(local.name))
            return {"columns": data.get("columns"), "rows": data.get("data")}
        except Exception as e:  # network / missing file — record, don't crash the run
            return {"error": f"{type(e).__name__}: {e}", "path": ref.get("path")}
    # already-inline (list/scalar) — pass through
    return ref


def extract_trace(run, summary, dl_root):
    """Pull the heavy per-run bits for a sidecar: the full per-step TRACE table (with
    real 6x6 Sigma per step) + the WAIT-spine feasibility curve, downloaded from their
    wandb Table artifacts and stored INLINE. Plus any remaining scalar heavy keys."""
    out = {}
    out["trace"] = _load_wandb_table(run, summary, "trace", dl_root)
    out["wait_spine"] = _load_wandb_table(run, summary, "wait_spine", dl_root)
    # remaining heavy scalar/vector keys that live directly in summary
    for k in HEAVY_KEYS:
        if k in ("trace", "wait_spine", "wait_spine_pc", "wait_spine_t_h"):
            continue
        if k in summary and not (isinstance(summary[k], dict)
                                 and summary[k].get("_type") == "table-file"):
            out[k] = summary[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", action="store_true",
                    help="also write heavy per-run trace sidecars (all finished runs)")
    ap.add_argument("--traces-only-mcts", action="store_true",
                    help="write trace sidecars only for policy_variant=mcts runs")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: cap runs per project")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=60)

    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    excluded = []                 # (project, run_id, state)
    state_tally = Counter()       # per project x state
    n_traces = 0

    for project in PROJECTS:
        t0 = time.time()
        runs = api.runs(project, per_page=500)
        total = len(runs)
        print(f"[{project}] {total} runs total — streaming config+summary ...",
              flush=True)
        seen = 0
        for r in runs:
            seen += 1
            if args.limit and seen > args.limit:
                break
            state = r.state
            state_tally[(project.split("/")[-1], state)] += 1
            if state != "finished":
                excluded.append((project.split("/")[-1], r.id, state))
                continue
            config = dict(r.config)
            summary = dict(r.summary._json_dict)
            row = flatten_row(r.id, r.name, project, state, config, summary)
            rows.append(row)

            want_trace = args.traces or (
                args.traces_only_mcts and row["policy_variant"] == "mcts")
            if want_trace:
                tdir = os.path.join(DATA_DIR, "traces", project.split("/")[-1])
                os.makedirs(tdir, exist_ok=True)
                dl_root = os.path.join(DATA_DIR, "traces", "_wandb_dl")
                tr = extract_trace(r, summary, dl_root)
                tr["run_id"] = r.id
                tr["policy_variant"] = row["policy_variant"]
                tr["case_id"] = row["case_id"]
                tr["sensor_quality"] = row["sensor_quality"]
                tr["cadence_h"] = row["cadence_h"]
                tr["seed"] = row["seed"]
                with open(os.path.join(tdir, f"{r.id}.json"), "w") as f:
                    json.dump(tr, f)
                n_traces += 1

            if seen % 200 == 0:
                print(f"  ... {seen}/{total} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{project}] done in {time.time()-t0:.0f}s "
              f"(finished so far: {len(rows)})", flush=True)

    # ---- write the scalar table (CSV + JSON) ----
    # --limit is a DEBUG cap; writing partial data over the canonical table would
    # silently truncate it, so a limited run writes to a _debug suffix instead.
    suffix = f"_debug_limit{args.limit}" if args.limit else ""
    csv_path = os.path.join(DATA_DIR, f"sweep_all{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    json_path = os.path.join(DATA_DIR, f"sweep_all{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=0)

    # ---- coverage report: per (variant x quality x cadence) cell ----
    cells = defaultdict(lambda: defaultdict(set))  # cell -> {"finished": {seeds}}
    for row in rows:
        key = (row["policy_variant"], str(row["sensor_quality"]),
               _norm_cadence(row["cadence_h"]))
        try:
            cells[key]["finished"].add(int(row["seed"]))
        except (TypeError, ValueError):
            cells[key]["finished"].add(row["seed"])
    # also per-case within a cell for the seed-set cross-check
    case_cells = defaultdict(set)  # (variant,quality,cadence,case) -> {seeds}
    for row in rows:
        key = (row["policy_variant"], str(row["sensor_quality"]),
               _norm_cadence(row["cadence_h"]), row["case_id"])
        try:
            case_cells[key].add(int(row["seed"]))
        except (TypeError, ValueError):
            case_cells[key].add(row["seed"])

    coverage = build_coverage(cells, case_cells, excluded)
    cov_path = os.path.join(DATA_DIR, f"sweep_coverage{suffix}.json")
    with open(cov_path, "w") as f:
        json.dump(coverage, f, indent=2)

    # ---- console report ----
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"finished runs written : {len(rows)}")
    print(f"excluded (non-finished): {len(excluded)}")
    print("\nstate tally by project:")
    for (proj, st), n in sorted(state_tally.items()):
        mark = "" if st == "finished" else "   <-- EXCLUDED"
        print(f"  {proj:28s} {st:12s} {n:5d}{mark}")
    if n_traces:
        print(f"\ntrace sidecars written: {n_traces}")
    print(f"\nwrote:\n  {csv_path}\n  {json_path}\n  {cov_path}")

    print_coverage_console(coverage)
    return 0


def _norm_cadence(c):
    try:
        return int(round(float(c)))
    except (TypeError, ValueError):
        return c


def build_coverage(cells, case_cells, excluded):
    """Per-cell finished/expected + hole flags, and a per-case seed-set check."""
    per_cell = []
    holes = []
    for variant in EXPECTED_VARIANTS:
        for quality in EXPECTED_QUALITY:
            for cadence in EXPECTED_CADENCE:
                key = (variant, quality, cadence)
                finished_seeds = cells.get(key, {}).get("finished", set())
                # expected = cases x seeds (all should be present per cell)
                expected = len(EXPECTED_CASES) * len(EXPECTED_SEEDS)
                got = 0
                # count finished runs in this cell across cases+seeds
                incomplete_cases = []
                for case in EXPECTED_CASES:
                    ck = (variant, quality, cadence, case)
                    seeds = case_cells.get(ck, set())
                    got += len(seeds)
                    missing = sorted(set(EXPECTED_SEEDS) - {int(s) for s in seeds
                                                            if _is_int(s)})
                    if missing:
                        incomplete_cases.append({"case": case,
                                                 "have_seeds": sorted(str(s) for s in seeds),
                                                 "missing_seeds": missing})
                cell = {
                    "policy_variant": variant, "sensor_quality": quality,
                    "cadence_h": cadence, "finished": got, "expected": expected,
                    "coverage_frac": round(got / expected, 3) if expected else None,
                    "incomplete_cases": incomplete_cases,
                }
                per_cell.append(cell)
                if got == 0:
                    holes.append({**{k: cell[k] for k in
                                     ("policy_variant", "sensor_quality", "cadence_h")},
                                  "severity": "ZERO", "finished": got, "expected": expected})
                elif got < expected:
                    holes.append({**{k: cell[k] for k in
                                     ("policy_variant", "sensor_quality", "cadence_h")},
                                  "severity": "LOW", "finished": got, "expected": expected})
    excl = Counter((p, st) for (p, _rid, st) in excluded)
    return {
        "expected_grid": {
            "cases": len(EXPECTED_CASES), "quality": EXPECTED_QUALITY,
            "cadence_h": EXPECTED_CADENCE, "seeds": EXPECTED_SEEDS,
            "variants": EXPECTED_VARIANTS,
            "total_episodes": (len(EXPECTED_CASES) * len(EXPECTED_QUALITY) *
                               len(EXPECTED_CADENCE) * len(EXPECTED_SEEDS) *
                               len(EXPECTED_VARIANTS)),
        },
        "excluded_by_project_state": {f"{p}/{st}": n for (p, st), n in excl.items()},
        "cells": per_cell,
        "holes": holes,
    }


def _is_int(s):
    try:
        int(s); return True
    except (TypeError, ValueError):
        return False


def print_coverage_console(coverage):
    eg = coverage["expected_grid"]
    print("\n" + "=" * 70)
    print("COVERAGE vs SWEEP GRID")
    print("=" * 70)
    print(f"expected total episodes: {eg['total_episodes']} "
          f"({eg['cases']} cases x {len(eg['quality'])} quality x "
          f"{len(eg['cadence_h'])} cadence x {len(eg['seeds'])} seeds x "
          f"{len(eg['variants'])} variants)")
    holes = coverage["holes"]
    zero = [h for h in holes if h["severity"] == "ZERO"]
    low = [h for h in holes if h["severity"] == "LOW"]
    print(f"\ncells with ZERO finished runs: {len(zero)}")
    for h in zero:
        print(f"  !! ZERO  {h['policy_variant']:16s} {h['sensor_quality']:7s} "
              f"{h['cadence_h']}h  (expected {h['expected']})")
    print(f"\ncells with LOW coverage (0 < got < expected): {len(low)}")
    for h in low:
        print(f"  !  LOW   {h['policy_variant']:16s} {h['sensor_quality']:7s} "
              f"{h['cadence_h']}h  {h['finished']}/{h['expected']}")
    if not zero and not low:
        print("  all cells fully covered.")


if __name__ == "__main__":
    sys.exit(main())
