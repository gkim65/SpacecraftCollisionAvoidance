#!/usr/bin/env python3
"""wandb_runner.py — the thin wandb client for the debris-cohort episode sweep.

One invocation = ONE receding-horizon episode. The runner:

  1. wandb.init(project, entity, config=<episode config>) — a sweep agent injects
     the swept axes (case / sensor_quality / cadence / seed) into wandb.config.
  2. writes that resolved config as a tiny Julia file and shells out to
     scripts/run_episode_entry.jl (the scheduler-agnostic "run one config" seam),
     which runs run_episode_metrics(cfg) and emits the flat metrics dict as JSON.
  3. logs the metrics: the SWEEP-COMPARABLE scalars go to wandb.summary (so the
     sweep dashboard groups/sorts on them) AND the FULL metrics dict is preserved
     two ways — every scalar as a summary field, and the nested trace / wait-spine
     as wandb.Tables — so nothing is lost for later analysis.
  4. finishes. FAIL-SOFT: a wandb auth/network failure must NOT lose the episode —
     the metrics JSON is always written to disk first (source of truth), wandb is
     logging on top (mirrors the RSSDA benchmark's pattern).

Both a LOCAL single run and a cluster SWEEP AGENT use this same entrypoint:

  # single run (wandb offline is fine for a local smoke test):
  WANDB_MODE=offline uv run python scripts/wandb_runner.py \
      --case data/cara_cdms/000040115_conj_000030660_...cdm \
      --sensor-quality best --cadence-h 8 --seed 1

  # sweep agent (config comes from the sweep controller):
  uv run python scripts/wandb_runner.py            # reads wandb.config

The Julia side needs the in-repo Julia project + PyCall/brahe env; this Python
side needs `wandb`. See scripts/SWEEP_LAUNCH.md for the cluster env + agent launch.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY_JL = os.path.join("scripts", "run_episode_entry.jl")

# wandb target. Mirrors the RSSDA benchmark (entity kmeans_gsopt); project is
# per-experiment. Override with --project / --entity or WANDB_PROJECT / WANDB_ENTITY.
DEFAULT_PROJECT = "spacecraftCA-belief-mcts"
DEFAULT_ENTITY = "kmeans_gsopt"

# The SWEEP-COMPARABLE summary scalars promoted to top-level summary fields (what a
# sweep dashboard groups/sorts across). The FULL scalar set is also written to
# summary below — this list is only the "headline" ordering the sweep cares about.
HEADLINE_KEYS = [
    "actual_decision", "right_call", "decision_matches_feasibility",
    "resolved_without_maneuver", "wait_feasible", "wait_durably_safe",
    "pc_at_tca", "peak_pc", "integrated_pc", "crossing_h",
    "total_dv_mps", "n_maneuvers", "first_maneuver_h",
    "lead_time_at_first_maneuver_h", "maneuver_mitigated", "maneuver_effective",
    "final_miss_m", "n_steps", "wall_time_s", "well_formed",
    "sec_class", "horizon_h", "miss_cdm_m", "relative_speed_mps", "pc_cdm", "valid_2d",
]

# Nested arrays/lists logged as wandb Tables (queryable, not clutter) rather than
# raw summary fields. Everything else scalar-ish goes to summary.
TABLE_KEYS = {"trace", "wait_spine_pc", "wait_spine_t_h", "per_maneuver_mitigated",
              "maneuver_timings_h"}


def _julia_config_literal(cfg):
    """Emit the resolved config as a Julia file defining CONFIG::Dict{String,Any}.

    Avoids a JSON parser on the Julia side (JSON.jl is only a transitive dep in this
    repo). Only flat scalars/strings/None appear in an episode config, so the mapping
    is direct: None->nothing, bool->true/false, str->quoted, num->literal.
    """
    def lit(v):
        if v is None:
            return "nothing"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        s = str(v).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    body = ",\n    ".join(f'"{k}" => {lit(v)}' for k, v in cfg.items())
    return "CONFIG = Dict{String,Any}(\n    " + body + ",\n)\n"


def run_one(cfg, julia="julia", timeout_s=3600):
    """Run ONE episode via the Julia entry script. Returns the metrics dict.

    Writes the config as a Julia file, calls run_episode_entry.jl, reads the metrics
    JSON back. Raises on a Julia failure (the caller decides fail-soft vs. abort).
    """
    with tempfile.TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "config.jl")
        out_path = os.path.join(td, "metrics.json")
        with open(cfg_path, "w") as f:
            f.write(_julia_config_literal(cfg))
        cmd = [julia, f"--project={REPO}", os.path.join(REPO, ENTRY_JL),
               cfg_path, out_path]
        print(f"[runner] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO, timeout=timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"Julia entry exited {proc.returncode} "
                               f"(after {time.time()-t0:.0f}s) — see the log above.")
        if not os.path.exists(out_path):
            raise RuntimeError("Julia entry produced no metrics JSON.")
        with open(out_path) as f:
            return json.load(f)


def _clean(v):
    """JSON NaN/Inf -> None so wandb.summary stays clean (json.load already turns
    the Julia `null` we write into None; this guards any float NaN that slips in)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def log_to_wandb(run, metrics):
    """Log the metrics dict: headline + all scalars to summary; nested arrays as
    Tables. Preserves the FULL dict (scalars on summary, arrays as Tables)."""
    import wandb

    # 1. headline scalars first (stable ordering for the sweep dashboard).
    for k in HEADLINE_KEYS:
        if k in metrics:
            run.summary[k] = _clean(metrics[k])

    # 2. every remaining scalar-ish field to summary too (nothing lost). Skip the
    #    Table keys and the nested `config` (already in wandb.config) / `trace`.
    for k, v in metrics.items():
        if k in TABLE_KEYS or k in ("config", "trace"):
            continue
        if isinstance(v, (dict, list)):
            continue  # non-scalar, non-table nesting: keep out of summary
        run.summary[k] = _clean(v)

    # 3. the per-step trace as a Table (queryable; the plotting source).
    trace = metrics.get("trace", [])
    if trace:
        cols = list(trace[0].keys())
        t = wandb.Table(columns=cols)
        for r in trace:
            t.add_data(*[_clean(r.get(c)) for c in cols])
        run.log({"trace": t})

    # 4. the WAIT-spine feasibility curve as a Table (t_h, pc) — the F1/F2 curve.
    tsp = metrics.get("wait_spine_t_h", [])
    psp = metrics.get("wait_spine_pc", [])
    if tsp and psp and len(tsp) == len(psp):
        st = wandb.Table(columns=["t_remaining_h", "wait_spine_pc"])
        for th, pc in zip(tsp, psp):
            st.add_data(_clean(th), _clean(pc))
        run.log({"wait_spine": st})

    # 5. maneuver timings + per-maneuver mitigation as a small Table.
    mt = metrics.get("maneuver_timings_h", [])
    pm = metrics.get("per_maneuver_mitigated", [])
    if mt:
        mtab = wandb.Table(columns=["maneuver_idx", "lead_time_h", "mitigated"])
        for i, th in enumerate(mt):
            mtab.add_data(i, _clean(th), pm[i] if i < len(pm) else None)
        run.log({"maneuvers": mtab})


def _build_cfg_from_args(args):
    """Assemble an episode config dict from CLI args (single-run / sweep-agent fallback
    when no sweep controller injected wandb.config). Only sets keys the user gave;
    the Julia side fills the rest from episode_config defaults."""
    cfg = {"case_path": args.case}
    if args.sensor_quality is not None:
        cfg["sensor_quality"] = args.sensor_quality
    if args.cadence_h is not None:
        cfg["cadence_secondary"] = float(args.cadence_h) * 3600.0
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.n_iterations is not None:
        cfg["n_iterations"] = int(args.n_iterations)
    cfg["sigma_mode"] = args.sigma_mode
    cfg["reward_mode"] = args.reward_mode
    cfg["constraint_mode"] = args.constraint_mode
    cfg["grid_mode"] = args.grid_mode
    if getattr(args, "p_arrival", None) is not None:
        cfg["p_arrival"] = float(args.p_arrival)
    # ROOT chance-constraint knobs. The SWEEP path injects these from wandb.config
    # (the YAML axis) with no code change; these are for the single-run / sanity path.
    if getattr(args, "root_rule", None) is not None:
        cfg["root_rule"] = args.root_rule
    if getattr(args, "alpha_cc", None) is not None:
        cfg["alpha_cc"] = float(args.alpha_cc)
    # DECISION POLICY (F3): a single flat variant string ("mcts" / "wait_feasibility" /
    # "delay_<N>h"). run_episode_entry.jl maps it to (policy, policy_params). Default
    # "mcts" keeps the pre-baseline behavior. (A sweep passes this via wandb.config;
    # this is the single-run / sanity-check path.)
    if getattr(args, "policy_variant", None) is not None:
        cfg["policy_variant"] = args.policy_variant
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", help="path to a CDM (repo-relative or absolute)")
    ap.add_argument("--sensor-quality", dest="sensor_quality",
                    choices=["best", "median", "worst"])
    ap.add_argument("--cadence-h", dest="cadence_h", type=float,
                    help="secondary measurement cadence in HOURS (e.g. 2,4,8,24)")
    ap.add_argument("--root-rule", dest="root_rule", default=None,
                    help="root decision rule: 'chance' (true chance constraint) or "
                         "'legacy' (soft-penalty argmax-Qa). Sweep sets it via the YAML.")
    ap.add_argument("--alpha-cc", dest="alpha_cc", type=float, default=None,
                    help="root chance-constraint risk level alpha (default 0.05).")
    ap.add_argument("--p-arrival", dest="p_arrival", type=float,
                    help="P(a scheduled DEBRIS measurement arrives); 1.0 (default) = "
                         "guaranteed, byte-identical to pre-arrival runs. <1 makes each "
                         "due debris fix a Bernoulli arrival (else predict-only that step).")
    ap.add_argument("--policy-variant", dest="policy_variant",
                    help="F3 decision policy: mcts | wait_feasibility | delay_<N>h "
                         "(default mcts). Mapped to (policy, policy_params) by "
                         "run_episode_entry.jl:policy_variant_spec.")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-iterations", dest="n_iterations", type=int, default=12)
    ap.add_argument("--sigma-mode", dest="sigma_mode", default="exact")
    ap.add_argument("--reward-mode", dest="reward_mode", default="terminal")
    ap.add_argument("--constraint-mode", dest="constraint_mode", default="penalize")
    ap.add_argument("--grid-mode", dest="grid_mode", default="measurement",
                    choices=["measurement", "adaptive"])
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    ap.add_argument("--julia", default=os.environ.get("JULIA_BIN", "julia"))
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=3600)
    ap.add_argument("--no-wandb", action="store_true",
                    help="skip wandb entirely; just run the episode + write the JSON")
    ap.add_argument("--out", help="also copy the metrics JSON here (default: none)")
    args = ap.parse_args()

    # ---- wandb.init: a sweep controller injects the swept axes into wandb.config ----
    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=args.project, entity=args.entity)
        except Exception as e:  # auth/network/import — fail soft, still run the episode
            print(f"[runner] wandb.init FAILED ({type(e).__name__}: {e}). Running the "
                  f"episode WITHOUT wandb; the metrics JSON is still written. Fix auth "
                  f"with `wandb login` or set WANDB_MODE=offline.", flush=True)
            run = None

    # ---- resolve the config: wandb.config (sweep) wins, else CLI args ----
    if run is not None and len(dict(run.config)) > 0:
        cfg = dict(run.config)
        # a sweep may pass cadence in hours for readability; normalize to seconds.
        if "cadence_h" in cfg and "cadence_secondary" not in cfg:
            cfg["cadence_secondary"] = float(cfg.pop("cadence_h")) * 3600.0
        if "case_path" not in cfg and "case" in cfg:
            cfg["case_path"] = cfg.pop("case")
    else:
        if not args.case:
            ap.error("--case is required for a single run (no sweep config present).")
        cfg = _build_cfg_from_args(args)
        if run is not None:
            run.config.update(cfg, allow_val_change=True)

    # ---- run the episode (source of truth is the metrics JSON) ----
    metrics = run_one(cfg, julia=args.julia, timeout_s=args.timeout_s)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f)
        print(f"[runner] wrote metrics -> {args.out}", flush=True)

    # ---- log on top of the JSON (fail-soft) ----
    if run is not None:
        try:
            log_to_wandb(run, metrics)
        except Exception as e:
            print(f"[runner] wandb logging FAILED ({type(e).__name__}: {e}). The episode "
                  f"+ metrics JSON are intact; only the wandb upload was skipped.", flush=True)
        finally:
            run.finish()

    d = metrics.get("actual_decision")
    print(f"[runner] done: decision={d} pc_at_tca={metrics.get('pc_at_tca'):.3e} "
          f"dv={metrics.get('total_dv_mps')} wall={metrics.get('wall_time_s')}s", flush=True)


if __name__ == "__main__":
    main()
