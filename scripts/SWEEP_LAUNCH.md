# Debris-cohort tracking-fidelity sweep — launch guide

The paper's **F2** figure: does a debris conjunction resolve by WAITing (no burn),
and how does that depend on **sensor quality × measurement cadence**, across the
8-case debris cohort. One wandb run = one receding-horizon episode; a wandb **sweep**
fans the `case × quality × cadence × seed` grid across N parallel `wandb agent`s.

## What the harness is

- `scripts/run_episode_entry.jl` — scheduler-agnostic "run ONE config" seam. Reads a
  config (a Julia `CONFIG` dict), runs `run_episode_metrics(cfg)`, writes the flat
  metrics dict as JSON. Works with or without wandb.
- `scripts/wandb_runner.py` — the thin wandb client. `wandb.init` → writes the config
  → shells out to the Julia entry → reads the JSON → logs (summary scalars + Tables)
  → `finish`. **Fail-soft**: the metrics JSON is the source of truth; a wandb
  auth/network failure never loses an episode.
- `scripts/sweep.yaml` — the grid (8 cases × 3 quality × 4 cadence × 5 seeds = 480).

## Env each agent node needs

An agent process runs `uv run python scripts/wandb_runner.py`, which launches Julia,
which calls brahe through PyCall. So a node needs **all** of:

1. **The repo** checked out (the agent's CWD must be the repo root, or set it in the
   agent's launch script).
2. **The Julia project** instantiated: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
   (once per node). Julia must be on `PATH` (or set `JULIA_BIN`).
3. **The uv/brahe env**: `uv sync` (installs brahe + numpy + wandb). PyCall must point
   at this env's Python — if PyCall was built against a different Python, rebuild it:
   `PYCALL_JL_RUNTIME_PYTHON=$(uv run which python) julia --project=. -e 'using Pkg; Pkg.build("PyCall")'`.
4. **wandb auth**: `WANDB_API_KEY` in the environment (from https://wandb.ai/authorize),
   or `wandb login` once on the node. Set `WANDB_ENTITY` / `WANDB_PROJECT` to override
   the sweep.yaml defaults (`kmeans_gsopt` / `spacecraftCA-belief-mcts`).

> **Cluster-env unknowns (flag for Grace):** this guide assumes the nodes can see the
> repo, run the in-repo uv/brahe env, and have wandb auth. If any of those is NOT true
> on the actual cluster, the *entrypoint* is still scheduler-agnostic — you can drive
> `run_episode_entry.jl` directly (no wandb) and collect the JSONs — but the wandb
> sweep specifically needs items 1–4 above on every agent node.

## Smoke test ONE run locally first (no cluster, wandb offline)

```bash
cd <repo>
WANDB_MODE=offline uv run python scripts/wandb_runner.py \
  --case data/cara_cdms/000040115_conj_000030660_20230721_100115_20230720_061903.cdm \
  --sensor-quality best --cadence-h 8 --seed 1 --out /tmp/metrics_smoke.json
```

Or with no wandb at all (just the episode + JSON):

```bash
uv run python scripts/wandb_runner.py --no-wandb \
  --case <cdm> --sensor-quality best --cadence-h 8 --seed 1 --out /tmp/m.json
```

## Launch the sweep

```bash
# 1. create the sweep (prints a SWEEP_ID like entity/project/xxxxxxxx):
wandb sweep scripts/sweep.yaml

# 2. start N agents — EACH agent pulls unclaimed cells until the grid is exhausted.
#    Run one agent per core/slot you want busy. On one node:
for i in $(seq 1 8); do wandb agent <SWEEP_ID> & done
#    On a SLURM-style cluster, submit N array tasks each running:
#      uv run wandb agent <SWEEP_ID>
```

## How many agents / parallelism (Grace's question)

Unlike the RSSDA benchmark (whose RS-SDA*/RS-MAA* solver held large T/O/R matrices in
RAM, capping parallelism), **each episode here is a belief-MCTS rollout with a small
footprint** — belief nodes + one brahe propagator per process, on the order of
**~0.5–1 GB per agent** (dominated by the Julia + PyCall/brahe runtime, not the
search). There is **no shared solver and no giant matrix**, so agents don't contend for
memory the way RSSDA did.

Rule of thumb per node: **N_agents ≈ min(cores − 1, RAM_GB / 1)**. E.g. a 16-core /
32 GB node comfortably runs ~15 agents. Throughput ≈ `N_agents × (1 episode / ~150 s
at :exact, 8 h cadence)`; finer cadences (2/4 h) are more decision steps → slower per
episode (the 4 h case was ~7 min). The full 480-cell grid at, say, 15 agents ≈ a few
hours wall-clock.

**Check progress every ~hour** (Grace) from the wandb sweep page, or:
`wandb sweep --stop <SWEEP_ID>` to halt early. Because the grid is deterministic per
cell, a stopped/failed cell can be re-run by just starting another agent.

## Adding more seeds later (if 5 looks stochastic)

Append seeds in `scripts/sweep.yaml` (`seed.values: [1,2,3,4,5,6,7,8]`), then
`wandb sweep scripts/sweep.yaml` again (new sweep id) OR create a seed-only sweep for
the new seeds and run agents. New cells are just more grid points; nothing else
changes. (best is ~deterministic — extra seeds there mostly repeat; the payoff is in
median/worst where the belief-drift decision distribution lives.)

## Fixed vs swept (so the runs are comparable)

Swept: `case_path`, `sensor_quality`, `cadence_h`, `seed`. Fixed for the whole sweep:
`:exact` sigma, `terminal` reward, `penalize` constraint, `measurement`-schedule grid,
`n_iterations=12`. :fast is intentionally NOT used (it crashes on the grid path;
deferred, non-blocking — the cluster episode-parallelism is the real speed lever, not
:fast's ~1.85×).
```
