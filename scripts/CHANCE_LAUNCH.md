# Chance-constraint cadence sweeps — launch cheatsheet (2026-08-11)

Two PAIRED sweeps, run SIDE BY SIDE, that differ ONLY in the root decision rule:

| file | root_rule | project | cells | agents |
|------|-----------|---------|-------|--------|
| `sweep_mcts_chance.yaml` | chance (true chance constraint) | `spacecraftCA-mcts-chance` | 480 | **76** |
| `sweep_mcts_legacy.yaml` | legacy (soft-penalty argmax-Qa) | `spacecraftCA-mcts-legacy` | 480 | **50** |

Both: cadence {2,4,8,24} × 8 cases × 3 quality × 5 seeds = 480, p_arrival=1.0,
n_iter=100, :exact, terminal reward, penalize (in-tree). Same seeds ⇒ every chance
cell has a legacy twin at the SAME (case,quality,seed,cadence) → the old-vs-new
decision difference is attributable to the RULE alone. SEPARATE projects (do not mix).

**48 h per-episode timeout** (`--timeout-s 172800`) — generous headroom so a slow
2 h-cadence n_iter=100 cell is never killed mid-episode. NOTE: the timeout is a
per-EPISODE wall-clock ceiling, NOT an OOM guard — it does not extend the total run,
and it does not prevent a node running out of RAM. **OOM is controlled by the AGENT
COUNT per node, not the timeout**: if a node OOMs on the heavy 2 h cells, reduce the
agents on that node (the 76/50 split below assumes many nodes / ample RAM).

## 0. Node env (once per node — same as SWEEP_LAUNCH.md / SPLIT_LAUNCH.md items 1–4)

Repo checked out at commit with the chance constraint (root_rule feature);
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`; `uv sync`; PyCall built against
the uv Python
(`PYCALL_JL_RUNTIME_PYTHON=$(uv run which python) PYTHON=$(uv run which python) julia --project=. -e 'using Pkg; Pkg.build("PyCall")'`);
`wandb login` (or `WANDB_API_KEY`).

## 1. Sanity-check ONE live cell of EACH rule FIRST (before fanning out)

```bash
cd <repo>
CASE=data/cara_cdms/000038771_conj_000030802_20201216_182131_20201215_171306.cdm

# chance rule — best quality, 8 h cadence (should DEFER; validated locally):
uv run python scripts/wandb_runner.py --case "$CASE" \
  --sensor-quality best --cadence-h 8 --seed 1 --out /tmp/chk_chance.json
# add root_rule/alpha to the config the runner writes — via env or wandb.config axis;
# the SWEEP path injects root_rule/alpha_cc from the YAML automatically. For a bare
# single-run check, set them in a CONFIG dict / rely on the module defaults
# (:chance, α=0.05). Confirm actual_decision=defer, well_formed=true,
# config.root_rule=chance, config.alpha_cc=0.05.

# legacy rule — same cell (should MANEUVER at step 1 on 38771 best; the contrast):
#   run the legacy sweep's single cell equivalently and confirm config.root_rule=legacy.
```

(The sweep path reads the whole `wandb.config` — root_rule + alpha_cc flow straight
from the YAML with no Python change. This bare-runner note is only for a manual check.)

## 2. Create BOTH sweeps (each prints an ID: entity/project/xxxxxxxx)

```bash
wandb sweep scripts/sweep_mcts_chance.yaml    # -> CHANCE_ID   (project ...-chance)
wandb sweep scripts/sweep_mcts_legacy.yaml    # -> LEGACY_ID   (project ...-legacy)
```

## 3. Launch agents — 76 for chance, 50 for legacy, side by side

```bash
# CHANCE — 76 agents, staggered (same tmux idiom as SPLIT_LAUNCH.md):
for i in $(seq 1 76); do
  tmux new-session -d "cd <repo> && wandb agent kmeans_gsopt/spacecraftCA-mcts-chance/CHANCE_ID"
  sleep 2
done

# LEGACY — 50 agents, staggered:
for i in $(seq 1 50); do
  tmux new-session -d "cd <repo> && wandb agent kmeans_gsopt/spacecraftCA-mcts-legacy/LEGACY_ID"
  sleep 2
done
```

Each agent pulls unclaimed cells until its sweep is drained. The two sweeps are
independent (different projects/IDs) so they run fully in parallel. Watch memory on
the 2 h-cadence cells (n_iter=100 makes them ~8× heavier than the old n_iter=12 runs);
if a node OOMs, cut the agent count for that node.

## 4. Monitor / stop / re-run

- Progress: the wandb sweep pages for both projects, or `wandb sweep <ID>`.
- Stop early: `wandb sweep --stop <ID>`.
- Each grid cell is DETERMINISTIC per (seed, config), so a stopped/failed cell re-runs
  by simply starting another agent on that sweep — no state to clean up.

## 5. After the runs — export + analysis

`scripts/export_sweep.py` pointed at EACH project (`spacecraftCA-mcts-chance`,
`spacecraftCA-mcts-legacy`), then the paired old-vs-new analysis: same
(case,quality,seed,cadence) cells, compare `actual_decision` / `n_maneuvers` /
`total_dv_mps` / `decision_matches_feasibility` between the two projects. The headline:
chance defers where legacy panic-burns (the 38771 best-quality contrast, at scale).
