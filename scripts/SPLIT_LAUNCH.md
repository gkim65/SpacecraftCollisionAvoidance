# F3 all-policies sweep — cost-split launch cheatsheet

The full policy grid (mcts + 5 gate baselines × 8 cases × 3 quality × 4 cadence × 5
seeds) is split into **3 sweeps by COST** so the expensive `mcts × 2 h` cells (which
oversubscribed memory on a prior run) run with FEWER agents. `scripts/sweep.yaml` is
the canonical single-sweep reference (all policies, one grid) — the 3 split files are
the one-off execution artifacts for THIS memory-constrained run.

All three log to the SAME project (`spacecraftCA-belief-mcts`) with the SAME metrics
schema → at figure time you query the project (or filter by `config.policy` /
`config.cadence_secondary_h`) and every run comes back together. The 3 sweep IDs are
an execution convenience, NOT a data split.

| file | cells | what | agents |
|------|-------|------|--------|
| `sweep_gates.yaml`   | 4800 | 5 gate variants × cadence {24,8,4,2} — CHEAP (no MCTS) | wide (~78) |
| `sweep_mcts.yaml`    | 360  | mcts × cadence {24,8,4} — moderate                     | wide (~78) |
| `sweep_mcts_2h.yaml` | 120  | mcts × cadence {2} ONLY — HEAVY (minutes each)         | **~28** (memory) |

## 0. Node env (once per node — same as SWEEP_LAUNCH.md items 1–4)

Repo checked out; `julia --project=. -e 'using Pkg; Pkg.instantiate()'`; `uv sync`;
PyCall built against the uv Python
(`PYCALL_JL_RUNTIME_PYTHON=$(uv run which python) PYTHON=$(uv run which python) julia --project=. -e 'using Pkg; Pkg.build("PyCall")'`);
`wandb login` (or `WANDB_API_KEY`).

## 1. Sanity-check ONE live cell FIRST (before fanning out)

Run one cell per policy KIND directly through the runner (wandb online, `--out` keeps
the JSON) and confirm it logs clean — decision sensible, `config.policy` /
`policy_params` correct, `well_formed=true`, the new `sigma_*` fields present.

```bash
cd <repo>
CASE=data/cara_cdms/000040115_conj_000030660_20230721_100115_20230720_061903.cdm

# mcts (the heavy kind) — one 8h cell:
uv run python scripts/wandb_runner.py --case "$CASE" \
  --sensor-quality best --cadence-h 8 --seed 1 \
  --policy-variant mcts --out /tmp/chk_mcts.json

# a gate + the oracle (cheap):
uv run python scripts/wandb_runner.py --case "$CASE" \
  --sensor-quality median --cadence-h 8 --seed 1 \
  --policy-variant delay_12h --out /tmp/chk_delay.json
uv run python scripts/wandb_runner.py --case "$CASE" \
  --sensor-quality best --cadence-h 24 --seed 1 \
  --policy-variant wait_feasibility --out /tmp/chk_oracle.json
```

Check each JSON: `config.policy`, `config.policy_params`, `actual_decision`,
`well_formed`, and that `trace[0].sigma_debris_pos_m` + `b0_sigma_debris_pos_m` exist.
(Offline instead of online: prefix `WANDB_MODE=offline`; `--no-wandb` skips wandb
entirely.)

## 2. Create the 3 sweeps (each prints an ID: entity/project/xxxxxxxx)

```bash
wandb sweep scripts/sweep_gates.yaml      # -> GATES_ID
wandb sweep scripts/sweep_mcts.yaml       # -> MCTS_ID
wandb sweep scripts/sweep_mcts_2h.yaml    # -> MCTS2H_ID
```

## 3. Launch agents — WIDE for the cheap sweeps, ~28 for mcts×2h

```bash
# gates (cheap, wide) — ~78 agents, staggered
for i in $(seq 1 78); do
  tmux new-session -d "cd <repo> && wandb agent kmeans_gsopt/spacecraftCA-belief-mcts/GATES_ID"
  sleep 2
done

# mcts non-2h (moderate, wide) — ~78 agents
for i in $(seq 1 78); do
  tmux new-session -d "cd <repo> && wandb agent kmeans_gsopt/spacecraftCA-belief-mcts/MCTS_ID"
  sleep 2
done

# mcts × 2h (HEAVY) — ONLY ~28 agents (this is the memory-safe cap)
for i in $(seq 1 28); do
  tmux new-session -d "cd <repo> && wandb agent kmeans_gsopt/spacecraftCA-belief-mcts/MCTS2H_ID"
  sleep 2
done
```

Run the gates + mcts-non-2h sweeps first (they drain fast); start the mcts×2h sweep
whenever, but keep it at ~28 agents so it never oversubscribes RAM.

## Notes
- `best` is ~deterministic (extra seeds repeat); the 5 seeds matter for median/worst.
  To trim: set `seed.values: [1]` for best-only would need a separate file — simplest
  is to leave all 5 and accept the repeats, or drop to `[1,2,3]` in each file.
- Adding seeds later: append to a file's `seed.values` and re-run `wandb agent <ID>` —
  the sweep picks up the new unclaimed cells, nothing else changes.
