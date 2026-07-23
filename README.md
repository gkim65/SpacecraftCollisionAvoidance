# Spacecraft Collision Avoidance — chance-constrained belief-space planner

A chance-constrained belief-space MCTS planner for spacecraft conjunction
avoidance, repository for AMOS 2026 paper.

The planner decides, at discrete steps between conjunction detection and the
time of closest approach (TCA), whether to WAIT or MANEUVER, so as to minimize
fuel use and trajectory deviation while keeping the probability of collision
(Pc) below a safety threshold **throughout** the planning window — not just at
TCA. Uncertainty in both objects' states is tracked as a Gaussian belief
`(μ, Σ)` and planned over directly.

See `notes/belief_mcts_architecture.md` for the full design (git-ignored;
distributed separately) and `CONSTANTS.md` for the sourced parameters.

## Status

| Phase | What | State |
|-------|------|-------|
| 0 | Repo setup; port dynamics / observation noise / Brahe setup | done |
| 1 | Chan (1997) Pc + cross-validation vs Python reference | done |
| 2 | Conjunction generator (head-on / cross-track) + Pc sanity check | done |
| 3 | Precomputed Σ(τ) covariance-vs-time-remaining table + Pc-trust check | done |
| 4 | Kalman predict/correct belief tracker (linear-Gaussian) | done |
| 5 | Baseline belief-space MCTS skeleton (miss-distance reward, no Pc yet) | done |
| 6 | Chance constraint: Pc-at-TCA reward + per-step constraint check | done |
| 7+ | Closed-loop episode driver, baselines, experiments | not started |

Only Phases 0–6 (plus the measurement-realism pass) have reproducible results as
of this commit. Phase 6 replaces the Phase-5 miss-distance placeholder with a
**Pc-at-TCA reward** and a **per-step chance-constraint check** (Pc evaluated from
each node's belief at every simulated step, penalized when Pc > threshold). A
node's Pc uses that node's **own accumulated belief covariance** — the Σ the
Kalman predict/correct cycle produced getting there — propagated the rest of the
way to TCA. The observation model is **asymmetric**: the satellite gets a ~10 m
GPS fix about every 2 h and the debris a ~1 km TLE fix about every 8 h, with the
belief predicted (Σ growing) between fixes. (The outer closed-loop episode driver,
proximity-ops geometry, and the nonlinear SSN observation model are deferred —
see `notes/TODOS.md`.)

## Layout

```
src/
  SpacecraftCollisionAvoidance.jl  top-level module includes
  SpacecraftCAPOMDP.jl             POMDP problem definition (state/action/params)
  states.jl  actions.jl            CAState, CAAction
  observations.jl                  Gaussian observation-noise model
  transitions.jl                   true-state dynamics (Brahe propagation)
  rewards.jl                       reward (miss-distance baseline; Pc added later)
  utils/
    genConjunctions.jl             Brahe / PyCall setup, conjunction generation
    computePc.jl                   Pc methods: Chan (1997), Foster, Monte Carlo
    covarianceTable.jl             precomputed Σ(τ) table + Pc-through-Σ(τ) check
    beliefTracker.jl               Kalman predict/correct belief tracker (Phase 4)
    beliefMCTS.jl                  belief-space MCTS: Pc-at-TCA reward + chance constraint (Phase 5/6)
  tests/
    test_chan_crossvalidation.jl   Julia-vs-Python Chan cross-validation
    test_conjunction_generator.jl  conjunction geometry + Pc-vs-miss sanity check
    test_from_orbits.jl            orbit-first closest-approach round-trip verification
    test_covariance_table.jl       Σ(τ) structure/health/growth + Pc-trust check
    test_belief_tracker.jl         Kalman predict/correct: shrinkage, z-independence, x-val
    test_belief_mcts.jl            MCTS mechanics (UCB/backup/widening/determinism) + Pc reward & chance constraint
CONSTANTS.md                       every physical constant + its source
figures/                           generated figures (local; not tracked in git)
```

This project implements a **custom, minimal belief-space MCTS**, not POMCPOW or
any POMDPs.jl-ecosystem solver (see architecture doc §6). The `POMDPs.jl` type
structure (`POMDP{S,A,O}`, `states`/`actions`/`observation`) is kept for clean
problem definition; the POMCPOW solver and particle-filter belief have been
removed.

## Setup

Julia dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Pc computation and conjunction generation call the Python
[`brahe`](https://github.com/duncaneddy/brahe) orbital-mechanics library
through `PyCall`. The Python side is a dedicated, reproducible in-repo
environment managed with [`uv`](https://docs.astral.sh/uv/): `pyproject.toml`
and `uv.lock` (both committed) pin `brahe==1.7.0` and a compatible `numpy` — the
only two packages the Julia code imports via `PyCall`. Build it with:

```bash
uv sync            # creates ./.venv from pyproject.toml + uv.lock
```

Then point `PyCall` at that in-repo `.venv` and rebuild it:

```bash
PYTHON="$PWD/.venv/bin/python" julia --project=. -e 'using Pkg; Pkg.build("PyCall")'

# confirm PyCall imports brahe 1.7.0 from the in-repo venv
julia --project=. -e 'using PyCall; println(pyimport("brahe").__version__)'   # => 1.7.0
```

The `.venv/` directory is git-ignored; `uv sync` rebuilds it identically from
the committed lockfile, so anyone cloning the repo reproduces the exact Python
environment.

## Reproducing results

### Phase 1 — Chan (1997) Pc cross-validation

Confirms the Julia Chan implementation in `src/utils/computePc.jl` agrees with
the original Python reference
(`../ProbofCollision/src/collision/chan1997.py`) on fixed
`(mean, covariance, HBR)` test cases, plus correctness properties and a
Brahe/PyCall matrix-orientation check.

```bash
julia --project=. src/tests/test_chan_crossvalidation.jl
```

The test runs three groups, all of which must pass:

- **Cross-validation** — 5 fixed cases; Julia vs Python Chan agree to well
  within the ~0.2% tolerance established in `ProbofCollision/FINDINGS.md`
  (observed ≤ 3e-13% relative error).
- **Correctness properties** — Pc ∈ [0,1], monotone in miss distance and
  hard-body radius, swap-symmetric, and the isotropic limit reduces to the
  noncentral chi-squared CDF.
- **PyCall / Brahe orientation** — matrices survive the Julia↔numpy round trip
  without transposition, `covariance_rtn` is symmetric, and hand-propagated
  `Φ Σ Φᵀ` matches Brahe's own covariance.

The Python reference values are computed at run time by invoking the
`ProbofCollision` implementation directly (no hard-coded expected numbers), so
this test requires:

- The sibling repo `../ProbofCollision` present, with its Python environment at
  `../ProbofCollision/.venv/` containing `numpy` and the `collision` package.
  (Override the interpreter path in `src/tests/chan_reference.py` /
  `test_chan_crossvalidation.jl`, or the `PROBOFCOLLISION_SRC` env var, if your
  layout differs.) **This is a separate interpreter from the PyCall/brahe
  binding** — it is the Python Chan *reference*, invoked as a subprocess, and
  stays pointed at `ProbofCollision`; it is not the in-repo `.venv`.
- `PyCall` able to import `brahe` from the in-repo `.venv` (see Setup) for the
  orientation group; that group is skipped with a warning if Brahe is
  unavailable.

### Phase 2 — conjunction generator sanity check

Confirms the deterministic conjunction generator in
`src/utils/genConjunctions.jl` (`sc1_eci_at_tca`, `place_debris_at_tca`,
`generate_conjunction_geometry`) places a debris object at TCA with the
requested miss distance and geometry, and that the resulting geometry feeds
sensibly into the Phase 1 Chan Pc.

```bash
julia --project=. src/tests/test_conjunction_generator.jl
```

The test runs three groups, all of which must pass (57 checks total):

- **Geometry construction** — the generated relative RTN state puts a head-on
  miss on the along-track axis and a cross-track miss on the radial axis, with
  the miss magnitude matching the request to sub-meter accuracy after the
  RTN→ECI round trip.
- **Pc vs miss distance** — Chan Pc is highest at the smallest miss, monotone
  non-increasing across a miss sweep, and negligible (<1e-6) at 50 km, for both
  geometries. The absolute peak Pc is modest (~0.07) because the combined
  hard-body radius (~20 m) is small next to the tens-of-meters combined
  position 1σ — the *trend* is the sanity check, not the peak value.
- **Determinism** — regenerating the same `(geometry, miss, v_rel)` returns
  byte-identical states (no hidden RNG).

This test only needs `PyCall` able to import `brahe` (see Setup); it is skipped
with a warning if Brahe is unavailable. It does not need the `ProbofCollision`
Python environment.

### Phase 2 — orbit-first closest-approach verification

Confirms that the geometry-first placements from `generate_conjunction_geometry`
are *dynamically real*. The orbit-first machinery in `src/utils/genConjunctions.jl`
(`_closest_approach`, `_reduce_rel_to_params`, `make_conjunction_from_orbits`)
turns a pair of Keplerian orbits into a dynamically-verified conjunction: it
propagates both objects under the accurate force model (drag/SRP) over a ±600 s
window around TCA and finds the true closest approach via a coarse
closing-speed-adaptive grid plus a golden-section refine (ported from RSSDA,
wired to this repo's accurate dynamics instead of RSSDA's two-body model).

```bash
julia --project=. src/tests/test_from_orbits.jl
```

The test round-trips `generate_conjunction_geometry` → ECI-to-KOE →
`make_conjunction_from_orbits` and checks (38 checks total):

- **Cross-track round trip** — the requested radial standoff is perpendicular to
  the along-track closing velocity, so the placement instant *is* the closest
  approach: the Brahe-measured true miss recovers the requested miss to <1 mm
  and the closest approach lands at TCA (|t_ca| < 1 s).
- **Head-on round trip** — the requested offset is *along-track*, parallel to the
  closing velocity, so the objects fly through: the true closest approach is far
  smaller than the requested offset and occurs ~`offset / v_rel` seconds off TCA.
  This is the correct dynamical answer, and the test asserts it.
- **Feasibility guard** — an obviously hyperbolic (e > 1), non-LEO (apogee above
  the ceiling), or over-eccentric secondary orbit is flagged infeasible with a
  reason string.

The tests use an along-track `v_rel` of 15 m/s (RSSDA's co-orbital default). A
larger along-track `v_rel` (e.g. the Phase 2 sanity-sweep's 200 m/s) drops the
debris perigee below the atmosphere and the guard correctly rejects it — a true
fast LEO crossing is a *cross-track* velocity, not an along-track one. Same
Brahe-only dependency as the other tests; skipped with a warning if Brahe is
unavailable.

### Phase 3 — precomputed Σ(τ) covariance table + Pc-trust check

`src/utils/covarianceTable.jl` builds the offline covariance-vs-time-remaining
table the belief-space planner looks up during search. `build_covariance_table`
anchors each object's initial covariance (`P0_sc` / `P0_debris`) at TCA with
Brahe's STM machinery enabled and propagates it forward under the accurate force
model (drag/SRP), reading `Σ(τ)` back at τ = `dt`, 2·`dt`, …, 24 hr remaining
until TCA — for both the spacecraft and the debris, in both RTN and ECI frames.
It reads Brahe's own propagated covariance (`covariance_rtn` / `covariance_gcrf`)
directly; Phase 1 proved that equals the hand-computed `Φ Σ₀ Φᵀ` to 0.0 relative
diff. The τ grid's step size is a **swappable `dt` argument** (defaults to
`pomdp.dt`), so the 30-min and 1-hr grids can both be built and compared.
`pc_through_table` then evaluates Chan Pc through the propagated `Σ(τ)` for a
fixed conjunction — the Pc-trust check folded in from Phase 2.

> **Validity:** this table is a pure lookup-by-time-remaining only under the
> **noiseless-maneuver** assumption (architecture doc §5/§8). Once maneuver
> execution uncertainty is added (Phase 8), Σ depends on the action sequence and
> the table must be tracked per-node instead.

```bash
julia --project=. src/tests/test_covariance_table.jl
```

The test builds both grids (1-hr = 24 steps, 30-min = 48 steps) and checks (976
checks total):

- **Structure + health** — every `Σ(τ)`, both objects, both frames, is 6×6,
  exactly symmetric, and positive-definite (Cholesky); `all_pd` holds and no
  forced symmetrization is needed (Brahe output is symmetric to ~1e-11 rel).
- **Growth** — along-track (RTN transverse) 1σ grows monotonically and stays
  bounded (debris: ~1.56 km @1h → ~30.2 km @24h). The power-law exponent is ≈1
  (σ ∝ τ, variance ∝ τ²), **not** the ~1.5 cubic-in-variance one might expect:
  the τ³ signature comes from semi-major-axis (energy) error dominating, whereas
  the current placeholder `P0` (per-axis position + velocity diagonal, no SMA
  term) is *velocity*-error dominated — which is the correct growth for this
  covariance. The exponent should move toward 1.5 once `P0` carries a real
  SMA-uncertainty term (the Phase 4 SSN-noise decision).
- **Pc-trust** — Chan Pc through `Σ(τ)` for a feasible co-orbital cross-track
  conjunction (miss 500 m, `v_rel` 15 m/s) is finite, in `[0,1]`, and evolves
  smoothly, rising sensibly toward TCA (9.4e-5 @24h → 1.65e-3 @1h) as the
  ballooning covariance shrinks back toward the still-nonzero miss.
- **Grid equivalence** — at every shared (whole-hour) τ the 30-min and 1-hr
  tables agree, confirming Σ(τ) is a pure function of time-remaining,
  independent of the step size used to build it.

Same Brahe-only dependency as the other tests.

### Phase 4 — Kalman predict/correct belief tracker

`src/utils/beliefTracker.jl` is the linear-Gaussian belief tracker the planner
carries through search. A `Belief` is **two independent 6×6 sub-beliefs** (one
spacecraft, one debris) plus time-remaining `t` — the objects are physically
independent, so a coupled 12×12 would only carry structural zeros. `predict`
propagates each `(μ, Σ)` one `dt` step (μ via the `transitions.jl` dynamics with a
+Δv·v̂ kick on the spacecraft if MANEUVER; `Σ⁻ = Φ Σ Φᵀ` via Brahe, no process
noise — so WAIT and MANEUVER give the same `Σ⁻`, differing only in μ).
`correct_linear` (the runtime path, H = I₆) and `correct_brahe` (a brahe
`ExtendedKalmanFilter` cross-validation oracle) both apply the Kalman update
against a genuine sampled observation from `sample_observation`.

> **Load-bearing property:** in the linear-Gaussian update, `Σ⁺ = (I−KH)Σ⁻`
> depends only on H and R, **not** on the sampled observation value — only μ⁺
> depends on z. This is exactly what makes the Phase 3 `Σ(τ)` lookup valid. It
> holds only for the linear update under noiseless maneuvers (a nonlinear SSN
> model, deferred to Phase 8.5, would break it).

```bash
julia --project=. src/tests/test_belief_tracker.jl
```

99 checks in 5 groups: correction shrinks Σ (`Σ⁻ − Σ⁺ ⪰ 0`); Σ⁺ is bitwise
identical across 40 random observation draws while μ⁺ varies (the z-independence
property); predict-then-correct tracks truth within 3σ with Σ staying sym + PD;
the hand-rolled and brahe filters agree to <1e-10; and predict's `Σ⁻` at τ = dt
matches the Phase 3 `build_covariance_table` entry to <1e-9.

Same Brahe-only dependency as the other tests.

### Phase 5 — belief-space MCTS skeleton (search mechanics)

`src/utils/beliefMCTS.jl` is the hand-rolled belief-space MCTS (architecture doc
§4/§6) — **not** POMCPOW. A `BeliefNode` carries a Phase-4 `Belief`, the sampled
true `CAState`, visit/value bookkeeping (`N`, per-action `Na`/`Qa`), and
observation-children per action. One simulation (`simulate!`) does the §4 loop:
select an action by **UCB** (`Q + c·√(ln N / n_a)`), expand via
`POMDPs.transition` (true state) + `predict` + (on a measurement step)
`sample_observation` + `correct_linear_*` (belief), recurse, and back up a
running average. Corrections follow an **asymmetric measurement cadence** (below,
Phase "measurement realism"): the belief is *predicted* every step but only
*corrected* for an object when that object's fix is due. Stochastic
observations are handled by **double progressive widening** — a node adds a new
observation-child while `n_children ≤ k·n_a^α`, else reuses one at random. The
UCB/MaxUCB and widening rules and their constants (`c=1`, `k=10`, `α=0.5`,
`tree_queries=1000`) are **borrowed from POMCPOW's published defaults**; the
solver itself is not adopted, so the belief stays visible to the reward at every
step (the whole point of §6).

> **Phase 5 built the mechanics on a miss-distance placeholder reward; Phase 6
> replaced that reward wholesale** (below). The mechanics (UCB, backup, widening,
> determinism, tree health) are unchanged and still tested. `dt` is a swappable
> planner argument (defaults `pomdp.dt`) — both the 1-hr and 30-min grids run
> end-to-end; no default is forced.

### Phase 6 — Pc-at-TCA reward + per-step chance constraint

Phase 6 replaces the Phase-5 miss-distance reward with a **Pc-at-TCA** reward and
a **per-step chance-constraint check** (architecture §4 steps 5–6, §7):

- `node_pc_at_tca(pomdp, node)` computes Pc-at-TCA from a node's belief via the
  Phase-1 Chan method. **Uncertainty model:** a node's Pc uses that node's OWN
  **accumulated belief Σ** — the covariance the Kalman predict/correct cycle
  actually produced getting to the node (predict-grown, measurement-shrunk) —
  propagated the rest of the way to TCA (`Σ_at_TCA = Φ(now→TCA)·Σ_belief·Φᵀ`, read
  from Brahe's `covariance_gcrf`). This is "Pc at TCA if we stop measuring now and
  coast" — the covariance the planner actually holds, not a fresh `P0`. The node's
  belief **mean** is propagated to TCA (that is what a maneuver moves). The Phase 3
  Σ(τ) table is *not* reused as a lookup here (it is anchored the other way in time
  and lands on a different orbital phase — the R/N covariance breathes once per
  orbit; per-node direct propagation is exact).
- `step_reward` = `−pc_weight·Pc − maneuver_cost` (per burn) `− pc_penalty` when
  `Pc > pomdp.pc_threshold`. The constraint is checked on **every** simulated
  step, at every depth — not once per node.
- `leaf_value` reuses the *same* Pc-at-TCA for both a true leaf (reached TCA) and
  a computational-budget cutoff (§7 — one consistent risk metric).
- `constraint_mode` (planner arg): `:penalize` (default — penalty but keep
  expanding, so a violate-then-recover wait-and-measure branch is not amputated),
  `:terminate` (also mark the branch terminal), `:off` (no-constraint baseline).
  Each `BeliefNode` records its `pc` / `violated` for ablation.
- `sigma_mode` (planner arg, **efficiency pass**): `:fast` (default) precomputes
  the **debris** Σ-at-TCA **per tree depth** once per `plan` (`build_sigma_tca_table`)
  and looks it up per node, propagating only the debris *mean* + the full satellite
  belief per node; `:exact` is the fully-per-node path kept as the correctness
  oracle. This is valid because the debris Σ-at-a-given-depth is **bitwise
  branch-invariant** (`predict`'s `Σ⁻=ΦΣΦᵀ` is maneuver-independent and the cadence
  correction's `Σ⁺=(I−K)Σ⁻` is observation-independent), so `:fast` gives a Pc
  **identical** to `:exact` (verified 0.0 relative diff) at roughly **1.85× less
  cost per node**. The satellite Σ is *not* tabled — it is propagated exactly per
  node (it is only weakly branch-invariant). **`:fast` is valid only under
  noiseless maneuvers; Phase 8's maneuver process noise makes even the debris Σ
  action-dependent, so Phase 8 must switch to `:exact`.**

```bash
julia --project=. src/tests/test_belief_mcts.jl
```

100 checks in 12 groups: the five Phase-5 mechanics groups above (**UCB
selection**, **backup arithmetic**, **progressive widening**, **tree health**,
**determinism**); five Phase-6 groups — **Pc-at-TCA from a node** (finite, in
[0,1], matches a direct `chan_pc` on the mean + the node's accumulated Σ propagated
to TCA, and diverges from a fresh-P0 at a deep node); **Pc grows the belief Σ over
the coast** (a node hours out has an appreciable Pc, a node at TCA is orders of
magnitude smaller, noting the once-per-orbit ripple); **per-step constraint**
(penalty fires above threshold, not below; `:off` disables it; `:terminate` marks
a violating child terminal); **leaf == cutoff** (identical Pc-at-TCA value);
**end-to-end + ablation** — on a real cross-track conjunction the planner prefers
the maneuver that lowers Pc-at-TCA, and `:terminate` prunes more violating nodes
than `:penalize`; the **asymmetric measurement cadence** group (below); and the
**fast Σ path** group — the per-depth debris Σ table reproduces the exact per-node
debris Σ-at-TCA, and `sigma_mode = :fast` yields a Pc (and an end-to-end action +
tree) identical to `:exact`.

Same Brahe-only dependency as the other tests. Note: Pc evaluation is the search
bottleneck (Brahe numerical propagations). The `:fast` `sigma_mode` (default,
above) cuts it to ~1.85× less per node by tabling the branch-invariant debris Σ,
but a `plan` over a multi-hour window is still minutes: the next efficiency lever
is **multiprocess root-parallel search** (one Brahe interpreter per worker
process — PyCall's GIL rules out in-process threading), deferred to its own task.

### Asymmetric measurement realism — sourced noise + per-object cadence

The observation model is strongly **asymmetric**, matching operational reality:
the satellite is an own-asset **GPS** fix (`σ_sc = 10 m`, a conservative bound;
Hauschild & Montenbruck 2021) corrected about every **2 h**, while the debris is
an SSN **TLE** (`σ_debris = 1 km` at the OD epoch; Flohrer 2008 / ESA SDC5)
corrected only about every **8 h**. Both are noisy full-state observations
(linear `H = I` — a TLE and a GPS solution are each a fitted state estimate, so
this is faithful, not a shortcut; EKF/UKF are deferred, see the design doc).

The MCTS applies the cadence in `expand_child`: the belief is **predicted every
step** (Σ grows), but an object is **corrected only when its fix is due**. Each
`BeliefNode` carries `since_sc` / `since_debris` (seconds since that object's last
fix); a correction (`correct_linear_sc` / `correct_linear_debris`) fires for an
object once its timer crosses the object's cadence, then resets. Between TLEs the
debris covariance grows freely — over an 8 h coast the debris position 1σ ramps
from ~50 m to ~5 km before the next TLE snaps it back — making the debris the
dominant, evolving uncertainty. `cadence_sc`, `cadence_debris`, and the
`correct_at_root` phase toggle are all swappable planner/POMDP arguments (like
`dt`), ready for the cadence and σ-magnitude ablations. Test group 11 checks that
a predict-only step advances the timers, a fix step resets the timer and shrinks
that object's Σ relative to the same step's predict-only counterpart (at
`σ_debris = 1 km` a single TLE fix is weak, so it need not pull Σ below its prior
value), and the `correct_at_root` phase behavior.