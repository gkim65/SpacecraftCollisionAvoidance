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
| 5+ | MCTS, chance constraint | not started |

Only Phases 0–4 have reproducible results as of this commit. (Proximity-ops
geometry and the nonlinear SSN observation model are deferred — see
`notes/TODOS.md`.)

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
  tests/
    test_chan_crossvalidation.jl   Julia-vs-Python Chan cross-validation
    test_conjunction_generator.jl  conjunction geometry + Pc-vs-miss sanity check
    test_from_orbits.jl            orbit-first closest-approach round-trip verification
    test_covariance_table.jl       Σ(τ) structure/health/growth + Pc-trust check
    test_belief_tracker.jl         Kalman predict/correct: shrinkage, z-independence, x-val
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