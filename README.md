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
| 3+ | Σ(τ) table, Kalman tracker, MCTS, chance constraint | not started |

Only Phases 0–2 have reproducible results as of this commit. (Proximity-ops
geometry is deferred — see `notes/TODOS.md`.)

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
  tests/
    test_chan_crossvalidation.jl   Julia-vs-Python Chan cross-validation
    test_conjunction_generator.jl  conjunction geometry + Pc-vs-miss sanity check
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
through `PyCall`. `PyCall` must point at a Python environment with `brahe`
(and `numpy`) installed:

```bash
# confirm PyCall can import brahe
julia --project=. -e 'using PyCall; pyimport("brahe"); println("brahe OK")'
```

If `brahe` is not found, rebuild `PyCall` against the correct Python:

```bash
PYTHON=/path/to/python-with-brahe julia --project=. -e 'using Pkg; Pkg.build("PyCall")'
```

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
  layout differs.)
- `PyCall` able to import `brahe` (see Setup) for the orientation group; that
  group is skipped with a warning if Brahe is unavailable.

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