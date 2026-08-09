# =========================================================================
# beliefTracker.jl — Phase 4: Kalman predict/correct belief tracker
#
# The belief-space planner (architecture doc §4) carries a Gaussian belief
# b = (μ, Σ) and updates it each step with a Kalman predict/correct cycle:
#
#   PREDICT   μ⁻ = f(μ, a)          (dynamics, maneuver Δv if MANEUVER)
#             Σ⁻ = Φ Σ Φᵀ           (STM propagation; no process noise yet)
#   CORRECT   K  = Σ⁻Hᵀ(HΣ⁻Hᵀ + R)⁻¹
#             μ⁺ = μ⁻ + K(z − Hμ⁻)  (depends on the SAMPLED observation z)
#             Σ⁺ = (I − KH)Σ⁻       (does NOT depend on z)
#
# TWO OBJECTS, TWO SEPARATE BELIEFS. The spacecraft and debris are physically
# independent objects, observed independently, with uncorrelated estimation
# errors — there is no dynamical coupling and no shared measurement. Every
# operation here (block-diagonal STM, block-diagonal H = I and
# R = diag([σ_sc²×6, σ_debris²×6]) from observations.jl) preserves that
# block-diagonal structure, so a monolithic 12×12 joint belief would carry only
# structural zeros off-diagonal. We therefore track two independent 6×6
# sub-beliefs — which is also what Phase 3 (build_covariance_table returns
# Σ_sc / Σ_debris separately) and chan_pc (takes two states + two covariances)
# already want.
#
# ─────────────────────────────────────────────────────────────────────────
# THE LOAD-BEARING PROPERTY (architecture doc §4/§5, and why it matters):
#
# In the linear-Gaussian update below, H and R are FIXED (H = I₆, R diagonal),
# so Σ⁺ = (I − KH)Σ⁻ depends only on H and R — NOT on the sampled observation
# value z. Only μ⁺ depends on z. This is a mathematical property of the linear
# Kalman filter, not a modeling assumption. It is EXACTLY what makes the Phase 3
# precomputed Σ(τ) lookup valid (§5): under noiseless maneuvers, Σ's evolution
# is a pure function of time-remaining, independent of which observations were
# sampled or which branch of the search tree you are in.
#
# THIS HOLDS ONLY FOR THE LINEAR-GAUSSIAN UPDATE UNDER NOISELESS MANEUVERS
# (§4/§5/§8). If a nonlinear observation model (e.g. az/el/range ground tracking
# via brahe's EKF/UKF with AzElRange/AzEl measurement models — deferred, see
# TODOS Phase 8.5) is adopted, the filter relinearizes per step, Σ⁺ becomes
# dependent on the state estimate x̂, and the Phase 3 global lookup no longer
# holds. Likewise, once maneuver execution noise Q(a) is added (Phase 8) the
# predict step gains Σ⁻ = Φ Σ Φᵀ + Q(a) and Σ depends on the action sequence.
# Neither is done here.
#
# ─────────────────────────────────────────────────────────────────────────
# TWO PARALLEL CORRECT IMPLEMENTATIONS (cross-validated in the unit tests):
#   (a) correct_linear  — hand-rolled linear-Gaussian update, the reference
#       implementation the architecture doc calls "newly built."
#   (b) correct_brahe   — brahe 1.7.0 ExtendedKalmanFilter with the LINEAR
#       InertialStateMeasurementModel (full-state, H = I₆). At H = I this is the
#       SAME filter as (a); wiring it confirms we are not reinventing a wheel
#       brahe already turns, and gives a Chan-style Julia-vs-brahe cross-check.
# The nonlinear SSN az/el/range path is intentionally NOT built here.
#
# WHICH PATH TO USE (decided 2026-07-22): use the hand-rolled `correct_linear`
# as the RUNTIME path — at H = I it is numerically identical to brahe's EKF
# (unit tests: <1e-10 rel), but it is microseconds vs. brahe's per-call
# propagator setup + PyCall marshalling, and it is transparent/controllable for
# the MCTS hot loop (Phase 5-6). `correct_brahe` earns its keep as a
# CROSS-VALIDATION ORACLE in the tests, not as the runtime path.
#   ⮕ FLIP TO BRAHE WHEN GOING NONLINEAR: once the SSN az/el/range observation
#     model is adopted (Phase 8.5), brahe's EKF/UKF does the per-step Jacobian /
#     sigma-point machinery for you — hand-rolling that is more work and more
#     error-prone, so at that point prefer brahe's filter over a hand-roll.
#
# TIMESTEP: predict uses `dt` (defaults to pomdp.dt) — swappable, never
# hard-coded, per the Phase 3 convention. Σ⁻ for a given τ = dt must be
# consistent with build_covariance_table at that τ (cross-checked in tests).
# =========================================================================

using LinearAlgebra
using Distributions
using Random

"""
    ObjBelief

Gaussian belief for a single 6-D ECI object: mean `μ` (m, m/s) and covariance
`Σ` (6×6). Σ is kept symmetric and positive-definite.
"""
struct ObjBelief
    μ::Vector{Float64}
    Σ::Matrix{Float64}
end

"""
    Belief

Joint belief for the conjunction, stored as TWO INDEPENDENT 6-D sub-beliefs —
one for the spacecraft, one for the debris (see module header for why they are
not a coupled 12×12). `t` is time remaining to TCA (s), carried alongside so a
belief knows the τ at which its Σ should match the Phase 3 table.
"""
struct Belief
    sc::ObjBelief
    debris::ObjBelief
    t::Float64
end

"""
    belief_from_pomdp(pomdp, sc_eci, debris_eci, t) -> Belief

Construct an initial belief anchored at the given true ECI states with the
POMDP's initial covariances P0_sc / P0_debris, at time-remaining `t` (s).
"""
function belief_from_pomdp(pomdp::SpacecraftCAPOMDP,
                           sc_eci::AbstractVector,
                           debris_eci::AbstractVector,
                           t::Real)
    return Belief(ObjBelief(collect(float.(sc_eci)), Matrix{Float64}(pomdp.P0_sc)),
                  ObjBelief(collect(float.(debris_eci)), Matrix{Float64}(pomdp.P0_debris)),
                  Float64(t))
end

# --- numerical hygiene: keep Σ exactly symmetric (roundoff snaps drift) -----
_sym(Σ::AbstractMatrix) = (Matrix(Σ) .+ transpose(Matrix(Σ))) ./ 2

# =========================================================================
# SNC PROCESS NOISE (Phase 8 growth fix, 2026-08-08).
#
# Our predict step was Σ⁻ = Φ Σ Φᵀ (Q = 0), a documented under-grower: real
# along-track OD covariance grows ~τ² (drag-driven), a noiseless linear STM only
# reaches ~τ^1.4 (notes/cara_covariance_growth_realism_findings.md §6). The fix
# is a tuned State-Noise-Compensation term (NASA CARA's own remedy; Zaidi &
# Hejduk 2016): Σ⁻ = Φ Σ Φᵀ + Q, with the standard SNC discrete matrix
#
#   Q_axis(dt) = q · [[dt³/3, dt²/2],
#                     [dt²/2, dt   ]]   on each (position, velocity) axis pair,
#
# where q (m²/s³) is a constant PSD and the ADDED matrix scales with dt (NOT a
# flat Q₀ — a flat Q₀ is step-size-inconsistent). The velocity variance
# accumulates q·dt each step; that accumulation is what bends the growth up
# toward τ² (verified in the growth study's 1-D prototype). q is ANISOTROPIC in
# RTN (along-track T dominant, small radial/cross floor) to match the measured
# RTN shape, keyed per-object per-conjunction via pomdp.q_rtn_{sc,debris}.
# =========================================================================

"""
    _rtn_rotation(state) -> 3×3

RTN→ECI rotation whose COLUMNS are the RTN basis in ECI (R radial, T in-track,
N cross-track), so `v_eci = R * v_rtn`. Self-contained copy of the cdmParser
helper so beliefTracker (included before cdmParser) does not depend on it.
"""
function _rtn_rotation(state::AbstractVector)
    r = state[1:3]; v = state[4:6]
    r_hat = r ./ norm(r)
    n_hat = cross(r, v) ./ norm(cross(r, v))
    t_hat = cross(n_hat, r_hat)
    return hcat(r_hat, t_hat, n_hat)   # columns R, T, N
end

"""
    snc_q_eci(q_rtn, dt, state) -> 6×6

Discrete SNC process-noise covariance for one object over a step `dt` (s),
built per RTN axis from the PSD 3-vector `q_rtn = [q_R, q_T, q_N]` (m²/s³) and
rotated into ECI at `state`. Each axis contributes the standard block
`q·[[dt³/3, dt²/2],[dt²/2, dt]]` coupling that axis's position and velocity.
`q_rtn == zeros(3)` ⇒ the zero matrix (Q = 0, the pre-Phase-8 behavior).
"""
function snc_q_eci(q_rtn::AbstractVector, dt::Real, state::AbstractVector)
    dt = Float64(dt)
    Q_rtn = zeros(6, 6)
    for i in 1:3
        q = Float64(q_rtn[i])
        q == 0.0 && continue
        Q_rtn[i, i]         = q * dt^3 / 3
        Q_rtn[i, 3 + i]     = q * dt^2 / 2
        Q_rtn[3 + i, i]     = q * dt^2 / 2
        Q_rtn[3 + i, 3 + i] = q * dt
    end
    all(Q_rtn .== 0.0) && return Q_rtn
    R = _rtn_rotation(state)
    A = zeros(6, 6); A[1:3, 1:3] = R; A[4:6, 4:6] = R
    return _sym(A * Q_rtn * transpose(A))
end

# =========================================================================
# PREDICT — propagate (μ, Σ) forward one dt step given an action.
#
# μ:  reuse the transitions.jl path — apply the maneuver Δv to the spacecraft
#     velocity (along +v̂) if MANEUVER, then propagate both objects forward by dt
#     under the accurate force model.
# Σ:  Σ⁻ = Φ Σ Φᵀ, read back via brahe's covariance_gcrf (Phase 1 proved this
#     equals the hand-computed Φ Σ Φᵀ to 0.0 rel diff). NO process noise —
#     WAIT and MANEUVER propagate Σ IDENTICALLY (noiseless maneuver, §8): the
#     maneuver only moves μ.
# =========================================================================

"""
    _predict_object(pomdp, μ, Σ, objParams, epoch_current, dt; Δv_applied) -> (μ⁻, Σ⁻)

Propagate one object's (μ, Σ) forward by `dt` (s) from `epoch_current`. `Σ⁻` is
read from brahe's propagated ECI covariance (== Φ Σ Φᵀ). If `Δv_applied` is a
6-vector it is added to the mean state before propagation (the maneuver on the
spacecraft); Σ is propagated unchanged either way (noiseless maneuver).
"""
function _predict_object(pomdp::SpacecraftCAPOMDP,
                         μ::AbstractVector, Σ::AbstractMatrix,
                         objParams::AbstractVector,
                         epoch_current, dt::Real;
                         Δv_applied::Union{AbstractVector,Nothing} = nothing,
                         q_rtn::AbstractVector = zeros(3))
    μ0 = collect(float.(μ))
    if Δv_applied !== nothing
        μ0 = μ0 .+ collect(float.(Δv_applied))
    end
    epoch_current_tuple = epoch_to_tuple(epoch_current)
    prop, ep0 = eci2orb_brahe(μ0, epoch_current_tuple, objParams,
                              pomdp.forceModel; initial_covariance = Matrix(Σ))
    ep_next = ep0 + Float64(dt)
    prop.propagate_to(ep_next)
    μ_next = collect(prop.current_state()[1:6])
    # Σ⁻ = Φ Σ Φᵀ + Q_snc(dt): the SNC term is built from the PRE-step state RTN
    # (the axes the process noise acts along over the step) and added to brahe's
    # propagated covariance. q_rtn == 0 ⇒ Q = 0 (byte-identical to the old path).
    Σ_next = _sym(collect(prop.covariance_gcrf(ep_next)) .+ snc_q_eci(q_rtn, dt, μ0))
    return μ_next, Σ_next
end

"""
    predict(pomdp, b::Belief, a::CAAction; dt=pomdp.dt) -> Belief

Predict step (architecture §4 step 2). Propagates both sub-beliefs forward one
`dt` step. On MANEUVER, the spacecraft mean gets a +Δv·v̂ kick (same convention
as transitions.jl); the debris is untouched. Σ is propagated via the STM for
both objects regardless of action (noiseless maneuver, §8) — so WAIT and
MANEUVER give the SAME Σ⁻, differing only in μ.

`dt` is swappable (defaults to pomdp.dt); the returned belief's `t` is
decremented by `dt`. For τ = dt the returned Σ⁻ is consistent with
`build_covariance_table` at that τ (cross-checked in the unit tests).
"""
function predict(pomdp::SpacecraftCAPOMDP, b::Belief, a::CAAction; dt::Real = pomdp.dt)
    bh = get_brahe()
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - b.t

    # Maneuver kick on the spacecraft mean velocity (matches transitions.jl).
    Δv_sc = nothing
    if a == MANEUVER
        v     = b.sc.μ[4:6]
        v_hat = v / norm(v)
        Δv_sc = vcat(zeros(3), pomdp.Δv .* v_hat)
    end

    μ_sc, Σ_sc = _predict_object(pomdp, b.sc.μ, b.sc.Σ, pomdp.satParams,
                                 epoch_current, dt; Δv_applied = Δv_sc,
                                 q_rtn = pomdp.q_rtn_sc)
    μ_db, Σ_db = _predict_object(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams,
                                 epoch_current, dt; q_rtn = pomdp.q_rtn_debris)

    return Belief(ObjBelief(μ_sc, Σ_sc), ObjBelief(μ_db, Σ_db), b.t - Float64(dt))
end

# =========================================================================
# CORRECT — Kalman update against a SAMPLED observation.
# =========================================================================

"""
    sample_observation(pomdp, a, s_true, rng) -> Vector{Float64}

Draw a genuine noisy observation of the true next state (architecture §4 step 3)
from the project's observation model (`POMDPs.observation`, an MvNormal over
[sc_eci; debris_eci] with diag([σ_sc²×6, σ_debris²×6])). This is a random draw,
NOT an expected value — the correct step's μ⁺ depends on it.

`s_true` must be a `CAState`. Returns the 12-vector [z_sc(6); z_debris(6)].
"""
function sample_observation(pomdp::SpacecraftCAPOMDP, a::CAAction, s_true::CAState,
                            rng::AbstractRNG)
    return rand(rng, POMDPs.observation(pomdp, a, s_true))
end

"""
    _kalman_correct_linear(μ⁻, Σ⁻, z, R) -> (μ⁺, Σ⁺)

Hand-rolled linear-Gaussian update for one object with H = I₆ (full-state
observation, matching observations.jl). The reference implementation.

    K  = Σ⁻Hᵀ(HΣ⁻Hᵀ + R)⁻¹  = Σ⁻(Σ⁻ + R)⁻¹    (H = I)
    μ⁺ = μ⁻ + K(z − μ⁻)                         (depends on z)
    Σ⁺ = (I − K)Σ⁻                              (does NOT depend on z)

Σ⁺ is symmetrized against roundoff (Joseph form is not needed here since H = I
and R is well-conditioned, but symmetry is snapped).
"""
function _kalman_correct_linear(μ::AbstractVector, Σ::AbstractMatrix,
                                z::AbstractVector, R::AbstractMatrix)
    S = Symmetric(Matrix(Σ) .+ Matrix(R))          # HΣ⁻Hᵀ + R, H = I
    K = Matrix(Σ) / S                               # Σ⁻(Σ⁻+R)⁻¹, solved not inverted
    μ⁺ = collect(μ) .+ K * (collect(z) .- collect(μ))
    Σ⁺ = _sym((I - K) * Matrix(Σ))
    return μ⁺, Σ⁺
end

"""
    correct_linear(pomdp, b⁻::Belief, z; sc_range=1:6, debris_range=7:12) -> Belief

Correct step via the hand-rolled linear-Gaussian update (implementation (a)).
Splits the 12-vector observation `z` into its spacecraft/debris blocks and
applies the per-object Kalman update with R = σ²·I from the POMDP (matching
observations.jl: R_sc = σ_sc²·I₆, R_debris = σ_debris²·I₆). `t` is unchanged
(a correction does not advance time).
"""
function correct_linear(pomdp::SpacecraftCAPOMDP, b::Belief, z::AbstractVector;
                        sc_range = 1:6, debris_range = 7:12)
    R_sc = Matrix{Float64}(pomdp.σ_sc^2 * I, 6, 6)
    R_db = Matrix{Float64}(pomdp.σ_debris^2 * I, 6, 6)
    μ_sc, Σ_sc = _kalman_correct_linear(b.sc.μ, b.sc.Σ, z[sc_range], R_sc)
    μ_db, Σ_db = _kalman_correct_linear(b.debris.μ, b.debris.Σ, z[debris_range], R_db)
    return Belief(ObjBelief(μ_sc, Σ_sc), ObjBelief(μ_db, Σ_db), b.t)
end

"""
    correct_linear_sc(pomdp, b⁻::Belief, z; sc_range=1:6) -> Belief
    correct_linear_debris(pomdp, b⁻::Belief, z; debris_range=7:12) -> Belief

Correct ONE object only, leaving the other sub-belief untouched. Used by the
MCTS asymmetric measurement cadence (architecture §4 / TODOS "measurement
realism"): the satellite (own-asset GPS) and debris (SSN/TLE) get fixes on
DIFFERENT schedules, so a given decision step may correct one, both, or neither
object. `z` is the full 12-vector observation (only the relevant block is read).
`t` is unchanged (a correction does not advance time). Equivalent to
`correct_linear` restricted to a single object.
"""
function correct_linear_sc(pomdp::SpacecraftCAPOMDP, b::Belief, z::AbstractVector;
                           sc_range = 1:6)
    R_sc = Matrix{Float64}(pomdp.σ_sc^2 * I, 6, 6)
    μ_sc, Σ_sc = _kalman_correct_linear(b.sc.μ, b.sc.Σ, z[sc_range], R_sc)
    return Belief(ObjBelief(μ_sc, Σ_sc), b.debris, b.t)
end

function correct_linear_debris(pomdp::SpacecraftCAPOMDP, b::Belief, z::AbstractVector;
                               debris_range = 7:12)
    R_db = Matrix{Float64}(pomdp.σ_debris^2 * I, 6, 6)
    μ_db, Σ_db = _kalman_correct_linear(b.debris.μ, b.debris.Σ, z[debris_range], R_db)
    return Belief(b.sc, ObjBelief(μ_db, Σ_db), b.t)
end

"""
    _kalman_correct_brahe(pomdp, μ⁻, Σ⁻, z, σ, objParams, t) -> (μ⁺, Σ⁺)

Correct one object via brahe 1.7.0's ExtendedKalmanFilter with the LINEAR
InertialStateMeasurementModel (H = I₆, R = diag([σ²×3, σ²×3])). Because the
model is linear, this is numerically the same filter as
`_kalman_correct_linear` (unit tests assert agreement to ~1e-13 rel).

The EKF is anchored at the belief's current epoch with (μ⁻, Σ⁻) and given a
single Observation at that same epoch, so no propagation happens inside the
filter — this isolates the CORRECT step (predict is done separately, above).
Process noise is disabled (EKFConfig(process_noise=None)) — noiseless.
"""
function _kalman_correct_brahe(pomdp::SpacecraftCAPOMDP,
                               μ::AbstractVector, Σ::AbstractMatrix,
                               z::AbstractVector, σ::Real,
                               objParams::AbstractVector, t::Real)
    bh = get_brahe()
    np = pyimport("numpy")

    epoch_tca = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch     = epoch_tca - Float64(t)

    prop_cfg  = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()
    force_cfg = pomdp.forceModel ? bh.ForceModelConfig.default() : bh.ForceModelConfig.two_body()
    meas      = bh.InertialStateMeasurementModel(Float64(σ), Float64(σ))
    ekf_cfg   = bh.EKFConfig(process_noise = nothing)

    ekf = bh.ExtendedKalmanFilter(
        epoch, np.array(collect(float.(μ))), np.array(Matrix(Σ)),
        prop_cfg, force_cfg, [meas], ekf_cfg,
        params = np.array(collect(float.(objParams))))

    ekf.process_observation(bh.Observation(epoch, np.array(collect(float.(z))), 0))

    μ⁺ = collect(ekf.current_state()[1:6])
    Σ⁺ = _sym(collect(ekf.current_covariance()))
    return μ⁺, Σ⁺
end

"""
    correct_brahe(pomdp, b⁻::Belief, z; sc_range=1:6, debris_range=7:12) -> Belief

Correct step via brahe's ExtendedKalmanFilter + linear InertialStateMeasurement
model (implementation (b)). Same interface and (up to ~1e-13 rel) same result as
`correct_linear`; kept as an independent path for cross-validation and to reuse
brahe's production filter rather than hand-rolling.
"""
function correct_brahe(pomdp::SpacecraftCAPOMDP, b::Belief, z::AbstractVector;
                       sc_range = 1:6, debris_range = 7:12)
    μ_sc, Σ_sc = _kalman_correct_brahe(pomdp, b.sc.μ, b.sc.Σ, z[sc_range],
                                       pomdp.σ_sc, pomdp.satParams, b.t)
    μ_db, Σ_db = _kalman_correct_brahe(pomdp, b.debris.μ, b.debris.Σ, z[debris_range],
                                       pomdp.σ_debris, pomdp.debrisParams, b.t)
    return Belief(ObjBelief(μ_sc, Σ_sc), ObjBelief(μ_db, Σ_db), b.t)
end
