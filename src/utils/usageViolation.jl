# usageViolation.jl — 2D-Pc conjunction-plane usage-violation detector
#
# Faithful Julia port of the RECTILINEAR tier of NASA CARA's
# `UsageViolationPc2D.m` (DistributedMatlab/ProbabilityOfCollision/Utils),
# together with the thresholding that `PcMultiStep.m` applies to turn the raw
# indicators into the boolean `ViolationsPc2D` flag NASA publishes.
#
# ---------------------------------------------------------------------------
# WHAT THIS DETECTS
# ---------------------------------------------------------------------------
# Given the same inputs `elrod_pc` takes (two ECI states, two ECI 6x6
# covariances, a combined hard-body radius), this reports whether the
# assumptions behind the 2D conjunction-plane Pc method are violated for a
# given conjunction, and which criterion fired. A violation means the 2D-Pc
# value should not be trusted — empirically (cara_2d3d_gap_findings.md) the
# flagged cases are ~5x too LOW vs NASA's 3D reference, the unsafe direction.
#
# ---------------------------------------------------------------------------
# WHY THE RECTILINEAR TIER ONLY (scope decision, 2026-08-08, w/ Grace)
# ---------------------------------------------------------------------------
# `UsageViolationPc2D.m` has two tiers:
#
#   * RECTILINEAR (this file): models both objects as moving in STRAIGHT LINES
#     at constant velocity through TCA. Under that assumption the encounter
#     "Q-function" (overlap probability vs time) is an exact parabola, so the
#     encounter half-duration `STRectilinear`, its peak time `T`, and the
#     Mahalanobis miss-in-sigma `MDTRectilinear` all fall out of a few matrix
#     products with the summed relative position covariance A. From these it
#     forms the `Extended` (encounter too long) and `Offset` (peak too far
#     from TCA) indicators. This is exactly NASA's own
#     `FullPc2DViolationAnalysis = false` path — a faithful port, not a
#     simplified heuristic.
#
#   * CURVILINEAR (NOT ported): replaces the straight-line model with the
#     objects' real curved orbits, via a ~3000-line two-body stack
#     (equinoctial elements + Jacobians + the iterative "peak overlap point"
#     solver `PeakOverlapPos`, Hall 2021). It refines Extended/Offset and adds
#     a third indicator, `Inaccurate`. On this 53-case CARA set it changes only
#     the 3 most extreme slow/deep-tail cases (the ones NASA codes `111`); the
#     other 50 already agree with the rectilinear estimate. Those 3 live in the
#     far-out (>3-day, high miss-in-sigma) regime the paper GATES-and-escalates
#     rather than trusts a number for (cara_2d3d_gap_findings.md §3, §6), so
#     the rectilinear tier is the right cost/coverage tradeoff. If the 53-label
#     validation shows the 3 `111` cases matter, revisit the curvilinear port.
#
# ---------------------------------------------------------------------------
# THE NASA `ViolationsPc2D` ENCODING
# ---------------------------------------------------------------------------
# NASA's published column is a 3-digit code [Extended Offset Inaccurate], each
# digit a boolean, e.g. 0 = none, 100 = Extended only, 111 = all three. The
# per-indicator booleans come from `PcMultiStep.m` thresholding the raw
# `UVIndicators` fields (PcMultiStep.m:1013-1023):
#     Extended   > Pc2DExtendedCutoff   (default 0.02)
#     Offset     > Pc2DOffsetCutoff     (default 0.01)
#     Inaccurate > Pc2DInaccurateCutoff (default 0.02)
# plus an NPD flag `any(NPDIssues)`. `AnyPc2DViolations` (PcMultiStep.m:1026)
# is the OR of all four. This file reproduces Extended, Offset, and NPD (the
# rectilinear-tier indicators); Inaccurate requires the curvilinear tier and
# is reported as `missing`.
#
# ---------------------------------------------------------------------------
# CONSTANTS (all from CARA, tracked in CONSTANTS.md)
# ---------------------------------------------------------------------------
#   Fclip             = 1e-4    UsageViolationPc2D.m:116  (eigenvalue clip factor)
#   ConjDurationGamma = 1e-16   UsageViolationPc2D.m:117  (Coppola bound tail prob)
#   Pc2DExtendedCutoff = 0.02   PcMultiStep.m:606
#   Pc2DOffsetCutoff   = 0.01   PcMultiStep.m:607
#   GM (EGM-96)       = 3.986004418e14 m^3/s^2  orbit_period.m:8
#
# Units: states in m / (m/s), covariances in m^2. Matches cdmParser.jl output
# and `elrod_pc`'s call signature. (CARA's MATLAB works internally in km; the
# rectilinear indicators are dimensionless ratios so the choice of length unit
# cancels — we stay in metres throughout for consistency with the rest of the
# project.)

using LinearAlgebra
using SpecialFunctions: erfcinv

# CARA constants (see header / CONSTANTS.md).
const UV_FCLIP              = 1e-4
const UV_CONJ_DURATION_GAMMA = 1e-16
const UV_EXTENDED_CUTOFF    = 0.02
const UV_OFFSET_CUTOFF      = 0.01
const UV_GM_EGM96           = 3.986004418e14   # m^3/s^2

"""
    UsageViolationResult

Result of a rectilinear 2D-Pc usage-violation analysis.

Fields (mirroring NASA's `UVIndicators` / `PcMultiStep` outputs):
- `any_violation` : `true` if any rectilinear-tier criterion fired (NPD, Extended,
  or Offset). This is the boolean to compare against NASA's `AnyPc2DViolations` /
  a nonzero `ViolationsPc2D` code (excluding the curvilinear-only Inaccurate bit).
- `npd`           : `true` if any of the three position covariances (primary,
  secondary, relative) is non-positive-definite.
- `extended`      : `true` if the Extended indicator exceeds its cutoff (0.02).
- `offset`        : `true` if the Offset indicator exceeds its cutoff (0.01).
- `inaccurate`    : `missing` — requires the (un-ported) curvilinear tier.

Raw indicator values (before thresholding), for diagnostics / regression:
- `extended_ind`  : `min(1, 2·dτ/PeriodMin)`  — encounter duration / orbit period.
- `offset_ind`    : `min(1, max(|Ta|,|Tb|)/PeriodMin)` — peak offset from TCA.
- `npd_issues`    : `(pri, sec, rel)` NPD booleans.

Rectilinear geometry (the quantities the gap study keys on):
- `md_tca`        : Mahalanobis miss distance in sigma at the encounter peak
  (`MDTRectilinear`) — the dominant regressor for the 2D→3D gap
  (cara_2d3d_gap_findings.md, Spearman −0.80).
- `t_peak`        : time of minimum modified Mahalanobis distance rel. to TCA (s).
- `sigma_t`       : 1-σ encounter half-width in time (`STRectilinear`, s).
- `ta`, `tb`      : Coppola conjunction time bounds (s), γ = 1e-16.
- `period_min`    : min of the two orbital periods (s).
- `converged`     : `false` if relative velocity is (near) zero (STRectilinear = Inf),
  where the rectilinear analysis is undefined; then indicators are `missing`.
"""
struct UsageViolationResult
    any_violation::Bool
    npd::Bool
    extended::Union{Bool,Missing}
    offset::Union{Bool,Missing}
    inaccurate::Missing
    extended_ind::Union{Float64,Missing}
    offset_ind::Union{Float64,Missing}
    npd_issues::NTuple{3,Bool}
    md_tca::Float64
    t_peak::Float64
    sigma_t::Float64
    ta::Float64
    tb::Float64
    period_min::Float64
    converged::Bool
end

"""
    orbit_period(r, v; GM=UV_GM_EGM96) -> Float64

Two-body orbital period (s) from an ECI position/velocity pair (m, m/s).
Port of CARA `orbit_period.m`. Returns `Inf` for a non-bound (β ≤ 0) state.
"""
function orbit_period(r::AbstractVector, v::AbstractVector; GM::Real = UV_GM_EGM96)
    r0   = sqrt(dot(r, r))
    v02  = dot(v, v)
    beta = 2 * GM / r0 - v02
    beta <= 0 && return Inf         # parabolic/hyperbolic — no finite period
    return (2π) * GM * beta^(-1.5)
end

"""
    _is_npd(C3; tol=0.0) -> Bool

Whether a 3x3 symmetric covariance is non-positive-definite, i.e. has any
eigenvalue ≤ 0. Mirrors NASA's `any(Leig <= 0)` test
(UsageViolationPc2D.m:326,329,274).
"""
function _is_npd(C3::AbstractMatrix)
    Leig = eigvals(Symmetric(Matrix(C3)))
    return any(<=(0.0), Leig)
end

"""
    usage_violation_pc2d(r1, v1, C1, r2, v2, C2, HBR;
                         Fclip=UV_FCLIP, gamma=UV_CONJ_DURATION_GAMMA,
                         extended_cutoff=UV_EXTENDED_CUTOFF,
                         offset_cutoff=UV_OFFSET_CUTOFF) -> UsageViolationResult

Rectilinear 2D-Pc usage-violation detector. Faithful port of the
rectilinear tier of NASA CARA `UsageViolationPc2D.m` (lines ~255–394),
plus the `PcMultiStep.m` cutoff thresholding.

Inputs match `elrod_pc`:
- `r1,v1` / `r2,v2` : primary / secondary ECI position (m) & velocity (m/s),
  each a length-3 vector, OR a length-6 state passed as `r`/`v` need not be
  split — but here pass them separately.
- `C1,C2`           : 6x6 ECI position/velocity covariances (m^2, m^2/s, ...).
  Only the 3x3 position blocks and the summed relative position block are used
  by the rectilinear tier; velocity blocks are accepted for signature parity.
- `HBR`             : combined hard-body radius (m), > 0.

Covariance cross-correlation is NOT applied (CARA's `apply_covXcorr_corrections`
is only meaningful with DCP sensitivity vectors, which CDMs do not carry — NASA
runs these test cases with it off; PcMultiStep_UnitTest.m sets
`apply_covXcorr_corrections = false`). So the summed relative position
covariance is simply `A = C1[1:3,1:3] + C2[1:3,1:3]`.
"""
function usage_violation_pc2d(r1::AbstractVector, v1::AbstractVector,
                              C1::AbstractMatrix,
                              r2::AbstractVector, v2::AbstractVector,
                              C2::AbstractMatrix,
                              HBR::Real;
                              Fclip::Real = UV_FCLIP,
                              gamma::Real = UV_CONJ_DURATION_GAMMA,
                              extended_cutoff::Real = UV_EXTENDED_CUTOFF,
                              offset_cutoff::Real = UV_OFFSET_CUTOFF)

    HBR > 0 || throw(ArgumentError("Combined HBR must be positive (got $HBR)."))

    r1 = Vector{Float64}(r1[1:3]); v1 = Vector{Float64}(v1[1:3])
    r2 = Vector{Float64}(r2[1:3]); v2 = Vector{Float64}(v2[1:3])

    # --- NPD indicators for primary, secondary, relative position covariances.
    # (UsageViolationPc2D.m:322-333.) Computed from the RAW input covariances,
    # before any eigenvalue clipping.
    C1pos = Symmetric(Matrix(C1)[1:3, 1:3])
    C2pos = Symmetric(Matrix(C2)[1:3, 1:3])
    A_raw = C1pos + C2pos                      # summed relative position cov (m^2)

    pri_npd = _is_npd(C1pos)
    sec_npd = _is_npd(C2pos)
    rel_npd = _is_npd(A_raw)
    npd_issues = (pri_npd, sec_npd, rel_npd)
    npd_any = pri_npd || sec_npd || rel_npd

    # --- Relative TCA state (secondary − primary). (UsageViolationPc2D.m:258-259.)
    r = r2 .- r1
    v = v2 .- v1

    # --- Invert A with CARA's eigenvalue-clip NPD remediation.
    # (UsageViolationPc2D.m:270-278.) Clip floor Lclip = (Fclip·HBR)^2 in m^2.
    Lclip = (Fclip * HBR)^2
    F = eigen(Symmetric(Matrix(A_raw)))
    Leig = collect(F.values)
    Veig = F.vectors
    Leig[Leig .< Lclip] .= Lclip               # clip small/negative eigenvalues
    Ainv = Veig * Diagonal(1.0 ./ Leig) * Veig'

    # --- Rectilinear min-MD^2 time T and Q-function width. (UVPc2D.m:281-294.)
    rT_Ai   = r' * Ainv
    rT_Ai_r = (rT_Ai * r)                       # scalar
    rT_Ai_v = (rT_Ai * v)                       # scalar
    vT_Ai_v = (v' * Ainv * v)                   # scalar

    T = rT_Ai_v == 0 ? 0.0 : -rT_Ai_v / vT_Ai_v

    # 1-σ Q-function width in time. SigmaSquaredRL = 1 / (vᵀA⁻¹v);
    # if the relative velocity is (near) zero, vᵀA⁻¹v→0 and this is Inf, meaning
    # the rectilinear analysis is undefined (UVPc2D.m:293-294, 387-394).
    sigma_t = vT_Ai_v <= 0 ? Inf : sqrt(1.0 / vT_Ai_v)

    # Mahalanobis miss-in-σ at the encounter peak (t = T). (UVPc2D.m:308-319.)
    # MD2(T) = rᵀA⁻¹r + 2T·rᵀA⁻¹v + T²·vᵀA⁻¹v, evaluated at the midpoint.
    MD2_T = rT_Ai_r + T * (2 * rT_Ai_v) + (T^2) * vT_Ai_v
    md_tca = sqrt(max(0.0, MD2_T))

    # --- Orbital periods. (UVPc2D.m:361-363.)
    period1 = orbit_period(r1, v1)
    period2 = orbit_period(r2, v2)
    period_min = min(period1, period2)

    # --- Coppola conjunction time bounds in the small-HBR limit.
    # (UVPc2D.m:355-358.) dτ = STRectilinear · √2 · erfcinv(γ).
    sqrt2_erfcinv_gamma = sqrt(2.0) * erfcinv(gamma)

    # Undefined-velocity guard: STRectilinear = Inf ⇒ rectilinear analysis does
    # not produce meaningful Extended/Offset (UVPc2D.m:387-394 returns early).
    if !isfinite(sigma_t)
        return UsageViolationResult(
            npd_any,                 # any_violation: only NPD is defined here
            npd_any, missing, missing, missing,
            missing, missing, npd_issues,
            md_tca, T, sigma_t, -Inf, Inf, period_min, false,
        )
    end

    dtau = sigma_t * sqrt2_erfcinv_gamma
    Ta = T - dtau
    Tb = T + dtau

    # --- Raw indicators. (UVPc2D.m:366-370.)
    extended_ind = min(1.0, 2 * dtau / period_min)
    Tab = max(abs(Ta), abs(Tb))
    offset_ind = min(1.0, Tab / period_min)

    # --- Threshold to booleans, per PcMultiStep.m:1013-1023.
    extended = extended_ind > extended_cutoff
    offset   = offset_ind   > offset_cutoff

    any_violation = npd_any || extended || offset

    return UsageViolationResult(
        any_violation,
        npd_any, extended, offset, missing,
        extended_ind, offset_ind, npd_issues,
        md_tca, T, sigma_t, Ta, Tb, period_min, true,
    )
end