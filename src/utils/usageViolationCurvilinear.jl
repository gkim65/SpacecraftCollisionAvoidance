# usageViolationCurvilinear.jl — full (rectilinear + CURVILINEAR) 2D-Pc
# usage-violation detector.
#
# Faithful Julia port of the curvilinear tier of NASA CARA `UsageViolationPc2D.m`
# (lines ~208–920), layered on the rectilinear tier (usageViolation.jl) and the
# equinoctial / POP machinery ported in:
#   equinoctial.jl        — cart<->equinoctial transforms + Jacobians (validated)
#   curvilinearUtils.jl   — CovRemEigValClip, cov_make_symmetric, TimeParabolaFit
#   peakOverlap.jl        — PeakOverlapPos / PeakOverlapMD2 (the iterative solver)
#
# WHAT THE CURVILINEAR TIER ADDS over the rectilinear one: it re-evaluates the
# encounter Q-function (modified Mahalanobis distance vs time) using the objects'
# real curved orbits at the peak-overlap point, refines the min-Q time/width, and
# computes the `Inaccurate` indicator — the log-ratio of the curvilinear 2D-Nc to
# the rectilinear 2D-Pc. On the 53-case CARA set, the `Inaccurate` bit drives
# ~26/29 of NASA's violations (the "2D-Pc underestimation" cases the rectilinear
# tier alone cannot see).
#
# UNITS: caller passes states in m / (m/s), covariances in m^2 (same as
# elrod_pc / usage_violation_pc2d). Internally converts to km for the
# equinoctial/POP machinery, matching CARA's EquinoctialMatrices.m.
#
# CROSS-CORRELATION: off (CDMs carry no DCP sensitivity vectors; NASA runs these
# cases with apply_covXcorr_corrections=false — see PcMultiStep_UnitTest.m).
# RETROGRADE REORIENTATION: not ported (never triggers on real CARA conjunctions;
# UsageViolationPc2D.m:196-199). Both are faithful omissions for this data set.

using LinearAlgebra
using SpecialFunctions: erfcinv

include(joinpath(@__DIR__, "equinoctial.jl"))
include(joinpath(@__DIR__, "curvilinearUtils.jl"))
include(joinpath(@__DIR__, "peakOverlap.jl"))
include(joinpath(@__DIR__, "usageViolation.jl"))   # rectilinear tier + constants

# Curvilinear-tier defaults (UsageViolationPc2D.m:122-128).
const UV_ACCEPTABLE_INIT_OFFSET_FACTOR = 10.0
const UV_TIME_SIGMA_SPACING            = 1.0
const UV_TIME_SIGMA_DIFF_LIMIT         = 2.5
const UV_QFUNCTION_DIFF_LIMIT          = 1.0
const UV_MAX_REFINEMENTS               = 20
const UV_INACCURATE_CUTOFF             = 0.02   # PcMultiStep.m:608

"""
    CurvilinearUVResult

Full usage-violation result including the curvilinear indicators.
- `any_violation` : NPD ∨ Extended ∨ Offset ∨ Inaccurate (matches NASA's
  `AnyPc2DViolations`, all four cutoffs applied).
- `extended`, `offset`, `inaccurate`, `npd` : the four thresholded booleans.
- raw indicators `extended_ind`, `offset_ind`, `inaccurate_ind`.
- `nc2d_hi_limit`, `nc2d_small_hbr`, `nc2d_lo_limit` : the curvilinear 2D-Nc
  estimates (the acceptance-gate quantities — `nc2d_hi_limit` ≳ NASA `Nc2D`).
- `log_pc_correction` : log(2D-Nc / 2D-Pc) small-HBR estimate.
- curvilinear geometry: `t_curv`, `q_t_curv` (min modified MD²), `sigma_t_curv`,
  `v_t_curv`; and rectilinear `md_tca`, `q_t_rect` carried through.
- `qconverged` : whether the curvilinear analysis converged (else curvilinear
  indicators fall back to the rectilinear values / `missing`).
- `refinement_level`, `rect` (the underlying rectilinear result).
"""
struct CurvilinearUVResult
    any_violation::Bool
    npd::Bool
    extended::Union{Bool,Missing}
    offset::Union{Bool,Missing}
    inaccurate::Union{Bool,Missing}
    extended_ind::Union{Float64,Missing}
    offset_ind::Union{Float64,Missing}
    inaccurate_ind::Union{Float64,Missing}
    nc2d_hi_limit::Float64
    nc2d_small_hbr::Float64
    nc2d_lo_limit::Float64
    log_pc_correction::Float64
    t_curv::Float64
    q_t_curv::Float64
    sigma_t_curv::Float64
    v_t_curv::Float64
    md_tca::Float64
    q_t_rect::Float64
    qconverged::Bool
    refinement_level::Int
    rect::UsageViolationResult
end

"""
    _equinoctial_matrices(r_m, v_m, C_m) -> (E, Q, Xkm)

Build the epoch equinoctial mean state `E=[n,af,ag,chi,psi,lM]` and 6x6
equinoctial covariance `Q = K·P·K'` (K = dE/dX) from an ECI state (m, m/s) and
covariance (m^2). Port of `EquinoctialMatrices.m` (no NPD equinoctial-cov
remediation, matching the default `remediate_NPD_TCA_eq_covariances=false`).
Returns the km-unit cartesian state too. Throws on equinoctial failure.
"""
function _equinoctial_matrices(r_m::AbstractVector, v_m::AbstractVector,
                               C_m::AbstractMatrix)
    Xkm = vcat(r_m[1:3] ./ 1e3, v_m[1:3] ./ 1e3)     # km, km/s
    P = Matrix(C_m) ./ 1e6                            # km^2, ...
    a, n, af, ag, chi, psi, lM, _ = cartesian_to_equinoctial(Xkm[1:3], Xkm[4:6])
    E = [n, af, ag, chi, psi, lM]
    J = jacobian_equinoctial_to_cartesian(E, Xkm)     # dX/dE
    K = J \ Matrix{Float64}(I, 6, 6)                  # dE/dX
    Q = cov_make_symmetric(K * P * K')
    return (E, Q, Xkm)
end

"""
    usage_violation_pc2d_curvilinear(r1, v1, C1, r2, v2, C2, HBR;
                                     gamma=UV_CONJ_DURATION_GAMMA,
                                     extended_cutoff=UV_EXTENDED_CUTOFF,
                                     offset_cutoff=UV_OFFSET_CUTOFF,
                                     inaccurate_cutoff=UV_INACCURATE_CUTOFF,
                                     Fclip=UV_FCLIP) -> CurvilinearUVResult

Full 2D-Pc usage-violation analysis (rectilinear + curvilinear). Same inputs as
`usage_violation_pc2d` / `elrod_pc` (states m/(m/s), covariances m^2, HBR m).

Port of `UsageViolationPc2D.m` with `FullPc2DViolationAnalysis=true`. Runs the
rectilinear tier first (via `usage_violation_pc2d`), then the curvilinear
Q-function analysis: 3-point POP-MD² evaluation about the rectilinear peak,
parabola fit for the min-Q time/width, refinement until the min stabilizes, and
the `Inaccurate` indicator + 2D-Nc limits at the converged min-Q point.
"""
function usage_violation_pc2d_curvilinear(
        r1::AbstractVector, v1::AbstractVector, C1::AbstractMatrix,
        r2::AbstractVector, v2::AbstractVector, C2::AbstractMatrix,
        HBR::Real;
        gamma::Real = UV_CONJ_DURATION_GAMMA,
        extended_cutoff::Real = UV_EXTENDED_CUTOFF,
        offset_cutoff::Real = UV_OFFSET_CUTOFF,
        inaccurate_cutoff::Real = UV_INACCURATE_CUTOFF,
        Fclip::Real = UV_FCLIP)

    HBR > 0 || throw(ArgumentError("Combined HBR must be positive (got $HBR)."))
    HBRkm = HBR / 1e3

    # --- Rectilinear tier (indicators, miss-in-σ, T, STRectilinear, periods) ---
    rect = usage_violation_pc2d(r1, v1, C1, r2, v2, C2, HBR;
                                Fclip = Fclip, gamma = gamma,
                                extended_cutoff = extended_cutoff,
                                offset_cutoff = offset_cutoff)

    # Rectilinear Q-function quantities we need for the correction factor.
    T = rect.t_peak
    ST_rect = rect.sigma_t
    period_min = rect.period_min
    # Rectilinear velocity magnitude (km/s) and Q(T).
    v_rel = (Vector{Float64}(v2[1:3]) .- Vector{Float64}(v1[1:3])) ./ 1e3
    VT_rect = norm(v_rel)
    md_tca = rect.md_tca
    # Rectilinear Q(T) = MD²(T) + log(Adet/1e18). We recompute Adet from the
    # summed position covariance with the same clip, to match UVPc2D.m:277,312.
    A_m = Matrix(C1)[1:3, 1:3] .+ Matrix(C2)[1:3, 1:3]     # m^2
    Lclip_m = (Fclip * HBR)^2
    Fe = eigen(Symmetric(A_m)); Le = collect(Fe.values)
    Le[Le .< Lclip_m] .= Lclip_m
    Adet_m = Le[1] * Le[2] * Le[3]
    logAdet = log(Adet_m / 1e18)                          # km-unit determinant
    QT_rect = md_tca^2 + logAdet

    sqrt2_erfcinv_gamma = sqrt(2.0) * erfcinv(gamma)

    # Defaults if the curvilinear analysis cannot run / converge.
    inaccurate_ind = missing
    nc2d_hi = NaN; nc2d_small = NaN; nc2d_lo = NaN; logPcCorr = NaN
    t_curv = NaN; qt_curv = NaN; st_curv = NaN; vt_curv = NaN
    qconverged = false
    refinement_level = 0
    extended_ind = rect.extended_ind
    offset_ind = rect.offset_ind

    # Bail to rectilinear-only if the rectilinear analysis was undefined
    # (zero relative velocity) — mirrors UVPc2D.m:387-394.
    if !rect.converged || !isfinite(ST_rect)
        any_v = rect.any_violation
        return CurvilinearUVResult(
            any_v, rect.npd, rect.extended, rect.offset, missing,
            extended_ind, offset_ind, missing,
            nc2d_hi, nc2d_small, nc2d_lo, logPcCorr,
            t_curv, qt_curv, st_curv, vt_curv, md_tca, QT_rect,
            false, -1, rect)
    end

    # --- Build equinoctial mean matrices for both objects ---
    local E1, Q1, E2, Q2
    try
        E1, Q1, _ = _equinoctial_matrices(r1, v1, C1)
        E2, Q2, _ = _equinoctial_matrices(r2, v2, C2)
    catch
        # Equinoctial failure ⇒ curvilinear analysis unavailable; rectilinear stands.
        return CurvilinearUVResult(
            rect.any_violation, rect.npd, rect.extended, rect.offset, missing,
            extended_ind, offset_ind, missing,
            nc2d_hi, nc2d_small, nc2d_lo, logPcCorr,
            t_curv, qt_curv, st_curv, vt_curv, md_tca, QT_rect, false, -1, rect)
    end

    popp = PeakOverlapParams(Fclip = Fclip)

    # --- 3-point curvilinear Q evaluation about the rectilinear peak T ---
    dt = UV_TIME_SIGMA_SPACING * ST_rect
    tpts = [T - dt, T, T + dt]; MidPoint = 2
    Qt = fill(NaN, 3)
    Xu_mid = Float64[]
    for nt in 1:3
        md2, Xu, _, _, _, conv, _ =
            peak_overlap_md2(tpts[nt], E1, Q1, E2, Q2, HBRkm; EMD2 = 1, params = popp)
        Qt[nt] = md2
        nt == MidPoint && (Xu_mid = Xu)
    end
    Qconv = !any(isnan, Qt)

    tcurv = copy(tpts); Qtcurv = copy(Qt)

    if Qconv
        refinement_level = 1
        Qdotdot = (Qtcurv[3] - 2 * Qtcurv[2] + Qtcurv[1]) / dt^2
        if Qdotdot <= 0
            st_curv = Inf
        else
            c, _, _, _ = time_parabola_fit(tcurv, Qtcurv)
            # NASA computes t0=-abc(2)/Qdotdot, Qt0=abc(3)-abc(2)^2/abc(1)/4 using
            # the numeric Qdotdot and the fitted parabola c=[a,b,c] (y=a t²+b t+c).
            # For the parabola vertex these reduce to t0=-b/2a, Qt0=c-b²/4a, which
            # we use directly (consistent with the fitted coefficients).
            a_, b_, c_ = c[1], c[2], c[3]
            t0 = -b_ / (2 * a_)
            Qt0 = c_ - b_^2 / (4 * a_)

            # Acceptable-init-offset clamp (UVPc2D.m:477-489).
            delt0max = UV_ACCEPTABLE_INIT_OFFSET_FACTOR * (tcurv[3] - tcurv[1])
            AcceptableInitOffset = true
            if t0 < tcurv[1] - delt0max
                AcceptableInitOffset = false
                t0 = tcurv[1] - delt0max
                Qt0 = a_ * t0^2 + b_ * t0 + c_
            elseif t0 > tcurv[3] + delt0max
                AcceptableInitOffset = false
                t0 = tcurv[3] + delt0max
                Qt0 = a_ * t0^2 + b_ * t0 + c_
            end

            t_curv = t0; qt_curv = Qt0
            st_curv = sqrt(2 / Qdotdot)
            vt_curv = norm(Xu_mid[4:6])

            # First correction-factor estimate.
            logPcCorr = log(st_curv / ST_rect) + log(vt_curv / VT_rect) +
                        (QT_rect - qt_curv) / 2

            NeedsRefinement = if AcceptableInitOffset
                TimeDifference = t_curv - T
                MaxTimeDifference = UV_TIME_SIGMA_DIFF_LIMIT * min(ST_rect, st_curv)
                QDifference = qt_curv - QT_rect
                abs(TimeDifference) >= MaxTimeDifference ||
                    abs(QDifference) >= UV_QFUNCTION_DIFF_LIMIT
            else
                true
            end

            # --- Refinement loop (UVPc2D.m:538-802) ---
            while NeedsRefinement
                refinement_level += 1
                _, ItMinimum = findmin(Qtcurv)

                tnew_c = t_curv
                Qnew, Xu_r, _, _, _, convr, _ =
                    peak_overlap_md2(tnew_c, E1, Q1, E2, Q2, HBRkm; EMD2 = 1, params = popp)

                if isnan(Qnew)
                    # POP failed at the new point — bisect toward the bracket.
                    min_t = minimum(tcurv); max_t = maximum(tcurv)
                    tnew = if t_curv < min_t
                        (t_curv + min_t) / 2
                    elseif t_curv > max_t
                        (t_curv + max_t) / 2
                    else
                        NaN
                    end
                    if isnan(tnew) || length(tcurv) > 3
                        NeedsRefinement = false
                    else
                        if refinement_level < UV_MAX_REFINEMENTS
                            NeedsRefinement = true
                            t_curv = tnew; qt_curv = NaN
                        else
                            NeedsRefinement = false
                        end
                    end
                else
                    Qsrt = sort(Qtcurv)
                    Qcut = Qsrt[3]
                    Qdecreasing = Qnew < Qcut

                    push!(tcurv, tnew_c); push!(Qtcurv, Qnew)

                    if Qdecreasing
                        vt_curv = norm(Xu_r[4:6])
                        yy, _, _, _ = time_parabola_fit(tcurv, Qtcurv)
                        if yy[1] <= 0
                            st_curv = Inf
                        else
                            t_curv = -yy[2] / yy[1] / 2
                            qt_curv = yy[3] - yy[2]^2 / yy[1] / 4
                            st_curv = sqrt(1 / yy[1])
                        end
                    else
                        NtC = length(tcurv) - 1
                        min_t = minimum(tcurv[1:NtC]); max_t = maximum(tcurv[1:NtC])
                        if t_curv < min_t
                            tnew = (t_curv + min_t) / 2
                            yy, _, _, _ = time_parabola_fit(tcurv, Qtcurv)
                            Qnew2 = yy[1] + tnew * (yy[2] + tnew * yy[3])
                            t_curv = tnew; qt_curv = Qnew2
                        elseif t_curv > max_t
                            tnew = (t_curv + max_t) / 2
                            yy, _, _, _ = time_parabola_fit(tcurv, Qtcurv)
                            Qnew2 = yy[1] + tnew * (yy[2] + tnew * yy[3])
                            t_curv = tnew; qt_curv = Qnew2
                        else
                            t_curv = tcurv[ItMinimum]; qt_curv = Qtcurv[ItMinimum]
                        end
                    end

                    if !isinf(st_curv)
                        logPcCorr = log(st_curv / ST_rect) + log(vt_curv / VT_rect) +
                                    (QT_rect - qt_curv) / 2
                    end

                    NeedsRefinement = false
                    if !isinf(st_curv) && refinement_level < UV_MAX_REFINEMENTS
                        TimeDifference = t_curv - tcurv[ItMinimum]
                        MaxTimeDifference = UV_TIME_SIGMA_DIFF_LIMIT * st_curv
                        QDifference = qt_curv - Qtcurv[ItMinimum]
                        NeedsRefinement = abs(TimeDifference) >= MaxTimeDifference ||
                                          abs(QDifference) >= UV_QFUNCTION_DIFF_LIMIT
                    end
                end
            end
        end
    end

    # Mark unconverged if final curvilinear sigma is undefined (UVPc2D.m:806-809).
    if isinf(st_curv) || isnan(st_curv)
        Qconv = false
    end

    # --- Exact Q(T) at the converged min-Q time + 2D-Nc limits (UVPc2D.m:811-854) ---
    if Qconv
        Qnew, Xu_f, _, Asdet, Asinv, convf, auxf =
            peak_overlap_md2(t_curv, E1, Q1, E2, Q2, HBRkm; EMD2 = 1, params = popp)
        if convf <= 0 || isnan(Qnew)
            Qconv = false
        else
            qt_curv = Qnew
            logPcCorr = log(st_curv / ST_rect) + log(vt_curv / VT_rect) +
                        (QT_rect - qt_curv) / 2

            HBR2 = HBRkm^2
            Nc2DCoef = HBR2 * vt_curv * st_curv / 2
            nc2d_small = min(1.0, Nc2DCoef * exp(-qt_curv / 2))

            AsLeig = auxf[:AsLeig]
            Asiru = Asinv * Xu_f[1:3]
            Aterm = 2 * HBRkm * sqrt(dot(Asiru, Asiru))
            QTmax = HBR2 / minimum(AsLeig) + Aterm + qt_curv
            nc2d_lo = min(1.0, Nc2DCoef * exp(-QTmax / 2))

            logAsdet = log(Asdet)
            MD2Curv = qt_curv - logAsdet
            MD2min = max(0.0, HBR2 / maximum(AsLeig) - Aterm + MD2Curv)
            QTmin = MD2min + logAsdet
            nc2d_hi = min(1.0, Nc2DCoef * exp(-QTmin / 2))
        end
    end

    qconverged = Qconv

    # --- Final indicators (UVPc2D.m:886-920) ---
    if qconverged
        if isinf(st_curv)
            extended_ind = 1.0
        else
            dtau = st_curv * sqrt2_erfcinv_gamma
            Ta = t_curv - dtau; Tb = t_curv + dtau
            extended_ind = min(1.0, (Tb - Ta) / period_min)
            offset_ind = min(1.0, max(abs(Ta), abs(Tb)) / period_min)
            inaccurate_ind = 1 - exp(-abs(logPcCorr))
        end
    end

    npd_b = rect.npd
    ext_b = extended_ind === missing ? missing : (extended_ind > extended_cutoff)
    off_b = offset_ind === missing ? missing : (offset_ind > offset_cutoff)
    ina_b = inaccurate_ind === missing ? missing : (inaccurate_ind > inaccurate_cutoff)

    any_v = npd_b ||
            (ext_b === true) || (off_b === true) || (ina_b === true)

    return CurvilinearUVResult(
        any_v, npd_b, ext_b, off_b, ina_b,
        extended_ind, offset_ind, inaccurate_ind,
        nc2d_hi, nc2d_small, nc2d_lo, logPcCorr,
        t_curv, qt_curv, st_curv, vt_curv, md_tca, QT_rect,
        qconverged, refinement_level, rect)
end