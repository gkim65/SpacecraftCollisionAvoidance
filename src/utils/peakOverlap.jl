# peakOverlap.jl — iterative peak-overlap-point (POP) solver + modified
# Mahalanobis distance, for the curvilinear 2D-Pc usage-violation analysis.
#
# Faithful Julia port of NASA CARA:
#   PeakOverlapPos.m  (ProbabilityOfCollision/Utils) — the iterative POP solver
#   PeakOverlapMD2.m  (same)                          — modified-MD² wrapper
#
# Reference: Hall, D.T. (2021) "Expected Collision Rates for Tracked
# Satellites", JSR 58(3):715-728; Coppola (2012a/b).
#
# WHAT IT DOES: given two objects' epoch equinoctial mean states + covariances,
# finds the inertial position where their (curved-orbit, first-order-linearized)
# position PDFs most overlap at time `t` — the center-of-linearization for
# Hall's curvilinear Pc/Nc. This replaces the rectilinear straight-line model
# with the real two-body geometry, and is what lets the usage-violation analysis
# compute the `Inaccurate` (Nc-vs-Pc) indicator.
#
# UNITS: km, km/s, mu in km^3/s^2. Depends on equinoctial.jl + curvilinearUtils.jl
# (assumed already included by the caller).
#
# ---------------------------------------------------------------------------
# ONE DELIBERATE SUBSTITUTION (flagged, not silent)
# ---------------------------------------------------------------------------
# `AdjustPOPCoL` limits the per-iteration relative orbital-energy change on very
# elongated covariances (NASA's 2024-DEC stabilization). When the nominal step
# exceeds `MaxRelEnergyChange`, NASA finds the fractional step via its 256-line
# general-purpose `refine_bounded_extrema.m`. That refiner is NOT ported. Instead
# we reproduce its ACTUAL use here: a 201-point grid scan of the same objective
# over x∈[0,1] (identical to NASA's `xinit`/`yinit`), then — only when the grid
# minimum sits in the first few points (NASA's `imin < Ninitcut` trigger) — a
# golden-section refine on the bracketing sub-interval. This matches NASA's
# intent and its default grid; it is not guaranteed bit-identical on the rare
# elongated cases that hit the refine branch. Those cases are exactly the
# high-anisotropy ones, so the Nc2D acceptance gate (validation against NASA's
# Nc2D column) is the check that this substitution is good enough.

using LinearAlgebra

const POP_GM_KM = 3.986004418e5   # EGM-96 mu, km^3/s^2

"""
    PeakOverlapParams

Execution parameters for `peak_overlap_pos`, defaulting to CARA's values.
- `MD2tol`            : (tight, loose) convergence tolerances on the POP-point
  Mahalanobis-distance² change. Default `(1e-12, 1e-6)`.
- `MaxRelEnergyChange`: max per-iteration relative orbital-energy change
  (stabilizes overshoot on elongated PDFs). Default `0.10`.
- `maxiter`           : max POP iterations. Default `100`.
- `Fclip`             : eigenvalue-clip factor; Lclip = (HBR·Fclip)². Default `1e-4`.
- `GM`                : gravitational parameter (km³/s²). Default EGM-96.
"""
Base.@kwdef struct PeakOverlapParams
    MD2tol::NTuple{2,Float64} = (1e-12, 1e-6)
    MaxRelEnergyChange::Float64 = 0.10
    maxiter::Int = 100
    Fclip::Float64 = 1e-4
    GM::Float64 = POP_GM_KM
end

"""
    POPResult

Peak-overlap-point solution. `conv` is the convergence flag (as CARA: `true`,
`false`, or a fractional code 0.5/0.1–0.35 for oscillatory/relaxed convergence
— all treated as "converged enough" by callers via `conv > 0`). `rpk` is the
POP inertial position (km); `v1pk`/`v2pk` the conditional mean velocities there.
`aux` carries the per-object POP states, Jacobians, epoch equinoctial states,
peak-overlap covariance, and the offset states `xu1`/`xu2` used downstream.
"""
struct POPResult
    conv::Float64
    rpk::Vector{Float64}
    v1pk::Vector{Float64}
    v2pk::Vector{Float64}
    aux::Dict{Symbol,Any}
end

"""
    _calc_col_energy(x, rsA, rsB, Xu, beta, GM) -> (Energy, rs)

Vectorized center-of-linearization orbit energy at intermediate positions
`rs = rsA + x·(rsB−rsA)`, using conditional velocity `vup = vu + beta·(rs−ru)`.
Port of `CalcCoLEnergy`. `x` may be a scalar or vector; returns energy per point
and the position(s). (For scalar `x`, `rs` is a 3-vector.)
"""
function _calc_col_energy(x, rsA::AbstractVector, rsB::AbstractVector,
                          Xu::AbstractVector, beta::AbstractMatrix, GM::Real)
    xs = x isa Number ? [Float64(x)] : collect(Float64, x)
    ru = Xu[1:3]; vu = Xu[4:6]
    drs = rsB .- rsA
    energies = Vector{Float64}(undef, length(xs))
    positions = Matrix{Float64}(undef, 3, length(xs))
    for (i, xi) in enumerate(xs)
        rs = rsA .+ xi .* drs
        positions[:, i] = rs
        vup = vu .+ beta * (rs .- ru)
        energies[i] = dot(vup, vup) / 2 - GM / norm(rs)
    end
    if x isa Number
        return (energies[1], positions[:, 1])
    end
    return (energies, positions)
end

"""
    _adjust_pop_col(mup, rs, beta, Xu, GM, MaxRelEnergyChange) -> (rsAdj, EnAdj, fracAdj)

Adjust the POP center-of-linearization point to cap the per-iteration relative
orbital-energy change. Port of `AdjustPOPCoL`. Returns the nominal `mup`
(fracAdj=1) when the energy change is acceptable or the original energy is
non-negative; otherwise finds the fractional step where the relative energy
change equals `MaxRelEnergyChange` (see the flagged substitution in the file
header for how the search differs from NASA's `refine_bounded_extrema`).
"""
function _adjust_pop_col(mup::AbstractVector, rs::AbstractVector,
                         beta::AbstractMatrix, Xu::AbstractVector,
                         GM::Real, MaxRelEnergyChange::Real)
    EnergyA, _ = _calc_col_energy(0.0, rs, mup, Xu, beta, GM)
    EnergyB, _ = _calc_col_energy(1.0, rs, mup, Xu, beta, GM)

    # No adjustment if original CoL energy is not negative (unbound).
    if EnergyA >= 0
        return (Vector{Float64}(mup), EnergyB, 1.0)
    end

    FracEnergyChange = abs(EnergyB - EnergyA) / abs(EnergyA)
    if FracEnergyChange <= MaxRelEnergyChange
        return (Vector{Float64}(mup), EnergyB, 1.0)
    end

    # Objective: (|Energy(x)/EnergyA − 1| − MaxRelEnergyChange)^2, minimized over
    # x∈[0,1]. NASA uses a 201-point grid (xinit/yinit) then refines only if the
    # grid min lands in the first ~5% of points. We do the same grid, then a
    # golden-section refine on the bracket when triggered.
    Ninitial = 201
    Ninitcut = max(5, round(Int, 0.05 * Ninitial))
    obj(xx) = begin
        E, _ = _calc_col_energy(xx, rs, mup, Xu, beta, GM)
        (abs(E / EnergyA - 1) - MaxRelEnergyChange)^2
    end
    xinit = range(0, 1; length = Ninitial)
    yinit = [obj(x) for x in xinit]
    imin = argmin(yinit)
    xmin = xinit[imin]

    if imin < Ninitcut
        # Refine on the sub-interval bracketing the grid minimum.
        lo = xinit[max(1, imin - 1)]
        hi = xinit[min(Ninitial, imin + 1)]
        xmin = _golden_section_min(obj, lo, hi; tol = 5e-4)
    end

    xmin <= 0 && (xmin = xinit[2])

    EnAdj, rsAdj = _calc_col_energy(xmin, rs, mup, Xu, beta, GM)
    return (rsAdj, EnAdj, xmin)
end

"Golden-section minimization of a unimodal `f` on `[a,b]` (used only in the rare CoL-adjust refine branch)."
function _golden_section_min(f, a::Real, b::Real; tol::Real = 5e-4, maxit::Int = 100)
    invphi = (sqrt(5) - 1) / 2
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c); fd = f(d)
    for _ in 1:maxit
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a); fd = f(d)
        end
        abs(b - a) < tol && break
    end
    return (a + b) / 2
end

"""
    peak_overlap_pos(t, xb1, Jb1, t01, Eb01, Qb01, xb2, Jb2, t02, Eb02, Qb02, HBR;
                     params=PeakOverlapParams()) -> POPResult

Iteratively find the peak-overlap position of the primary/secondary position
PDFs at time `t` (offsets `t01`,`t02` are the epochs of the equinoctial PDFs;
here always 0). Port of `PeakOverlapPos.m`. All inputs in km / km/s; equinoctial
states `Eb0*` = [n,af,ag,chi,psi,lM]; `Qb0*` = 6x6 equinoctial covariances;
`Jb*` = dX/dE at the mean state; `xb*` = mean cartesian state.
"""
function peak_overlap_pos(t::Real,
                          xb1::AbstractVector, Jb1::AbstractMatrix, t01::Real,
                          Eb01::AbstractVector, Qb01::AbstractMatrix,
                          xb2::AbstractVector, Jb2::AbstractMatrix, t02::Real,
                          Eb02::AbstractVector, Qb02::AbstractMatrix,
                          HBR::Real; params::PeakOverlapParams = PeakOverlapParams())

    avgiter = min(35, round(Int, params.maxiter * 0.35))
    acciter = min(25, round(Int, params.maxiter * 0.25))
    osciter = min(15, round(Int, params.maxiter * 0.15))

    SigpRem0 = Diagonal(fill(HBR^2, 3))
    Lclip = (HBR * params.Fclip)^2
    GM = params.GM
    twopi = 2π
    I3x3 = Matrix{Float64}(I, 3, 3)

    sinLb01 = sin(Eb01[6]); sinLb02 = sin(Eb02[6])
    cosLb01 = cos(Eb01[6]); cosLb02 = cos(Eb02[6])

    dt01 = t - t01; dt02 = t - t02

    xs1 = Vector{Float64}(xb1); Js1 = Matrix{Float64}(Jb1); Es01 = Vector{Float64}(Eb01)
    xs2 = Vector{Float64}(xb2); Js2 = Matrix{Float64}(Jb2); Es02 = Vector{Float64}(Eb02)

    iterating = true; iteration = 0
    converged = 0.0                       # 0/false; set to 1.0/0.5/frac on success
    failure = 0
    mup_old = fill(NaN, 3); mup_old2 = fill(NaN, 3); iterAdj = -Inf
    dE01 = zeros(6); dE02 = zeros(6)

    # Values that must survive the loop for output assembly.
    mup = fill(NaN, 3); vu1p = fill(NaN, 3); vu2p = fill(NaN, 3)
    Ps1 = zeros(6, 6); Ps2 = zeros(6, 6)
    Sigp = zeros(3, 3); Sigpinv = zeros(3, 3)
    SigpRem = zeros(3, 3); SigpReminv = zeros(3, 3)
    vu1p_old = fill(NaN, 3); vu2p_old = fill(NaN, 3)

    while iterating
        # Offset states for the current iteration.
        if iteration == 0
            dE01 = zeros(6); dE02 = zeros(6)
            xu1 = copy(xs1); xu2 = copy(xs2)
        else
            dE01[1:5] = Eb01[1:5] .- Es01[1:5]
            dE01[6] = asin(sinLb01 * cos(Es01[6]) - cosLb01 * sin(Es01[6]))
            dE02[1:5] = Eb02[1:5] .- Es02[1:5]
            dE02[6] = asin(sinLb02 * cos(Es02[6]) - cosLb02 * sin(Es02[6]))
            xu1 = xs1 .+ Js1 * dE01
            xu2 = xs2 .+ Js2 * dE02
        end

        ru1 = xu1[1:3]; vu1 = xu1[4:6]
        ru2 = xu2[1:3]; vu2 = xu2[4:6]

        Ps1 = Js1 * Qb01 * Js1'; Ps1 = (Ps1 + Ps1') / 2
        Ps2 = Js2 * Qb02 * Js2'; Ps2 = (Ps2 + Ps2') / 2

        As1 = Ps1[1:3, 1:3]; Bs1 = Ps1[4:6, 1:3]
        As2 = Ps2[1:3, 1:3]; Bs2 = Ps2[4:6, 1:3]

        As1inv = _clip_inv(As1, Lclip)
        As2inv = _clip_inv(As2, Lclip)

        Sigpinv = As1inv + As2inv
        Sigp = _clip_inv(Sigpinv, Lclip)

        mup = Sigp * (As1inv * ru1 + As2inv * ru2)

        SigpRem = Sigp + SigpRem0
        SigpReminv = SigpRem \ I3x3

        if params.maxiter <= 1
            iterating = false; converged = 1.0
            vu1p = vu1; vu2p = vu2
        else
            # mu-point averaging to accelerate slow convergence.
            if iteration > avgiter && iteration > iterAdj + 2
                mup = 0.5 * (mup .+ mup_old)
            end

            beta1 = Bs1 * As1inv
            rs1, Energy1, Adj1 = _adjust_pop_col(mup, xs1[1:3], beta1, xu1, GM, params.MaxRelEnergyChange)
            beta2 = Bs2 * As2inv
            rs2, Energy2, Adj2 = _adjust_pop_col(mup, xs2[1:3], beta2, xu2, GM, params.MaxRelEnergyChange)
            if Adj1 != 1 || Adj2 != 1
                iterAdj = iteration
            end

            vu1p = vu1 .+ beta1 * (rs1 .- ru1)
            vu2p = vu2 .+ beta2 * (rs2 .- ru2)

            if Energy1 >= 0 || Energy2 >= 0
                iterating = false; converged = 0.0
                failure = 10 * (Energy1 >= 0) + (Energy2 >= 0)
            else
                xs1 = vcat(rs1, vu1p)
                xs2 = vcat(rs2, vu2p)

                if iteration > 0
                    dmup = mup_old .- mup
                    dMD2 = dmup' * SigpReminv * dmup

                    if dMD2 <= params.MD2tol[1] && iteration != iterAdj
                        iterating = false; converged = 1.0
                    elseif iteration >= params.maxiter
                        iterating = false; converged = 0.0
                    elseif iteration > osciter && iteration > iterAdj + 2
                        dmuposc = mup_old2 .- mup
                        dMD2osc = dmuposc' * SigpReminv * dmuposc
                        if dMD2osc <= params.MD2tol[1]
                            iterating = false; converged = 0.5
                        else
                            if iteration > avgiter
                                MD2cut = params.MD2tol[2]; omfrc = 0.0
                            elseif iteration <= acciter
                                MD2cut = params.MD2tol[1]; omfrc = 1.0
                            else
                                frc = (iteration - acciter) / (avgiter - acciter)
                                omfrc = 1 - frc
                                MD2cut = exp(omfrc * log(params.MD2tol[1]) +
                                             frc * log(params.MD2tol[2]))
                            end
                            if dMD2 <= MD2cut
                                iterating = false; converged = 0.1 + omfrc / 4
                            end
                        end
                    end
                end

                # Equinoctial elements at the current POP expansion-center states.
                bad1s = false; bad2s = false
                local e1, e2
                try
                    e1 = cartesian_to_equinoctial(xs1[1:3], xs1[4:6])
                catch
                    bad1s = true
                end
                try
                    e2 = cartesian_to_equinoctial(xs2[1:3], xs2[4:6])
                catch
                    bad2s = true
                end

                if bad1s || bad2s
                    iterating = false; converged = 0.0
                    failure = 1000 * bad1s + 100 * bad2s
                else
                    a1s, n1s, af1s, ag1s, chi1s, psi1s, lM1s, _ = e1
                    a2s, n2s, af2s, ag2s, chi2s, psi2s, lM2s, _ = e2
                    esq1s = af1s^2 + ag1s^2; unbound1s = (a1s <= 0) || esq1s >= 1
                    esq2s = af2s^2 + ag2s^2; unbound2s = (a2s <= 0) || esq2s >= 1

                    if unbound1s || unbound2s
                        iterating = false; converged = 0.0
                        failure = 10 * unbound1s + unbound2s
                    else
                        lM10s = mod(lM1s - n1s * dt01, twopi)
                        lM20s = mod(lM2s - n2s * dt02, twopi)
                        Es01 = [n1s, af1s, ag1s, chi1s, psi1s, lM10s]
                        Es02 = [n2s, af2s, ag2s, chi2s, psi2s, lM20s]

                        Js1, _ = jacobian_E0_to_Xt(dt01, Es01)
                        Js2, _ = jacobian_E0_to_Xt(dt02, Es02)

                        if iterating
                            mup_old2 = copy(mup_old)
                            mup_old = copy(mup)
                            vu1p_old = copy(vu1p)
                            vu2p_old = copy(vu2p)
                            iteration += 1
                        end
                    end
                end
            end
        end
    end

    aux = Dict{Symbol,Any}(:converged => converged, :iteration => iteration,
                           :failure => failure, :iterAdj => iterAdj)
    if converged > 0
        aux[:xs1] = xs1; aux[:Js1] = Js1; aux[:Es01] = Es01
        aux[:xs2] = xs2; aux[:Js2] = Js2; aux[:Es02] = Es02
        aux[:Sigp] = Sigp; aux[:Sigpinv] = Sigpinv
        aux[:SigpRem] = SigpRem; aux[:SigpReminv] = SigpReminv
        aux[:Ps1] = Ps1; aux[:Ps2] = Ps2

        if params.maxiter <= 1
            aux[:xu1] = Vector{Float64}(xb1); aux[:dE01] = zeros(6)
            aux[:xu2] = Vector{Float64}(xb2); aux[:dE02] = zeros(6)
        else
            dE01[1:5] = Eb01[1:5] .- Es01[1:5]
            dE01[6] = asin(sinLb01 * cos(Es01[6]) - cosLb01 * sin(Es01[6]))
            dE02[1:5] = Eb02[1:5] .- Es02[1:5]
            dE02[6] = asin(sinLb02 * cos(Es02[6]) - cosLb02 * sin(Es02[6]))
            aux[:xu1] = xs1 .+ Js1 * dE01; aux[:dE01] = copy(dE01)
            aux[:xu2] = xs2 .+ Js2 * dE02; aux[:dE02] = copy(dE02)
        end
    end

    return POPResult(converged, mup, vu1p, vu2p, aux)
end

"Eigenvalue-clip inverse of a symmetric 3x3 (matches the inline clip in PeakOverlapPos)."
function _clip_inv(A::AbstractMatrix, Lclip::Real)
    F = eigen(Symmetric(Matrix(A)))
    L = collect(F.values); V = F.vectors
    L[L .< Lclip] .= Lclip
    return V * Diagonal(1.0 ./ L) * V'
end

"""
    peak_overlap_md2(t, Eb10, Qb10, Eb20, Qb20, HBR; EMD2=1, params) -> (MD2, Xu, Ps, Asdet, Asinv, converged, aux)

Effective (modified) Mahalanobis distance² at the peak-overlap point for time
`t`. Port of `PeakOverlapMD2.m` with cross-correlation processing OFF (CDMs
carry no DCP sensitivity vectors). Epoch times t10=t20=0.

`EMD2` controls the "modified" MD²:
  0 → plain MD²;  1 → MD² + log|As|;  2 → MD² + log|As| − 2·log|vu|.
The usage-violation analysis uses `EMD2=1` (the Q-function). Returns `MD2=NaN`
and empty companions when the POP solve does not converge.
"""
function peak_overlap_md2(t::Real,
                          Eb10::AbstractVector, Qb10::AbstractMatrix,
                          Eb20::AbstractVector, Qb20::AbstractMatrix,
                          HBR::Real; EMD2::Int = 1,
                          params::PeakOverlapParams = PeakOverlapParams())
    Jb1, xb1 = jacobian_E0_to_Xt(t, Eb10)
    Jb2, xb2 = jacobian_E0_to_Xt(t, Eb20)

    pop = peak_overlap_pos(t, xb1, Jb1, 0.0, Eb10, Qb10,
                              xb2, Jb2, 0.0, Eb20, Qb20, HBR; params = params)

    if pop.conv > 0
        Xu = pop.aux[:xu2] .- pop.aux[:xu1]
        ru = Xu[1:3]
        Ps = pop.aux[:Ps1] .+ pop.aux[:Ps2]         # XCprocessing off

        As = Ps[1:3, 1:3]
        Lclip = (params.Fclip * HBR)^2
        F = eigen(Symmetric(Matrix(As)))
        Leig = collect(F.values); Veig = F.vectors
        Leig[Leig .< Lclip] .= Lclip
        Asdet = Leig[1] * Leig[2] * Leig[3]
        Asinv = Veig * Diagonal(1.0 ./ Leig) * Veig'
        aux = Dict{Symbol,Any}(:As => As, :AsLeig => Leig, :AsVeig => Veig,
                               :AsLclip => Lclip)

        MD2 = ru' * Asinv * ru
        aux[:MD2actual] = MD2
        if EMD2 == 1
            MD2 = MD2 + log(Asdet)
        elseif EMD2 == 2
            MD2 = MD2 + log(Asdet) - 2 * log(norm(Xu[4:6]))
        end
        return (MD2, Xu, Ps, Asdet, Asinv, pop.conv, aux)
    else
        return (NaN, Float64[], zeros(0, 0), NaN, zeros(0, 0), pop.conv,
                Dict{Symbol,Any}())
    end
end