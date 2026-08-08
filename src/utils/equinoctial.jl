# equinoctial.jl — equinoctial-element orbital mechanics for the curvilinear
# usage-violation analysis (foundation layer for the PeakOverlapPos port).
#
# Faithful Julia port of NASA CARA's equinoctial transforms + Jacobians:
#   convert_cartesian_to_equinoctial.m   (DistributedMatlab/Utils/OrbitTransformations)
#   convert_equinoctial_to_cartesian.m   (same)
#   jacobian_equinoctial_to_cartesian.m  (same)
#   jacobian_E0_to_Xt.m                  (ProbabilityOfCollision/Utils)
#
# Reference: Vallado & Alfano (2015), AAS 15-537, "Updated Analytical Partials
# for Covariance Transformations and Optimization"; Broucke & Cefola (1972).
#
# WHY EQUINOCTIAL (not Keplerian / not Brahe): Hall (2021)'s peak-overlap-point
# method linearizes the two-body motion in equinoctial elements
# (n, af, ag, chi, psi, lM), which stay non-singular at zero eccentricity and
# zero inclination — exactly the near-circular LEO regime of conjunction
# objects, where classical Keplerian elements (and Brahe's Keplerian API) go
# singular. These are pure closed-form transforms; no external dependency.
#
# UNITS: this module works in km, km/s, and mu in km^3/s^2 (EGM-96), matching
# CARA's internal convention (EquinoctialMatrices.m divides the m-unit state by
# 1e3 before calling these). Callers in metres must convert.
#
# The equinoctial state vector is E = [n, af, ag, chi, psi, lM] with:
#   n   = mean motion (rad/s)
#   af, ag = eccentricity-vector components in the equinoctial frame
#   chi, psi = node/inclination components
#   lM  = mean longitude (rad)
# fr = +1 (prograde). Retrograde (fr = -1) is not exercised by real CARA
# conjunctions (see UsageViolationPc2D.m:196-199) and is not supported here.

using LinearAlgebra

# EGM-96 gravitational parameter in km^3/s^2 (orbit_period.m uses the m-unit
# value 3.986004418e14; here we are in km).
const EQ_MU_KM = 3.986004418e5

"""
    cartesian_to_equinoctial(rvec, vvec; mu=EQ_MU_KM) -> (a, n, af, ag, chi, psi, lM, F)

Convert a cartesian state (km, km/s) to equinoctial elements. Prograde (fr=+1).
Throws for unbound / equatorial-retrograde orbits (mirrors CARA's early returns).

Port of `convert_cartesian_to_equinoctial.m` (bound-orbit path, fr=+1).
"""
function cartesian_to_equinoctial(rvec::AbstractVector, vvec::AbstractVector;
                                  mu::Real = EQ_MU_KM)
    fr = 1
    rvec = Vector{Float64}(rvec[1:3]); vvec = Vector{Float64}(vvec[1:3])

    r  = sqrt(rvec[1]^2 + rvec[2]^2 + rvec[3]^2)
    v2 = vvec[1]^2 + vvec[2]^2 + vvec[3]^2
    a  = mu * r / (2 * mu - v2 * r)
    (a <= 1e-5 || !isfinite(a)) &&
        throw(ArgumentError("Unbound orbit: nonpositive/infinite semimajor axis a=$a"))

    rdv = dot(rvec, vvec)
    rcv = [rvec[2] * vvec[3] - rvec[3] * vvec[2],
           rvec[3] * vvec[1] - rvec[1] * vvec[3],
           rvec[1] * vvec[2] - rvec[2] * vvec[1]]

    n = sqrt(mu / a^3)
    evec = (1 / mu) * ((v2 - mu / r) * rvec - rdv * vvec)
    dot(evec, evec) >= 1 &&
        throw(ArgumentError("Unbound orbit: ecc^2 = e·e >= 1"))

    what = rcv / sqrt(rcv[1]^2 + rcv[2]^2 + rcv[3]^2)
    (what[3] + fr <= 1e-10) &&
        throw(ArgumentError("Equatorial retrograde orbit (i≈180°) not supported"))

    cpden = 1 + fr * what[3]
    chi =  what[1] / cpden
    psi = -what[2] / cpden

    chi2 = chi^2; psi2 = psi^2; C = 1 + chi2 + psi2
    fhat = [1 - chi2 + psi2, 2 * chi * psi,     -2 * fr * chi] / C
    ghat = [2 * fr * chi * psi, (1 + chi2 - psi2) * fr, 2 * psi] / C

    af = dot(fhat, evec)
    ag = dot(ghat, evec)
    af2 = af^2; ag2 = ag^2; ec2 = ag2 + af2
    ec2 >= 1 && throw(ArgumentError("Unbound orbit: ecc^2 = af^2+ag^2 >= 1"))

    X = dot(fhat, rvec)
    Y = dot(ghat, rvec)

    safg = sqrt(1 - ag2 - af2)
    b = 1 / (1 + safg)
    Fden = a * safg
    bagaf = b * ag * af

    sinF = ag + ((1 - ag2 * b) * Y - bagaf * X) / Fden
    cosF = af + ((1 - af2 * b) * X - bagaf * Y) / Fden
    F = atan(sinF, cosF)
    lM = F + ag * cosF - af * sinF

    return (a, n, af, ag, chi, psi, lM, F)
end

"""
    _equinoctial_kepeq(lam, af, ag; Ftol=100*eps(2π), maxiter=100) -> (F, cF, sF, converged)

Solve Kepler's equation in equinoctial elements for the eccentric longitude F,
by the Newton iteration of Vallado & Alfano (2015) eq. 10. Port of the
`equinoctial_kepeq` subfunction of `convert_equinoctial_to_cartesian.m`.

The MATLAB falls back to `fminbnd` only when the Newton step fails to converge,
which happens for ecc ≳ 0.9. Conjunction objects are near-circular LEO, so the
Newton step converges; if it ever does not, we throw rather than silently return
a wrong F (a wrong F would corrupt the whole curvilinear analysis).
"""
function _equinoctial_kepeq(lam::Real, af::Real, ag::Real;
                            Ftol::Real = 100 * eps(2π), maxiter::Int = 100)
    lam = mod(lam, 2π)
    ecc2 = af^2 + ag^2
    ecc2 >= 1 && throw(ArgumentError("equinoctial_kepeq: eccentricity^2 >= 1"))

    F = lam; cF = cos(F); sF = sin(F)
    iter = 0
    converged = false
    while true
        Fdel = (F + ag * cF - af * sF - lam) / (1 - ag * sF - af * cF)
        F -= Fdel; cF = cos(F); sF = sin(F)
        absFdel = abs(Fdel)
        if absFdel < Ftol
            converged = true
            break
        elseif iter >= maxiter || absFdel >= π
            break
        else
            iter += 1
        end
    end
    converged || throw(ArgumentError(
        "equinoctial_kepeq failed to converge (Newton); high-ecc fminbnd " *
        "fallback is not ported — not expected for near-circular LEO."))
    return (F, cF, sF, converged)
end

"""
    equinoctial_to_cartesian(E, T; mu=EQ_MU_KM) -> (rvec, vvec)

Propagate an epoch equinoctial state `E = [n,af,ag,chi,psi,lM]` forward by time
offset `T` (s) under two-body motion and return the cartesian state (km, km/s).
Prograde (fr=+1). Port of `convert_equinoctial_to_cartesian.m` (single T).
"""
function equinoctial_to_cartesian(E::AbstractVector, T::Real; mu::Real = EQ_MU_KM)
    fr = 1
    n, af, ag, chi, psi, lam0 = E[1], E[2], E[3], E[4], E[5], E[6]

    a3 = mu / n^2
    a = cbrt(a3)
    na = n * a

    lam = lam0 + n * T

    ag2 = ag^2; af2 = af^2
    B = sqrt(1 - ag2 - af2)
    b = 1 / (1 + B)
    omag2b = 1 - ag2 * b
    omaf2b = 1 - af2 * b
    afagb = af * ag * b

    chi2 = chi^2; psi2 = psi^2; C = 1 + chi2 + psi2
    fhat = [1 - chi2 + psi2, 2 * chi * psi,      -2 * fr * chi] / C
    ghat = [2 * fr * chi * psi, (1 + chi2 - psi2) * fr, 2 * psi] / C

    F, cF, sF, _ = _equinoctial_kepeq(lam, af, ag)

    X = a * (omag2b * cF + afagb * sF - af)
    Y = a * (omaf2b * sF + afagb * cF - ag)
    rvec = X * fhat + Y * ghat

    na2or = na / (1 - af * cF - ag * sF)
    Xdot = na2or * (afagb * cF - omag2b * sF)
    Ydot = na2or * (omaf2b * cF - afagb * sF)
    vvec = Xdot * fhat + Ydot * ghat

    return (rvec, vvec)
end

"""
    jacobian_equinoctial_to_cartesian(E, X; mu=EQ_MU_KM) -> J (6x6)

Jacobian J = dX/dE between equinoctial elements `E = [n,af,ag,chi,psi,lM]` and
cartesian state `X = [r; v]` (km, km/s), evaluated at the pair (E, X) that map
to each other. Prograde (fr=+1). Port of `jacobian_equinoctial_to_cartesian.m`.
"""
function jacobian_equinoctial_to_cartesian(E::AbstractVector, X::AbstractVector;
                                           mu::Real = EQ_MU_KM)
    fr = 1
    n, af, ag, chi, psi = E[1], E[2], E[3], E[4], E[5]
    rvec = Vector{Float64}(X[1:3]); vvec = Vector{Float64}(X[4:6])

    r2 = dot(rvec, rvec); r = sqrt(r2); r3 = r2 * r
    a3 = mu / n^2; a = cbrt(a3); A = n * a^2

    ag2 = ag^2; af2 = af^2; B = sqrt(1 - ag2 - af2)
    chi2 = chi^2; psi2 = psi^2; C = 1 + chi2 + psi2

    fhat = [1 - chi2 + psi2, 2 * chi * psi,      -2 * fr * chi] / C
    ghat = [2 * fr * chi * psi, (1 + chi2 - psi2) * fr, 2 * psi] / C
    what = [2 * chi, -2 * psi, (1 - chi2 - psi2) * fr] / C

    Xc = dot(fhat, rvec)
    Yc = dot(ghat, rvec)
    Xd = dot(fhat, vvec)
    Yd = dot(ghat, vvec)

    AB = A * B; Bp1 = B + 1; nBp1 = n * Bp1; Aor3 = A / r3

    dXdaf =  ag * Xd / nBp1 + a * (Yc * Xd / AB - 1)
    dYdaf =  ag * Yd / nBp1 - a * (Xc * Xd / AB)
    dXdag = -af * Xd / nBp1 + a * (Yc * Yd / AB)
    dYdag = -af * Yd / nBp1 - a * (Xc * Yd / AB + 1)

    dXddaf =  a * Xd * Yd / AB - Aor3 * (a * ag * Xc / Bp1 + Xc * Yc / B)
    dYddaf = -a * Xd * Xd / AB - Aor3 * (a * ag * Yc / Bp1 - Xc * Xc / B)
    dXddag =  a * Yd * Yd / AB + Aor3 * (a * af * Xc / Bp1 - Yc * Yc / B)
    dYddag = -a * Xd * Yd / AB + Aor3 * (a * af * Yc / Bp1 + Xc * Yc / B)

    J = Matrix{Float64}(undef, 6, 6)

    # d/dn
    cv = 1 / (3n); cr = -2cv
    J[1:3, 1] = cr * rvec
    J[4:6, 1] = cv * vvec
    # d/daf
    J[1:3, 2] = dXdaf * fhat + dYdaf * ghat
    J[4:6, 2] = dXddaf * fhat + dYddaf * ghat
    # d/dag
    J[1:3, 3] = dXdag * fhat + dYdag * ghat
    J[4:6, 3] = dXddag * fhat + dYddag * ghat
    # d/dchi
    cc = 2 / C
    J[1:3, 4] = cc * (fr * psi * (Yc * fhat - Xc * ghat) - Xc * what)
    J[4:6, 4] = cc * (fr * psi * (Yd * fhat - Xd * ghat) - Xd * what)
    # d/dpsi
    J[1:3, 5] = cc * (fr * chi * (Xc * ghat - Yc * fhat) + Yc * what)
    J[4:6, 5] = cc * (fr * chi * (Xd * ghat - Yd * fhat) + Yd * what)
    # d/dlM
    J[1:3, 6] = vvec / n
    J[4:6, 6] = (-n * a3 / r3) * rvec

    return J
end

"""
    jacobian_E0_to_Xt(T, E0; mu=EQ_MU_KM) -> (JT, XT)

Given epoch equinoctial state `E0`, compute the cartesian state `XT = [r; v]`
(km, km/s) at time offset `T` (s) and the Jacobian `JT = dX(T)/dE0` used as the
first-order Taylor expansion of two-body motion about a center-of-linearization:
    X(t) = Xcent(t) + JT · (E0 − Ecent0).
Port of `jacobian_E0_to_Xt.m` (single T). Prograde (fr=+1).
"""
function jacobian_E0_to_Xt(T::Real, E0::AbstractVector; mu::Real = EQ_MU_KM)
    rT, vT = equinoctial_to_cartesian(E0, T; mu = mu)
    XT = vcat(rT, vT)

    # dE(t)/dE(t0) STM: identity except the mean-longitude drift lM(t)=lM0+n·T,
    # so ∂lM(t)/∂n = T (phi[6,1] = T).
    phi = Matrix{Float64}(I, 6, 6)
    phi[6, 1] = T

    # Equinoctial state at time t: only lM advances.
    ET = collect(Float64, E0)
    ET[6] = E0[6] + T * E0[1]

    J = jacobian_equinoctial_to_cartesian(ET, XT; mu = mu)   # dX(t)/dE(t)
    JT = J * phi                                             # dX(t)/dE0
    return (JT, XT)
end