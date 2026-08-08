# computePc.jl — probability-of-collision methods.
#
# Four 2D Pc methods live here. They all evaluate the same underlying integral —
# a 2D Gaussian over the hard-body disk in the conjunction plane — and differ
# only in how they do it. `chan_pc`, `elrod_pc` and `numeric_pc` take identical
# arguments (two ECI states, two ECI covariances, a combined HBR) and so are
# directly interchangeable.
#
#   chan_pc     Chan (1997) noncentral chi-squared series. FAST. What the MCTS
#               planner calls. Accurate to ~1% on well-conditioned geometry, but
#               degrades badly when the encounter-plane covariance is highly
#               elongated — see the anisotropy warning on the function itself.
#
#   elrod_pc    Chebyshev-Gauss quadrature ("error function" method). Also fast,
#               and does NOT have the anisotropy weakness. Port of NASA CARA's
#               PcElrod.m.
#
#   numeric_pc  Direct numerical integration. Ground truth, but SLOW —
#               a test oracle, not a rollout method. See its docstring.
#
#   fosterPcAnalytical
#               Legacy, PROBABLY INCORRECT — see the warning above it. Not used
#               by the planner. Do not use without fixing it first.
#
# All of the above were cross-validated against NASA CARA's 53 real conjunctions;
# findings and the accuracy envelope are written up in
# notes/cara_validation_findings.md.

using HCubature
using Distributions
using LinearAlgebra
using Random
using SpecialFunctions: erfc

# -------------------------------------------------------
# Helper
# -------------------------------------------------------

function epoch_to_tuple(epoch)
    dt = epoch.to_datetime()
    return (dt[1], dt[2], dt[3], dt[4], dt[5], dt[6], dt[7])
end

# -------------------------------------------------------
# Core integration for Foster method
# -------------------------------------------------------

function integrate_circle(gaussian, radius)
    f(polarcoords) = polarcoords[1] * pdf(gaussian, 
        [polarcoords[1] * cos(polarcoords[2]), 
         polarcoords[1] * sin(polarcoords[2])])
    result, _ = hcubature(f, (0, 0), (radius, 2*π))
    return result
end

# -------------------------------------------------------
# Foster Analytical Pc
#
# !! WARNING — THIS FUNCTION IS SUSPECT. NEEDS CHECKING BEFORE ANY USE. !!
#
# It predates the current work and has not been validated. A read-through
# against the real Foster/Estes (1992) formulation and against NASA CARA's
# Pc2D_Foster.m turned up three concrete problems:
#
#   1. It is not actually the Foster method. Real Foster projects the COMBINED
#      position covariance onto the plane perpendicular to the relative
#      velocity. This builds an ad-hoc axis set from cross(v1, v2) and the
#      difference of velocity unit vectors, then assembles a 2x2 covariance
#      from hand-picked RTN variance entries with sin^2/cos^2 weights.
#
#   2. It discards the off-diagonal term (`uw_cov = [u_var 0; 0 w_var]`). Real
#      conjunction covariances are strongly correlated in the encounter plane;
#      dropping that correlation can move Pc by an order of magnitude.
#
#   3. It sums the two objects' RTN covariances component-wise. Each object has
#      its OWN RTN frame — those are different bases, so the sum is not
#      meaningful. Covariances must be combined in a common frame (ECI).
#      `chan_pc` does this correctly; this does not.
#
# It also degenerates when v1 is nearly parallel to v2 (the low-relative-
# velocity case), where cross(v1, v2) is numerically meaningless and the guard
# below silently returns 0.0 — reporting perfect safety on exactly the hardest
# conjunctions.
#
# The planner does NOT call this: every live call site (beliefMCTS.jl,
# covarianceTable.jl, beliefTracker.jl) routes to `chan_pc`. This is reachable
# only via the `compute_pc` dispatcher, used by src/examples/test_Pc.jl.
#
# Recommendation: delete it, or point `:foster` at `elrod_pc`, rather than try
# to repair it. Kept for now only so old example scripts still run.
# -------------------------------------------------------

function fosterPcAnalytical(object1_x, object1_Σ, object2_x, object2_Σ;
                             object1_radius=10, object2_radius=10)
    # Safety check for NaN/Inf in inputs
    if any(isnan.(object1_x)) || any(isinf.(object1_x)) || 
       any(isnan.(object2_x)) || any(isinf.(object2_x)) ||
       any(isnan.(object1_Σ)) || any(isinf.(object1_Σ)) ||
       any(isnan.(object2_Σ)) || any(isinf.(object2_Σ))
        return 0.0
    end

    u_var = object1_Σ[1, 1] + object2_Σ[1, 1]

    u_axis = cross(object1_x[4:6], object2_x[4:6])
    u_norm = norm(u_axis)
    if u_norm < 1e-10
        return 0.0
    end
    u_axis /= u_norm

    v_axis = object2_x[4:6] - object1_x[4:6]
    v_norm = norm(v_axis)
    if v_norm < 1e-10
        return 0.0
    end
    v_axis /= v_norm

    w_axis = cross(u_axis, v_axis)
    R_inv = hcat(u_axis, v_axis, w_axis)

    if any(isnan.(R_inv)) || any(isinf.(R_inv))
        return 0.0
    end

    R = inv(R_inv)

    debris_pos = object2_x[1:3] - object1_x[1:3]
    debris_pos = R * debris_pos
    U0 = debris_pos[1]
    W0 = debris_pos[3]

    theta_r_spacecraft = acos(clamp(dot(v_axis, object1_x[4:6]) / norm(object1_x[4:6]), -1.0, 1.0))
    theta_r_debris     = acos(clamp(dot(v_axis, object2_x[4:6]) / norm(object2_x[4:6]), -1.0, 1.0))

    w_var = object1_Σ[2, 2] * sin(theta_r_spacecraft)^2 + 
            object1_Σ[3, 3] * cos(theta_r_spacecraft)^2 + 
            object2_Σ[2, 2] * sin(theta_r_debris)^2 + 
            object2_Σ[3, 3] * cos(theta_r_debris)^2

    uw_mean = [U0, W0]
    uw_cov  = [u_var 0.0; 0.0 w_var]
    d_hb    = object1_radius + object2_radius

    gaussian = MvNormal(uw_mean, uw_cov)
    Pc = integrate_circle(gaussian, d_hb)


    println("    U0=$U0, W0=$W0")
    println("    u_var=$u_var, w_var=$w_var")
    println("    d_hb=$d_hb")
    return Pc
end

# -------------------------------------------------------
# Foster wrapper — spins up temporary propagators
# -------------------------------------------------------

function compute_pc_foster(pomdp::SpacecraftCAPOMDP,
                           sc_eci::Vector{Float64},
                           debris_eci::Vector{Float64},
                           epoch_current,
                           epoch_tca,
                           Σ_sc::Matrix{Float64},
                           Σ_debris::Matrix{Float64})

    epoch_current_tuple = epoch_to_tuple(epoch_current)
    prop_sc_tmp, _     = eci2orb_brahe(sc_eci, epoch_current_tuple, pomdp.satParams, 
                                        pomdp.forceModel, initial_covariance=Σ_sc)
    prop_debris_tmp, _ = eci2orb_brahe(debris_eci, epoch_current_tuple, pomdp.debrisParams, 
                                        pomdp.forceModel, initial_covariance=Σ_debris)
    # Propagate temporaries to TCA
    prop_sc_tmp.propagate_to(epoch_tca)
    prop_debris_tmp.propagate_to(epoch_tca)

    # Absolute ECI states at TCA for conjunction geometry
    sc_eci_tca     = collect(prop_sc_tmp.current_state()[1:6])
    debris_eci_tca = collect(prop_debris_tmp.current_state()[1:6])

    # RTN covariances at TCA from brahe
    P_sc_rtn     = collect(prop_sc_tmp.covariance_rtn(epoch_tca))
    P_debris_rtn = collect(prop_debris_tmp.covariance_rtn(epoch_tca))
    println("  Σ_debris[1,1] = $(Σ_debris[1,1])")
    println("  Σ_sc[1,1]     = $(Σ_sc[1,1])")
    println("  P_debris_rtn[1,1] at TCA = $(collect(prop_debris_tmp.covariance_rtn(epoch_tca))[1,1])")
    
    
    return fosterPcAnalytical(sc_eci_tca, P_sc_rtn, debris_eci_tca, P_debris_rtn,
                              object1_radius = pomdp.R_hard_body_sc,
                              object2_radius = pomdp.R_hard_body_debris)
end

# -------------------------------------------------------
# Monte Carlo Pc — STM-based, cheap
# -------------------------------------------------------
function compute_pc_mc(pomdp::SpacecraftCAPOMDP,
                       sc_eci::Vector{Float64},
                       debris_eci::Vector{Float64},
                       epoch_current,
                       epoch_tca,
                       Σ_sc::Matrix{Float64},
                       Σ_debris::Matrix{Float64};
                       N::Int = 10000)

    epoch_current_tuple = epoch_to_tuple(epoch_current)
    prop_sc_tmp, _     = eci2orb_brahe(sc_eci, epoch_current_tuple, pomdp.satParams, 
                                        pomdp.forceModel, initial_covariance=Σ_sc)
    prop_debris_tmp, _ = eci2orb_brahe(debris_eci, epoch_current_tuple, pomdp.debrisParams, 
                                        pomdp.forceModel, initial_covariance=Σ_debris)

    prop_sc_tmp.propagate_to(epoch_tca)
    prop_debris_tmp.propagate_to(epoch_tca)

    sc_eci_tca     = collect(prop_sc_tmp.current_state()[1:6])
    debris_eci_tca = collect(prop_debris_tmp.current_state()[1:6])
    P_sc_rtn       = collect(prop_sc_tmp.covariance_rtn(epoch_tca))
    P_debris_rtn   = collect(prop_debris_tmp.covariance_rtn(epoch_tca))

    # Build UVW conjunction plane — same as Foster
    u_axis = cross(sc_eci_tca[4:6], debris_eci_tca[4:6])
    u_norm = norm(u_axis)
    if u_norm < 1e-10; return 0.0; end
    u_axis /= u_norm

    v_axis = debris_eci_tca[4:6] - sc_eci_tca[4:6]
    v_norm = norm(v_axis)
    if v_norm < 1e-10; return 0.0; end
    v_axis /= v_norm

    w_axis = cross(u_axis, v_axis)
    R      = inv(hcat(u_axis, v_axis, w_axis))

    # Project mean into UW plane
    debris_pos = debris_eci_tca[1:3] - sc_eci_tca[1:3]
    debris_uvw = R * debris_pos
    U0 = debris_uvw[1]
    W0 = debris_uvw[3]

    # Build 2D covariance in UW plane
    theta_r_sc     = acos(clamp(dot(v_axis, sc_eci_tca[4:6])     / norm(sc_eci_tca[4:6]),     -1.0, 1.0))
    theta_r_debris = acos(clamp(dot(v_axis, debris_eci_tca[4:6]) / norm(debris_eci_tca[4:6]), -1.0, 1.0))

    u_var = P_sc_rtn[1,1] + P_debris_rtn[1,1]
    w_var = P_sc_rtn[2,2]     * sin(theta_r_sc)^2     +
            P_sc_rtn[3,3]     * cos(theta_r_sc)^2     +
            P_debris_rtn[2,2] * sin(theta_r_debris)^2 +
            P_debris_rtn[3,3] * cos(theta_r_debris)^2

    μ_uw = [U0, W0]
    Σ_uw = [u_var 0.0; 0.0 w_var]
    R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris

    # Target distribution p — same Gaussian as Foster
    p = MvNormal(μ_uw, Σ_uw)

    # Proposal distribution q — centered at origin, std = R_combined
    q = MvNormal([0.0, 0.0], diagm([R_combined^2, R_combined^2]))

    # Importance sampling
    samples  = rand(q, N)
    weights  = [pdf(p, samples[:, i]) / pdf(q, samples[:, i]) for i in 1:N]
    inside   = [norm(samples[:, i]) < R_combined for i in 1:N]

    Pc = mean(inside .* weights)

    return Pc
end

# -------------------------------------------------------
# Chan (1997) closed-form Pc
#
# Ported from ProbofCollision/src/collision/chan1997.py, which was validated
# against Fowler (1993) numerical integration to ~0.2% (see
# ProbofCollision/FINDINGS.md). This is a faithful port: same encounter-plane
# construction, same 2D projection, same noncentral-chi-squared series, so that
# it cross-validates against the Python reference on identical inputs.
#
# Reference: Chan, F.K. (1997), "Spacecraft Collision Probability," AAS 97-173.
# -------------------------------------------------------

"""
    chan_pc(sc1_eci, sc2_eci, cov1, cov2, hard_body_radius) -> Float64

Analytic probability of collision via the Chan (1997) series expansion.

Evaluates the same 2D-Gaussian-over-disk integral as the Foster/Fowler method
but in closed form, via the noncentral chi-squared CDF. Inputs are ECI states
and full 6x6 ECI covariances at TCA; only the 3x3 position blocks are used.

Arguments
- `sc1_eci`, `sc2_eci` : 6-element ECI states at TCA (m, m/s)
- `cov1`, `cov2`       : 6x6 ECI position-velocity covariances (m^2, ...)
- `hard_body_radius`   : combined hard-body radius of both objects (m)

Returns Pc in [0, 1]. Throws if the relative speed at TCA is zero (the
encounter plane is undefined).

!!! warning "Accuracy degrades on elongated covariances"
    Chan's series rescales the encounter-plane ellipse to a circle and applies a
    `σ1/σ2` correction. That approximation is good for round-ish ellipses and
    degrades as they elongate. Measured against NASA CARA reference values on 53
    real conjunctions (see `notes/cara_validation_findings.md`):

    | anisotropy σ2/σ1 | typical error |
    |------------------|---------------|
    | < 50             | ~1%           |
    | ~120             | tens of %     |
    | ~570             | up to 20x     |

    Real conjunction covariances are routinely elongated, because along-track
    uncertainty far exceeds radial. Roughly 13% of the CARA cases exceeded 100x.
    Use `covariance_anisotropy` to check, and `elrod_pc` when it is large.
"""
function chan_pc(sc1_eci::AbstractVector, sc2_eci::AbstractVector,
                 cov1::AbstractMatrix, cov2::AbstractMatrix,
                 hard_body_radius::Real)

    r1 = sc1_eci[1:3]; v1 = sc1_eci[4:6]
    r2 = sc2_eci[1:3]; v2 = sc2_eci[4:6]

    r_rel = r1 .- r2
    v_rel = v1 .- v2

    v_rel_mag = norm(v_rel)
    if v_rel_mag == 0.0
        error("Relative speed at TCA is zero — encounter plane is undefined.")
    end

    # ------------------------------------------------------------------
    # 1. Encounter-plane orthonormal basis (identical to chan1997.py)
    # ------------------------------------------------------------------
    z_hat = v_rel ./ v_rel_mag

    r_perp = r_rel .- dot(r_rel, z_hat) .* z_hat
    r_perp_mag = norm(r_perp)

    if r_perp_mag < 1e-10
        arbitrary = [1.0, 0.0, 0.0]
        if abs(dot(arbitrary, z_hat)) > 0.9
            arbitrary = [0.0, 1.0, 0.0]
        end
        r_perp = arbitrary .- dot(arbitrary, z_hat) .* z_hat
        r_perp_mag = norm(r_perp)
    end

    x_hat = r_perp ./ r_perp_mag
    y_hat = cross(z_hat, x_hat)

    # ------------------------------------------------------------------
    # 2. Combined position covariance and 2D projection
    # ------------------------------------------------------------------
    C_pos = cov1[1:3, 1:3] .+ cov2[1:3, 1:3]   # (3, 3)
    B = permutedims(hcat(x_hat, y_hat))         # (2, 3), rows = [x_hat; y_hat]
    C_2d = B * C_pos * transpose(B)             # (2, 2)
    miss_2d = B * r_rel                         # (2,)

    # ------------------------------------------------------------------
    # 3. Chan (1997) series
    # ------------------------------------------------------------------
    return _chan_series(miss_2d, C_2d, hard_body_radius)
end

"""
    _chan_series(miss, cov, radius) -> Float64

Integral of a 2D Gaussian (mean=`miss`, covariance=`cov`) over a disk of
`radius` centred at the origin, via diagonalisation to principal axes and the
Chan (1997) noncentral-chi-squared result:

    u  = (x0/σ1)^2 + (y0/σ2)^2      (Mahalanobis noncentrality)
    v  = R^2 / σ1^2                  (normalised HBR^2 on the smaller axis)
    Pc = (σ1/σ2) * ncx2.cdf(v; df=2, nc=u)
"""
function _chan_series(miss::AbstractVector, cov::AbstractMatrix, radius::Real)
    # eigen on a Symmetric matrix returns ascending eigenvalues (σ1^2 ≤ σ2^2),
    # matching numpy.linalg.eigh in the Python reference.
    F = eigen(Symmetric(cov))
    s1_sq = F.values[1]   # smaller variance
    s2_sq = F.values[2]   # larger  variance

    if s1_sq <= 0.0 || s2_sq <= 0.0
        return 0.0
    end

    # Miss vector in the principal-axis frame: eigvecs' * miss
    miss_p = transpose(F.vectors) * miss
    x0 = miss_p[1]
    y0 = miss_p[2]

    u = x0^2 / s1_sq + y0^2 / s2_sq   # total Mahalanobis noncentrality
    v = radius^2 / s1_sq               # normalised HBR^2 (smaller axis)

    aniso_correction = sqrt(s1_sq / s2_sq)   # σ1/σ2 ≤ 1

    pc = aniso_correction * cdf(NoncentralChisq(2, u), v)
    return clamp(pc, 0.0, 1.0)
end

# ---------------------------------------------------------------------------
# Shared: projection onto the 2D encounter plane
# ---------------------------------------------------------------------------

"""
    encounter_plane(r1, v1, r2, v2, C1, C2) -> (miss_2d, C_2d)

Project the relative position and combined position covariance onto the 2D
conjunction plane — the plane perpendicular to the relative velocity vector.

This is the common front end of every 2D Pc method: the methods differ only in
how they integrate the resulting 2D Gaussian over the hard-body disk.

The basis follows CARA's relative encounter frame (PcElrod.m):
  y = relative velocity direction
  z = relative angular momentum direction (r_rel x v_rel)
  x = y x z   (completes the right-handed set; lies in the miss-distance plane)
The Pc integral is taken over the x-z plane, i.e. the plane normal to y.

Returns the 2-element projected miss vector and the 2x2 projected covariance.
"""
function encounter_plane(r1::AbstractVector, v1::AbstractVector,
                         r2::AbstractVector, v2::AbstractVector,
                         C1::AbstractMatrix, C2::AbstractMatrix)

    r_rel = r1 .- r2
    v_rel = v1 .- v2

    vmag = norm(v_rel)
    vmag == 0.0 && error("Relative speed at TCA is zero — encounter plane is undefined.")

    h = cross(r_rel, v_rel)
    hmag = norm(h)
    if hmag < 1e-12
        # Exactly head-on: the relative angular momentum vanishes and the
        # in-plane axes are arbitrary. Any basis perpendicular to v_rel works.
        y_hat = v_rel ./ vmag
        tmp = abs(y_hat[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
        z_hat = normalize(cross(y_hat, tmp))
        x_hat = cross(y_hat, z_hat)
    else
        y_hat = v_rel ./ vmag
        z_hat = h ./ hmag
        x_hat = cross(y_hat, z_hat)
    end

    # Rows are the two in-plane basis vectors, so B maps a 3-vector to the plane.
    B = permutedims(hcat(x_hat, z_hat))          # (2, 3)
    C_pos = C1[1:3, 1:3] .+ C2[1:3, 1:3]         # combine in ECI, then project
    C_2d = B * C_pos * transpose(B)              # (2, 2)
    miss_2d = B * r_rel                          # (2,)

    return miss_2d, C_2d
end

# ---------------------------------------------------------------------------
# Elrod: Chebyshev-Gauss quadrature / error-function method
# ---------------------------------------------------------------------------

"""
    gauss_chebyshev_nodes(n) -> (nodes, weights)

Gauss-Chebyshev quadrature nodes and weights, matching CARA's `GenGCQuad.m`.
`n` must be even. Only the upper half of the (symmetric) node set is returned,
since the Elrod summation exploits that symmetry.

The weight convention includes the 1/sqrt(8*pi) normalisation CARA folds in.
"""
function gauss_chebyshev_nodes(n::Int)
    iseven(n) || error("Chebyshev order must be even, got $n")
    c = pi / (n + 1)
    v = c .* (n:-1:1)
    x = cos.(v)
    y = sqrt.(max.(0.0, 1 .- x .^ 2))
    w = c .* y ./ sqrt(8pi)
    half = (n ÷ 2 + 1):n
    return collect(x[half]), collect(w[half])
end

"""
    revchol2x2(C) -> (u11, u12, u22)

Reverse Cholesky factorisation of a 2x2 symmetric positive-definite matrix:
finds upper-triangular U with `U * U' = C` (as opposed to the standard
Cholesky's `L * L' = C`). Returns the three distinct entries.

Returns `nothing` if C is not positive-definite in this factorisation.
"""
function revchol2x2(C::AbstractMatrix)
    c11, c12, c22 = C[1, 1], C[1, 2], C[2, 2]
    c22 <= 0 && return nothing
    u22 = sqrt(c22)
    u12 = c12 / u22
    rad = c11 - u12^2
    rad <= 0 && return nothing
    u11 = sqrt(rad)
    return (u11, u12, u22)
end

"""
    remediate_2x2(C, hbr) -> (C_rem, is_pos_def, is_remediated)

Eigenvalue-clipping remediation for a non-positive-definite 2x2 covariance,
following CARA's `RemediateCovariance2x2.m`. Negative or vanishing eigenvalues
are clipped to `(f * hbr)^2`, trying progressively larger clipping factors until
the result factorises.

Validation against 53 real CDMs found 0 non-positive-definite cases, so this
path is rarely exercised on operational data — but Kalman-propagated covariances
can drift out of positive-definiteness numerically, so the guard matters here.
"""
function remediate_2x2(C::AbstractMatrix, hbr::Real)
    revchol2x2(C) !== nothing && minimum(eigvals(Symmetric(Matrix(C)))) > 0 &&
        return (C, true, false)

    for f in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
        Lclip = (f * hbr)^2
        F = eigen(Symmetric(Matrix(C)))
        vals = max.(F.values, Lclip)
        C_rem = F.vectors * Diagonal(vals) * transpose(F.vectors)
        if revchol2x2(C_rem) !== nothing
            return (C_rem, true, true)
        end
    end
    return (C, false, true)
end

"""
    elrod_pc(sc1_eci, sc2_eci, cov1, cov2, hard_body_radius; order=64) -> Float64

Probability of collision via the Chebyshev-Gauss quadrature (error function)
method of Elrod, as implemented in NASA CARA's `PcElrod.m`.

Evaluates the same 2D-Gaussian-over-disk integral as `chan_pc`, but with
Gauss-Chebyshev quadrature over the disk rather than Chan's noncentral
chi-squared series. It is accurate across the full range of covariance
anisotropy, where the Chan series degrades badly for elongated ellipses.

Arguments
- `sc1_eci`, `sc2_eci` : 6-element ECI states at TCA (m, m/s)
- `cov1`, `cov2`       : 6x6 (or 3x3) ECI covariances (m^2, ...)
- `hard_body_radius`   : combined hard-body radius of both objects (m)
- `order`              : Chebyshev order, must be even (default 64, as CARA)

Returns Pc in [0, 1].
"""
function elrod_pc(sc1_eci::AbstractVector, sc2_eci::AbstractVector,
                  cov1::AbstractMatrix, cov2::AbstractMatrix,
                  hard_body_radius::Real; order::Int = 64)

    hard_body_radius <= 0 && return 0.0

    r1, v1 = sc1_eci[1:3], sc1_eci[4:6]
    r2, v2 = sc2_eci[1:3], sc2_eci[4:6]

    _, C_2d = encounter_plane(r1, v1, r2, v2, cov1, cov2)

    C_rem, is_pd, _ = remediate_2x2(C_2d, hard_body_radius)
    is_pd || return 0.0

    U = revchol2x2(C_rem)
    U === nothing && return 0.0
    u11, u12, u22 = U

    nodes, weights = gauss_chebyshev_nodes(order)

    # CARA parameterises the miss distance as the full 3D relative range, with
    # the encounter-frame geometry carried by the reverse Cholesky factors.
    x0 = norm(r1 .- r2)

    denom = u11 * sqrt(2.0)
    s = hard_body_radius / u22
    hbr2 = hard_body_radius^2

    total = 0.0
    for k in eachindex(nodes)
        z = nodes[k] * s
        radical = sqrt(max(0.0, hbr2 - u22^2 * z^2))
        t1 = erfc((x0 - u12 * z - radical) / denom)
        t2 = erfc((x0 + u12 * z - radical) / denom)
        t3 = erfc((x0 - u12 * z + radical) / denom)
        t4 = erfc((x0 + u12 * z + radical) / denom)
        total += weights[k] * exp(-z^2 / 2) * (t1 + t2 - t3 - t4)
    end

    return clamp(total * s, 0.0, 1.0)
end

# ---------------------------------------------------------------------------
# Numerical integration — test oracle
# ---------------------------------------------------------------------------

"""
    numeric_pc(sc1_eci, sc2_eci, cov1, cov2, hard_body_radius; rtol=1e-10) -> Float64

Probability of collision by direct numerical integration of the projected 2D
Gaussian over the hard-body disk, in polar coordinates.

This makes no series or quadrature approximation beyond the tolerance of the
integrator, so it serves as ground truth for the other 2D methods.

!!! warning "Do not call this in a rollout"
    This is a TEST ORACLE. `hcubature` at `rtol = 1e-10` is orders of magnitude
    slower than `chan_pc` or `elrod_pc`, and MCTS evaluates Pc thousands of
    times per rollout — using it there would make the planner unusably slow.

    Use it to check the fast methods offline, to generate reference values for
    unit tests, or to investigate a specific suspicious conjunction. For the
    planner, use `chan_pc` (fast, watch the anisotropy caveat) or `elrod_pc`
    (fast, robust to anisotropy).

Validated against NASA CARA's `Pc2D` reference values on 53 real conjunctions:
agreement to 0.00% including on the cases where `chan_pc` errs by up to 20x.
"""
function numeric_pc(sc1_eci::AbstractVector, sc2_eci::AbstractVector,
                    cov1::AbstractMatrix, cov2::AbstractMatrix,
                    hard_body_radius::Real; rtol::Real = 1e-10,
                    maxevals::Int = 10^7)

    hard_body_radius <= 0 && return 0.0

    r1, v1 = sc1_eci[1:3], sc1_eci[4:6]
    r2, v2 = sc2_eci[1:3], sc2_eci[4:6]

    miss, C = encounter_plane(r1, v1, r2, v2, cov1, cov2)

    detC = det(C)
    detC <= 0 && return 0.0
    Cinv = inv(C)
    nrm = 1 / (2pi * sqrt(detC))

    # Polar integrand over the disk: the extra factor of rho is the Jacobian.
    function integrand(p)
        rho, phi = p[1], p[2]
        d = [rho * cos(phi) - miss[1], rho * sin(phi) - miss[2]]
        return rho * nrm * exp(-0.5 * dot(d, Cinv * d))
    end

    val, _ = hcubature(integrand, (0.0, 0.0), (hard_body_radius, 2pi);
                       rtol = rtol, maxevals = maxevals)
    return clamp(val, 0.0, 1.0)
end

# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

"""
    covariance_anisotropy(sc1_eci, sc2_eci, cov1, cov2) -> Float64

Ratio of the long to the short principal axis of the encounter-plane covariance
(sigma2 / sigma1, always >= 1).

This is the condition number that governs `chan_pc` accuracy. Empirically, from
the CARA validation set: below ~50 the Chan series agrees with ground truth to
about 1%; by ~120 the error reaches tens of percent; at ~570 it reaches 20x.

Use it to decide when to escalate from `chan_pc` to `elrod_pc`.
"""
function covariance_anisotropy(sc1_eci::AbstractVector, sc2_eci::AbstractVector,
                               cov1::AbstractMatrix, cov2::AbstractMatrix)
    _, C = encounter_plane(sc1_eci[1:3], sc1_eci[4:6],
                           sc2_eci[1:3], sc2_eci[4:6], cov1, cov2)
    ev = eigvals(Symmetric(Matrix(C)))
    (ev[1] <= 0 || ev[2] <= 0) && return Inf
    return sqrt(maximum(ev) / minimum(ev))
end

# -------------------------------------------------------
# Dispatcher
# -------------------------------------------------------

function compute_pc(pomdp::SpacecraftCAPOMDP,
                    sc_eci::Vector{Float64},
                    debris_eci::Vector{Float64},
                    epoch_current,
                    epoch_tca;
                    method::Symbol = :foster,
                    Σ_sc = nothing,
                    Σ_debris = nothing,
                    N::Int = 10000)

    # Fall back to initial covariances if not provided
    Σ_sc_use     = Σ_sc     === nothing ? pomdp.P0_sc     : Σ_sc
    Σ_debris_use = Σ_debris === nothing ? pomdp.P0_debris : Σ_debris

    if method == :foster
        return compute_pc_foster(pomdp, sc_eci, debris_eci, epoch_current, epoch_tca, 
                                  Σ_sc_use, Σ_debris_use)
    elseif method == :mc
        return compute_pc_mc(pomdp, sc_eci, debris_eci, epoch_current, epoch_tca, 
                              Σ_sc_use, Σ_debris_use, N= N)
    else
        error("Unknown Pc method: $method. Use :foster or :mc")
    end
end