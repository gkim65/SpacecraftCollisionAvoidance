using HCubature
using Distributions
using LinearAlgebra
using Random

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