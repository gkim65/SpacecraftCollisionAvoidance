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