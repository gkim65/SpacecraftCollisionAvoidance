using Distributions 
using LinearAlgebra
using PyCall
using Random

# Import brahe Python library - lazy initialization
const _bh_cache = Ref{Union{PyObject, Nothing}}(nothing)

function get_brahe()
    if _bh_cache[] === nothing
        _bh_cache[] = pyimport("brahe")
        # Initialize EOP and space weather data
        _bh_cache[].initialize_eop()
        _bh_cache[].initialize_sw()
    end
    return _bh_cache[]
end


function spacecraftECIgen(pomdp::SpacecraftCAPOMDP)
    """
    Using a reasonable set of orbital parameters for LEO, generate an initial ephemeris position
    Returns state in m and m/s
    """
    bh = get_brahe()
    
    Random.seed!(pomdp.seed)
    
    # Create orbital elements array
    np = pyimport("numpy")
    oe = np.array([bh.R_EARTH+pomdp.R_alt, pomdp.e, pomdp.i, pomdp.Ω, pomdp.ω, pomdp.M])
    
    # Convert to ECI (returns in meters and m/s)
    new_eci_m = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    
    return collect(new_eci_m)
end

function generate_tca_relative(pomdp::SpacecraftCAPOMDP)
    """
    Generate a debris ECI state at TCA relative to spacecraft.

    Returns:
        spacecraft_eci::Vector{Float64}  # 6-element [r,v] in meters/m/s
        debris_eci::Vector{Float64}      # 6-element [r,v] in meters/m/s
    """
    # Generate spacecraft ECI
    spacecraft_eci = spacecraftECIgen(pomdp)
    Random.seed!(pomdp.seed)

    # Noise scale for small perturbations (1% of rMag)
    noise_scale = pomdp.rMag * 0.01

    # Relative position in RTN frame
    r_rel = zeros(3)
    if pomdp.conjunctionType == "head-on"
        r_rel .= [randn()*noise_scale, -pomdp.rMag, randn()*noise_scale]
    elseif pomdp.conjunctionType == "overtaking"
        r_rel .= [randn()*noise_scale, pomdp.rMag, randn()*noise_scale]
    elseif pomdp.conjunctionType == "crossing"
        r_rel .= [randn()*noise_scale, randn()*noise_scale, pomdp.rMag]
    else
        r_rel .= pomdp.rMag .* (2rand(3).-1)  # completely random
    end

    # Relative velocity perpendicular to r_rel
    tmp = randn(3)
    v_rel = tmp - (dot(tmp,r_rel)/dot(r_rel,r_rel)) * r_rel  # project out along r_rel
    v_rel /= norm(v_rel)
    v_rel *= pomdp.vMag

    # Combine into RTN state
    x_rel_rtn = vcat(r_rel, v_rel)

    # Transform to absolute ECI state for debris
    bh = get_brahe()
    debris_eci = bh.state_rtn_to_eci(spacecraft_eci, x_rel_rtn)

    return spacecraft_eci, debris_eci
end


function eci2orb_brahe(eci, epoch::Tuple, objParams, forceModel; 
                        initial_covariance=nothing)
    bh = get_brahe()
    
    bh_epoch = bh.Epoch.from_datetime(epoch[1], epoch[2], epoch[3], 
                                       epoch[4], epoch[5], epoch[6], 
                                       epoch[7], bh.TimeSystem.UTC)

    # STM needed if we have a covariance to propagate
    if initial_covariance !== nothing
        prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()
    else
        prop_config = bh.NumericalPropagationConfig.default()
    end

    if forceModel
        force_config = bh.ForceModelConfig.default()
    else
        force_config = bh.ForceModelConfig.two_body()
    end

    np = pyimport("numpy")

    if initial_covariance !== nothing
        prop = bh.NumericalOrbitPropagator(
            bh_epoch, eci, prop_config, force_config,
            params = np.array(objParams),
            initial_covariance = np.array(initial_covariance)
        )
    else
        prop = bh.NumericalOrbitPropagator(
            bh_epoch, eci, prop_config, force_config,
            params = np.array(objParams)
        )
    end

    return prop, bh_epoch
end

function generate_conjunction(pomdp::SpacecraftCAPOMDP)
    Random.seed!(pomdp.seed)

    eci_spacecraft, eci_debris = generate_tca_relative(pomdp)

    prop_spacecraft, epoch_spacecraft = eci2orb_brahe(eci_spacecraft, pomdp.epochTCA, pomdp.satParams, pomdp.forceModel)
    prop_debris, epoch_debris = eci2orb_brahe(eci_debris, pomdp.epochTCA, pomdp.debrisParams, pomdp.forceModel)

    bh = get_brahe()
    epoch_start = epoch_spacecraft - pomdp.TCA_max

    prop_spacecraft.propagate_to(epoch_start)
    prop_debris.propagate_to(epoch_start)

    spacecraft_eci_t0 = collect(prop_spacecraft.current_state()[1:6])
    debris_eci_t0 = collect(prop_debris.current_state()[1:6])

    # Use brahe RTN conversion instead of manual ECI difference
    x_rel_rtn = collect(bh.state_eci_to_rtn(spacecraft_eci_t0, debris_eci_t0))

    return eci_spacecraft, eci_debris, prop_spacecraft, prop_debris, epoch_spacecraft, epoch_debris, x_rel_rtn, epoch_start
end

