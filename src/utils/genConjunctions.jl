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


# -------------------------------------------------------------------------
# Deterministic conjunction placement — ported from RSSDA
# (RSSDA/benchmarks/spacecraftCA/conjunction_generator.py +
#  spacecraft_transition_v2.make_sc2_rel_state_at_tca).
#
# This is the generalized (miss, geometry-angle θ, v_rel) construction, NOT the
# single-radial-miss RSSDA/spacecraftCA/spacecraft_matrices.py version. It lives
# ALONGSIDE the older stochastic generate_tca_relative (which is still used by
# generate_conjunction). A conjunction is specified at TCA by:
#     perp = miss · sin θ     (sideways standoff, placed on RADIAL R)
#     dt0  = miss · cos θ     (along-track offset, placed on transverse T)
#     closing velocity        (magnitude v_rel, placed on T as -v_rel)
# so θ = 0° → head-on (all along-track), θ = 90° → cross-track (all sideways).
# Frame convention matches Brahe's state_rtn_to_eci: RTN = [Radial, Transverse
# (along-track), Normal (cross-track)] — identical to what this repo already
# uses in generate_tca_relative / generate_conjunction.
# -------------------------------------------------------------------------

"""
    sc1_eci_at_tca(pomdp) -> Vector{Float64}

Spacecraft (SC1 / primary) ECI state at TCA from the POMDP's orbital elements.
Deterministic (no RNG); the `randAdd` element jitter in the POMDP constructor
already fixes the elements, so this just converts them KOE→ECI. Returns a
6-element [r, v] state in meters / m·s⁻¹. Port of RSSDA `sc1_eci_at_tca`.
"""
function sc1_eci_at_tca(pomdp::SpacecraftCAPOMDP)
    bh = get_brahe()
    np = pyimport("numpy")
    oe = np.array([bh.R_EARTH + pomdp.R_alt, pomdp.e, pomdp.i, pomdp.Ω, pomdp.ω, pomdp.M])
    return collect(bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES))
end

"""
    place_debris_at_tca(sc1_eci, miss_m, angle_deg, v_rel; closing=true) -> Vector{Float64}

Place the debris (SC2 / secondary) ECI state at TCA relative to `sc1_eci`, for a
conjunction with total miss distance `miss_m` (m) at geometry angle `angle_deg`.

The relative state is built in the RTN frame (RSSDA
`make_sc2_rel_state_at_tca`), then converted to ECI via Brahe:
    δr_RTN = [perp, dt0, 0]   with  perp = miss·sin θ (radial), dt0 = miss·cos θ (along-track)
    δv_RTN = [0, ∓v_rel, 0]   (along-track closing speed; -v_rel by default so debris closes)

- `angle_deg = 0`   → head-on   (miss entirely along-track)
- `angle_deg = 90`  → cross-track (miss entirely sideways/radial standoff)

Deterministic. Returns a 6-element ECI [r, v] state (m, m·s⁻¹).
"""
function place_debris_at_tca(sc1_eci::AbstractVector, miss_m::Real,
                             angle_deg::Real, v_rel::Real; closing::Bool = true)
    θ = deg2rad(angle_deg)
    perp = miss_m * sin(θ)   # radial standoff
    dt0  = miss_m * cos(θ)   # along-track offset
    v_sign = closing ? -1.0 : 1.0
    rtn_rel = [perp, dt0, 0.0, 0.0, v_sign * v_rel, 0.0]

    bh = get_brahe()
    return collect(bh.state_rtn_to_eci(collect(sc1_eci), rtn_rel))
end

"""
    generate_conjunction_geometry(pomdp; geometry, miss_m, v_rel) -> (sc1_eci, debris_eci)

Deterministic conjunction generator for a named `geometry` at a specified
`miss_m` (m) and along-track relative speed `v_rel` (m/s), both at TCA. This is
the generalized RSSDA placement adapted to the two geometries the AMOS abstract
needs this phase:

- `:head_on`     → angle 0°   (miss entirely along-track; θ from CONSTANTS.md)
- `:cross_track` → angle 90°  (miss entirely cross-track / sideways standoff)

(Proximity-operations is deferred — see TODOS.md Phase 2.)

Returns `(sc1_eci, debris_eci)`, each a 6-element ECI [r, v] state (m, m·s⁻¹) at
TCA. Both states are deterministic given the POMDP's (jittered) orbital elements.
"""
function generate_conjunction_geometry(pomdp::SpacecraftCAPOMDP;
                                       geometry::Symbol = :head_on,
                                       miss_m::Real = pomdp.rMag,
                                       v_rel::Real = pomdp.vMag)
    angle_deg = if geometry == :head_on
        0.0
    elseif geometry == :cross_track
        90.0
    else
        error("Unknown geometry $geometry. Use :head_on or :cross_track " *
              "(proximity-ops deferred — see TODOS.md Phase 2).")
    end

    sc1_eci = sc1_eci_at_tca(pomdp)
    debris_eci = place_debris_at_tca(sc1_eci, miss_m, angle_deg, v_rel)
    return sc1_eci, debris_eci
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

