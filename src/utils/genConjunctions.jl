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

# =========================================================================
# ORBIT-FIRST closest-approach machinery — ported from RSSDA
# (RSSDA/benchmarks/spacecraftCA/conjunction_generator.py: _closest_approach,
#  _reduce_rel_to_params, make_conjunction_from_orbits; plus propagate_batch_to
#  from RSSDA/benchmarks/spacecraftCA/spacecraft_matrices.py).
#
# Purpose: turn a pair of orbits into a *dynamically-verified* conjunction, so
# we can confirm the geometry-first generate_conjunction_geometry placements are
# dynamically real — i.e. that the requested miss really IS the closest approach
# and that it lands at TCA.
#
# Two intentional differences from the RSSDA source (decided with Grace):
#   1. Propagate each object ONCE across the whole ±span window, then read the
#      state at every grid epoch via Brahe's trajectory interpolation
#      (prop.state(epoch)) — NOT a re-propagation per grid point. The search hits
#      ~40-60 grid epochs and this path is called a lot, so a single propagate +
#      cheap interpolated lookups is the speedup.
#
#      NB: this replaces RSSDA's `par_propagate_to` batch call. That Brahe API
#      only accepts Keplerian/SGP propagators, which is incompatible with the
#      accurate numerical (drag/SRP) force model below — so the single-propagate
#      + trajectory-interpolation path is used instead. It achieves the same goal
#      (no per-grid-point re-propagation) while keeping the accurate dynamics.
#   2. Accurate force model (drag/SRP) via eci2orb_brahe with the POMDP's
#      satParams/debrisParams, NOT the two-body model RSSDA hardcodes. This wires
#      the search to the SAME dynamics the rest of the pipeline uses.
#
# Note on KOE: the KOE ⇄ ECI conversions here (and the perigee/apogee/ecc
# feasibility check) are a change of *coordinates* at a single instant, not a
# dynamics choice — they do not propagate anything and do not discard drag/SRP.
# All propagation between epochs uses the accurate numerical force model.
#
# Frame convention (Brahe): RTN = [Radial, Transverse (along-track), Normal
# (cross-track)] — same as the rest of this file.
# =========================================================================

"""
    _closest_approach(pomdp, sc1_eci, sc2_eci; span_s, tca_epoch) -> (min_dist_m, t_ca_s, rel_rtn)

Find the true closest approach between SC1 (`sc1_eci`) and SC2 (`sc2_eci`), both
given at the nominal TCA epoch, over a window ±`span_s` around that epoch. Port
of RSSDA `_closest_approach`: coarse grid → golden-section refine (no external
optimizer), with the **closing-speed-adaptive grid spacing** preserved from the
source.

Propagation path (see the block comment above): each object is propagated ONCE,
from `TCA - span_s` to `TCA + span_s`, using the accurate numerical force model
(`eci2orb_brahe` with the POMDP's per-object params). The separation at any grid
offset is then read from the stored trajectory via `prop.state(epoch)` — cheap
interpolation, no re-propagation per grid point.

Returns `(min_dist_m, t_ca_s, rel_rtn)`:
- `min_dist_m` : 3-D closest-approach distance (m)
- `t_ca_s`     : offset of the true closest approach from the nominal TCA (s)
- `rel_rtn`    : SC2's RTN state relative to SC1 at the true closest approach
                 (6-vector [R,T,N, Ṙ,Ṫ,Ṅ], m / m·s⁻¹)

Why the adaptive grid (kept verbatim from RSSDA): a fixed grid whose spacing is
coarse relative to how fast the pair closes lets a sharp km-scale minimum for a
multi-km/s crossing fall BETWEEN two samples, so the golden bracket misses it
(RSSDA observed a ~6 km/s crossing's true 4.87 km min reported as 5.00 km with a
30 s grid). We size the grid from the closing speed so spacing is ≲0.2 km of
along-track travel, clamped to [41, 60001] points.
"""
function _closest_approach(pomdp::SpacecraftCAPOMDP,
                           sc1_eci::AbstractVector, sc2_eci::AbstractVector;
                           span_s::Real = 600.0, tca_epoch = nothing)
    bh = get_brahe()
    np = pyimport("numpy")
    ep_tca = tca_epoch === nothing ? _tca_epoch(pomdp) : tca_epoch

    # Build one propagator per object, anchored at TCA - span_s, and propagate
    # ONCE to TCA + span_s so the whole search window is stored in the trajectory.
    # (state(epoch) requires the requested epoch to already be covered.) We must
    # therefore back-propagate each object's TCA state to TCA - span_s first.
    ep_lo = ep_tca - float(span_s)
    ep_hi = ep_tca + float(span_s)

    function build_prop(eci, params)
        # anchor at TCA, back-propagate to ep_lo, then forward-propagate the
        # whole window so state(epoch) is valid across [ep_lo, ep_hi].
        prop, _ = eci2orb_brahe(collect(eci), pomdp.epochTCA, params, pomdp.forceModel)
        prop.propagate_to(ep_lo)
        eci_lo = collect(prop.state(ep_lo))[1:6]
        prop2, _ = eci2orb_brahe(eci_lo, _epoch_to_tuple(ep_lo), params, pomdp.forceModel)
        prop2.propagate_to(ep_hi)
        return prop2
    end

    prop1 = build_prop(sc1_eci, pomdp.satParams)
    prop2 = build_prop(sc2_eci, pomdp.debrisParams)

    # separation (m) at offset dt seconds from the nominal TCA
    function sep(dt)
        ep = ep_tca + float(dt)
        s1 = collect(prop1.state(ep))[1:6]
        s2 = collect(prop2.state(ep))[1:6]
        return norm(s1[1:3] - s2[1:3])
    end

    # closing speed (km/s) by finite difference at t=0, to size the grid
    v_close_kms = abs(sep(1.0) - sep(-1.0)) / 2.0 / 1e3
    if !isfinite(v_close_kms)          # escape/hyperbolic → NaN separation
        v_close_kms = 0.05
    end
    n_grid = Int(clamp(ceil(2 * span_s * max(v_close_kms, 0.05) / 0.2), 41, 60001))

    grid = range(-span_s, span_s; length = n_grid)
    dvals = [sep(t) for t in grid]
    k = argmin(dvals)
    lo = grid[max(k - 1, 1)]
    hi = grid[min(k + 1, length(grid))]

    # golden-section refine within [lo, hi]
    gr = (sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = sep(c), sep(d)
    for _ in 1:40
        abs(b - a) < 1e-3 && break
        if fc < fd
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = sep(c)
        else
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = sep(d)
        end
    end
    t_ca = 0.5 * (a + b)

    ep = ep_tca + float(t_ca)
    s1 = collect(prop1.state(ep))[1:6]
    s2 = collect(prop2.state(ep))[1:6]
    rel_rtn = collect(bh.state_eci_to_rtn(np.array(s1), np.array(s2)))
    d_min = norm(s1[1:3] - s2[1:3])
    return d_min, t_ca, rel_rtn
end

# Brahe Epoch at the nominal TCA, from the POMDP's epochTCA tuple.
function _tca_epoch(pomdp::SpacecraftCAPOMDP)
    bh = get_brahe()
    e = pomdp.epochTCA
    return bh.Epoch.from_datetime(e[1], e[2], e[3], e[4], e[5], e[6], e[7],
                                  bh.TimeSystem.UTC)
end

# A Brahe Epoch → the (Y, M, D, h, m, s, ns) tuple form eci2orb_brahe expects.
# Used to re-anchor a propagator at TCA - span (the back-propagated state) so the
# whole search window is stored forward of the propagator's initial epoch.
# Brahe's to_datetime() already returns exactly this 7-tuple (second is a float,
# nanosecond is separate), matching eci2orb_brahe's from_datetime signature.
function _epoch_to_tuple(ep)
    c = ep.to_datetime()   # (year, month, day, hour, minute, second_float, nanosecond)
    return (Int(c[1]), Int(c[2]), Int(c[3]), Int(c[4]), Int(c[5]),
            float(c[6]), float(c[7]))
end

"""
    _reduce_rel_to_params(rel_rtn) -> (perp_m, dt0_m, v_rel_ms)

Reduce a full RTN relative state `[R,T,N, Ṙ,Ṫ,Ṅ]` (m, m·s⁻¹) to the model's
`(perp, dt0, v_rel)`. Port of RSSDA `_reduce_rel_to_params` (kept in meters
here; RSSDA returns km). LOSSY by design:
- `perp  = √(R² + N²)`  — sideways standoff (radial + cross-track magnitude)
- `dt0   = T`           — along-track offset
- `v_rel = -Ṫ`          — along-track closing speed
The Ṙ/Ṅ velocity components are dropped (they live in the geometry + gain, not
the reduced state).
"""
function _reduce_rel_to_params(rel_rtn::AbstractVector)
    R, T, N = rel_rtn[1], rel_rtn[2], rel_rtn[3]
    perp = hypot(R, N)
    dt0 = T
    v_rel = -rel_rtn[5]
    return perp, dt0, v_rel
end

# Feasibility floors/ceilings for SC2's orbit (LEO near-circular regime).
# Ported from RSSDA conjunction_generator.py (PERIGEE_FLOOR_KM / APOGEE_CEIL_KM /
# ECC_MAX). Tracked in CONSTANTS.md.
const PERIGEE_FLOOR_KM = 150.0    # below this: into the atmosphere / "through Earth"
const APOGEE_CEIL_KM   = 2000.0   # above this: leaves LEO, reduced model unvalidated
const ECC_MAX          = 0.25     # near-circular co-orbital regime

"""
    make_conjunction_from_orbits(pomdp, sc1_koe, sc2_koe; span_s) -> NamedTuple

Turn two Keplerian orbits into a dynamically-verified conjunction. Port of RSSDA
`make_conjunction_from_orbits`: convert both KOE→ECI at the nominal TCA epoch,
find the true closest approach over ±`span_s` (accurate dynamics, batch path),
read off the geometry, reduce to `(perp, dt0, v_rel)`, and apply the
perigee/apogee/eccentricity feasibility guard on SC2's orbit.

`sc1_koe` / `sc2_koe` are 6-element `[a, e, i, Ω, ω, M]` with `a` in meters and
angles in degrees, anomaly `M` phased so the encounter falls in the window.

Like the RSSDA source, this does NOT guarantee a conjunction — it REPORTS
whatever closest approach the two orbits have in the window. A huge
`true_miss_m` just means the orbits don't actually conjunct there.

Returns a NamedTuple:
- `true_miss_m` : brahe 3-D closest-approach distance (m)
- `t_ca_s`      : offset of closest approach from nominal TCA (s)
- `at_tca`      : |t_ca_s| < 1 s (closest approach pinned to TCA)
- `perp_m`, `dt0_m`, `v_rel_ms` : reduced model params (m, m, m·s⁻¹)
- `quad_miss_m` : √(perp² + dt0²), the model's quadrature miss
- `angle_deg`   : geometry angle atan2(perp, |dt0|) in degrees (0=head-on, 90=cross-track)
- `feasible`    : SC2 a LEO-resident near-circular orbit?
- `reason`      : why infeasible / verify note (empty string if clean)
- `rel_rtn`     : full RTN relative state at closest approach (6-vector)
"""
function make_conjunction_from_orbits(pomdp::SpacecraftCAPOMDP,
                                      sc1_koe::AbstractVector, sc2_koe::AbstractVector;
                                      span_s::Real = 600.0)
    bh = get_brahe()
    np = pyimport("numpy")
    sc1_eci = collect(bh.state_koe_to_eci(np.array(collect(sc1_koe)), bh.AngleFormat.DEGREES))
    sc2_eci = collect(bh.state_koe_to_eci(np.array(collect(sc2_koe)), bh.AngleFormat.DEGREES))

    true_miss, t_ca, rel_rtn = _closest_approach(pomdp, sc1_eci, sc2_eci; span_s = span_s)
    at_tca = abs(t_ca) < 1.0
    perp, dt0, v_rel = _reduce_rel_to_params(rel_rtn)
    quad_miss = hypot(perp, dt0)

    # feasibility of SC2's own (already-real) orbit
    a, e = float(sc2_koe[1]), float(sc2_koe[2])
    feasible, reason = true, ""
    peri_km = (a * (1.0 - e) - bh.R_EARTH) / 1e3
    apo_km  = (a * (1.0 + e) - bh.R_EARTH) / 1e3
    if !(isfinite(a) && isfinite(e))
        feasible, reason = false, "non-finite elements"
    elseif e >= 1.0
        feasible, reason = false, "hyperbolic/escape (e=$(round(e, digits=3)))"
    elseif e > ECC_MAX
        feasible, reason = false, "e=$(round(e, digits=3)) > $ECC_MAX (non-co-orbital)"
    elseif peri_km < PERIGEE_FLOOR_KM
        feasible, reason = false, "perigee $(round(peri_km, digits=0))km < $(PERIGEE_FLOOR_KM)km (into atmosphere)"
    elseif apo_km > APOGEE_CEIL_KM
        feasible, reason = false, "apogee $(round(apo_km, digits=0))km > $(APOGEE_CEIL_KM)km (leaves LEO)"
    end

    angle_deg = (perp != 0.0 || dt0 != 0.0) ? rad2deg(atan(perp, abs(dt0))) : 0.0

    return (true_miss_m = true_miss, t_ca_s = t_ca, at_tca = at_tca,
            perp_m = perp, dt0_m = dt0, v_rel_ms = v_rel,
            quad_miss_m = quad_miss, angle_deg = angle_deg,
            feasible = feasible, reason = reason, rel_rtn = rel_rtn)
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

