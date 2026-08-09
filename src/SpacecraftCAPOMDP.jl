"""
CAState - Collision avoidance state at a given time
"""
struct CAState
    sc_eci::Vector{Float64}      # true spacecraft ECI (m, m/s)
    debris_eci::Vector{Float64}  # true debris ECI (m, m/s)
    t::Float64                   # time remaining (s)
    # pc::Float64               # probability of collision
    terminal::Bool
end
# Convenience constructors
# CAState(x_rel, sc_eci, t, pc) = CAState(x_rel, sc_eci, t, pc, false)
# CAState(x_rel, sc_eci, t)     = CAState(x_rel, sc_eci, t, 0.0, false)
CAState(sc_eci, debris_eci, t)     = CAState(sc_eci, debris_eci, t, false)

@enum CAAction WAIT=1 MANEUVER=2

# --- SNC process-noise PSD presets (m²/s³ per RTN axis, [q_R, q_T, q_N]) ------
# The predict/grow step adds Q_rtn = diag(q)⊗[[dt³/3,dt²/2],[dt²/2,dt]] (rotated
# RTN→ECI) to Φ Σ Φᵀ. `q` is a constant PSD; the ADDED matrix scales with dt (NOT
# a flat Q₀). Anisotropic: along-track (T) carries the growth, R/N a small floor
# so those axes stay ~flat (matching the measured real RTN shape). Calibrated
# against the real CARA growth curves — see CONSTANTS.md "Filter process noise Q".
#
# Q_RTN_WELL_TRACKED — the well-tracked / primary-like level. INTENDED to make
# along-track σ ∝ τ^~1.9 over 0.5–5.5 d, matching the real primary curve
# (notes/cara_covariance_growth_realism_findings.md; payload secondaries grow
# similarly, p≈1.65). **NOT YET CALIBRATED — held at zeros(3) (Q = 0), so the
# default belief growth is byte-identical to the pre-Phase-8 STM.** The 2026-08-08
# session wired the full SNC-Q machinery + back-prop and confirmed the calibration
# is a standalone fit (our STM is the no-drag growth; a radial seed maps to large
# in-track by Keplerian δa→along-track drift, and the once-per-orbit STM ripple
# competes with the SNC accumulation so a naive q_T sweep tops out ~p1.4–1.5, not
# the smooth real τ^1.9). Calibrating it — orbital-phase averaging / DMC vs SNC /
# seed treatment + a χ² containment check — is the next task (see the session
# note + growth-notes TODO #1/#2). Set a per-conjunction q via q_rtn_sc/q_rtn_debris
# once calibrated. See CONSTANTS.md "Filter process noise Q (SNC)".
const Q_RTN_WELL_TRACKED = [0.0, 0.0, 0.0]  # [q_R, q_T, q_N] — Q=0 until calibrated


struct SpacecraftCAPOMDP <: POMDP{CAState, CAAction, Vector{Float64}}  # POMDP{State, Action, Observation}
    satParams::Vector{Float64}         # Spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
    debrisParams::Vector{Float64}         # Spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
    epochTCA::Tuple
    forceModel::Bool
    R_alt::Float64
    e::Float64
    i::Float64
    Ω::Float64
    ω::Float64
    M::Float64
    seed::Int64
    randAdd::Bool
    conjunctionType::String
    rMag::Float64  # scalar, magnitude of relative position at TCA (m)
    vMag::Float64  # scalar, magnitude of relative velocity at TCA (m/s)
    TCA_max::Real    # max time to closest approach [s]
    Δv::Float64   # delta-v magnitude for maneuver (m/s)
    maneuver_cost::Float64
    P0_sc::Matrix{Float64}      # initial spacecraft covariance
    P0_debris::Matrix{Float64}  # initial debris covariance
    σ_sc::Float64      # spacecraft measurement noise std (m, m/s)
    σ_debris::Float64  # debris measurement noise std (m, m/s)
    R_hard_body_sc::Float64      # spacecraft hard body radius (m), default 5.0
    R_hard_body_debris::Float64  # debris hard body radius (m), default 15.0
    # SNC (state-noise-compensation) process-noise PSD per RTN axis (m²/s³), one
    # 3-vector [q_R, q_T, q_N] per object. The predict/grow step adds
    #   Q_rtn = diag(q_i)⊗[[dt³/3, dt²/2],[dt²/2, dt]]  (rotated RTN→ECI),
    # bending the Q=0 STM growth (σ∝τ^~1.4) up toward the drag-driven ~τ² real
    # growth. Anisotropic (along-track T dominant) to match the measured RTN
    # shape. Per-conjunction changeable (forwarded via load_cdm_scenario kwargs).
    # See CONSTANTS.md "Filter process noise Q (SNC)". q_rtn = zeros(3) ⇒ Q=0.
    q_rtn_sc::Vector{Float64}
    q_rtn_debris::Vector{Float64}
    pc_threshold::Float64 # TODO: not working right now
    dt::Float64  # timestep in seconds, default 30 minutes
    cadence_sc::Float64      # satellite measurement cadence (s) — onboard GPS, ~2 h
    cadence_debris::Float64  # debris measurement cadence (s) — SSN/TLE, ~8 h
    correct_at_root::Bool    # fix both objects at detection (root); timers start there
    γ::Float64
end


# Custom constructor to handle dynamic initialization
function SpacecraftCAPOMDP(;
    satParams = [500.0, 2.0, 2.2, 2.0, 1.3],
    debrisParams = [500.0, 2.0, 2.2, 2.0, 1.3],
    epochTCA = (2026, 1, 1, 12, 0, 0.0, 0.0),
    forceModel = true,
    R_alt = 400e3,  # meters
    e = 0.01,
    i = 75.0,  # degrees
    Ω = 45.0,  # degrees
    ω = 30.0,  # degrees
    M = 360.0,  # degrees
    seed = 42,
    randAdd = true,
    conjunctionType="crossing", # "head-on", "overtaking", "crossing"
    rMag = 100,
    vMag = 100,
    TCA_max = 10*60*60, # 10 hours
    Δv = 0.1, # m/s
    maneuver_cost = 10 , # CHANGE TODO
    P0_sc = diagm([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001]),      # 10m, 0.01 m/s TODO
    P0_debris = diagm([1e6, 1e6, 1e6, 0.01, 0.01, 0.01]),  # 1 km pos, 0.1 m/s vel (TLE-sourced; see CONSTANTS.md)
    # initial debris belief COMES FROM a TLE, so it can be no tighter than σ_debris (1 km).
    # pos 1σ = 1 km (Flohrer 2008 / ESA SDC5, isotropic mid-band); vel 1σ = 0.1 m/s
    # (conservative — SGP4-OD near-epoch radial vel error is ~1-3 cm/s, growing over days).
    σ_sc = 10.0, # own-asset onboard GPS ~1-10 m (Hauschild & Montenbruck 2021); 10 m conservative
    σ_debris = 1000.0, # SSN/TLE ~1 km at OD epoch (Flohrer 2008 / ESA SDC5)
    R_hard_body_sc = 5.0,  # meters, typical spacecraft radius TODO
    R_hard_body_debris = 15.0,  # meters, typical debris radius TODO
    # SNC process-noise PSD per RTN axis (m²/s³). Default = the well-tracked
    # (primary-like) level, calibrated so Φ Σ Φᵀ + Q grows along-track σ ∝ τ^~1.9
    # to match the real CARA primary curve (notes/cara_covariance_growth_realism
    # _findings.md). Along-track (T) dominant; small R/N floor keeps those axes
    # ~flat as in the real data. Debris (steeper, altitude-split) presets are in
    # CONSTANTS.md — pass q_rtn_debris=... per case. See Q_RTN_* consts below.
    q_rtn_sc = Q_RTN_WELL_TRACKED,
    q_rtn_debris = Q_RTN_WELL_TRACKED,
    pc_threshold = 1e-5, # Similar to spacex TODO
    dt = 30*60,  # 30 minute steps
    cadence_sc = 2*60*60,      # 2 h — GPS operator-contact cadence (swappable; ablation)
    cadence_debris = 8*60*60,  # 8 h — representative TLE refresh cadence (swappable)
    correct_at_root = true,    # fix both objects at detection; cadence timers start at root
    γ = 0.99

)
    Random.seed!(seed)
    if randAdd
        R_alt = R_alt + 200e3*rand() # meters
        e = e + 0.1*rand()
        i = i + 15.0*rand() # degrees
        Ω =Ω + 45.0*rand() # degrees
        ω =ω + 30.0*rand() # degrees
        M =M * rand() # degrees
        TCA_max = TCA_max + (5*60*60*rand())
    end
	return SpacecraftCAPOMDP(satParams,debrisParams,epochTCA,forceModel,
                            R_alt,e,i,Ω,ω,M,seed,randAdd,
                            conjunctionType,rMag,vMag,
                            TCA_max, Δv,maneuver_cost,
                            P0_sc, P0_debris,σ_sc, σ_debris,R_hard_body_sc, R_hard_body_debris,
                            collect(float.(q_rtn_sc)), collect(float.(q_rtn_debris)),
                            pc_threshold, dt,
                            cadence_sc, cadence_debris, correct_at_root, γ)
end

POMDPs.discount(pomdp::SpacecraftCAPOMDP) = pomdp.γ
