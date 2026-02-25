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
    pc_threshold::Float64 # TODO: not working right now
    dt::Float64  # timestep in seconds, default 30 minutes
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
    P0_debris = diagm([2500.0, 2500.0, 2500.0, 0.01, 0.01, 0.01]),  # 50m, 0.1 m/s,
    # P0_debris = diagm([10000.0, 10000.0, 10000.0, 0.01, 0.01, 0.01]),   # 100m, 0.1 m/s TODO
    σ_sc = 10.0, # TODO
    σ_debris = 100.0, # TODO
    R_hard_body_sc = 5.0,  # meters, typical spacecraft radius TODO
    R_hard_body_debris = 15.0,  # meters, typical debris radius TODO
    pc_threshold = 1e-5, # Similar to spacex TODO
    dt = 30*60,  # 30 minute steps
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
                            P0_sc, P0_debris,σ_sc, σ_debris,R_hard_body_sc, R_hard_body_debris, pc_threshold, dt, γ)
end

POMDPs.discount(pomdp::SpacecraftCAPOMDP) = pomdp.γ
