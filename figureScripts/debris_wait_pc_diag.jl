# debris_wait_pc_diag.jl — DIAGNOSTIC: is the WAIT Pc-drop real (covariance dilution)
# or the forward-growth Pc→0 artifact? Print the FULL Pc-vs-time curve + the debris Σ
# along-track σ at each step, for ONE debris case, under (a) NO measurements and (b) 2h
# cadence. Grace's question: "with no measurement Pc goes lower — that doesn't feel right."

using LinearAlgebra, Printf, Random, PyCall, POMDPs, POMDPTools, Distributions

include(joinpath(@__DIR__, "..", "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "sensorTiers.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "genConjunctions.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "computePc.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "covarianceTable.jl"))
include(joinpath(@__DIR__, "..", "src", "states.jl"))
include(joinpath(@__DIR__, "..", "src", "actions.jl"))
include(joinpath(@__DIR__, "..", "src", "rewards.jl"))
include(joinpath(@__DIR__, "..", "src", "observations.jl"))
include(joinpath(@__DIR__, "..", "src", "transitions.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefTracker.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefMCTS.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "cdmScenario.jl"))

const CDM = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms",
    "000040059_conj_000035921_20220326_194122_20220325_215435.cdm"))

# RTN σ of a 6×6 ECI covariance at `state` (m): [radial, in-track, cross-track].
function rtn_sigmas(Σ, state)
    R = rtn_rotation(state)                 # cols R,T,N in ECI
    Σp = Matrix(Σ)[1:3,1:3]
    d = [sqrt(max(0.0, (R[:,i]' * Σp * R[:,i]))) for i in 1:3]
    return round.(d, digits=1)
end

# miss distance (m) between the two means grown to TCA.
function diag_walk(pomdp, b0; measure::Bool, dt_s)
    b = b0; since_db = 0.0; nfix = 0
    nsteps = round(Int, b0.t/dt_s)
    println(@sprintf("%5s %10s %12s %28s %28s %8s", "t_h", "Pc@TCA", "miss@TCA_m",
            "sc RTN σ (root) m", "db RTN σ (root) m", "dbfixes"))
    for k in 0:nsteps
        # Pc at TCA (exact grow)
        st = CAState(copy(b.sc.μ), copy(b.debris.μ), b.t)
        pc = node_pc_at_tca(pomdp, BeliefNode(b, st, false))
        # grown means at TCA for miss distance
        μs,_ = _grow_belief_to_tca(pomdp, b.sc.μ, b.sc.Σ, pomdp.satParams, b.t; q_rtn=pomdp.q_rtn_sc)
        μd,_ = _grow_belief_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, b.t; q_rtn=pomdp.q_rtn_debris)
        miss = norm(μs[1:3] .- μd[1:3])
        println(@sprintf("%5.1f %10.2e %12.0f %28s %28s %8d",
            b.t/3600, pc, miss, string(rtn_sigmas(b.sc.Σ,b.sc.μ)),
            string(rtn_sigmas(b.debris.Σ,b.debris.μ)), nfix))
        b.t <= 1.0 && break
        b = predict(pomdp, b, WAIT; dt=dt_s)
        since_db += dt_s
        if measure && since_db >= pomdp.cadence_debris
            z = vcat(b.sc.μ, b.debris.μ)
            b = correct_linear_debris(pomdp, b, z); since_db = 0.0; nfix += 1
        end
    end
end

sc = load_cdm_scenario(CDM; dt=3600.0, sensor_quality=:median)
println("case: ", first(basename(CDM),40))
println("lead = ", round(sc.t_horizon/3600,digits=1), " h  miss(CDM)=", round(sc.miss_distance),
        " m  hbr=", sc.hbr, " m  CARA Pc=", sc.pc_cdm)
println("anchor Pc@TCA (CDM cov, at TCA) = ", node_pc_at_tca(sc.pomdp, BeliefNode(sc.b_tca, sc.s_true, false)))
println("root seed debris RTN σ = ", rtn_sigmas(sc.b0.debris.Σ, sc.b0.debris.μ), " m")
println("root seed sc     RTN σ = ", rtn_sigmas(sc.b0.sc.Σ, sc.b0.sc.μ), " m")
println()
println("###### (a) NO MEASUREMENTS — pure WAIT/predict ######")
diag_walk(sc.pomdp, sc.b0; measure=false, dt_s=3600.0)
println()
println("###### (b) WITH 8h debris cadence (median) ######")
diag_walk(sc.pomdp, sc.b0; measure=true, dt_s=3600.0)
