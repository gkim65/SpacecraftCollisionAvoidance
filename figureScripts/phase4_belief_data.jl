# =========================================================================
# phase4_belief_data.jl — generate Phase 4 belief-tracker validation data.
#
# Julia is the SOURCE OF TRUTH: this runs the actual beliefTracker.jl and dumps
# JSON that phase4_belief_plot.py renders. Kept OUT of the pinned brahe/numpy
# venv (plotting is a separate `uv run --with matplotlib` step, per Figures.md).
#
# Run from the repo root:
#   julia --project=. figureScripts/phase4_belief_data.jl
# writes figureScripts/phase4_data.json
#
# Fixture: SpacecraftCAPOMDP(seed=42, randAdd=false), Phase-3 feasible co-orbital
# cross-track conjunction (miss 500 m, v_rel 15 m/s), dt = 1 hr, 24 h window.
# =========================================================================
using LinearAlgebra
using Random
using PyCall
using POMDPs
using POMDPTools

# Minimal JSON writer (avoid adding JSON to the project's direct deps).
_json(x::Bool) = x ? "true" : "false"
_json(x::Real) = isfinite(x) ? string(x) : "null"
_json(x::AbstractString) = "\"$x\""
_json(v::AbstractVector) = "[" * join(_json.(v), ",") * "]"
_json(m::AbstractMatrix) = "[" * join([_json(collect(m[i, :])) for i in 1:size(m, 1)], ",") * "]"
_json(d::AbstractDict) = "{" * join(["\"$k\":" * _json(v) for (k, v) in d], ",") * "}"

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(REPO, "src", "utils", "genConjunctions.jl"))
include(joinpath(REPO, "src", "utils", "computePc.jl"))
include(joinpath(REPO, "src", "utils", "covarianceTable.jl"))
include(joinpath(REPO, "src", "states.jl"))
include(joinpath(REPO, "src", "actions.jl"))
include(joinpath(REPO, "src", "observations.jl"))
include(joinpath(REPO, "src", "transitions.jl"))
include(joinpath(REPO, "src", "utils", "beliefTracker.jl"))

pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)
sc_eci, debris_eci = generate_conjunction_geometry(pomdp;
    geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)

dt = 3600.0
t0 = 24 * 3600.0
n  = Int(t0 / dt)

pos_sig(Σ) = sqrt(maximum(diag(Matrix(Σ))[1:3]))
vel_sig(Σ) = sqrt(maximum(diag(Matrix(Σ))[4:6]))

# ---------------------------------------------------------------------------
# Run a predict/correct sweep with a given measurement cadence.
#   correct_every = 1  -> measurement every step
#   correct_every = K  -> measurement only every K-th step
#   correct_every = 0  -> NEVER correct (predict-only; uncertainty just grows)
# Records σ AFTER each step's processing and whether that step had a measurement.
# ---------------------------------------------------------------------------
function run_cadence(correct_every::Int; seed = 20260722)
    rng = MersenneTwister(seed)
    b = belief_from_pomdp(pomdp, sc_eci, debris_eci, t0)
    τ = Float64[]; sc_pos = Float64[]; db_pos = Float64[]
    sc_vel = Float64[]; db_vel = Float64[]; measured = Bool[]
    for k in 1:n
        b_pred = predict(pomdp, b, WAIT; dt = dt)
        did_measure = correct_every > 0 && (k % correct_every == 0)
        if did_measure
            s_true = CAState(b_pred.sc.μ, b_pred.debris.μ, b_pred.t)
            z = sample_observation(pomdp, WAIT, s_true, rng)
            b = correct_linear(pomdp, b_pred, z)
        else
            b = b_pred
        end
        push!(τ, b.t / 3600)
        push!(sc_pos, pos_sig(b.sc.Σ)); push!(db_pos, pos_sig(b.debris.Σ))
        push!(sc_vel, vel_sig(b.sc.Σ)); push!(db_vel, vel_sig(b.debris.Σ))
        push!(measured, did_measure)
    end
    return Dict("tau_hr" => τ, "sc_pos" => sc_pos, "db_pos" => db_pos,
                "sc_vel" => sc_vel, "db_vel" => db_vel, "measured" => measured)
end

every1   = run_cadence(1)     # measurement every hour (continuous tracking)
every6   = run_cadence(6)     # measurement every 6 hours (sparse)
predonly = run_cadence(0)     # never (predict-only: Σ grows unbounded)

# ---------------------------------------------------------------------------
# PANEL B: Σ⁺ independent of z — one predict step, many random-z corrects.
# ---------------------------------------------------------------------------
b0 = belief_from_pomdp(pomdp, sc_eci, debris_eci, t0)
bp = predict(pomdp, b0, WAIT; dt = dt)
s_true = CAState(bp.sc.μ, bp.debris.μ, bp.t)

rng2 = MersenneTwister(999)
mu_x = Float64[]; mu_y = Float64[]; Σplus = nothing
for i in 1:200
    z = sample_observation(pomdp, WAIT, s_true, rng2)
    bc = correct_linear(pomdp, bp, z)
    push!(mu_x, bc.sc.μ[1] - bp.sc.μ[1]); push!(mu_y, bc.sc.μ[2] - bp.sc.μ[2])
    global Σplus = bc.sc.Σ
end
z1 = sample_observation(pomdp, WAIT, s_true, MersenneTwister(5))
ba = correct_linear(pomdp, bp, z1)
bb = correct_brahe(pomdp, bp, z1)

data = Dict(
    "dt_hr" => dt / 3600, "n" => n,
    "every1" => every1, "every6" => every6, "predonly" => predonly,
    "muB_x" => mu_x, "muB_y" => mu_y,
    "Sigma_plus_xy" => [Σplus[1,1] Σplus[1,2]; Σplus[2,1] Σplus[2,2]],
    "ab_mu_a" => ba.sc.μ[1:2] .- bp.sc.μ[1:2],
    "ab_mu_b" => bb.sc.μ[1:2] .- bp.sc.μ[1:2],
    "P0_sc_pos" => sqrt(pomdp.P0_sc[1,1]), "P0_db_pos" => sqrt(pomdp.P0_debris[1,1]),
    "sigma_sc" => pomdp.σ_sc, "sigma_debris" => pomdp.σ_debris,
)

out = joinpath(@__DIR__, "phase4_data.json")
open(out, "w") do io; write(io, _json(data)); end
println("wrote $out")
println("measurement cadence: every1 = 1 hr, every6 = 6 hr, predonly = never")
println("predict-only final debris pos 1σ: ", round(predonly["db_pos"][end], digits=1), " m",
        "  (vs every-hour: ", round(every1["db_pos"][end], digits=1), " m)")
