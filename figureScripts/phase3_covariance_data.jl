# =========================================================================
# phase3_covariance_data.jl — generate Phase 3 Σ(τ)-table validation data.
#
# Julia is the SOURCE OF TRUTH: this runs the actual build_covariance_table /
# pc_through_table from src/utils/covarianceTable.jl and dumps JSON that
# phase3_covariance_plot.py renders. Kept OUT of the pinned brahe/numpy venv
# (plotting is a separate `uv run --with matplotlib` step, per Figures.md).
#
# Run from the repo root:
#   julia --project=. figureScripts/phase3_covariance_data.jl
# writes figureScripts/phase3_data.json
#
# Fixture: SpacecraftCAPOMDP(seed=42, randAdd=false), Phase-3 feasible co-orbital
# cross-track conjunction (miss 500 m, v_rel 15 m/s), same as the Phase 3 test.
# Two grids are computed: the 1-hr grid the planner actually samples, and a fine
# 6-min grid that resolves the once-per-orbit radial/cross-track oscillation.
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

pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)
sc_eci, debris_eci = generate_conjunction_geometry(pomdp;
    geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)

# Orbital period (for the "oscillation == orbital period" annotation).
a_sma  = pomdp.R_alt + 6.3781363e6      # R_alt above Earth radius (m)
mu_g   = 3.986004418e14                 # GM_earth (m^3/s^2)
Torb_h = 2 * pi * sqrt(a_sma^3 / mu_g) / 3600

# RTN 1σ per component from a table entry.
sig(tbl, fld, i) = [sqrt(getfield(tbl, fld)[k][i, i]) for k in 1:tbl.n_steps]

# ---------------------------------------------------------------------------
# Build a grid's series: τ (hr) + RTN R/T/N 1σ for both objects + Pc through Σ(τ).
# RTN axis 1 = radial (R), 2 = transverse / along-track (T), 3 = normal / cross-track (N).
# ---------------------------------------------------------------------------
function grid_series(dt)
    tbl = build_covariance_table(pomdp, sc_eci, debris_eci; dt = dt, verbose = false)
    pc  = pc_through_table(pomdp, sc_eci, debris_eci, tbl)
    return Dict(
        "tau_hr" => tbl.τ_s ./ 3600,
        "sc_R" => sig(tbl, :Σ_sc_rtn, 1), "sc_T" => sig(tbl, :Σ_sc_rtn, 2), "sc_N" => sig(tbl, :Σ_sc_rtn, 3),
        "db_R" => sig(tbl, :Σ_debris_rtn, 1), "db_T" => sig(tbl, :Σ_debris_rtn, 2), "db_N" => sig(tbl, :Σ_debris_rtn, 3),
        "pc" => pc,
    )
end

coarse = grid_series(3600.0)   # 1-hr grid: what the planner actually samples
fine   = grid_series(360.0)    # 6-min grid: resolves the once-per-orbit oscillation

data = Dict(
    "coarse" => coarse, "fine" => fine,
    "Torb_h" => Torb_h, "miss_m" => 500.0, "v_rel" => 15.0,
)

out = joinpath(@__DIR__, "phase3_data.json")
open(out, "w") do io; write(io, _json(data)); end
println("wrote $out")
println("orbital period = ", round(Torb_h * 60, digits=1), " min")
println("debris along-track 1σ: ", round(coarse["db_T"][1], digits=1), " m (1 h) → ",
        round(coarse["db_T"][end], digits=1), " m (24 h)")
println("Pc through Σ(τ): ", round(coarse["pc"][end], sigdigits=3), " (24 h) → ",
        round(coarse["pc"][1], sigdigits=3), " (1 h)")
