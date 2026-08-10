# =========================================================================
# run_episode_entry.jl — the SCHEDULER-AGNOSTIC "run ONE config" entrypoint.
#
# Reads ONE episode config, runs `run_episode_metrics(cfg)` (beliefExecutor.jl),
# and writes the resulting flat metrics dict as JSON. This is the seam the Python
# wandb runner (scripts/wandb_runner.py) shells out to: a wandb sweep agent pulls
# a config, the Python runner writes it here, calls this script, reads the JSON
# back, and logs it. Nothing wandb-specific lives in Julia.
#
# CONFIG IN (avoids a JSON parser — JSON.jl is only a transitive dep here): the
# config is passed as a small Julia file that defines `CONFIG::Dict{String,Any}`
# (the Python runner emits it). We `include` it. See scripts/wandb_runner.py.
#
# METRICS OUT: hand-rolled JSON (same convention as figureScripts/*.jl — JSON only
# a transitive dep, so we do not `import JSON`).
#
# Usage (direct, no wandb):
#   julia --project=. scripts/run_episode_entry.jl <config.jl> <out.json>
#
# The <config.jl> file must define e.g.
#   CONFIG = Dict{String,Any}("case_path"=>"data/cara_cdms/....cdm",
#                             "sensor_quality"=>"median", "cadence_secondary"=>28800.0,
#                             "seed"=>1, "sigma_mode"=>"exact", "n_iterations"=>12, ...)
# Any key `episode_config` / `run_episode_metrics` understands is accepted; unset
# keys fall back to the `episode_config` defaults (grid_mode defaults :measurement).
# =========================================================================

using LinearAlgebra, Random, PyCall, POMDPs, POMDPTools, Distributions, Dates, Printf

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "SpacecraftCollisionAvoidance.jl"))

# --- hand-rolled JSON writer (project convention: JSON only a transitive dep) ---
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_j(x::Nothing) = "null"
_j(::Missing) = "null"
_j(x::Symbol) = _j(string(x))
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
_j(d::AbstractDict) = "{" * join([_j(string(k)) * ":" * _j(v) for (k, v) in d], ",") * "}"

function main()
    length(ARGS) >= 2 ||
        error("usage: julia --project=. scripts/run_episode_entry.jl <config.jl> <out.json>")
    config_path = ARGS[1]
    out_path    = ARGS[2]

    isfile(config_path) || error("run_episode_entry: config file not found: $config_path")
    # The config file's last expression IS the config dict (`CONFIG = Dict(...)` — an
    # assignment returns its value). `Base.include` returns that value, so we capture
    # it directly rather than depending on the global-binding visibility of Main.CONFIG
    # inside this function's world age.
    raw = Base.include(Main, abspath(config_path))
    raw isa AbstractDict ||
        error("run_episode_entry: $config_path must END with a Dict{String,Any} " *
              "(e.g. `CONFIG = Dict{String,Any}(...)` as the last expression); got $(typeof(raw))")

    # Rebuild a normalized config through episode_config so defaults + types are
    # applied consistently (a sweep only sets the keys it varies). `case_path` is
    # resolved relative to the repo root if not absolute.
    cp = String(raw["case_path"])
    isabspath(cp) || (cp = normpath(joinpath(REPO, cp)))
    isfile(cp) || error("run_episode_entry: case_path not found: $cp")

    getk(k, d) = haskey(raw, k) && raw[k] !== nothing ? raw[k] : d
    cfg = episode_config(;
        case_path         = cp,
        cadence_secondary = getk("cadence_secondary", nothing),
        coarse            = getk("coarse", 8 * 3600),
        fine              = getk("fine", nothing),
        sensor_quality    = Symbol(getk("sensor_quality", "median")),
        seed              = Int(getk("seed", 20240809)),
        sigma_mode        = Symbol(getk("sigma_mode", "exact")),
        parallel          = Bool(getk("parallel", false)),
        n_iterations      = Int(getk("n_iterations", 12)),
        reward_mode       = Symbol(getk("reward_mode", "terminal")),
        constraint_mode   = Symbol(getk("constraint_mode", "penalize")),
        k                 = Float64(getk("k", 2.0)),
        truncate_safe     = Bool(getk("truncate_safe", false)),
        dt                = Float64(getk("dt", 60 * 60)),
        grid_mode         = Symbol(getk("grid_mode", "measurement")),
        p_arrival         = Float64(getk("p_arrival", 1.0)),
        verbose           = Bool(getk("verbose", true)),
    )
    # pass-through of the optional overrides episode_config keys but leaves nothing
    sc_over = getk("sec_class_override", nothing)
    sc_over !== nothing && (cfg["sec_class_override"] = String(sc_over))
    th = getk("t_horizon", nothing); th !== nothing && (cfg["t_horizon"] = Float64(th))
    pt = getk("pc_threshold", nothing); pt !== nothing && (cfg["pc_threshold"] = Float64(pt))
    ms = getk("max_steps", nothing); ms !== nothing && (cfg["max_steps"] = Int(ms))

    # DECISION POLICY (F3 baseline comparison). Two ways to set it from a config:
    #  • `policy_variant` (the sweep-friendly SINGLE flat axis, Grace's call): one
    #    string like "mcts" / "wait_feasibility" / "delay_12h" mapped to
    #    (policy, policy_params) via `policy_variant_spec` — so ALL baselines live in
    #    ONE sweep axis with no invalid grid cells.
    #  • explicit `policy` + `policy_params` (direct override; wins if given).
    # Absent both → defaults to "mcts" (unchanged behavior).
    pv = getk("policy_variant", nothing)
    if getk("policy", nothing) !== nothing
        cfg["policy"] = String(raw["policy"])
        pp = getk("policy_params", nothing)
        cfg["policy_params"] = pp === nothing ? nothing : Dict{String,Any}(String(k) => v for (k, v) in pp)
    elseif pv !== nothing
        pk, pp = policy_variant_spec(String(pv))
        cfg["policy"] = pk
        cfg["policy_params"] = pp
    end

    @info "run_episode_entry: running one episode" case=basename(cp) quality=cfg["sensor_quality"] cadence_h=(cfg["cadence_secondary"]===nothing ? "tier" : cfg["cadence_secondary"]/3600) seed=cfg["seed"] grid_mode=cfg["grid_mode"]

    metrics = run_episode_metrics(cfg)

    mkpath(dirname(abspath(out_path)))
    open(out_path, "w") do io; write(io, _j(metrics)); end
    @info "run_episode_entry: wrote metrics" out=out_path wall_s=metrics["wall_time_s"] decision=metrics["actual_decision"] pc_at_tca=metrics["pc_at_tca"]
    return nothing
end

main()
