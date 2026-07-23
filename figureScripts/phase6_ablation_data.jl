# =========================================================================
# phase6_ablation_data.jl — constraint-mode ablation data (Fig 6c).
#
# Runs the SAME conjunction + seed under the three constraint_modes and records
# what the chance constraint does to the search:
#   :off       — no constraint (Phase-5-style): optimize Pc reward only.
#   :penalize  — violating branches kept but down-weighted (default).
#   :terminate — violating branches amputated (marked terminal).
#
# For each mode we walk the built tree and collect: total nodes, violating nodes,
# terminal-violating nodes, the chosen action, and the root Qa[WAIT]/Qa[MANEUVER]
# (the Q-margin). This is the quantitative "how much does the constraint change
# the decision / how much does it prune" figure Grace asked for.
#
# Julia is the SOURCE OF TRUTH. Kept OUT of the pinned brahe/numpy venv.
#
# Run from the repo root:
#   julia --project=. figureScripts/phase6_ablation_data.jl
# writes figureScripts/phase6_ablation_data.json
#
# Sims kept MODEST (Pc eval ~172 ms/node): a short window (4 h / 1-hr grid) and a
# small tree_queries so 3 trees finish in a few minutes.
# =========================================================================
using LinearAlgebra
using Random
using PyCall
using POMDPs
using POMDPTools

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
include(joinpath(REPO, "src", "rewards.jl"))
include(joinpath(REPO, "src", "observations.jl"))
include(joinpath(REPO, "src", "transitions.jl"))
include(joinpath(REPO, "src", "utils", "beliefTracker.jl"))
include(joinpath(REPO, "src", "utils", "beliefMCTS.jl"))

# Walk the whole tree, apply `f` to every node (from test_belief_mcts.jl).
function walk(f, node::BeliefNode)
    f(node)
    for (_, kids) in node.children, ch in kids
        walk(f, ch)
    end
end

# Fixture (make_conjunction_state from test_belief_mcts.jl): both objects at TCA
# at the requested miss, propagated back by t so the planner has a window.
function make_conjunction_state(pomdp::SpacecraftCAPOMDP;
                                miss_m = 200.0, v_rel = 15.0,
                                geometry = :cross_track, t = 3 * 60 * 60)
    sc_tca, db_tca = generate_conjunction_geometry(pomdp; geometry = geometry,
                                                   miss_m = miss_m, v_rel = v_rel)
    bh = get_brahe()
    epoch_tca = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    et = epoch_to_tuple(epoch_tca)
    prop_sc, ep0 = eci2orb_brahe(sc_tca, et, pomdp.satParams, pomdp.forceModel)
    prop_db, _   = eci2orb_brahe(db_tca, et, pomdp.debrisParams, pomdp.forceModel)
    prop_sc.propagate_to(ep0 - Float64(t))
    prop_db.propagate_to(ep0 - Float64(t))
    sc0 = collect(prop_sc.current_state()[1:6])
    db0 = collect(prop_db.current_state()[1:6])
    return CAState(sc0, db0, Float64(t))
end

# Δv = 5 m/s so a burn visibly lowers Pc-at-TCA (end-to-end fixture). Short 4-h
# window, 1-hr grid.
#
# GEOMETRY CHOICE (miss = 20 m, not 200/500 m): for the ablation to have anything
# to show, WAIT branches must actually VIOLATE the threshold inside the tree. At
# 200/500 m miss the at-TCA Pc collapses into the once-per-orbit covariance nulls
# (Fig 6a) — so most in-tree nodes sit at Pc ~1e-5 or below and nothing violates.
# At a 20 m miss the Pc stays high and persistent (Pc-at-TCA ~0.07, still ~0.008
# an hour out — measured), so WAIT-to-TCA genuinely and repeatedly exceeds a
# modest threshold while a 5 m/s burn (mean → ~90+ km separation, Fig 6b) clears
# it. pc_threshold FIXED at 1e-3 (below the WAIT Pc, above the cleared MANEUVER Pc).
pc_threshold = 1.0e-3
pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                          TCA_max = 4 * 60 * 60, Δv = 5.0,
                          pc_threshold = pc_threshold)
s0 = make_conjunction_state(pomdp; miss_m = 20.0, v_rel = 15.0,
                            geometry = :cross_track, t = 4 * 60 * 60)
nsteps = Int(round(s0.t / pomdp.dt))
pc_wait_root = node_pc_at_tca(pomdp, root_from_pomdp(pomdp, s0))

const NITER = 80        # modest sim budget (Pc ~172 ms/node × tree)
const SEED  = 11

function run_mode(mode::Symbol)
    root = root_from_pomdp(pomdp, s0)
    planner = MCTSPlanner(pomdp; n_iterations = NITER, max_depth = nsteps,
                          dt = pomdp.dt, constraint_mode = mode)
    best_a, _ = plan(planner, root, MersenneTwister(SEED))
    n_nodes = Ref(0); n_viol = Ref(0); n_term_viol = Ref(0)
    # Pc collected per depth (time-remaining bucket) to show WHERE violations sit.
    pcs = Float64[]; taus_hr = Float64[]; viol_flags = Bool[]
    walk(root) do nd
        n_nodes[] += 1
        if nd.violated
            n_viol[] += 1
            nd.is_terminal && (n_term_viol[] += 1)
        end
        if !isnan(nd.pc)
            push!(pcs, nd.pc); push!(taus_hr, nd.belief.t / 3600); push!(viol_flags, nd.violated)
        end
    end
    return Dict(
        "mode" => String(mode),
        "best_action" => best_a == MANEUVER ? "MANEUVER" : "WAIT",
        "q_wait" => get(root.Qa, WAIT, NaN),
        "q_maneuver" => get(root.Qa, MANEUVER, NaN),
        "n_nodes" => n_nodes[],
        "n_violating" => n_viol[],
        "n_terminal_violating" => n_term_viol[],
        "na_wait" => get(root.Na, WAIT, 0),
        "na_maneuver" => get(root.Na, MANEUVER, 0),
        "node_pc" => pcs, "node_tau_hr" => taus_hr, "node_violated" => viol_flags,
    )
end

off  = run_mode(:off)
pen  = run_mode(:penalize)
term = run_mode(:terminate)

data = Dict(
    "pc_threshold" => pc_threshold, "pc_wait_root" => pc_wait_root,
    "miss_m" => 20.0, "v_rel" => 15.0, "dv_ms" => pomdp.Δv,
    "n_iterations" => NITER, "window_hr" => 4, "dt_hr" => 1,
    "off" => off, "penalize" => pen, "terminate" => term,
)

out = joinpath(@__DIR__, "phase6_ablation_data.json")
open(out, "w") do io; write(io, _json(data)); end
println("wrote $out")
for m in (off, pen, term)
    println(m["mode"], ": action=", m["best_action"],
            "  Q(W)=", round(m["q_wait"]; digits = 2),
            "  Q(M)=", round(m["q_maneuver"]; digits = 2),
            "  nodes=", m["n_nodes"], "  viol=", m["n_violating"],
            "  term_viol=", m["n_terminal_violating"])
end
