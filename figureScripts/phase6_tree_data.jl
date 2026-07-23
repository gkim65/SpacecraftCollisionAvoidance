# =========================================================================
# phase6_tree_data.jl — dump the full MCTS tree structure for the three
# constraint modes (Fig 6c redesign: draw the actual trees, not bar charts).
#
# Runs the SAME conjunction + seed under :off / :penalize / :terminate at a
# SMALL depth (4) and modest sim budget so the tree is legible when drawn as a
# node-link diagram. For every node we record: an id, its parent id, the action
# on the edge into it (WAIT/MANEUVER/root), its depth, time-remaining, Pc-at-TCA,
# whether it violated the threshold, whether it's terminal, and its visit count.
#
# The plotter (phase6_tree_plot.py) lays each mode's tree out and colors nodes by
# Pc, marks violating nodes (X) and terminal nodes, so :terminate visibly
# amputates the violating WAIT subtree that :penalize / :off keep expanding.
#
# Julia is the SOURCE OF TRUTH. Kept OUT of the pinned brahe/numpy venv.
#
# Run from the repo root:
#   julia --project=. figureScripts/phase6_tree_data.jl
# writes figureScripts/phase6_tree_data.json
#
# Fixture: 20 m cross-track miss (WAIT-to-TCA Pc stays high ~0.07 and does NOT
# collapse into the once-per-orbit nulls, so WAIT branches genuinely violate a
# 1e-3 threshold while a 5 m/s burn clears it — see phase6_ablation_data.jl).
# =========================================================================
using LinearAlgebra
using Random
using PyCall
using POMDPs
using POMDPTools

_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
_json(x::Real) = isfinite(x) ? string(x) : "null"
_json(x::AbstractString) = "\"$x\""
_json(v::AbstractVector) = "[" * join(_json.(v), ",") * "]"
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

function make_conjunction_state(pomdp::SpacecraftCAPOMDP;
                                miss_m = 20.0, v_rel = 15.0,
                                geometry = :cross_track, t = 4 * 60 * 60)
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

# Generate the tree to depth 10 so the FULL data is on disk — the plotter can
# subset to a shallower depth (e.g. 4) for a legible figure without re-running
# this expensive Julia job. Keep observation widening tight (small k) and sims
# modest so the depth-10 tree doesn't explode in node count / Pc-eval cost
# (Pc ~172 ms/node). SEED/fixture identical across modes so the trees are
# directly comparable.
const DEPTH = 10
const NITER = 45
const SEED  = 7
const K_OBS = 2.0        # tighter widening than the 10.0 default → drawable fan-out

pc_threshold = 1.0e-3
pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                          TCA_max = DEPTH * 60 * 60, Δv = 5.0,
                          pc_threshold = pc_threshold)
s0 = make_conjunction_state(pomdp; miss_m = 20.0, v_rel = 15.0,
                            geometry = :cross_track, t = DEPTH * 60 * 60)

# Serialize the tree by a DFS that assigns integer ids. Each record carries the
# edge action into the node (WAIT/MANEUVER/"root"), parent id, depth, τ (hr),
# Pc, violated, terminal, visits.
function dump_tree(root::BeliefNode)
    nodes = Vector{Dict{String,Any}}()
    next_id = Ref(0)
    function visit(node::BeliefNode, parent_id::Int, edge_action::String, depth::Int)
        id = next_id[]; next_id[] += 1
        push!(nodes, Dict{String,Any}(
            "id" => id, "parent" => parent_id, "action" => edge_action,
            "depth" => depth, "tau_hr" => node.belief.t / 3600,
            "pc" => isnan(node.pc) ? NaN : node.pc,
            "violated" => node.violated, "terminal" => node.is_terminal,
            "visits" => node.N,
        ))
        # children grouped by action; order WAIT then MANEUVER for stable layout
        for a in (WAIT, MANEUVER)
            kids = get(node.children, a, BeliefNode[])
            aname = a == WAIT ? "WAIT" : "MANEUVER"
            for ch in kids
                visit(ch, id, aname, depth + 1)
            end
        end
        return id
    end
    visit(root, -1, "root", 0)
    return nodes
end

function run_mode(mode::Symbol)
    root = root_from_pomdp(pomdp, s0)
    planner = MCTSPlanner(pomdp; n_iterations = NITER, max_depth = DEPTH,
                          dt = pomdp.dt, k = K_OBS, constraint_mode = mode)
    best_a, _ = plan(planner, root, MersenneTwister(SEED))
    nodes = dump_tree(root)
    nviol = count(n -> n["violated"] === true, nodes)
    nterm = count(n -> n["terminal"] === true, nodes)
    ntermviol = count(n -> n["violated"] === true && n["terminal"] === true, nodes)
    return Dict{String,Any}(
        "mode" => String(mode),
        "best_action" => best_a == MANEUVER ? "MANEUVER" : "WAIT",
        "q_wait" => get(root.Qa, WAIT, NaN),
        "q_maneuver" => get(root.Qa, MANEUVER, NaN),
        "n_nodes" => length(nodes), "n_violating" => nviol,
        "n_terminal" => nterm, "n_terminal_violating" => ntermviol,
        "nodes" => nodes,
    )
end

off  = run_mode(:off)
pen  = run_mode(:penalize)
term = run_mode(:terminate)

data = Dict{String,Any}(
    "pc_threshold" => pc_threshold, "miss_m" => 20.0, "v_rel" => 15.0,
    "dv_ms" => pomdp.Δv, "depth" => DEPTH, "n_iterations" => NITER,
    "k_obs" => K_OBS, "seed" => SEED,
    "off" => off, "penalize" => pen, "terminate" => term,
)

out = joinpath(@__DIR__, "phase6_tree_data.json")
open(out, "w") do io; write(io, _json(data)); end
println("wrote $out")
for m in (off, pen, term)
    println(m["mode"], ": action=", m["best_action"],
            "  nodes=", m["n_nodes"], "  viol=", m["n_violating"],
            "  term=", m["n_terminal"], "  term_viol=", m["n_terminal_violating"])
end
