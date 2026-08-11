# treezoom_probe.jl — FEASIBILITY PROBE for the FIG C "tree-zoom" idea.
#
# Question (Grace): can we even EXTRACT the per-rollout data to plot one depth of
# the MCTS tree — the WAIT vs MANEUVER children, each child's Pc-at-TCA, whether it
# tripped the chance-constraint (violated), and the resulting per-action Q — WITHOUT
# instrumenting the planner? The `plan(planner, root, rng)` call RETURNS the populated
# root BeliefNode (root.Qa / root.Na / root.children[a] -> Vector{BeliefNode}, each
# child carrying .pc and .violated), so in principle we just read them off. This probe
# confirms that and dumps one root's children to JSON so the figure code can plot it.
#
# Reproduces the REAL flip case from the sweep (000038771_conj_000030802, cadence 8 h,
# seed 1) at BEST and WORST sensor quality, using the sweep's default planner config
# (n_iterations=12, reward_mode=:terminal, constraint_mode=:penalize, k=2.0,
# grid_mode=:measurement, sigma_mode=:exact). If best defers (root picks WAIT) and
# worst maneuvers (root picks MANEUVER), the extraction matches the committed outcome.
#
# Usage:  julia --project=. figureScripts/treezoom_probe.jl
# Writes: figureScripts/data/treezoom/<quality>.json  + a console summary.

using LinearAlgebra, Random, PyCall, POMDPs, POMDPTools, Distributions

# --- hand-rolled JSON writer (project convention: JSON.jl is only a transitive
#     dep, so figureScripts/*.jl + scripts/*.jl hand-roll it). Handles nested
#     Dict/Vector, which is all this probe emits. ---
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_j(x::Nothing) = "null"
_j(x::Symbol) = _j(string(x))
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
_j(d::AbstractDict) = "{" * join([_j(string(k)) * ":" * _j(v) for (k, v) in d], ",") * "}"

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
include(joinpath(@__DIR__, "..", "src", "utils", "beliefExecutor.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "cdmScenario.jl"))

const CDM = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms",
    "000038771_conj_000030802_20201216_182131_20201215_171306.cdm"))
const CADENCE_H = 8.0
# Seeds to probe. Default just seed 1; override via env TREEZOOM_SEEDS="1,2,3" to check
# whether the root decision (and the n=50 flip) is seed-robust or a single-draw fluke.
const SEEDS = let e = get(ENV, "TREEZOOM_SEEDS", "1")
    parse.(Int, split(e, ","))
end
# n_iterations to probe. Default sweep uses 12 ("~12 rollouts"); pass more on the CLI
# (e.g. `julia … treezoom_probe.jl 12 50 200 500`) to watch the tree fill out and check
# whether the root decision holds. Files are written per (quality, n_iter[, seed]).
const NITERS = isempty(ARGS) ? [12] : parse.(Int, ARGS)

debris_pos_sigma(b) = sqrt(tr(b.debris.Σ[1:3, 1:3]))   # position 1σ magnitude (m)

# Recursively serialize the WHOLE tree built under `node` for ONE plan call, so the
# figure can draw the full branching (not just the root layer): every node reports
# its Pc-at-TCA, the chance-constraint `violated` flag, debris σ, tree depth, visit
# counts, and per-action Q; each node lists its children grouped by action so the
# WAIT/MANEUVER fanout at every level is preserved. This is the whole point of the
# "one decision, whole tree" zoom Grace asked for.
function dump_node(node, depth)
    kids = Dict{String,Any}()
    for (a, cs) in node.children
        kids[String(Symbol(a))] = [dump_node(c, depth + 1) for c in cs]
    end
    return Dict(
        "depth"     => depth,
        "pc"        => node.pc,
        "violated"  => node.violated,
        "sigma_debris_pos_m" => debris_pos_sigma(node.belief),
        "N"         => node.N,
        "Qa"        => Dict(String(Symbol(a)) => q for (a, q) in node.Qa),
        "Na"        => Dict(String(Symbol(a)) => n for (a, n) in node.Na),
        "children"  => kids,
    )
end

# tree size / max depth, for a quick "is this figure-able?" readout.
function tree_stats(node)
    n = 1
    dmax = 0
    for (_, cs) in node.children, c in cs
        cn, cd = tree_stats(c)
        n += cn
        dmax = max(dmax, cd + 1)
    end
    return n, dmax
end

function probe(quality::Symbol, niter::Int, seed::Int)
    sc = load_cdm_scenario(CDM; dt = 60*60, sensor_quality = quality)
    pomdp = sc.pomdp
    sec_cad = CADENCE_H * 3600
    root_grid = decision_grid(sc.t_horizon; cadence_secondary = sec_cad)
    grid_steps = grid_depth_count(root_grid)

    planner = MCTSPlanner(pomdp; n_iterations = niter, max_depth = grid_steps,
                          c = MCTS_UCB_C, k = 2.0, dt = 60*60, sigma_mode = :exact,
                          parallel = false, reward_mode = :terminal,
                          constraint_mode = :penalize, grid = root_grid)

    since_sc     = pomdp.correct_at_root ? 0.0 : pomdp.cadence_sc
    since_debris = pomdp.correct_at_root ? 0.0 : pomdp.cadence_debris
    root = BeliefNode(sc.b0, sc.s_true, isterminal(pomdp, sc.s_true);
                      since_sc = since_sc, since_debris = since_debris)

    a, root = plan(planner, root, MersenneTwister(seed))

    root_pc = node_pc_at_tca(pomdp, root)
    tree = dump_node(root, 0)
    nnodes, dmax = tree_stats(root)

    out = Dict(
        "case_id"        => "000038771_conj_000030802",
        "sensor_quality" => String(quality),
        "cadence_h"      => CADENCE_H,
        "seed"           => seed,
        "n_iterations"   => niter,
        "grid_steps"     => grid_steps,
        "t_horizon_h"    => sc.t_horizon / 3600,
        "pc_threshold"   => pomdp.pc_threshold,
        "root_pc_at_tca" => root_pc,
        "chosen_action"  => String(Symbol(a)),
        "n_nodes"        => nnodes,
        "max_depth"      => dmax,
        "tree"           => tree,   # the WHOLE tree under this one plan() call
    )

    outdir = joinpath(@__DIR__, "data", "treezoom")
    isdir(outdir) || mkpath(outdir)
    # Backward-compatible names: seed 1 keeps the old naming (plain for n=12, _n<ni>
    # otherwise, what the v2/v3 sketches read); other seeds always carry _s<seed>.
    base = niter == 12 ? "$(quality)" : "$(quality)_n$(niter)"
    fname = seed == 1 ? "$(base).json" : "$(base)_s$(seed).json"
    open(joinpath(outdir, fname), "w") do io
        write(io, _j(out))
    end

    println("=== quality=", quality, "  n_iter=", niter, "  seed=", seed,
            "  δ=", pomdp.pc_threshold, " ===")
    println("  root Pc-at-TCA = ", round(root_pc, sigdigits=3),
            "   root debris σ = ", round(debris_pos_sigma(root.belief), digits=1), " m")
    println("  chosen = ", a,
            "   Q(WAIT)=", round(get(root.Qa, WAIT, NaN), sigdigits=5),
            "  Q(MAN)=", round(get(root.Qa, MANEUVER, NaN), sigdigits=5),
            "   Na(W/M)=", get(root.Na, WAIT, 0), "/", get(root.Na, MANEUVER, 0))
    println("  full tree: ", nnodes, " nodes, max depth ", dmax)
    # per-depth node counts + how many violated, so we can see the penalty pattern
    bydepth = Dict{Int,Vector{Int}}()   # depth -> [n, n_violated]
    walk(nd) = begin
        v = get!(bydepth, nd["depth"], [0, 0]); v[1] += 1; nd["violated"] && (v[2] += 1)
        for (_, cs) in nd["children"], c in cs; walk(c); end
    end
    walk(tree)
    for d in sort(collect(keys(bydepth)))
        n, nv = bydepth[d]
        println("    depth ", lpad(d, 2), ": ", lpad(n, 3), " nodes, ",
                lpad(nv, 3), " violated (Pc>δ)")
    end
    println()
    flush(stdout)
    return out
end

println("TREE-ZOOM PROBE — full tree off the returned root; swept over n_iters × seeds.")
println("case ", basename(CDM), "  cadence ", CADENCE_H, " h  seeds ", SEEDS,
        "  n_iters ", NITERS); println(); flush(stdout)
summary = Tuple{Symbol,Int,Int,String,Int}[]   # (quality, niter, seed, chosen, nnodes)
for niter in NITERS, seed in SEEDS, q in (:best, :worst)
    o = probe(q, niter, seed)
    push!(summary, (q, niter, seed, o["chosen_action"], o["n_nodes"]))
end
println("\n=== SUMMARY (does the decision hold across seeds / iterations?) ===")
println(rpad("quality", 8), rpad("n_iter", 8), rpad("seed", 6), rpad("chosen", 10), "nodes")
for (q, ni, sd, ch, nn) in summary
    println(rpad(String(q), 8), rpad(ni, 8), rpad(sd, 6), rpad(ch, 10), nn)
end
println("Reading: if best→WAIT and worst→MANEUVER (matching the sweep), the FULL-TREE")
println("extraction is faithful and the 'one decision, whole tree' zoom is buildable.")
println("The per-depth violated counts show where the chance-constraint penalty fires")
println("along the rollouts, and Q at the root is what that propagates back into.")
