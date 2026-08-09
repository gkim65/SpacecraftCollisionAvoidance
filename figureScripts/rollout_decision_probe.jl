# rollout_decision_probe.jl — DIAGNOSTIC (read-only, no commit). Two questions:
#   (1) Is the executed Pc actually Pc-AT-TCA? Print each node's belief time `b.t`
#       (must be > 0 pre-TCA so node_pc_at_tca GROWS Σ to TCA) alongside the Pc.
#   (2) Why does the 40115 episode MANEUVER when the WAIT-spine says defer? Print
#       Q(WAIT) vs Q(MANEUVER) at each executed step's root + the WAIT-spine FROM
#       that root (real noisy belief), to see if a noisy measurement flips it.
#
# Reproduces run_episode's per-step loop with the SAME seed so it matches the run.
# Usage: julia --project=. figureScripts/rollout_decision_probe.jl [file]

using LinearAlgebra, Random, PyCall, POMDPs, POMDPTools, Distributions, Printf

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

const CDMDIR = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms"))
const FILE   = length(ARGS) >= 1 ? ARGS[1] :
    "000040115_conj_000030660_20230721_100115_20230720_061903.cdm"
const SEED   = 20240809
const NITER  = 12

sc = load_cdm_scenario(joinpath(CDMDIR, FILE))
p  = sc.pomdp
sec_cad = p.cadence_debris

println("case ", sc.name1, " vs ", sc.name2, "  horizon ",
        round(sc.t_horizon/3600, digits=2), " h  thr ", p.pc_threshold)

grid_builder = t -> begin
    g, _ = adaptive_decision_grid(p, sc.b0, sc.s_true, t;
             cadence_secondary = sec_cad, coarse = 8*3600, fine = sec_cad,
             truncate_safe = false); g
end
root_grid, _ = adaptive_decision_grid(p, sc.b0, sc.s_true, sc.t_horizon;
                 cadence_secondary = sec_cad, coarse = 8*3600, fine = sec_cad,
                 truncate_safe = false)
grid_steps = grid_depth_count(root_grid)

s_true = sc.s_true
belief = belief_from_pomdp(p, s_true.sc_eci, s_true.debris_eci, s_true.t)
since_sc = p.correct_at_root ? 0.0 : p.cadence_sc
since_debris = p.correct_at_root ? 0.0 : p.cadence_debris
rng = MersenneTwister(SEED)

step = 0
while !isterminal(p, s_true) && step < grid_steps
    global step, s_true, belief, since_sc, since_debris
    step += 1
    t_rem = s_true.t
    g = grid_builder(t_rem)
    pl = MCTSPlanner(p; n_iterations = NITER, max_depth = grid_steps, k = 2.0,
                     dt = p.dt, sigma_mode = :exact, parallel = false,
                     reward_mode = :terminal, constraint_mode = :penalize, grid = g)
    root = BeliefNode(belief, s_true, isterminal(p, s_true);
                      since_sc = since_sc, since_debris = since_debris)
    a, r = plan(pl, root, rng)
    qw = get(r.Qa, WAIT, NaN); qm = get(r.Qa, MANEUVER, NaN)
    spine = wait_spine_pc(p, belief, s_true, g.t_epochs)
    pc_here = node_pc_at_tca(p, root)   # Pc-at-TCA from the CURRENT belief (this root)
    @printf("\nstep %d  t_rem=%.2fh  belief.t=%.2fh (>0 ⇒ grown to TCA)  grid=%d steps\n",
            step, t_rem/3600, belief.t/3600, grid_depth_count(g))
    @printf("   Pc-at-TCA(now)=%.3e   Q(WAIT)=%.4g  Q(MANEUVER)=%.4g  Na(W/M)=%d/%d  → %s\n",
            pc_here, qw, qm, get(r.Na,WAIT,0), get(r.Na,MANEUVER,0), a)
    @printf("   WAIT-spine from here (real belief): %s\n",
            join([@sprintf("%.2e", x) for x in spine], " "))
    exec_dt = g.dts[1]
    sp = rand(rng, transition_dt(p, s_true, a, exec_dt))
    belief, since_sc, since_debris = step_belief(p, belief, a, sp, rng;
        dt = exec_dt, cadence_sc = p.cadence_sc, cadence_debris = p.cadence_debris,
        since_sc = since_sc, since_debris = since_debris, grid = g, grid_depth = 1)
    post = BeliefNode(belief, sp, isterminal(p, sp))
    @printf("   after %s: post-belief.t=%.2fh  Pc-at-TCA(post)=%.3e  miss=%.1f km\n",
            a, belief.t/3600, node_pc_at_tca(p, post), miss_distance(sp)/1000)
    s_true = sp
end
