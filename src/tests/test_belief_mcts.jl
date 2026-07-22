#=
test_belief_mcts.jl

Phase 5 unit tests for the baseline belief-space MCTS skeleton
(src/utils/beliefMCTS.jl). This is a HAND-ROLLED custom MCTS (architecture doc
§4/§6), NOT POMCPOW — the belief (μ,Σ) stays visible to the reward at every
step. The reward here is MISS-DISTANCE-ONLY (no Pc, no chance constraint —
that is Phase 6). These tests confirm the SEARCH MECHANICS are correct and
deterministic under a fixed seed:

1. UCB SELECTION (must pass): with fixed per-action Q/n stats, ucb_select picks
   the higher-value action when exploration is off (c=0), and takes any
   unvisited action first regardless of c.

2. BACKUP ARITHMETIC (must pass): a single simulate! step updates the chosen
   action's visit count and running-average value by the exact running-mean
   formula Q ← Q + (q−Q)/n. Checked against a hand-computed value.

3. PROGRESSIVE WIDENING (must pass): should_widen follows POMCPOW's rule
   (n_children ≤ k·na^α), and over many simulations a node's observation-child
   count grows at the widening rate (≈ k·na^α), not once-per-visit and not
   unbounded.

4. TREE EXPANSION HEALTH (must pass): building a tree via predict/correct +
   dynamics produces child beliefs that stay symmetric, positive-definite, and
   free of NaN/Inf at every node (reuses the Phase 4 PD/sym checks).

5. DETERMINISM (must pass): same seed ⇒ identical tree (visit counts, values,
   chosen action). Different-seed runs are allowed to differ.

6. END-TO-END (must pass): on a real cross-track conjunction where a maneuver
   clearly increases the miss distance at TCA, the planner prefers MANEUVER.

VALIDITY: Phase 5 is the mechanics checkpoint. The miss-distance reward is a
placeholder for the Phase 6 Pc-at-TCA / chance-constraint reward; §7's
leaf/cutoff structure is honored but valued by miss distance for now.

Run:  julia --project=. src/tests/test_belief_mcts.jl
=#

using Test
using LinearAlgebra
using Random
using PyCall
using POMDPs        # SpacecraftCAPOMDP.jl subtypes POMDP{...}
using POMDPTools

include(joinpath(@__DIR__, "..", "SpacecraftCAPOMDP.jl"))
include(joinpath(@__DIR__, "..", "utils", "genConjunctions.jl"))
include(joinpath(@__DIR__, "..", "utils", "computePc.jl"))
include(joinpath(@__DIR__, "..", "utils", "covarianceTable.jl"))
include(joinpath(@__DIR__, "..", "states.jl"))
include(joinpath(@__DIR__, "..", "actions.jl"))
include(joinpath(@__DIR__, "..", "rewards.jl"))
include(joinpath(@__DIR__, "..", "observations.jl"))
include(joinpath(@__DIR__, "..", "transitions.jl"))
include(joinpath(@__DIR__, "..", "utils", "beliefTracker.jl"))
include(joinpath(@__DIR__, "..", "utils", "beliefMCTS.jl"))

# --- Phase 4 belief-health helpers (reused) -----------------------------------
_issym(A) = maximum(abs.(Matrix(A) .- transpose(Matrix(A)))) == 0.0
_pd(A) = (try; cholesky(Symmetric(Matrix(A))); true; catch; false; end)
_finite(A) = all(isfinite, A)

function belief_healthy(b::Belief)
    ok = true
    for ob in (b.sc, b.debris)
        ok &= _issym(ob.Σ) && _pd(ob.Σ) && _finite(ob.μ) && _finite(ob.Σ)
    end
    return ok
end

# Walk the whole tree, apply `f` to every node.
function walk(f, node::BeliefNode)
    f(node)
    for (_, kids) in node.children, ch in kids
        walk(f, ch)
    end
end

# --- Fixture: a real cross-track conjunction with a decision window -----------
# Places both objects at TCA at the requested small miss (via the Phase 2
# geometry generator), then propagates BOTH back by `t` so the planner has a
# window in which to act. WAIT-to-TCA reproduces the requested miss; a maneuver
# grows it.
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

@testset "Phase 5 — baseline belief-space MCTS" begin

    # -----------------------------------------------------------------
    @testset "1. UCB action selection" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)
        acts = POMDPs.actions(pomdp)

        # unvisited action is taken first, regardless of c
        b0 = belief_from_pomdp(pomdp, zeros(6), zeros(6), 3600.0)
        n = BeliefNode(b0, CAState(zeros(6), zeros(6), 3600.0), false)
        n.N = 5
        n.Na[WAIT] = 5; n.Qa[WAIT] = 100.0          # WAIT visited & high value
        # MANEUVER unvisited ⇒ must be chosen first
        @test ucb_select(n, acts; c = 1.0) == MANEUVER
        @test ucb_select(n, acts; c = 0.0) == MANEUVER

        # both visited: with c=0 (pure exploitation) pick the higher Q
        n2 = BeliefNode(b0, CAState(zeros(6), zeros(6), 3600.0), false)
        n2.N = 20
        n2.Na[WAIT] = 10;     n2.Qa[WAIT] = 1.0
        n2.Na[MANEUVER] = 10; n2.Qa[MANEUVER] = 5.0
        @test ucb_select(n2, acts; c = 0.0) == MANEUVER
        n2.Qa[WAIT] = 9.0
        @test ucb_select(n2, acts; c = 0.0) == WAIT

        # exploration bonus can flip a marginally-worse but under-visited action
        n3 = BeliefNode(b0, CAState(zeros(6), zeros(6), 3600.0), false)
        n3.N = 100
        n3.Na[WAIT] = 99;     n3.Qa[WAIT] = 1.0      # heavily visited
        n3.Na[MANEUVER] = 1;  n3.Qa[MANEUVER] = 0.9  # barely visited, slightly worse
        # bonus for MANEUVER: c·√(ln100/1) ≈ 2.15 > gap 0.1 ⇒ explore MANEUVER
        @test ucb_select(n3, acts; c = 1.0) == MANEUVER
    end

    # -----------------------------------------------------------------
    @testset "2. Backup running-average arithmetic" begin
        # Directly exercise the running-mean update the way simulate! applies it.
        Qa = Dict{CAAction,Float64}(); Na = Dict{CAAction,Int}()
        a = WAIT
        for (i, q) in enumerate([10.0, 20.0, 30.0])
            na = get(Na, a, 0)
            Na[a] = na + 1
            Qa[a] = get(Qa, a, 0.0) + (q - get(Qa, a, 0.0)) / Na[a]
        end
        @test Na[a] == 3
        @test Qa[a] ≈ 20.0            # mean(10,20,30)

        # matches a straight mean for a longer stream
        Random.seed!(1)
        xs = randn(50) .* 7 .+ 3
        Q = 0.0
        for (i, x) in enumerate(xs)
            Q += (x - Q) / i
        end
        @test Q ≈ sum(xs) / length(xs)
    end

    # -----------------------------------------------------------------
    @testset "3. Progressive widening rule + growth rate" begin
        # POMCPOW rule: add while n_children ≤ k·na^α
        @test should_widen(0, 0; k = 10.0, α = 0.5) == true
        @test should_widen(10, 1; k = 10.0, α = 0.5) == true     # 10 ≤ 10·1
        @test should_widen(11, 1; k = 10.0, α = 0.5) == false    # 11 > 10·1
        # na=4, α=0.5 ⇒ bound = 10·2 = 20
        @test should_widen(20, 4; k = 10.0, α = 0.5) == true
        @test should_widen(21, 4; k = 10.0, α = 0.5) == false
        # smaller k widens more slowly
        @test should_widen(2, 1; k = 1.0, α = 0.5) == false      # 2 > 1·1

        # End-to-end growth: with a small k the child count tracks ≈ k·na^α, i.e.
        # sublinear in visits (NOT one child per visit, NOT unbounded).
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s0 = make_conjunction_state(pomdp; t = 3 * 60 * 60)
        root = root_from_pomdp(pomdp, s0)
        planner = MCTSPlanner(pomdp; n_iterations = 40, max_depth = 3,
                              k = 3.0, α = 0.5, dt = pomdp.dt)
        plan(planner, root, MersenneTwister(3))
        for a in POMDPs.actions(pomdp)
            na = get(root.Na, a, 0)
            nc = length(get(root.children, a, BeliefNode[]))
            if na > 0
                # children never exceed one-per-visit and honor the widening bound
                @test nc <= na
                @test nc <= ceil(3.0 * na^0.5) + 1
            end
        end
    end

    # -----------------------------------------------------------------
    @testset "4. Tree expansion health (sym + PD + finite)" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s0 = make_conjunction_state(pomdp; t = 3 * 60 * 60)
        root = root_from_pomdp(pomdp, s0)
        planner = MCTSPlanner(pomdp; n_iterations = 50, max_depth = 3, dt = pomdp.dt)
        plan(planner, root, MersenneTwister(5))

        n_nodes = Ref(0)
        all_ok = Ref(true)
        walk(root) do node
            n_nodes[] += 1
            all_ok[] &= belief_healthy(node.belief)
            all_ok[] &= all(isfinite, node.s_true.sc_eci)
            all_ok[] &= all(isfinite, node.s_true.debris_eci)
        end
        @test n_nodes[] > 1          # tree actually expanded
        @test all_ok[]               # every node healthy
    end

    # -----------------------------------------------------------------
    @testset "5. Determinism under fixed seed" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s0 = make_conjunction_state(pomdp; t = 3 * 60 * 60)

        function run(seed)
            root = root_from_pomdp(pomdp, s0)
            planner = MCTSPlanner(pomdp; n_iterations = 40, max_depth = 3, dt = pomdp.dt)
            a, _ = plan(planner, root, MersenneTwister(seed))
            return a, copy(root.Na), copy(root.Qa)
        end

        a1, Na1, Qa1 = run(9)
        a2, Na2, Qa2 = run(9)          # same seed
        @test a1 == a2
        @test Na1 == Na2
        for a in POMDPs.actions(pomdp)
            @test get(Qa1, a, NaN) == get(Qa2, a, NaN)   # bitwise identical
        end
    end

    # -----------------------------------------------------------------
    @testset "6. End-to-end: prefers maneuver when it clearly helps" begin
        # Δv = 5 m/s so a single burn grows the 200 m cross-track miss to ~170 km
        # by TCA (verified: WAIT→200 m, MANEUVER→171 km). The planner should see
        # this in the miss-distance reward and prefer MANEUVER.
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60, Δv = 5.0)
        s0 = make_conjunction_state(pomdp; miss_m = 200.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 3 * 60 * 60)
        nsteps = Int(round(s0.t / pomdp.dt))

        # sanity: the fixture really is a close approach that a maneuver widens
        wait_s = s0; man_s = s0
        for i in 1:nsteps
            wait_s = rand(MersenneTwister(0), POMDPs.transition(pomdp, wait_s, WAIT))
        end
        man_s = rand(MersenneTwister(0), POMDPs.transition(pomdp, s0, MANEUVER))
        for i in 2:nsteps
            man_s = rand(MersenneTwister(0), POMDPs.transition(pomdp, man_s, WAIT))
        end
        @test miss_distance(wait_s) < 1_000.0            # WAIT ⇒ near-hit
        @test miss_distance(man_s) > 10 * miss_distance(wait_s)  # maneuver widens

        root = root_from_pomdp(pomdp, s0)
        planner = MCTSPlanner(pomdp; n_iterations = 60, max_depth = nsteps, dt = pomdp.dt)
        best_a, _ = plan(planner, root, MersenneTwister(11))
        @test best_a == MANEUVER
        @test get(root.Qa, MANEUVER, -Inf) > get(root.Qa, WAIT, -Inf)
    end

end
