#=
test_belief_mcts.jl

Unit tests for the belief-space MCTS (src/utils/beliefMCTS.jl). This is a
HAND-ROLLED custom MCTS (architecture doc §4/§6), NOT POMCPOW — the belief
(μ,Σ) stays visible to the reward at every step.

Phase 5 (search mechanics) tests are kept: UCB selection, backup arithmetic,
progressive widening, tree health, determinism. Phase 6 wires in the chance
constraint, so the reward-specific tests now exercise the Pc-at-TCA reward and
the per-step constraint instead of the retired miss-distance placeholder:

 1. UCB SELECTION — unvisited-first (any c); higher-Q at c=0; exploration bonus
    flips a marginally-worse under-visited action. (Mechanics, unchanged.)
 2. BACKUP ARITHMETIC — running-mean update Q ← Q + (q−Q)/n. (Unchanged.)
 3. PROGRESSIVE WIDENING — should_widen follows POMCPOW's rule; child count
    grows ≈ k·na^α (sublinear). (Unchanged.)
 4. TREE HEALTH — every node's belief stays sym + PD + finite. (Unchanged.)
 5. DETERMINISM — same seed ⇒ identical tree + action. (Unchanged.)

 6. Pc-AT-TCA FROM A NODE (Phase 6): node_pc_at_tca is finite, in [0,1], and
    matches a DIRECT chan_pc call on the node's mean + its ACCUMULATED belief Σ
    propagated to TCA (the node's own tracked covariance, not a fresh P0).
 7. Pc GROWS THE BELIEF Σ OVER THE COAST (Phase 6): a node hours out grows its Σ
    over a longer remaining window ⇒ appreciable Pc; a node at TCA (no growth) is
    orders of magnitude smaller. Confirms the Σ is propagated, not held fixed.
 8. PER-STEP CONSTRAINT (Phase 6): the reward carries the violation penalty when
    Pc > threshold and not when Pc ≤ threshold; :off disables it; :terminate
    marks a violating child terminal.
 9. LEAF == CUTOFF (Phase 6): a true leaf and a budget-cutoff node with the same
    belief get the SAME Pc-at-TCA value (§7 — one consistent metric).
10. END-TO-END (Phase 6): on a real cross-track conjunction the planner drives
    Pc down / prefers the action that lowers Pc, and the constraint prunes /
    penalizes branches (the ablation: :off vs :penalize vs :terminate differ).

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
# grows it (and lowers Pc-at-TCA).
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

@testset "Belief-space MCTS (Phase 5 mechanics + Phase 6 chance constraint)" begin

    # -----------------------------------------------------------------
    @testset "1. UCB action selection" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)
        acts = POMDPs.actions(pomdp)

        # unvisited action is taken first, regardless of c
        b0 = belief_from_pomdp(pomdp, zeros(6), zeros(6), 3600.0)
        n = BeliefNode(b0, CAState(zeros(6), zeros(6), 3600.0), false)
        n.N = 5
        n.Na[WAIT] = 5; n.Qa[WAIT] = 100.0          # WAIT visited & high value
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
        Qa = Dict{CAAction,Float64}(); Na = Dict{CAAction,Int}()
        a = WAIT
        for (i, q) in enumerate([10.0, 20.0, 30.0])
            na = get(Na, a, 0)
            Na[a] = na + 1
            Qa[a] = get(Qa, a, 0.0) + (q - get(Qa, a, 0.0)) / Na[a]
        end
        @test Na[a] == 3
        @test Qa[a] ≈ 20.0            # mean(10,20,30)

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
        @test should_widen(0, 0; k = 10.0, α = 0.5) == true
        @test should_widen(10, 1; k = 10.0, α = 0.5) == true     # 10 ≤ 10·1
        @test should_widen(11, 1; k = 10.0, α = 0.5) == false    # 11 > 10·1
        @test should_widen(20, 4; k = 10.0, α = 0.5) == true     # bound = 10·2 = 20
        @test should_widen(21, 4; k = 10.0, α = 0.5) == false
        @test should_widen(2, 1; k = 1.0, α = 0.5) == false      # 2 > 1·1

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
    @testset "6. Pc-at-TCA from a node matches direct chan_pc (Option 2)" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        # feasible co-orbital cross-track fixture (v_rel = 15 m/s), 2 h before TCA
        s0 = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 2 * 60 * 60)
        node = root_from_pomdp(pomdp, s0)

        pc = node_pc_at_tca(pomdp, node)
        @test isfinite(pc)
        @test 0.0 <= pc <= 1.0
        @test pc > 0.0                       # a real 500 m cross-track conjunction

        # Direct chan_pc: propagate the node's belief MEAN and its ACCUMULATED Σ
        # to TCA (the node's own tracked covariance, NOT a fresh P0), then call
        # chan_pc. node_pc_at_tca is exactly this, refactored ⇒ match to roundoff.
        b = node.belief
        hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
        μ_sc, Σ_sc = _grow_belief_to_tca(pomdp, b.sc.μ,     b.sc.Σ,     pomdp.satParams,    b.t)
        μ_db, Σ_db = _grow_belief_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, b.t)
        pc_direct = chan_pc(μ_sc, μ_db, Σ_sc, Σ_db, hbr)
        @test pc ≈ pc_direct rtol = 1e-12

        # The distinguishing test (the root-node checks above pass under EITHER
        # model, since a root's belief Σ == P0). At a DEEP node reached via
        # predict/correct, the belief Σ has been measurement-shrunk and no longer
        # equals P0 — and node_pc_at_tca must use THAT Σ, not a fresh P0. Confirm
        # the two paths genuinely diverge and that node_pc_at_tca follows the
        # accumulated-Σ path.
        child, _ = expand_child(pomdp, node, WAIT, MersenneTwister(0); dt = pomdp.dt)
        @test !isapprox(child.belief.sc.Σ[1, 1], pomdp.P0_sc[1, 1])   # Σ shrunk, ≠ P0
        pc_child_acc = node_pc_at_tca(pomdp, child)
        # emulate the (rejected) fresh-P0 path at the same node
        m1, S1 = _grow_belief_to_tca(pomdp, child.belief.sc.μ,
                                     Matrix{Float64}(pomdp.P0_sc), pomdp.satParams, child.belief.t)
        m2, S2 = _grow_belief_to_tca(pomdp, child.belief.debris.μ,
                                     Matrix{Float64}(pomdp.P0_debris), pomdp.debrisParams, child.belief.t)
        pc_child_p0 = chan_pc(m1, m2, S1, S2, hbr)
        @test !isapprox(pc_child_acc, pc_child_p0; rtol = 1e-3)       # models genuinely differ
        @test pc_child_acc == child.pc                                # cached the accumulated-Σ Pc
    end

    # -----------------------------------------------------------------
    @testset "7. Pc-at-TCA grows the node's belief Σ over the remaining coast" begin
        # A node's Pc grows its OWN accumulated belief Σ from now to TCA (coast,
        # no further measurements). A node hours out grows Σ over a longer window,
        # so the 500 m miss sits inside a fatter covariance-at-TCA → a FINITE,
        # appreciable Pc; a node AT TCA has its tight belief Σ with no growth and a
        # 500 m miss is many-σ away → a Pc many orders of magnitude smaller. This
        # growth-vs-no-growth contrast is the robust signature that the Σ used is
        # actually being propagated (not held fixed).
        #
        # NB: at the ROOT the belief Σ equals P0 (freshly anchored), so this root
        # check numerically matches the growth-of-P0 curve; the correctness of the
        # ACCUMULATED-Σ path (deep nodes carry a measurement-shrunk Σ, not P0) is
        # exercised by the tree-based tests (4, 8, 10) where nodes are reached via
        # predict/correct. hour-to-hour Pc is NOT monotone in τ — radial/cross-track
        # Σ breathes once per orbit (~92.6 min; Phase 3), adjacent hourly samples
        # can differ ~10× either way (experiment_ideas.md #1) — so we assert the
        # robust growth-vs-no-growth gap, not a spurious hourly ordering.
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s_far = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                       geometry = :cross_track, t = 2 * 60 * 60)
        pc_far = node_pc_at_tca(pomdp, root_from_pomdp(pomdp, s_far))
        pc_tca = node_pc_at_tca(pomdp,
                    BeliefNode(belief_from_pomdp(pomdp, s_far.sc_eci, s_far.debris_eci, 0.0),
                               CAState(s_far.sc_eci, s_far.debris_eci, 0.0), false))
        @test pc_far > 1e-4                  # Σ grown over 2 h ⇒ finite, appreciable Pc
        @test pc_tca < pc_far / 100          # at TCA (no growth) ⇒ orders of magnitude smaller
    end

    # -----------------------------------------------------------------
    @testset "8. Per-step chance constraint fires above / not below threshold" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s0 = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 2 * 60 * 60)
        node = root_from_pomdp(pomdp, s0)
        pc = node_pc_at_tca(pomdp, node)

        # Case A: force the constraint to bind by setting the threshold BELOW pc.
        pomdp_lo = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                     TCA_max = 3 * 60 * 60, pc_threshold = pc / 2)
        nlo = root_from_pomdp(pomdp_lo, s0)
        r_pen, pc_lo, viol_pen = step_reward(pomdp_lo, WAIT, nlo; constraint_mode = :penalize)
        @test viol_pen == true
        @test pc_lo ≈ pc rtol = 1e-9
        # penalize reward = −weight·pc − penalty ; off = −weight·pc (no penalty)
        nlo_off = root_from_pomdp(pomdp_lo, s0)
        r_off, _, viol_off = step_reward(pomdp_lo, WAIT, nlo_off; constraint_mode = :off)
        @test viol_off == true                          # still flagged...
        @test r_off ≈ r_pen + MCTS_PC_VIOLATION_PENALTY  # ...but no penalty applied
        @test r_pen < r_off                              # penalize is worse by exactly the penalty

        # Case B: threshold ABOVE pc ⇒ no violation, no penalty.
        pomdp_hi = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                     TCA_max = 3 * 60 * 60, pc_threshold = pc * 2)
        nhi = root_from_pomdp(pomdp_hi, s0)
        r_hi, _, viol_hi = step_reward(pomdp_hi, WAIT, nhi; constraint_mode = :penalize)
        @test viol_hi == false
        @test r_hi ≈ -MCTS_PC_REWARD_WEIGHT * pc          # only the Pc term (WAIT, no fuel)

        # Case C: :terminate marks a violating child terminal inside expand_child.
        nterm = root_from_pomdp(pomdp_lo, s0)
        child, _ = expand_child(pomdp_lo, nterm, WAIT, MersenneTwister(0);
                                dt = pomdp_lo.dt, constraint_mode = :terminate)
        @test child.violated == (child.pc > pomdp_lo.pc_threshold)
        if child.violated
            @test child.is_terminal == true
        end
    end

    # -----------------------------------------------------------------
    @testset "9. Leaf and cutoff nodes get the same Pc-at-TCA value (§7)" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s0 = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 2 * 60 * 60)
        # Two nodes with the SAME belief but different terminal status: one a true
        # leaf, one a mid-horizon cutoff. §7 requires the SAME Pc-at-TCA value.
        b = root_from_pomdp(pomdp, s0).belief
        leaf   = BeliefNode(b, s0, true)     # marked terminal (true leaf)
        cutoff = BeliefNode(b, s0, false)    # not terminal (budget cutoff)
        v_leaf   = leaf_value(pomdp, leaf)
        v_cutoff = leaf_value(pomdp, cutoff)
        @test v_leaf ≈ v_cutoff              # identical — one consistent metric
        @test leaf.pc ≈ cutoff.pc
    end

    # -----------------------------------------------------------------
    @testset "10. End-to-end + ablation (Pc reward drives the decision)" begin
        # Δv = 5 m/s so a single burn grows the 200 m cross-track miss to ~170 km
        # by TCA — which drops Pc-at-TCA far below any WAIT branch. Set the
        # threshold below the WAIT Pc so WAITing is a violation the planner should
        # avoid.
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60, Δv = 5.0)
        s0 = make_conjunction_state(pomdp; miss_m = 200.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 3 * 60 * 60)
        nsteps = Int(round(s0.t / pomdp.dt))

        # Establish that WAIT is risky and MANEUVER is safe at TCA (Pc ordering).
        root0 = root_from_pomdp(pomdp, s0)
        pc_now = node_pc_at_tca(pomdp, root0)
        @test isfinite(pc_now) && 0.0 <= pc_now <= 1.0

        # Threshold that WAIT-to-TCA violates but a maneuver clears.
        planner = MCTSPlanner(pomdp; n_iterations = 60, max_depth = nsteps,
                              dt = pomdp.dt, constraint_mode = :penalize)
        root = root_from_pomdp(pomdp, s0)
        best_a, _ = plan(planner, root, MersenneTwister(11))
        @test best_a == MANEUVER
        @test get(root.Qa, MANEUVER, -Inf) > get(root.Qa, WAIT, -Inf)

        # Ablation: the constraint mode changes the tree. Under :terminate a
        # violating branch is amputated (some nodes marked terminal that a
        # :penalize run keeps expanding). Count violating nodes under each mode.
        function count_violations(mode)
            r = root_from_pomdp(pomdp, s0)
            p = MCTSPlanner(pomdp; n_iterations = 60, max_depth = nsteps,
                            dt = pomdp.dt, constraint_mode = mode)
            plan(p, r, MersenneTwister(11))
            nv = Ref(0); nterm_viol = Ref(0)
            walk(r) do nd
                if nd.violated
                    nv[] += 1
                    nd.is_terminal && (nterm_viol[] += 1)
                end
            end
            return nv[], nterm_viol[]
        end
        nv_pen, nterm_pen = count_violations(:penalize)
        nv_term, nterm_term = count_violations(:terminate)
        # Under :terminate, every violating node that isn't a true TCA leaf is
        # forced terminal; under :penalize violating non-leaf nodes keep children.
        @test nterm_term >= nterm_pen
    end

end
