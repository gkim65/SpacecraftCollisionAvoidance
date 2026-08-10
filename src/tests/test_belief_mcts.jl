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
11. ASYMMETRIC CADENCE (measurement realism): the sat (GPS, ~2 h) and debris
    (TLE, ~8 h) get corrected on different schedules; predict-only steps grow Σ,
    fix steps shrink it relative to the same step's predict-only counterpart.
12. FAST Σ PATH (efficiency pass): the precompute-per-depth debris Σ table
    reproduces the exact per-node debris Σ-at-TCA, and sigma_mode = :fast gives a
    Pc (and an end-to-end action + tree) identical to :exact — the fast path
    introduces NO approximation (debris Σ branch-invariant, sat propagated exact).
13. ROOT-PARALLEL MCTS (efficiency pass): the split/seed/Na-weighted-Q-merge
    helpers are arithmetically correct; the local-fallback parallel plan is
    reproducible under a fixed master seed and picks the same action as a serial
    plan; and a real 2-worker multiprocess plan is reproducible under a fixed
    master seed (same merged Qa, run to run) and agrees with serial on the action.

Run:  julia --project=. src/tests/test_belief_mcts.jl
=#

using Test
using LinearAlgebra
using Random
using PyCall
using POMDPs        # SpacecraftCAPOMDP.jl subtypes POMDP{...}
using POMDPTools
using Distributed   # group 13: root-parallel MCTS (multiprocess)

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
include(joinpath(@__DIR__, "..", "utils", "beliefExecutor.jl"))

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
    @testset "6. Pc-at-TCA from a node matches direct elrod_pc (Option 2)" begin
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

        # Direct elrod_pc: propagate the node's belief MEAN and its ACCUMULATED Σ
        # to TCA (the node's own tracked covariance, NOT a fresh P0), then call
        # elrod_pc — the Pc method node_pc_at_tca now uses (swapped from chan_pc
        # for anisotropy robustness). node_pc_at_tca is exactly this, refactored
        # ⇒ match to roundoff.
        b = node.belief
        hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
        μ_sc, Σ_sc = _grow_belief_to_tca(pomdp, b.sc.μ,     b.sc.Σ,     pomdp.satParams,    b.t)
        μ_db, Σ_db = _grow_belief_to_tca(pomdp, b.debris.μ, b.debris.Σ, pomdp.debrisParams, b.t)
        pc_direct = elrod_pc(μ_sc, μ_db, Σ_sc, Σ_db, hbr)
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
        pc_child_p0 = elrod_pc(m1, m2, S1, S2, hbr)
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
        # NB this testset covers the LEGACY :per_step reward (the per-step Pc term +
        # per-step violation penalty), so it passes reward_mode = :per_step
        # explicitly. The DEFAULT is now :terminal (Pc charged at the leaf, per-step
        # reward = fuel only) — covered by testset 8b below.
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
        r_pen, pc_lo, viol_pen = step_reward(pomdp_lo, WAIT, nlo;
                                             reward_mode = :per_step, constraint_mode = :penalize)
        @test viol_pen == true
        @test pc_lo ≈ pc rtol = 1e-9
        # penalize reward = −weight·pc − penalty ; off = −weight·pc (no penalty)
        nlo_off = root_from_pomdp(pomdp_lo, s0)
        r_off, _, viol_off = step_reward(pomdp_lo, WAIT, nlo_off;
                                         reward_mode = :per_step, constraint_mode = :off)
        @test viol_off == true                          # still flagged...
        @test r_off ≈ r_pen + MCTS_PC_VIOLATION_PENALTY  # ...but no penalty applied
        @test r_pen < r_off                              # penalize is worse by exactly the penalty

        # Case B: threshold ABOVE pc ⇒ no violation, no penalty.
        pomdp_hi = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                     TCA_max = 3 * 60 * 60, pc_threshold = pc * 2)
        nhi = root_from_pomdp(pomdp_hi, s0)
        r_hi, _, viol_hi = step_reward(pomdp_hi, WAIT, nhi;
                                       reward_mode = :per_step, constraint_mode = :penalize)
        @test viol_hi == false
        @test r_hi ≈ -MCTS_PC_REWARD_WEIGHT * pc          # only the Pc term (WAIT, no fuel)

        # Case C: :terminate marks a violating child terminal inside expand_child.
        nterm = root_from_pomdp(pomdp_lo, s0)
        child, _ = expand_child(pomdp_lo, nterm, WAIT, MersenneTwister(0);
                                dt = pomdp_lo.dt, reward_mode = :per_step,
                                constraint_mode = :terminate)
        @test child.violated == (child.pc > pomdp_lo.pc_threshold)
        if child.violated
            @test child.is_terminal == true
        end
    end

    # -----------------------------------------------------------------
    @testset "8b. Terminal-only Pc reward (:terminal, the default)" begin
        # The redesign default (2026-08-09): per-step reward is FUEL COST ONLY (no
        # Pc term, no per-step violation penalty); the whole Pc term + a SOFT over-δ
        # penalty live in leaf_value at TCA.
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 3 * 60 * 60)
        s0 = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 2 * 60 * 60)
        pc = node_pc_at_tca(pomdp, root_from_pomdp(pomdp, s0))

        # Force a violation (threshold below pc). Per-STEP reward is fuel-only
        # regardless of the violation: WAIT ⇒ 0, MANEUVER ⇒ −maneuver_cost.
        plo = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                TCA_max = 3 * 60 * 60, pc_threshold = pc / 2)
        r_w, pc_w, viol_w = step_reward(plo, WAIT, root_from_pomdp(plo, s0);
                                        reward_mode = :terminal, constraint_mode = :penalize)
        r_m, _,    _      = step_reward(plo, MANEUVER, root_from_pomdp(plo, s0);
                                        reward_mode = :terminal, constraint_mode = :penalize)
        @test viol_w == true                 # still FLAGGED (pc cached), just not charged per step
        @test pc_w ≈ pc rtol = 1e-9
        @test r_w ≈ 0.0                       # WAIT: no Pc term, no fuel
        @test r_m ≈ -plo.maneuver_cost        # MANEUVER: fuel only

        # leaf_value carries the terminal Pc term + the SOFT over-δ penalty.
        leaf_v = BeliefNode(root_from_pomdp(plo, s0).belief, s0, false)
        v_viol = leaf_value(plo, leaf_v; reward_mode = :terminal, constraint_mode = :penalize)
        @test v_viol ≈ -MCTS_PC_REWARD_WEIGHT * pc - MCTS_TERMINAL_PENALTY
        # :off suppresses the soft penalty (no-constraint arm).
        leaf_o = BeliefNode(root_from_pomdp(plo, s0).belief, s0, false)
        v_off  = leaf_value(plo, leaf_o; reward_mode = :terminal, constraint_mode = :off)
        @test v_off ≈ -MCTS_PC_REWARD_WEIGHT * pc
        # Feasible leaf (threshold above pc): no over-δ penalty.
        phi = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                TCA_max = 3 * 60 * 60, pc_threshold = pc * 2)
        leaf_f = BeliefNode(root_from_pomdp(phi, s0).belief, s0, false)
        v_feas = leaf_value(phi, leaf_f; reward_mode = :terminal, constraint_mode = :penalize)
        @test v_feas ≈ -MCTS_PC_REWARD_WEIGHT * pc
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

    # -----------------------------------------------------------------
    @testset "11. Asymmetric measurement cadence (predict-only vs. fix steps)" begin
        # The satellite (GPS, ~2 h) and debris (TLE, ~8 h) get corrected on
        # DIFFERENT schedules. Between an object's fixes it is predict-only and
        # its Σ GROWS; on a fix step its Σ SHRINKS. With dt = 1 h, cadence_sc = 2 h,
        # cadence_debris = 8 h and correct_at_root = true (timers start at 0):
        #   step 1 (elapsed 1 h): neither due   → both predict-only (Σ grows)
        #   step 2 (elapsed 2 h): sat due        → sat shrinks, debris still grows
        #   ... debris first due at step 8 (elapsed 8 h).
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 10 * 60 * 60,
                                  cadence_sc = 2 * 60 * 60, cadence_debris = 8 * 60 * 60,
                                  correct_at_root = true)
        s0 = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 10 * 60 * 60)
        root = root_from_pomdp(pomdp, s0)
        @test root.since_sc == 0.0 && root.since_debris == 0.0

        Σsc0 = root.belief.sc.Σ[1, 1]
        Σdb0 = root.belief.debris.Σ[1, 1]

        # --- Step 1: neither cadence reached (elapsed 1 h) ⇒ predict-only. ------
        c1, _ = expand_child(pomdp, root, WAIT, MersenneTwister(0); dt = pomdp.dt)
        @test c1.since_sc == 60.0 * 60 && c1.since_debris == 60.0 * 60   # timers advanced
        @test c1.belief.sc.Σ[1, 1]     > Σsc0    # SC Σ grew (no fix)
        @test c1.belief.debris.Σ[1, 1] > Σdb0    # debris Σ grew (no fix; first step is growth)

        # --- Step 2 from c1: sat cadence reached (elapsed 2 h), debris not. -----
        c2, _ = expand_child(pomdp, c1, WAIT, MersenneTwister(0); dt = pomdp.dt)
        @test c2.since_sc == 0.0                       # sat fix taken → timer reset
        @test c2.since_debris == 2 * 60.0 * 60         # debris still waiting (no fix, timer carried)
        @test c2.belief.sc.Σ[1, 1]     < c1.belief.sc.Σ[1, 1]      # SC Σ SHRANK (fix)
        # NB: we deliberately do NOT assert the debris Σ "grew" on this predict-only
        # step. Only the ALONG-TRACK axis grows monotonically (secular drift); the
        # two minor RADIAL/CROSS-TRACK axes are bounded CW modes that BREATHE once
        # per orbit, so their trace can dip on a given step even with no measurement
        # (verified numerically 2026-07-23). The timer-not-reset check above is the
        # thing that proves the debris got no fix. CAVEAT: the per-step SHRINK itself
        # is not yet independently trusted — it wants the window-Pc / finer-grid
        # verification (experiment_ideas #1) before any logic relies on it; here we
        # only rely on it NOT holding a monotone-growth assertion.

        # --- Walk out to the step where the debris fix is due (step 8, 8 h). ---
        # NB: at σ_debris = 1 km the measurement is WEAK (R = 1e6 ≫ Σ⁻), and 8 h
        # of growth is large, so the TLE fix need NOT pull Σ below its prior value
        # — it only shrinks Σ RELATIVE TO THAT STEP'S PREDICT. That weak-fix regime
        # is exactly the intended physics (debris is the dominant, growing
        # uncertainty). So the correct check compares the fix step against a
        # predict-only counterpart from the SAME parent over the SAME step: the
        # correction is Σ⁺ = (I−K)Σ⁻ ⪯ Σ⁻, so a fix step's Σ < a predict-only step's.
        node = root
        for step in 1:7                       # advance to the node just before the debris fix
            node, _ = expand_child(pomdp, node, WAIT, MersenneTwister(step); dt = pomdp.dt)
        end
        @test node.since_debris == 7 * 60.0 * 60      # debris not yet fixed (due next step)
        # step 8 WITH the debris fix (cadence 8 h reached this step)
        fix, _ = expand_child(pomdp, node, WAIT, MersenneTwister(8); dt = pomdp.dt)
        @test fix.since_debris == 0.0                 # debris fixed at step 8
        # step 8 WITHOUT a debris fix (same parent, debris cadence pushed out of reach)
        pomdp_nodfix = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                         TCA_max = 10 * 60 * 60,
                                         cadence_sc = 2 * 60 * 60,
                                         cadence_debris = 100 * 60 * 60,  # never fires here
                                         correct_at_root = true)
        predonly, _ = expand_child(pomdp_nodfix, node, WAIT, MersenneTwister(8); dt = pomdp.dt)
        @test predonly.since_debris > 0.0             # debris NOT fixed (predict-only)
        @test fix.belief.debris.Σ[1, 1] < predonly.belief.debris.Σ[1, 1]  # the fix shrank Σ this step

        # --- correct_at_root = false: a fix lands on the very first step. -------
        # Same weak-fix caveat, so again compare fix vs. predict-only at step 1.
        pomdp_stale = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                        TCA_max = 10 * 60 * 60,
                                        cadence_sc = 2 * 60 * 60, cadence_debris = 8 * 60 * 60,
                                        correct_at_root = false)
        root_stale = root_from_pomdp(pomdp_stale, s0)
        @test root_stale.since_sc == pomdp_stale.cadence_sc
        @test root_stale.since_debris == pomdp_stale.cadence_debris
        cs, _ = expand_child(pomdp_stale, root_stale, WAIT, MersenneTwister(0); dt = pomdp_stale.dt)
        @test cs.since_sc == 0.0 && cs.since_debris == 0.0          # both fixed on step 1
        # predict-only counterpart: timers start at 0 (correct_at_root = true) and
        # cadences pushed out of reach, so no fix fires on step 1.
        pomdp_nofix = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                        TCA_max = 10 * 60 * 60,
                                        cadence_sc = 100 * 60 * 60, cadence_debris = 100 * 60 * 60,
                                        correct_at_root = true)
        root_nofix = root_from_pomdp(pomdp_nofix, s0)
        cn, _ = expand_child(pomdp_nofix, root_nofix, WAIT, MersenneTwister(0); dt = pomdp_nofix.dt)
        @test cn.since_debris > 0.0                                 # confirm NO debris fix
        @test cs.belief.debris.Σ[1, 1] < cn.belief.debris.Σ[1, 1]   # fix shrank Σ vs. predict-only
    end

    # -----------------------------------------------------------------
    @testset "12. Fast Σ path (efficiency pass) == exact per-node path" begin
        # The efficiency pass precomputes the branch-invariant DEBRIS Σ-at-TCA per
        # depth ONCE (build_sigma_tca_table) and looks it up per node, propagating
        # only the debris MEAN + the full satellite belief per node (sigma_mode =
        # :fast). Because the debris Σ is bitwise branch-invariant and the sat is
        # propagated exactly, the fast path's Pc must EQUAL the exact per-node path
        # (node_pc_at_tca / sigma_mode = :exact) — NO approximation. This is the
        # correctness assertion that lets :fast be the default while :exact stays
        # the oracle (and the only valid mode once Phase 8 adds maneuver noise).
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 8 * 60 * 60, Δv = 5.0)
        s0 = make_conjunction_state(pomdp; miss_m = 200.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 8 * 60 * 60)
        root = root_from_pomdp(pomdp, s0)
        max_d = 7
        table = build_sigma_tca_table(pomdp, root; dt = pomdp.dt, max_depth = max_d)

        # (a) The precomputed per-depth DEBRIS Σ-at-TCA reproduces a fresh per-node
        #     propagation of the SAME (branch-invariant) belief EXACTLY, on BOTH a
        #     WAIT and a MANEUVER spine (debris Σ is maneuver-/z-independent).
        function spine(first_action)
            node = root_from_pomdp(pomdp, s0)
            chain = [(0, node)]
            a = first_action
            for d in 1:max_d
                node, _ = expand_child(pomdp, node, a, MersenneTwister(50 + d);
                                       dt = pomdp.dt, sigma_mode = :exact)
                push!(chain, (d, node))
                a = WAIT
            end
            return chain
        end

        for first_action in (WAIT, MANEUVER)
            for (d, node) in spine(first_action)
                # exact debris Σ-at-TCA for THIS node's belief
                _, Σdb_exact = _grow_belief_to_tca(pomdp, node.belief.debris.μ,
                                                   node.belief.debris.Σ,
                                                   pomdp.debrisParams, node.belief.t)
                # the table entry (built along the WAIT reference) must match it
                @test table.Σ_db[d + 1] ≈ Σdb_exact rtol = 1e-12

                # full Pc: fast == exact to roundoff
                node.pc = NaN
                pc_exact = node_pc(pomdp, node; sigma_mode = :exact)
                node.pc = NaN
                pc_fast = node_pc(pomdp, node; depth = d, table = table, sigma_mode = :fast)
                @test pc_fast ≈ pc_exact rtol = 1e-10
            end
        end

        # (b) A depth beyond the table falls back to the exact path (no crash).
        deep = spine(WAIT)[end][2]
        deep.pc = NaN
        @test node_pc(pomdp, deep; depth = max_d + 5, table = table, sigma_mode = :fast) ≈
              node_pc_at_tca(pomdp, deep) rtol = 1e-10

        # (c) End-to-end: :fast and :exact planners produce the SAME action and a
        #     bitwise-identical tree under the same seed (fast is exact, and the
        #     RNG stream is untouched by the Σ mode).
        nsteps = max_d
        rf = root_from_pomdp(pomdp, s0)
        re = root_from_pomdp(pomdp, s0)
        pf = MCTSPlanner(pomdp; n_iterations = 20, max_depth = nsteps, dt = pomdp.dt,
                         sigma_mode = :fast)
        pe = MCTSPlanner(pomdp; n_iterations = 20, max_depth = nsteps, dt = pomdp.dt,
                         sigma_mode = :exact)
        af, _ = plan(pf, rf, MersenneTwister(7))
        ae, _ = plan(pe, re, MersenneTwister(7))
        @test af == ae
        for a in POMDPs.actions(pomdp)
            @test get(rf.Qa, a, NaN) ≈ get(re.Qa, a, NaN) rtol = 1e-9
        end
    end

    # -----------------------------------------------------------------
    @testset "13. Root-parallel MCTS (multiprocess)" begin
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)
        nsteps = 3
        s0 = make_conjunction_state(pomdp; t = nsteps * pomdp.dt)

        # (a) Pure combinators: budget split, seeds, and the Na-weighted Q merge.
        @test split_iterations(10, 3) == [4, 3, 3]      # remainder front-loaded
        @test split_iterations(9, 3)  == [3, 3, 3]
        @test sum(split_iterations(1000, 7)) == 1000     # budget conserved
        @test split_iterations(40, 1) == [40]            # single chunk = whole budget
        # deterministic + distinct per-chunk seeds from one master seed
        @test worker_seeds(123, 4) == worker_seeds(123, 4)
        @test length(unique(worker_seeds(123, 4))) == 4
        @test worker_seeds(123, 4) != worker_seeds(124, 4)
        # Na-weighted running-average combine: Q(a) = Σ Na·Q / Σ Na, Na summed.
        Na1 = Dict(WAIT => 4, MANEUVER => 2); Qa1 = Dict(WAIT => -1.0, MANEUVER => -3.0)
        Na2 = Dict(WAIT => 6, MANEUVER => 0); Qa2 = Dict(WAIT => -2.0)
        Nt, Qm = merge_action_stats([(Na1, Qa1), (Na2, Qa2)])
        @test Nt[WAIT] == 10 && Nt[MANEUVER] == 2
        @test Qm[WAIT] ≈ (4 * -1.0 + 6 * -2.0) / 10       # = -1.6
        @test Qm[MANEUVER] ≈ -3.0                          # only chunk 1 visited it
        # an action never visited by ANY chunk carries no value ⇒ omitted from the merge
        Nt0, _ = merge_action_stats([(Dict(WAIT => 5), Dict(WAIT => -1.0)),
                                     (Dict(WAIT => 5), Dict(WAIT => -1.0))])
        @test haskey(Nt0, WAIT) && !haskey(Nt0, MANEUVER)

        # (b) Local-fallback parallel plan: `parallel = true` with NO extra workers
        #     attached yet runs the whole budget locally (still through the merge
        #     path). Must be reproducible under a fixed master seed and pick the same
        #     ACTION as a serial plan (the merged decision matches serial in
        #     distribution). Runs before (c) adds any procs, so n_pool == 0 here.
        pser = MCTSPlanner(pomdp; n_iterations = 40, max_depth = nsteps, dt = pomdp.dt,
                           parallel = false)
        rser = root_from_pomdp(pomdp, s0)
        aser, _ = plan(pser, rser, MersenneTwister(123))

        ppar = MCTSPlanner(pomdp; n_iterations = 40, max_depth = nsteps, dt = pomdp.dt,
                           parallel = true)
        rp1 = root_from_pomdp(pomdp, s0); a_p1, _ = plan(ppar, rp1, MersenneTwister(123))
        rp2 = root_from_pomdp(pomdp, s0); a_p2, _ = plan(ppar, rp2, MersenneTwister(123))
        @test a_p1 == a_p2                                 # determinism (fixed master seed)
        @test rp1.Qa == rp2.Qa                             # merged Qa bitwise identical
        @test rp1.Na == rp2.Na
        @test rp1.N == sum(values(rp1.Na))                 # visits conserved through merge
        @test a_p1 == aser                                 # same action as serial (distributional)

        # (c) Real multiprocess: 2 worker processes, each its OWN brahe interpreter.
        #     Skips cleanly if procs can't be added (CI without the venv reachable).
        srcdir = joinpath(@__DIR__, "..")
        added = Int[]
        try
            added = addprocs(2)
            @everywhere added begin
                using LinearAlgebra, Random, PyCall, POMDPs, POMDPTools
                let d = $srcdir
                    include(joinpath(d, "SpacecraftCAPOMDP.jl"))
                    include(joinpath(d, "utils", "genConjunctions.jl"))
                    include(joinpath(d, "utils", "computePc.jl"))
                    include(joinpath(d, "utils", "covarianceTable.jl"))
                    include(joinpath(d, "states.jl"))
                    include(joinpath(d, "actions.jl"))
                    include(joinpath(d, "rewards.jl"))
                    include(joinpath(d, "observations.jl"))
                    include(joinpath(d, "transitions.jl"))
                    include(joinpath(d, "utils", "beliefTracker.jl"))
                    include(joinpath(d, "utils", "beliefMCTS.jl"))
                end
            end

            pmp = MCTSPlanner(pomdp; n_iterations = 40, max_depth = nsteps, dt = pomdp.dt,
                              parallel = true)
            rm1 = root_from_pomdp(pomdp, s0); am1, _ = plan(pmp, rm1, MersenneTwister(123))
            rm2 = root_from_pomdp(pomdp, s0); am2, _ = plan(pmp, rm2, MersenneTwister(123))
            @test am1 == am2                               # reproducible across real workers
            @test rm1.Qa == rm2.Qa                         # merged Qa identical, run to run
            @test rm1.Na == rm2.Na
            @test am1 == aser                              # agrees with serial on the action
        finally
            isempty(added) || rmprocs(added)
        end
    end

    # -----------------------------------------------------------------
    @testset "14. Closed-loop episode driver (Phase 6.5 MPC executor)" begin
        # A short executed episode on the Phase-5/6 end-to-end fixture: a few
        # steps, modest sims, sigma_mode = :fast, serial. Asserts the trace is
        # WELL-FORMED — finite, terminates, Pc in [0,1], Δv only on MANEUVER —
        # and that the shared belief update keeps the executor consistent with the
        # planner's internal step.
        nsteps = 4
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = nsteps * 60 * 60)
        s0 = make_conjunction_state(pomdp; miss_m = 200.0, v_rel = 15.0,
                                    geometry = :cross_track, t = nsteps * pomdp.dt)
        planner = MCTSPlanner(pomdp; n_iterations = 30, max_depth = nsteps,
                              dt = pomdp.dt, sigma_mode = :fast)

        trace = run_episode(planner, pomdp, s0, MersenneTwister(7))

        # (a) terminates: nonempty, no longer than the horizon, and the last step
        #     reaches TCA (t_remaining - dt ≤ 0) or a collision was hit early.
        @test !isempty(trace)
        @test length(trace) <= nsteps
        @test trace[end].t_remaining - pomdp.dt <= 0.0 || length(trace) < nsteps

        # (b) step indices are 1..n and time-remaining decreases by dt each step.
        for (i, s) in enumerate(trace)
            @test s.step == i
            @test s.action == WAIT || s.action == MANEUVER
        end
        for i in 2:length(trace)
            @test trace[i].t_remaining ≈ trace[i-1].t_remaining - pomdp.dt
        end
        @test trace[1].t_remaining ≈ s0.t

        # (c) every logged quantity is finite; Pc is a probability in [0, 1].
        for s in trace
            @test isfinite(s.t_remaining) && isfinite(s.Δv) &&
                  isfinite(s.pc) && isfinite(s.miss)
            @test 0.0 <= s.pc <= 1.0
            @test s.miss >= 0.0
        end

        # (d) Δv is spent ONLY on MANEUVER steps, and is exactly pomdp.Δv there.
        for s in trace
            if s.action == MANEUVER
                @test s.Δv ≈ pomdp.Δv
            else
                @test s.Δv == 0.0
            end
        end

        # (e) episode_summary rolls the trace up consistently.
        summ = episode_summary(trace)
        @test summ.n_steps == length(trace)
        @test summ.total_Δv ≈ sum(s.Δv for s in trace)
        @test summ.n_maneuvers == count(s -> s.action == MANEUVER, trace)
        @test summ.total_Δv ≈ summ.n_maneuvers * pomdp.Δv

        # (f) the dt-consistency guard fires when planner.dt ≠ pomdp.dt (the
        #     executed world steps by pomdp.dt; the planner must match it).
        bad = MCTSPlanner(pomdp; n_iterations = 5, max_depth = nsteps,
                          dt = pomdp.dt / 2, sigma_mode = :fast)
        @test_throws AssertionError run_episode(bad, pomdp, s0, MersenneTwister(7))
    end

    @testset "15. Probabilistic measurement arrival (p_arrival)" begin
        # A SCHEDULED debris (SSN/TLE) fix arrives with prob p_arrival; on a
        # non-arrival the debris is predict-only that step (Σ grows, no innovation)
        # and the cadence timer STILL resets. The satellite GPS fix is NOT gated.
        # p_arrival = 1.0 must be byte-identical to the pre-arrival path (the
        # Bernoulli draw is skipped, so the RNG stream — and every synthetic-suite
        # result — is unchanged). :fast is invalid under p<1 (the branch-invariant
        # debris Σ table no longer holds once arrival is a per-branch random draw).
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                  TCA_max = 10 * 60 * 60,
                                  cadence_sc = 2 * 60 * 60, cadence_debris = 8 * 60 * 60,
                                  correct_at_root = true)
        s0 = make_conjunction_state(pomdp; miss_m = 500.0, v_rel = 15.0,
                                    geometry = :cross_track, t = 10 * 60 * 60)
        root = root_from_pomdp(pomdp, s0)

        # Walk to the node just before the debris fix is due (step 8, 8 h elapsed).
        node = root
        for step in 1:7
            node, _ = expand_child(pomdp, node, WAIT, MersenneTwister(step); dt = pomdp.dt)
        end
        @test node.since_debris == 7 * 60.0 * 60      # due next step

        # (a) p_arrival = 1.0 is BYTE-IDENTICAL to the default (no kwarg): same RNG
        #     seed → same belief AND same reward. Proves the p=1.0 guard skips the
        #     Bernoulli draw (no RNG stream shift). Checked on the debris-due step.
        cdef, rdef = expand_child(pomdp, node, WAIT, MersenneTwister(8); dt = pomdp.dt)
        cp1,  rp1  = expand_child(pomdp, node, WAIT, MersenneTwister(8); dt = pomdp.dt,
                                  p_arrival = 1.0)
        @test cp1.belief.debris.Σ == cdef.belief.debris.Σ     # bitwise-identical debris Σ
        @test cp1.belief.sc.Σ     == cdef.belief.sc.Σ         # bitwise-identical sat Σ
        @test cp1.belief.debris.μ == cdef.belief.debris.μ
        @test rp1 == rdef                                     # identical step reward
        @test cdef.since_debris == 0.0                        # debris fix taken at p=1

        # (b) p_arrival = 0.0: the scheduled debris fix NEVER arrives → the debris is
        #     predict-only (its Σ equals the no-fix counterpart), yet the timer STILL
        #     resets (the fetch was scheduled, just returned nothing).
        cp0, _ = expand_child(pomdp, node, WAIT, MersenneTwister(8); dt = pomdp.dt,
                              p_arrival = 0.0)
        pomdp_nodfix = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = 60 * 60,
                                         TCA_max = 10 * 60 * 60,
                                         cadence_sc = 2 * 60 * 60,
                                         cadence_debris = 100 * 60 * 60,  # debris never due
                                         correct_at_root = true)
        predonly, _ = expand_child(pomdp_nodfix, node, WAIT, MersenneTwister(8); dt = pomdp.dt)
        @test cp0.belief.debris.Σ == predonly.belief.debris.Σ  # no innovation ⇒ = predict-only
        @test cp0.since_debris == 0.0                          # timer reset (scheduled fix)
        @test cdef.belief.debris.Σ[1, 1] < cp0.belief.debris.Σ[1, 1]  # arrived fix shrank Σ

        # (f) :fast is INVALID under probabilistic arrival (stochastic per-branch
        #     debris Σ breaks the branch-invariant Σ table) — the guard must error.
        pln_fast = MCTSPlanner(pomdp; n_iterations = 10, max_depth = 8,
                               dt = pomdp.dt, sigma_mode = :fast, p_arrival = 0.5)
        rfast = root_from_pomdp(pomdp, s0)
        @test_throws ErrorException plan(pln_fast, rfast, MersenneTwister(1))
    end

end
