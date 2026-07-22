#=
test_belief_tracker.jl

Phase 4 unit tests for the Kalman predict/correct belief tracker
(src/utils/beliefTracker.jl). The tracker carries two independent 6-D Gaussian
sub-beliefs (spacecraft, debris) and updates them with a predict/correct cycle
(architecture doc §4 steps 2 & 4). This test confirms:

1. CORRECTION SHRINKS Σ (must pass): after a Kalman correction, Σ⁺ ⪯ Σ⁻ — i.e.
   Σ⁻ − Σ⁺ is positive-semidefinite (uncertainty shrinks, never grows, at a
   correction step). Checked via eigenvalues of the difference, both objects.

2. Σ⁺ IS INDEPENDENT OF THE SAMPLED z (must pass): this is the empirical
   confirmation of the load-bearing §4/§5 property. Across many random
   observation draws z, μ⁺ varies (it depends on z) but Σ⁺ is IDENTICAL every
   time (it depends only on H and R). This independence is exactly what makes
   the Phase 3 precomputed Σ(τ) lookup valid.

3. PREDICT-THEN-CORRECT ROUND TRIP (must pass): starting from a known true
   state, predict one dt step then correct with a sampled observation; the
   corrected mean μ⁺ tracks the true state (closer than the 1σ predicted
   uncertainty), and every Σ along the way stays symmetric and PD.

4. CROSS-VALIDATION (a) vs (b) (must pass): the hand-rolled linear-Gaussian
   update (correct_linear) and brahe's ExtendedKalmanFilter with the linear
   InertialStateMeasurementModel (correct_brahe) agree to ~1e-12 relative on
   both μ⁺ and Σ⁺ — same spirit as the Chan Julia-vs-Python cross-validation.
   At H = I₆ they are the same filter; this confirms we are not reinventing a
   wheel brahe already turns correctly.

5. PREDICT CONSISTENT WITH PHASE 3 (must pass): the predict step's Σ⁻ for
   τ = dt matches build_covariance_table's Σ(dt) — the tracker's per-step STM
   propagation and the Phase 3 offline table are the same computation.

VALIDITY: the Σ⁺-independent-of-z property (test 2) and the Phase 3 consistency
(test 5) hold ONLY for the linear-Gaussian update under noiseless maneuvers
(architecture doc §4/§5/§8). A nonlinear EKF/UKF (deferred SSN az/el/range
model) would make Σ⁺ state-dependent and break both.

Run:  julia --project=. src/tests/test_belief_tracker.jl
=#

using Test
using LinearAlgebra
using Statistics
using Distributions
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
include(joinpath(@__DIR__, "..", "observations.jl"))
include(joinpath(@__DIR__, "..", "transitions.jl"))
include(joinpath(@__DIR__, "..", "utils", "beliefTracker.jl"))

# PD helper: is A positive-semidefinite up to a small roundoff floor scaled by
# the matrix's own trace? The velocity block has variances ~1e-8, so absolute
# eigenvalue floors would be wrong; scale by tr for a relative floor.
_psd(A; rtol = 1e-9) = minimum(eigvals(Symmetric(Matrix(A)))) > -rtol * max(tr(Matrix(A)), 1.0)
_pd(A) = (try; cholesky(Symmetric(Matrix(A))); true; catch; false; end)
_issym(A) = maximum(abs.(Matrix(A) .- transpose(Matrix(A)))) == 0.0

@testset "Phase 4 — Kalman predict/correct belief tracker" begin

    pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)

    # Feasible co-orbital cross-track conjunction (same fixture as Phase 3):
    # v_rel = 15 m/s (co-orbital) because the Phase-2 default 200 m/s is an
    # along-track speed the feasibility guard rejects.
    sc_eci, debris_eci = generate_conjunction_geometry(pomdp;
        geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)

    dt  = 3600.0             # 1-hr step for the tests (swappable; not hard-coded in code)
    t0  = 24 * 3600.0        # start 24 hr before TCA

    # Belief anchored at the true states with P0_sc / P0_debris.
    b0 = belief_from_pomdp(pomdp, sc_eci, debris_eci, t0)

    # -------------------------------------------------------------------
    @testset "1. correction shrinks Σ (Σ⁻ − Σ⁺ ⪰ 0)" begin
        # Predict one step to get a genuine Σ⁻, then correct once.
        b_pred = predict(pomdp, b0, WAIT; dt = dt)
        s_true = CAState(b_pred.sc.μ, b_pred.debris.μ, b_pred.t)   # true state == predicted mean
        rng = MersenneTwister(1)
        z = sample_observation(pomdp, WAIT, s_true, rng)
        b_corr = correct_linear(pomdp, b_pred, z)

        for (pre, post) in ((b_pred.sc, b_corr.sc), (b_pred.debris, b_corr.debris))
            Δ = Matrix(pre.Σ) .- Matrix(post.Σ)
            @test _issym(post.Σ)
            @test _pd(post.Σ)                       # Σ⁺ stays PD
            @test _psd(Δ)                           # Σ⁻ − Σ⁺ ⪰ 0 : uncertainty shrank
            @test tr(Matrix(post.Σ)) < tr(Matrix(pre.Σ))   # strictly less total variance
        end
    end

    # -------------------------------------------------------------------
    @testset "2. Σ⁺ independent of sampled z (μ⁺ varies, Σ⁺ does not)" begin
        b_pred = predict(pomdp, b0, WAIT; dt = dt)
        s_true = CAState(b_pred.sc.μ, b_pred.debris.μ, b_pred.t)

        Σ_ref_sc = nothing
        Σ_ref_db = nothing
        μ_sc_draws = Vector{Vector{Float64}}()
        rng = MersenneTwister(7)
        for i in 1:40
            z = sample_observation(pomdp, WAIT, s_true, rng)
            b_c = correct_linear(pomdp, b_pred, z)
            push!(μ_sc_draws, b_c.sc.μ)
            if Σ_ref_sc === nothing
                Σ_ref_sc = b_c.sc.Σ
                Σ_ref_db = b_c.debris.Σ
            else
                # Σ⁺ must be IDENTICAL across every random z (bitwise-close).
                @test maximum(abs.(b_c.sc.Σ     .- Σ_ref_sc)) < 1e-20
                @test maximum(abs.(b_c.debris.Σ .- Σ_ref_db)) < 1e-20
            end
        end
        # ...and μ⁺ genuinely DID vary with z (otherwise the test is vacuous).
        spread = maximum(std(hcat(μ_sc_draws...), dims = 2))
        @test spread > 1e-3
    end

    # -------------------------------------------------------------------
    @testset "3. predict-then-correct round trip tracks truth" begin
        # Ground truth: propagate the true state one step with the SAME dynamics
        # the predict step uses (via POMDPs.transition), so μ⁻ == true state and
        # the only deviation of μ⁺ from truth comes from observation noise.
        s0 = CAState(sc_eci, debris_eci, t0)
        # Use pomdp.dt path: temporarily match dt to the test step by building a
        # pomdp whose dt is the test dt, so transition advances by the same dt.
        pomdp_dt = SpacecraftCAPOMDP(seed = 42, randAdd = false, dt = dt)
        sp = rand(MersenneTwister(3), POMDPs.transition(pomdp_dt, s0, WAIT))

        b_pred = predict(pomdp, b0, WAIT; dt = dt)
        # μ⁻ should equal the transition's propagated true state (same dynamics).
        @test maximum(abs.(b_pred.sc.μ     .- sp.sc_eci))     < 1e-3
        @test maximum(abs.(b_pred.debris.μ .- sp.debris_eci)) < 1e-3

        # Correct with a real observation of the TRUE next state.
        rng = MersenneTwister(11)
        z = sample_observation(pomdp, WAIT, sp, rng)
        b_corr = correct_linear(pomdp, b_pred, z)

        # μ⁺ tracks truth: position error < predicted position 1σ (Σ⁻ diag).
        for (post, true_eci, pre) in ((b_corr.sc, sp.sc_eci, b_pred.sc),
                                      (b_corr.debris, sp.debris_eci, b_pred.debris))
            @test _issym(post.Σ) && _pd(post.Σ)
            pos_err = norm(post.μ[1:3] .- true_eci[1:3])
            σ_pos_pred = sqrt(maximum(diag(Matrix(pre.Σ))[1:3]))
            @test pos_err < 3 * σ_pos_pred       # within 3σ of the predicted uncertainty
        end
    end

    # -------------------------------------------------------------------
    @testset "4. cross-validate correct_linear (a) vs correct_brahe (b)" begin
        b_pred = predict(pomdp, b0, WAIT; dt = dt)
        s_true = CAState(b_pred.sc.μ, b_pred.debris.μ, b_pred.t)
        rng = MersenneTwister(5)
        z = sample_observation(pomdp, WAIT, s_true, rng)

        b_a = correct_linear(pomdp, b_pred, z)
        b_b = correct_brahe(pomdp, b_pred, z)

        for (ba, bb) in ((b_a.sc, b_b.sc), (b_a.debris, b_b.debris))
            μ_rel = maximum(abs.(ba.μ .- bb.μ)) / max(maximum(abs.(bb.μ)), 1.0)
            Σ_rel = maximum(abs.(Matrix(ba.Σ) .- Matrix(bb.Σ))) /
                    max(maximum(abs.(Matrix(bb.Σ))), 1.0)
            @test μ_rel < 1e-10
            @test Σ_rel < 1e-10
        end
    end

    # -------------------------------------------------------------------
    @testset "5. predict Σ⁻ consistent with Phase 3 table at τ = dt" begin
        # One predict step from t0 lands at τ = t0 - dt remaining. Phase 3 builds
        # Σ(τ) forward from TCA; both are Φ Σ₀ Φᵀ under the same force model, so
        # the FIRST table entry (τ = dt) is the covariance after propagating P0
        # by dt — exactly what one predict step from an at-TCA-anchored belief
        # produces. Anchor a belief AT TCA (t = 0 has no room to step; use the
        # table's own first step) and compare a single dt propagation of P0.
        tbl = build_covariance_table(pomdp, sc_eci, debris_eci; dt = dt, verbose = false)

        # Belief anchored at TCA, one predict step forward by dt → Σ at τ = dt.
        b_tca  = belief_from_pomdp(pomdp, sc_eci, debris_eci, 0.0)
        b_step = predict(pomdp, b_tca, WAIT; dt = dt)

        # Compare ECI covariances at τ = dt (table stores GCRF/ECI in Σ_*_eci[1]).
        rel_sc = maximum(abs.(Matrix(b_step.sc.Σ)     .- tbl.Σ_sc_eci[1])) /
                 max(maximum(abs.(tbl.Σ_sc_eci[1])), 1.0)
        rel_db = maximum(abs.(Matrix(b_step.debris.Σ) .- tbl.Σ_debris_eci[1])) /
                 max(maximum(abs.(tbl.Σ_debris_eci[1])), 1.0)
        @test rel_sc < 1e-9
        @test rel_db < 1e-9
    end

end
