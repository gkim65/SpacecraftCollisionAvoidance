# test_cdm_scenario.jl — the CDM → POMDP scenario loader (pipeline assembly, step 1).
#
# Loads ONE clean PAYLOAD-VS-PAYLOAD NASA CARA conjunction (SWIFT vs JILIN-01
# GAOFEN 2A) through `load_cdm_scenario` (src/utils/cdmScenario.jl) and asserts
# the loaded scenario is well-formed, then confirms the planner runs on it:
#
#   1. WELL-FORMED SCENARIO — finite ECI states, positive-definite belief
#      covariances, the real HBR preserved from the CDM (sum of the two hard-body
#      radii equals COMMENT HBR exactly), the secondary class tagged, and the
#      usage-violation validity flag set. The horizon equals the CDM's lead time
#      (creation → TCA).
#   2. BELIEF ANCHORING — with the covariance fix (backprop=true default) the
#      loaded belief b0 is the DETECTION-epoch seed (≠ CDM covariance); `b_tca`
#      holds the untouched CDM TCA endpoint. backprop=false restores the old
#      near-TCA design (belief == CDM covariance) for input-fidelity checks.
#   3. Pc ANCHOR — elrod on `b_tca` (the real TCA states + covariances + HBR)
#      reproduces CARA's operational Pc; this anchor must stay exact.
#   4. BACK-PROP ROUND-TRIP — the detection seed b0, forward-grown to TCA by the
#      planner's node_pc path, recovers CARA's Pc (meaningful, non-zero) and the
#      seed covariance forward-grows back to the CDM TCA covariance. (This is the
#      step-2b covariance fix; it replaces the old @test_skip planner-run block.)
#
# This is the "behaving sensibly" bar for the loader — NOT a sweep, NOT all 53,
# NOT baselines.
#
# Usage:  julia --project=. src/tests/test_cdm_scenario.jl

using Test
using LinearAlgebra
using Random
using PyCall
using POMDPs
using POMDPTools
using Distributions

include(joinpath(@__DIR__, "..", "SpacecraftCAPOMDP.jl"))
include(joinpath(@__DIR__, "..", "utils", "sensorTiers.jl"))
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
include(joinpath(@__DIR__, "..", "utils", "cdmScenario.jl"))

# The clean payload-vs-payload case: primary SWIFT (NASA payload) vs secondary
# JILIN-01 GAOFEN 2A. Low encounter-plane anisotropy (~63×), non-violation,
# symmetric uncertainty (sec/prim ~1×), tight miss (~193 m), lead ~33 h — the
# well-conditioned regime the current pipeline is most trustworthy in
# (notes/cara_cdm_deepdive_findings.md).
const JILIN_CDM = normpath(joinpath(@__DIR__, "..", "..", "data", "cara_cdms",
    "000028485_conj_000044777_20220407_231108_20220406_140506.cdm"))

_pd(A) = (try; cholesky(Symmetric(Matrix(A))); true; catch; false; end)

@testset "CDM scenario loader (payload-vs-payload: SWIFT vs JILIN)" begin
    @test isfile(JILIN_CDM)
    sc = load_cdm_scenario(JILIN_CDM)

    @testset "1. well-formed scenario" begin
        # finite true ECI states (m, m/s), 6-vectors
        @test length(sc.s_true.sc_eci) == 6
        @test length(sc.s_true.debris_eci) == 6
        @test all(isfinite, sc.s_true.sc_eci)
        @test all(isfinite, sc.s_true.debris_eci)

        # positive-definite belief covariances (the CDM TCA covariances)
        @test _pd(sc.b0.sc.Σ)
        @test _pd(sc.b0.debris.Σ)
        @test all(isfinite, sc.b0.sc.Σ)
        @test all(isfinite, sc.b0.debris.Σ)

        # real HBR preserved from the CDM — the SUM of the two hard-body radii
        # (all any Pc call uses) equals the CDM's COMMENT HBR exactly.
        @test sc.hbr ≈ 8.7 atol = 1e-6
        @test sc.pomdp.R_hard_body_sc + sc.pomdp.R_hard_body_debris ≈ sc.hbr atol = 1e-9

        # secondary class tagged (JILIN-01 GAOFEN 2A is an active payload)
        @test sc.sec_class == :payload

        # validity flag set (this case is a non-violation in the deep-dive)
        @test isa(sc.valid, Bool)
        @test sc.valid == !sc.validity.any_violation
        @test sc.valid == true   # SWIFT/JILIN is 2D-Pc-valid (no usage violation)

        # horizon = CDM lead time (creation → TCA), ~33.1 h for this case
        @test sc.t_horizon > 0
        @test isapprox(sc.t_horizon / 3600, 33.1; atol = 0.1)
        @test sc.b0.t == sc.t_horizon
        @test sc.s_true.t == sc.t_horizon
    end

    @testset "1b. class-tiered anisotropic sensor R (audit C1)" begin
        _possig(R) = sort(sqrt.(eigvals(Symmetric(Matrix(R)[1:3, 1:3]))))

        # The primary is the own asset → GPS-grade isotropic R (10 m on all pos
        # axes), cadence continuous (= dt).
        @test _possig(sc.pomdp.R_sc) ≈ [10.0, 10.0, 10.0] atol = 1e-6
        @test sc.pomdp.cadence_sc == sc.pomdp.dt

        # SWIFT/JILIN secondary is a PAYLOAD → GPS-grade isotropic R, 2 h cadence.
        @test sc.sec_class == :payload
        @test _possig(sc.pomdp.R_debris) ≈ [10.0, 10.0, 10.0] atol = 1e-6
        @test sc.pomdp.cadence_debris == 2 * 60 * 60.0

        # Forcing the secondary to DEBRIS gives the anisotropic SSN-radar R:
        # one tight (range→radial) axis < 100 m, two loose (angular) axes > 100 m,
        # and an 8 h SSN/TLE cadence — the class routing actually switches R.
        sc_deb = load_cdm_scenario(JILIN_CDM; sec_class_override = :debris)
        σd = _possig(sc_deb.pomdp.R_debris)
        @test σd[1] < 100.0            # radial (range) tight
        @test σd[3] > 100.0            # cross-range (angular) loose
        @test σd[3] / σd[1] > 5.0      # clearly anisotropic
        @test sc_deb.pomdp.cadence_debris == 8 * 60 * 60.0
        # primary R is unchanged by the secondary's class
        @test _possig(sc_deb.pomdp.R_sc) ≈ [10.0, 10.0, 10.0] atol = 1e-6

        # The measurement-QUALITY knob sweeps the SSN grade: worse grade → looser R.
        sc_best  = load_cdm_scenario(JILIN_CDM; sec_class_override = :debris, sensor_quality = :best)
        sc_worst = load_cdm_scenario(JILIN_CDM; sec_class_override = :debris, sensor_quality = :worst)
        @test maximum(_possig(sc_best.pomdp.R_debris)) < maximum(_possig(sc_worst.pomdp.R_debris))
    end

    @testset "2. belief anchoring (backprop default vs near-TCA)" begin
        # DEFAULT (backprop=true, covariance-fix step 2b): the loaded belief is the
        # DETECTION-epoch seed that forward-grows to the CDM TCA belief, NOT the CDM
        # covariance directly. `b_tca` holds the untouched real endpoint (the CDM
        # TCA covariance) — elrod on it reproduces CARA (block 3).
        @test sc.b_tca.sc.Σ == sc.pomdp.P0_sc            # b_tca IS the CDM TCA cov
        @test sc.b_tca.debris.Σ == sc.pomdp.P0_debris
        @test sc.b_tca.t == 0.5                          # anchored at TCA
        # the detection seed differs from the TCA covariance (it back-propagated)
        @test sc.b0.t == sc.t_horizon
        @test sc.b0.sc.Σ != sc.pomdp.P0_sc               # seed ≠ endpoint
        @test all(isfinite, sc.b0.sc.Σ) && _pd(sc.b0.sc.Σ)
        @test all(isfinite, sc.b0.debris.Σ) && _pd(sc.b0.debris.Σ)
        # truth means back-propagated to detection (== the seed belief means)
        @test sc.b0.sc.μ == sc.s_true.sc_eci
        @test sc.b0.debris.μ == sc.s_true.debris_eci

        # OLD near-TCA design still available via backprop=false: belief == CDM cov.
        sc_nb = load_cdm_scenario(JILIN_CDM; backprop = false)
        @test sc_nb.b0.sc.Σ == sc_nb.pomdp.P0_sc
        @test sc_nb.b0.debris.Σ == sc_nb.pomdp.P0_debris
        @test sc_nb.b0.sc.μ == sc_nb.s_true.sc_eci
    end

    @testset "3. Pc anchor: elrod at TCA == CARA (untouched real endpoint)" begin
        # The scenario's real endpoint (`b_tca`, the CDM TCA states + covariances +
        # HBR) must reproduce CARA's operational Pc — this is the input-fidelity
        # anchor the back-prop is built to recover, and must stay exact regardless
        # of the covariance fix.
        pc_tca = node_pc(sc.pomdp, BeliefNode(sc.b_tca, sc.s_true, false); sigma_mode = :exact)
        @test isfinite(pc_tca) && 0.0 <= pc_tca <= 1.0
        @test sc.pc_cdm > 0
        @test abs(log10(pc_tca) - log10(sc.pc_cdm)) < 1.0   # elrod ~0.02% on this case
    end

    @testset "4. back-prop round-trip: seed forward-grows to CDM Pc" begin
        # THE covariance-fix deliverable (step 2b): the DETECTION-epoch seed b0,
        # forward-grown to TCA by the planner's own node_pc path, must recover
        # CARA's operational Pc — i.e. growing the seed forward lands on the real
        # conjunction geometry (the old direct-anchor root grew it 33 h PAST TCA →
        # Pc→0, block-4 was @test_skip). At the Q=0 default the recovery is ~3%
        # (0.014 dex); the ~105 m mean integrator floor sets the residual.
        pc_fwd = node_pc(sc.pomdp, BeliefNode(sc.b0, sc.s_true, false); sigma_mode = :exact)
        @test isfinite(pc_fwd) && 0.0 <= pc_fwd <= 1.0
        @test pc_fwd > 0                                       # meaningful, NOT ~0
        @test abs(log10(pc_fwd) - log10(sc.pc_cdm)) < 0.3      # recovers CARA (< 2×)

        # and the seed covariance forward-grows back to the CDM TCA covariance
        μsc, Σsc = _grow_belief_to_tca(sc.pomdp, sc.b0.sc.μ, sc.b0.sc.Σ,
                                       sc.pomdp.satParams, sc.b0.t; q_rtn = sc.pomdp.q_rtn_sc)
        @test norm(Σsc - sc.pomdp.P0_sc) / norm(sc.pomdp.P0_sc) < 0.05   # <5% cov recover
    end
end
