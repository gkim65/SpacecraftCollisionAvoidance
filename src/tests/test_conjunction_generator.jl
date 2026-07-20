#=
test_conjunction_generator.jl

Phase 2 sanity check for the deterministic conjunction generator ported from
RSSDA (place_debris_at_tca / generate_conjunction_geometry in
src/utils/genConjunctions.jl).

The generator places a debris object at TCA relative to the spacecraft at a
specified miss distance and geometry (head-on / cross-track), in the Brahe RTN
frame. This test confirms that the geometry it produces feeds sensibly into the
Phase 1 Chan (1997) Pc: Pc is highest at the smallest miss distance and decays
to near-zero at a large miss distance, monotonically in between. (The absolute
peak Pc is modest because the combined hard-body radius ~20 m is small next to
the tens-of-meters combined 1σ — the trend, not the peak value, is the check.)

Because we only sanity-check the geometry→Pc coupling (not covariance growth,
which is Phase 3), covariances are taken as the POMDP's initial (μ, Σ) values
evaluated directly at TCA — no STM propagation. Chan takes covariances as direct
inputs, so this is self-contained and needs no Brahe covariance propagation.

Three groups of checks:

1. GEOMETRY CONSTRUCTION (must pass): the generated relative state at TCA has
   the expected RTN structure — head-on puts the miss on the along-track axis,
   cross-track on the radial (sideways) axis — and the miss magnitude matches
   what was requested, to sub-meter tolerance, after the RTN→ECI round trip.

2. Pc vs MISS DISTANCE (must pass, the headline sanity check): for each
   geometry, Chan Pc is high at a very small miss, near-zero at a large miss,
   and monotonically decreasing across a sweep in between.

3. DETERMINISM (must pass): regenerating the same (geometry, miss, v_rel)
   returns byte-identical states — the generator carries no hidden RNG.

Run:  julia --project=. src/tests/test_conjunction_generator.jl
=#

using Test
using LinearAlgebra
using Distributions
using Random
using PyCall
using POMDPs        # SpacecraftCAPOMDP.jl subtypes POMDP{...}
using POMDPTools

include(joinpath(@__DIR__, "..", "SpacecraftCAPOMDP.jl"))       # SpacecraftCAPOMDP type
include(joinpath(@__DIR__, "..", "utils", "computePc.jl"))      # chan_pc
include(joinpath(@__DIR__, "..", "utils", "genConjunctions.jl")) # generator

# ---------------------------------------------------------------------------
# Brahe availability — the generator needs state_koe_to_eci / state_rtn_to_eci.
# ---------------------------------------------------------------------------

const BRAHE_OK = try
    get_brahe()
    true
catch err
    @warn "Brahe not importable through PyCall; skipping conjunction-generator tests." exception = err
    false
end

# A fixed POMDP with randAdd off so orbital elements (and hence SC1 at TCA) are
# deterministic and the geometry checks are exactly reproducible.
make_pomdp() = SpacecraftCAPOMDP(seed = 42, randAdd = false)

# Combined hard-body radius used throughout (m).
hbr(p) = p.R_hard_body_sc + p.R_hard_body_debris

"""
Chan Pc for a conjunction generated at `miss_m` / `geometry`, using the POMDP's
initial covariances directly at TCA (no propagation — this is a geometry sanity
check, not a covariance-growth check).
"""
function pc_for_miss(p, geometry, miss_m; v_rel = 200.0)
    sc1, deb = generate_conjunction_geometry(p; geometry = geometry,
                                              miss_m = miss_m, v_rel = v_rel)
    return chan_pc(sc1, deb, p.P0_sc, p.P0_debris, hbr(p))
end

@testset "Phase 2 — conjunction generator" begin

    if !BRAHE_OK
        @test_skip "Brahe unavailable — conjunction generator not exercised."
    else
        p = make_pomdp()

        # ------------------------------------------------------------------
        # 1. Geometry construction
        # ------------------------------------------------------------------
        @testset "geometry construction (RTN structure + miss magnitude)" begin
            bh = get_brahe()
            for geometry in (:head_on, :cross_track), miss_m in (200.0, 1000.0)
                sc1, deb = generate_conjunction_geometry(p; geometry = geometry,
                                                         miss_m = miss_m, v_rel = 200.0)

                @test length(sc1) == 6
                @test length(deb) == 6
                @test all(isfinite, sc1)
                @test all(isfinite, deb)

                # Recover the relative RTN state Brahe would report and confirm
                # the miss landed on the expected axis with the right magnitude.
                rtn = collect(bh.state_eci_to_rtn(collect(sc1), collect(deb)))
                r_off, t_off, n_off = rtn[1], rtn[2], rtn[3]
                miss_recovered = norm(rtn[1:3])

                @test isapprox(miss_recovered, miss_m; atol = 1.0)   # sub-meter round trip
                @test abs(n_off) < 1.0                                # nothing on cross-track (N)

                if geometry == :head_on
                    # miss entirely along-track (T), essentially none radial (R)
                    @test abs(t_off) > 0.99 * miss_m
                    @test abs(r_off) < 1.0
                else  # :cross_track
                    # miss entirely radial standoff (R), none along-track (T)
                    @test abs(r_off) > 0.99 * miss_m
                    @test abs(t_off) < 1.0
                end
            end
        end

        # ------------------------------------------------------------------
        # 2. Pc vs miss distance — the headline sanity check
        # ------------------------------------------------------------------
        @testset "Pc vs miss distance (small→high, large→~0, monotone)" begin
            # Miss distances (m) from well inside the hard-body radius out to
            # many km. Combined HBR ≈ 20 m and the combined position 1σ is tens
            # of meters, so the *absolute* peak Pc is modest (a 20 m disk holds
            # only a fraction of the Gaussian mass), but the trend must be
            # unambiguous: highest at the smallest miss, negligible far out.
            misses = [10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 5_000.0, 50_000.0]

            for geometry in (:head_on, :cross_track)
                pcs = [pc_for_miss(p, geometry, m) for m in misses]

                # All valid probabilities.
                @test all(0.0 .<= pcs .<= 1.0)

                # Highest risk at the smallest miss; negligible at the largest.
                @test pcs[1] == maximum(pcs)   # smallest miss is the peak
                @test pcs[1] > 1e-3            # small miss: clearly non-negligible
                @test pcs[end] < 1e-6         # 50 km miss: negligible

                # Monotonically non-increasing across the sweep (tiny numerical slack).
                for k in 1:(length(pcs) - 1)
                    @test pcs[k] >= pcs[k + 1] - 1e-12
                end

                @info "Pc sweep ($geometry)" miss_m = misses pc = pcs
            end

            # Cross-track miss (pure radial standoff) projects fully into the 2D
            # encounter plane, so it drops off with miss distance faster than a
            # head-on (along-track) miss, which is nearly parallel to the
            # relative velocity and barely projects in. Confirm that ordering at
            # a moderate miss.
            pc_head  = pc_for_miss(p, :head_on, 500.0)
            pc_cross = pc_for_miss(p, :cross_track, 500.0)
            @test pc_cross < pc_head
        end

        # ------------------------------------------------------------------
        # 3. Determinism
        # ------------------------------------------------------------------
        @testset "determinism (no hidden RNG)" begin
            a1, b1 = generate_conjunction_geometry(p; geometry = :head_on,
                                                   miss_m = 300.0, v_rel = 150.0)
            a2, b2 = generate_conjunction_geometry(p; geometry = :head_on,
                                                   miss_m = 300.0, v_rel = 150.0)
            @test a1 == a2
            @test b1 == b2
        end
    end
end
