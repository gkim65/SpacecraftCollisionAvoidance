#=
test_from_orbits.jl

Round-trip verification for the orbit-first closest-approach machinery ported
from RSSDA (propagate_batch_to / _closest_approach / _reduce_rel_to_params /
make_conjunction_from_orbits in src/utils/genConjunctions.jl).

The geometry-first generator (generate_conjunction_geometry) PLACES a debris
object at a requested miss/geometry in the Brahe RTN frame. That placement is
geometrically clean but not verified against dynamics: nothing there confirms
the requested miss is actually the closest approach, or that it happens at TCA.
This test closes that gap by round-tripping through the dynamics:

    generate_conjunction_geometry (head-on / cross-track, several miss levels)
      -> convert both ECI states to KOE (Brahe state_eci_to_koe)
      -> make_conjunction_from_orbits (accurate-dynamics closest-approach search)
      -> assert the Brahe-measured true miss ~= the requested miss, and that the
         closest approach lands at TCA (|t_ca| small).

Note on geometry: generate_conjunction_geometry places the debris with a purely
ALONG-TRACK closing velocity. For CROSS-TRACK geometry the miss (a radial
standoff) is perpendicular to that velocity, so the placement instant already IS
the closest approach and t_ca ~= 0 tightly. For HEAD-ON geometry the miss is
along-track — parallel to the relative velocity — so the objects are still
closing at the placement instant and the true closest approach is a pure
fly-through: the true miss collapses toward the (near-zero) radial/cross-track
separation, NOT the requested along-track offset. That is the correct dynamical
answer, and the test asserts each regime on its own terms rather than forcing a
single tolerance onto both.

A final group confirms the feasibility guard flags an obviously non-LEO /
hyperbolic secondary orbit.

A note on the closing speed: these tests use v_rel = 15 m/s (RSSDA's co-orbital
default), NOT the 200 m/s "sanity-check" value from the Phase 2 Pc sweep. That
200 m/s value is an ALONG-TRACK velocity, and a 200 m/s along-track kick on a
LEO orbit drops the debris perigee ~200 km (below the Earth's surface) — the
feasibility guard correctly rejects it as "into atmosphere". This is itself a
finding of the round-trip verification: the Phase 2 test-default v_rel does not
produce a dynamically-resident LEO conjunction. A true fast LEO crossing is a
CROSS-TRACK (plane-difference) velocity, not an along-track one (see the RSSDA
crossing-velocity derivation) — representing those is future work.

Run:  julia --project=. src/tests/test_from_orbits.jl
=#

using Test
using LinearAlgebra
using Distributions
using Random
using PyCall
using POMDPs        # SpacecraftCAPOMDP.jl subtypes POMDP{...}
using POMDPTools

include(joinpath(@__DIR__, "..", "SpacecraftCAPOMDP.jl"))
include(joinpath(@__DIR__, "..", "utils", "genConjunctions.jl"))

# ---------------------------------------------------------------------------
# Brahe availability — the search needs state_koe_to_eci / par_propagate_to.
# ---------------------------------------------------------------------------

const BRAHE_OK = try
    get_brahe()
    true
catch err
    @warn "Brahe not importable through PyCall; skipping from-orbits tests." exception = err
    false
end

# randAdd off so SC1's elements (and hence the conjunction) are deterministic.
make_pomdp() = SpacecraftCAPOMDP(seed = 42, randAdd = false)

# ECI (m, m/s) -> KOE [a, e, i, Ω, ω, M] (a m, angles deg) via Brahe.
function eci_to_koe(eci)
    bh = get_brahe()
    np = pyimport("numpy")
    return collect(bh.state_eci_to_koe(np.array(collect(eci)), bh.AngleFormat.DEGREES))
end

@testset "Orbit-first closest-approach verification" begin

    if !BRAHE_OK
        @test_skip "Brahe unavailable — from-orbits machinery not exercised."
    else
        p = make_pomdp()

        # ------------------------------------------------------------------
        # 1. Cross-track round trip: placement instant IS closest approach,
        #    so the true miss must recover the requested radial standoff and
        #    the closest approach must land at TCA.
        # ------------------------------------------------------------------
        @testset "cross-track round trip (true miss == requested, CA at TCA)" begin
            for miss_m in (200.0, 1_000.0, 5_000.0)
                sc1, deb = generate_conjunction_geometry(p; geometry = :cross_track,
                                                         miss_m = miss_m, v_rel = 15.0)
                sc1_koe = eci_to_koe(sc1)
                deb_koe = eci_to_koe(deb)

                conj = make_conjunction_from_orbits(p, sc1_koe, deb_koe; span_s = 600.0)

                @test conj.feasible
                @test isempty(conj.reason)
                # true miss recovers the requested standoff to sub-km / a few %
                @test isapprox(conj.true_miss_m, miss_m;
                               atol = 50.0, rtol = 0.02)
                # closest approach pinned to TCA
                @test conj.at_tca
                @test abs(conj.t_ca_s) < 1.0
                # geometry reads as cross-track (perp dominates, ~90°)
                @test conj.perp_m > 0.9 * miss_m
                @test conj.angle_deg > 80.0

                @info "cross-track" miss_req = miss_m true_miss = conj.true_miss_m t_ca = conj.t_ca_s perp = conj.perp_m dt0 = conj.dt0_m angle = conj.angle_deg
            end
        end

        # ------------------------------------------------------------------
        # 2. Head-on round trip: miss is along-track (parallel to the closing
        #    velocity), so the placement instant is NOT closest approach — the
        #    objects fly through and the true miss collapses to the (near-zero)
        #    perpendicular separation. Assert the dynamics say exactly that:
        #    a fly-through much closer than the requested along-track offset,
        #    with the true CA offset in time by ~= dt0 / v_rel.
        #
        #    At v_rel = 15 m/s a 1 km along-track offset clears in ~67 s, well
        #    inside the ±600 s window, and the constant-closing-speed estimate
        #    dt0/v_rel holds closely. (Larger along-track offsets take so long to
        #    close that orbital curvature bends the fly-through and the linear
        #    estimate degrades — that regime is left out here.)
        # ------------------------------------------------------------------
        @testset "head-on round trip (along-track offset is a fly-through)" begin
            v_rel = 15.0
            for miss_m in (500.0, 1_000.0)
                sc1, deb = generate_conjunction_geometry(p; geometry = :head_on,
                                                         miss_m = miss_m, v_rel = v_rel)
                sc1_koe = eci_to_koe(sc1)
                deb_koe = eci_to_koe(deb)

                conj = make_conjunction_from_orbits(p, sc1_koe, deb_koe; span_s = 600.0)

                @test conj.feasible
                # true closest approach is far smaller than the requested
                # along-track offset (it's a fly-through)
                @test conj.true_miss_m < 0.2 * miss_m
                # and it happens away from the nominal TCA, by ~= dt0 / v_rel
                @test !conj.at_tca
                @test isapprox(abs(conj.t_ca_s), miss_m / v_rel; rtol = 0.15)

                @info "head-on" miss_req = miss_m true_miss = conj.true_miss_m t_ca = conj.t_ca_s expected_tca = miss_m / v_rel
            end
        end

        # ------------------------------------------------------------------
        # 3. Reduction arithmetic: _reduce_rel_to_params is trivial but exact.
        # ------------------------------------------------------------------
        @testset "_reduce_rel_to_params arithmetic" begin
            # [R, T, N, Ṙ, Ṫ, Ṅ]
            perp, dt0, v_rel = _reduce_rel_to_params([3.0, 40.0, 4.0, 1.0, -7.0, 2.0])
            @test isapprox(perp, hypot(3.0, 4.0))   # = 5
            @test dt0 == 40.0
            @test v_rel == 7.0                        # = -Ṫ
        end

        # ------------------------------------------------------------------
        # 4. Feasibility guard: an obviously non-LEO / hyperbolic secondary
        #    must be flagged infeasible with a reason.
        # ------------------------------------------------------------------
        @testset "feasibility guard flags bad secondary orbits" begin
            bh = get_brahe()
            sc1, _ = generate_conjunction_geometry(p; geometry = :cross_track,
                                                   miss_m = 500.0, v_rel = 200.0)
            sc1_koe = eci_to_koe(sc1)

            # hyperbolic secondary (e > 1) at the same semi-major-axis-ish anchor
            hyper_koe = copy(sc1_koe)
            hyper_koe[2] = 1.5
            c_hyper = make_conjunction_from_orbits(p, sc1_koe, hyper_koe; span_s = 600.0)
            @test !c_hyper.feasible
            @test occursin("hyperbolic", c_hyper.reason)

            # way-out-of-LEO secondary (huge semi-major axis → apogee ceiling)
            geo_koe = copy(sc1_koe)
            geo_koe[1] = bh.R_EARTH + 35_786e3   # GEO altitude
            geo_koe[2] = 0.001
            c_geo = make_conjunction_from_orbits(p, sc1_koe, geo_koe; span_s = 600.0)
            @test !c_geo.feasible
            @test occursin("apogee", c_geo.reason)

            # eccentric-but-bound secondary above ECC_MAX
            ecc_koe = copy(sc1_koe)
            ecc_koe[2] = 0.4
            c_ecc = make_conjunction_from_orbits(p, sc1_koe, ecc_koe; span_s = 600.0)
            @test !c_ecc.feasible
            @test occursin("e=", c_ecc.reason)

            @info "feasibility" hyper = c_hyper.reason geo = c_geo.reason ecc = c_ecc.reason
        end
    end
end
