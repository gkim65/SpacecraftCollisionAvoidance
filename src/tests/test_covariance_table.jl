#=
test_covariance_table.jl

Phase 3 sanity check for the precomputed covariance-vs-time-remaining table
Σ(τ) (build_covariance_table / pc_through_table in
src/utils/covarianceTable.jl).

The table propagates the initial covariances P0_sc / P0_debris forward under the
accurate numerical force model (drag/SRP) via Brahe's STM machinery and reads
Σ(τ) back at τ = dt, 2·dt, …, 24 hr remaining until TCA. This test confirms:

1. STRUCTURE + HEALTH (must pass): the table has the right shape for both the
   1-hr grid (24 steps) and the 30-min grid (48 steps); every Σ(τ) is symmetric
   and positive-definite; the sweep reports all_pd and (given clean brahe output)
   no forced symmetrization.

2. GROWTH (must pass): along-track (RTN transverse) uncertainty grows
   monotonically with τ and does not blow up or stay flat. NOTE on the growth
   law: the architecture/Phase-3 spec anticipated ~cubic-in-variance (σ ∝ τ^1.5)
   growth, which is the signature of semi-major-axis (energy) error dominating.
   The current placeholder P0 (independent per-axis position + velocity diagonal,
   no dedicated SMA/energy term) is instead VELOCITY-error dominated, so along-
   track σ grows ~linearly (variance ∝ τ²). That is the correct growth FOR THIS
   P0 — not flat, not exploding — so we assert a power-law exponent in a band
   that admits both the velocity-dominated (≈1) and SMA-dominated (≈1.5) regimes,
   and separately assert monotonic, bounded growth. (When P0 later carries a real
   SMA-uncertainty term — Phase 4 SSN-noise decision — the exponent should move
   toward 1.5; this test will still pass and the band documents the expectation.)

3. Pc-TRUST (must pass, folded in from Phase 2): Chan Pc evaluated through the
   propagated Σ(τ) for a fixed FEASIBLE conjunction (co-orbital cross-track,
   v_rel = 15 m/s — the Phase-2 default 200 m/s is an along-track speed the
   feasibility guard rejects) is finite, in [0,1], and evolves smoothly with τ
   (no NaN/Inf, no wild jumps). This is where the Phase-2 "not-yet-trusted" Pc
   values are confirmed to behave sensibly as uncertainty grows toward TCA.

4. GRID EQUIVALENCE (must pass): where the 30-min and 1-hr grids share a τ
   (whole-hour τ), they produce the same Σ(τ) — the table is a pure function of
   time-remaining, independent of the step size used to build it (the property
   that makes the lookup valid, architecture doc §5).

VALIDITY: the whole table is valid only under the noiseless-maneuver assumption
(architecture doc §5/§8). Phase 8 revisits it once maneuver execution
uncertainty is added.

Run:  julia --project=. src/tests/test_covariance_table.jl
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

@testset "Phase 3 — Σ(τ) covariance table" begin

    pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)

    # Feasible co-orbital cross-track conjunction. v_rel = 15 m/s (co-orbital):
    # the Phase-2 default 200 m/s is an along-track speed that sinks the debris
    # perigee ~200 km underground and the feasibility guard rejects it
    # (from_orbits finding). Cross-track standoff keeps the miss ⊥ the closing
    # velocity so the requested miss really is the closest approach.
    sc_eci, debris_eci = generate_conjunction_geometry(pomdp;
        geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)

    tbl_1hr  = build_covariance_table(pomdp, sc_eci, debris_eci; dt = 3600.0, verbose = false)
    tbl_30m  = build_covariance_table(pomdp, sc_eci, debris_eci; dt = 1800.0, verbose = false)

    # -------------------------------------------------------------------
    @testset "1. structure + health" begin
        @test tbl_1hr.n_steps == 24
        @test tbl_30m.n_steps == 48
        @test length(tbl_1hr.τ_s) == 24
        @test length(tbl_30m.Σ_debris_rtn) == 48

        # τ grid ascending, evenly spaced, ends at 24 hr.
        @test issorted(tbl_1hr.τ_s)
        @test tbl_1hr.τ_s[1] ≈ 3600.0
        @test tbl_1hr.τ_s[end] ≈ 24 * 3600.0

        # Every Σ(τ), both objects, both frames: 6×6, symmetric, PD.
        for tbl in (tbl_1hr, tbl_30m)
            @test tbl.all_pd
            for k in 1:tbl.n_steps
                for Σ in (tbl.Σ_sc_rtn[k], tbl.Σ_debris_rtn[k],
                          tbl.Σ_sc_eci[k], tbl.Σ_debris_eci[k])
                    @test size(Σ) == (6, 6)
                    @test maximum(abs.(Σ .- transpose(Σ))) == 0.0   # snapped exactly symmetric
                    # Positive-definite up to floating-point roundoff. The
                    # velocity block has variances ~1e-8, so eigenvalues can dip
                    # to a tiny negative (~-1e-7) purely from roundoff while the
                    # matrix is still PD by Cholesky (which is what all_pd uses).
                    # Assert eigenvalues exceed a small floor scaled to the trace.
                    F = eigen(Symmetric(Σ))
                    @test minimum(F.values) > -1e-9 * tr(Σ)
                end
            end
        end
        @info "  structure/health OK: all Σ(τ) 6×6, exactly symmetric, PD; " *
              "any_symmetrized(1hr)=$(tbl_1hr.any_symmetrized)"
    end

    # -------------------------------------------------------------------
    @testset "2. along-track growth is monotone, bounded, sensible" begin
        aτ  = tbl_1hr.τ_s ./ 3600
        σal = [sqrt(tbl_1hr.Σ_debris_rtn[k][2, 2]) for k in 1:tbl_1hr.n_steps]  # RTN axis 2 = along-track

        # Monotone increasing (uncertainty grows with time-to-go).
        @test issorted(σal)
        # Not flat: 24 hr σ is many× the 1 hr σ.
        @test σal[end] > 5 * σal[1]
        # Not exploding: stays physically bounded (< Earth radius scale).
        @test σal[end] < 1e6   # < 1000 km along-track 1σ over 24 hr — sane for LEO OD

        # Power-law exponent of σ vs τ. Velocity-dominated P0 → ~1; an
        # SMA/energy-dominated P0 would push toward ~1.5. Band admits both and
        # rules out flat (0) or runaway (>2).
        lx = log.(aτ); ly = log.(σal)
        slope = (mean(lx .* ly) - mean(lx) * mean(ly)) / (mean(lx .^ 2) - mean(lx)^2)
        @test 0.7 < slope < 1.7
        @info "  along-track σ: $(round(σal[1],digits=1)) m (1h) → " *
              "$(round(σal[end],digits=1)) m (24h); power-law exponent ≈ $(round(slope,digits=2)) " *
              "(≈1 = velocity-dominated P0, as expected for the current placeholder covariance)"
    end

    # -------------------------------------------------------------------
    @testset "3. Pc through the table is finite, in [0,1], and smooth" begin
        pcs = pc_through_table(pomdp, sc_eci, debris_eci, tbl_1hr)
        @test length(pcs) == tbl_1hr.n_steps
        @test all(isfinite, pcs)
        @test all(0.0 .<= pcs .<= 1.0)
        @test all(pcs .>= 0.0)

        # Smoothness: no wild step-to-step jumps (ratio between consecutive Pc
        # stays within an order of magnitude), i.e. Pc evolves believably as Σ
        # grows toward TCA rather than jumping around.
        for k in 2:length(pcs)
            if pcs[k-1] > 1e-12 && pcs[k] > 1e-12
                @test 0.1 < pcs[k] / pcs[k-1] < 10.0
            end
        end
        aτ = tbl_1hr.τ_s ./ 3600
        @info "  Pc(τ): 24h=$(pcs[findmin(abs.(aτ .- 24))[2]]), " *
              "12h=$(pcs[findmin(abs.(aτ .- 12))[2]]), " *
              "1h=$(pcs[findmin(abs.(aτ .- 1))[2]]) — finite, in [0,1], smooth"
    end

    # -------------------------------------------------------------------
    @testset "4. grid equivalence: Σ(τ) independent of step size" begin
        # At every whole-hour τ, the 30-min grid (even index) must equal the
        # 1-hr grid — Σ is a pure function of time-remaining (architecture §5).
        for k in 1:tbl_1hr.n_steps
            τ = tbl_1hr.τ_s[k]
            j = findfirst(≈(τ), tbl_30m.τ_s)
            @test j !== nothing
            for (A, B) in ((tbl_1hr.Σ_debris_rtn[k], tbl_30m.Σ_debris_rtn[j]),
                           (tbl_1hr.Σ_sc_eci[k],      tbl_30m.Σ_sc_eci[j]))
                @test maximum(abs.(A .- B)) ≤ 1e-6 * maximum(abs.(B))
            end
        end
        @info "  grid equivalence OK: 30-min and 1-hr tables agree at shared τ " *
              "(Σ depends only on time-remaining)"
    end

end
