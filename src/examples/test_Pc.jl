using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("../SpacecraftCollisionAvoidance.jl")

using Test
using LinearAlgebra

function test_compute_pc()
    println("=== Testing compute_pc ===\n")
    
    # Create a basic POMDP instance
    pomdp = SpacecraftCAPOMDP(
        seed = 42,
        randAdd = false,  # keep deterministic for testing
        conjunctionType = "crossing",
        rMag = 100.0,     # 100m miss distance at TCA
        vMag = 100.0,
        TCA_max = 10*60*60
    )
    println("TCA_max: $(pomdp.TCA_max / 60) minutes")
    println("P0_debris[1,1]: $(pomdp.P0_debris[1,1])")
    println("P0_sc[1,1]: $(pomdp.P0_sc[1,1])")
    # Generate conjunction to get real ECI states and epochs
    eci_sc, eci_debris, prop_sc, prop_debris, epoch_tca, _, _, epoch_start = generate_conjunction(pomdp)

    sc_eci_t0     = collect(prop_sc.current_state()[1:6])
    debris_eci_t0 = collect(prop_debris.current_state()[1:6])

    println("Test 1: Foster vs MC consistency")
    pc_foster = compute_pc(pomdp, sc_eci_t0, debris_eci_t0, epoch_start, epoch_tca, method=:foster)
    pc_mc     = compute_pc(pomdp, sc_eci_t0, debris_eci_t0, epoch_start, epoch_tca, method=:mc, N=100000000)
    println("  Foster Pc: $(round(pc_foster, sigdigits=4))")
    println("  MC Pc:     $(round(pc_mc,     sigdigits=4))")
    println("  Ratio:     $(round(pc_foster/max(pc_mc, 1e-10), sigdigits=3))")
    @test abs(pc_foster - pc_mc) < 0.01  # within 1%




    println("\nTest 2: Close conjunction should have high Pc")
    pomdp_close = SpacecraftCAPOMDP(seed=42, randAdd=false, rMag=1.0, vMag=100.0, TCA_max = 30*60)
    eci_sc_close, eci_debris_close, prop_sc_close, prop_debris_close, epoch_tca_close, _, _, epoch_start_close = generate_conjunction(pomdp_close)
    sc_close     = collect(prop_sc_close.current_state()[1:6])
    debris_close = collect(prop_debris_close.current_state()[1:6])
    pc_close = compute_pc(pomdp_close, sc_close, debris_close, epoch_start_close, epoch_tca_close)
    println("  rMag=1m Pc: $(round(pc_close, sigdigits=4))")
    @test pc_close > pc_foster  # closer should be higher Pc

    println("\nDiagnostic: actual miss distance at TCA for pomdp_close")
    prop_sc_tmp, _     = eci2orb_brahe(collect(prop_sc_close.current_state()[1:6]), 
                                        epoch_to_tuple(epoch_start_close), pomdp_close.satParams, pomdp_close.forceModel)
    prop_debris_tmp, _ = eci2orb_brahe(collect(prop_debris_close.current_state()[1:6]), 
                                        epoch_to_tuple(epoch_start_close), pomdp_close.debrisParams, pomdp_close.forceModel)
    prop_sc_tmp.propagate_to(epoch_tca_close)
    prop_debris_tmp.propagate_to(epoch_tca_close)

    r_sc_tca     = collect(prop_sc_tmp.current_state()[1:3])
    r_debris_tca = collect(prop_debris_tmp.current_state()[1:3])
    println("  Miss distance at TCA: $(norm(r_debris_tca - r_sc_tca)) m")
    println("  Expected rMag: $(pomdp_close.rMag) m")



    println("\nTest 3: Distant conjunction should have low Pc")
    pomdp_far = SpacecraftCAPOMDP(seed=42, randAdd=false, rMag=10000.0, vMag=100.0, TCA_max = 30*60)
    eci_sc_far, eci_debris_far, prop_sc_far, prop_debris_far, epoch_tca_far, _, _, epoch_start_far = generate_conjunction(pomdp_far)
    sc_far     = collect(prop_sc_far.current_state()[1:6])
    debris_far = collect(prop_debris_far.current_state()[1:6])
    pc_far = compute_pc(pomdp_far, sc_far, debris_far, epoch_start_far, epoch_tca_far)
    println("  rMag=10km Pc: $(round(pc_far, sigdigits=4))")
    @test pc_far < pc_foster  # farther should be lower Pc




    println("\nTest 4: Covariance sensitivity check")
    P_large = diagm([1000000.0, 1000000.0, 1000000.0, 1.0, 1.0, 1.0])
    pc_large_cov = compute_pc(pomdp, sc_eci_t0, debris_eci_t0, epoch_start, epoch_tca,
                            Σ_debris=P_large)
    println("  Default cov Pc: $(round(pc_foster,    sigdigits=4))")
    println("  Large cov Pc:   $(round(pc_large_cov, sigdigits=4))")
    # Just check it's a valid probability
    @test 0.0 <= pc_large_cov <= 1.0
    println("\nDiagnostic: actual miss distance at TCA")
    prop_sc_tmp, _     = eci2orb_brahe(collect(prop_sc.current_state()[1:6]), 
                                        epoch_to_tuple(epoch_start), pomdp.satParams, pomdp.forceModel)
    prop_debris_tmp, _ = eci2orb_brahe(collect(prop_debris.current_state()[1:6]), 
                                        epoch_to_tuple(epoch_start), pomdp.debrisParams, pomdp.forceModel)
    prop_sc_tmp.propagate_to(epoch_tca)
    prop_debris_tmp.propagate_to(epoch_tca)

    r_sc_tca     = collect(prop_sc_tmp.current_state()[1:3])
    r_debris_tca = collect(prop_debris_tmp.current_state()[1:3])
    println("  Miss distance at TCA: $(norm(r_debris_tca - r_sc_tca)) m")
    println("  Expected rMag: $(pomdp.rMag) m")

    println("\n=== All tests passed! ===")
end

test_compute_pc()