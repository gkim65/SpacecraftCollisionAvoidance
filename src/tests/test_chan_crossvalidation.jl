#=
test_chan_crossvalidation.jl

Phase 1 validation for the Chan (1997) closed-form Pc ported into
src/utils/computePc.jl.

Three groups of checks:

1. CROSS-VALIDATION (must pass): on fixed (mean, covariance, HBR) test cases,
   the Julia `chan_pc` agrees with the original Python reference
   (ProbofCollision/src/collision/chan1997.py) to within 0.2%. The Python
   value is computed at run time on the identical inputs — nothing is
   hard-coded — so both sides see the same numbers.

2. CORRECTNESS PROPERTIES (must pass): Pc in [0,1], Pc rises as miss shrinks,
   Pc rises with HBR, swap symmetry, and the isotropic-covariance limit
   reduces to the plain noncentral chi-squared CDF.

3. PYCALL MATRIX ORIENTATION (must pass): matrices passed to Brahe via PyCall
   and matrices returned from Brahe (STM, covariance_rtn) are not silently
   transposed by the row-major/column-major boundary. Checked directly on the
   matrices, not just on final Pc. Skipped with a clear message only if Brahe
   cannot be imported through PyCall in this environment.

Run:  julia --project=. src/tests/test_chan_crossvalidation.jl
=#

using Test
using LinearAlgebra
using Distributions
using Random
using POMDPs        # SpacecraftCAPOMDP.jl subtypes POMDP{...} and defines POMDPs.discount
using POMDPTools

include(joinpath(@__DIR__, "..", "SpacecraftCAPOMDP.jl"))   # SpacecraftCAPOMDP type
include(joinpath(@__DIR__, "..", "utils", "computePc.jl"))  # chan_pc, _chan_series

# ---------------------------------------------------------------------------
# Python reference bridge
# ---------------------------------------------------------------------------

const PY_REF = normpath(joinpath(@__DIR__, "..", "..", "..", "ProbofCollision",
                                 ".venv", "bin", "python"))
const PY_SCRIPT = joinpath(@__DIR__, "chan_reference.py")

"Serialize a vector of Floats as a JSON array."
_json_vec(v) = "[" * join(string.(Float64.(v)), ",") * "]"

"Serialize a matrix as a JSON array-of-arrays (row-major)."
function _json_mat(M)
    rows = [ _json_vec(M[i, :]) for i in 1:size(M, 1) ]
    return "[" * join(rows, ",") * "]"
end

"Compute Pc from the original Python Chan implementation on identical inputs."
function python_chan_pc(sc1, sc2, cov1, cov2, hbr)
    payload = string("{",
        "\"sc1\":",  _json_vec(sc1),  ",",
        "\"sc2\":",  _json_vec(sc2),  ",",
        "\"cov1\":", _json_mat(cov1), ",",
        "\"cov2\":", _json_mat(cov2), ",",
        "\"hbr\":",  string(Float64(hbr)),
        "}")
    out = read(pipeline(IOBuffer(payload), `$PY_REF $PY_SCRIPT`), String)
    m = match(r"\"pc\"\s*:\s*([-+0-9.eE]+)", out)
    m === nothing && error("Could not parse Pc from Python reference output: $out")
    return parse(Float64, m.captures[1])
end

# ---------------------------------------------------------------------------
# Fixed test cases: (name, sc1, sc2, cov1, cov2, hbr)
#
# Chosen to span the regimes exercised in ProbofCollision's own tests:
# head-on tail, moderate Pc, high Pc (zero miss + large HBR), anisotropic
# covariance, and a large-miss near-zero case.
# ---------------------------------------------------------------------------

iso(sp, sv) = diagm(vcat(fill(sp^2, 3), fill(sv^2, 3)))

const CASES = [
    (name = "head-on tail (100 m miss, tight cov)",
     sc1  = [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
     sc2  = [7.0e6 + 100.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0],
     cov1 = iso(10.0, 0.01), cov2 = iso(10.0, 0.01), hbr = 10.0),

    (name = "moderate Pc (200 m miss, 100 m cov)",
     sc1  = [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
     sc2  = [7.0e6 + 200.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0],
     cov1 = iso(100.0, 0.01), cov2 = iso(100.0, 0.01), hbr = 10.0),

    (name = "high Pc (zero miss, large HBR)",
     sc1  = [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
     sc2  = [7.0e6, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0],
     cov1 = iso(1.0, 0.01), cov2 = iso(1.0, 0.01), hbr = 20.0),

    (name = "anisotropic covariance (RTN-like)",
     sc1  = [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
     sc2  = [7.0e6 + 150.0, 50.0, 0.0, 0.0, 7500.0 - 400.0, 20.0],
     cov1 = diagm([100.0^2, 500.0^2, 50.0^2, 0.1^2, 0.5^2, 0.05^2]),
     cov2 = diagm([80.0^2,  300.0^2, 40.0^2, 0.1^2, 0.3^2, 0.04^2]),
     hbr  = 10.0),

    (name = "large miss (near-zero Pc)",
     sc1  = [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
     sc2  = [7.0e6 + 100e3, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0],
     cov1 = iso(10.0, 0.01), cov2 = iso(10.0, 0.01), hbr = 10.0),
]

# ---------------------------------------------------------------------------
# 1. Cross-validation against the Python reference
# ---------------------------------------------------------------------------

@testset "Chan (1997) Julia vs Python cross-validation" begin
    @test isfile(PY_REF)   # Python env with ProbofCollision + numpy available
    TOL = 0.002            # 0.2% relative tolerance

    for c in CASES
        pc_jl = chan_pc(c.sc1, c.sc2, c.cov1, c.cov2, c.hbr)
        pc_py = python_chan_pc(c.sc1, c.sc2, c.cov1, c.cov2, c.hbr)

        if pc_py < 1e-20
            # Both essentially zero — agree in absolute terms.
            @test pc_jl < 1e-15
            @info "  $(c.name): both ≈ 0  (jl=$(pc_jl), py=$(pc_py))"
        else
            rel = abs(pc_jl - pc_py) / pc_py
            @test rel <= TOL
            @info "  $(c.name): jl=$(pc_jl)  py=$(pc_py)  rel_err=$(round(rel*100, sigdigits=3))%"
        end
    end
end

# ---------------------------------------------------------------------------
# 2. Correctness properties of the Julia implementation
# ---------------------------------------------------------------------------

@testset "Chan Pc correctness properties" begin
    sc1 = [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]
    cov = iso(100.0, 0.01)

    # Pc in [0, 1]
    for c in CASES
        pc = chan_pc(c.sc1, c.sc2, c.cov1, c.cov2, c.hbr)
        @test 0.0 <= pc <= 1.0
    end

    # Pc rises as the miss distance shrinks
    sc2_far  = [7.0e6 + 500.0, 0.0, 0.0, 0.0, 7000.0, 0.0]
    sc2_near = [7.0e6 +  50.0, 0.0, 0.0, 0.0, 7000.0, 0.0]
    @test chan_pc(sc1, sc2_near, cov, cov, 10.0) > chan_pc(sc1, sc2_far, cov, cov, 10.0)

    # Pc rises with hard-body radius
    sc2 = [7.0e6 + 100.0, 0.0, 0.0, 0.0, 7000.0, 0.0]
    @test chan_pc(sc1, sc2, cov, cov, 50.0) > chan_pc(sc1, sc2, cov, cov, 5.0)

    # Swap symmetry: Pc(1,2) == Pc(2,1)
    pc12 = chan_pc(sc1, sc2, cov, cov, 10.0)
    pc21 = chan_pc(sc2, sc1, cov, cov, 10.0)
    @test abs(pc12 - pc21) < 1e-12

    # Zero relative speed → error (encounter plane undefined)
    sc2_costream = [7.0e6 + 100.0, 0.0, 0.0, 0.0, 7500.0, 0.0]  # same velocity as sc1
    @test_throws ErrorException chan_pc(sc1, sc2_costream, cov, cov, 10.0)

    # Isotropic-covariance limit: Pc reduces to ncx2.cdf(R^2/σ^2; 2, miss^2/σ^2).
    # With isotropic 3D position covariance, the 2D projection is isotropic too,
    # so the anisotropy factor is 1 and the formula is the plain noncentral χ².
    σ = 100.0
    covi = diagm([σ^2, σ^2, σ^2, 0.01^2, 0.01^2, 0.01^2])
    sc2i = [7.0e6 + 200.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0]
    pc_chan = chan_pc(sc1, sc2i, covi, covi, 10.0)
    # Encounter-plane miss magnitude for this head-on geometry is the radial 200 m.
    C_pos = covi[1:3, 1:3] .+ covi[1:3, 1:3]   # combined isotropic → 2σ² on the plane
    σ2d_sq = C_pos[1, 1]
    miss = 200.0
    pc_ncx2 = cdf(NoncentralChisq(2, miss^2 / σ2d_sq), 10.0^2 / σ2d_sq)
    @test isapprox(pc_chan, pc_ncx2; rtol = 1e-6)
    @info "  isotropic limit: chan=$(pc_chan)  ncx2=$(pc_ncx2)"
end

# ---------------------------------------------------------------------------
# 3. PyCall matrix-orientation check (Brahe STM / covariance_rtn)
#
# The Chan math above takes covariances as direct inputs and never touches
# Brahe. But the covariances fed to it in the real planner come from Brahe
# through PyCall, and PyCall bridges numpy (row-major) and Julia (column-major).
# A silent transpose there would corrupt every downstream Pc. Verify directly:
#   (a) a known non-symmetric matrix survives the Julia→numpy→Julia round trip
#       with orientation intact;
#   (b) Brahe's covariance_rtn output is symmetric and matches its own transpose
#       (a transposed read of a non-symmetric internal buffer would break this),
#       and the STM Φ is returned with the correct orientation by checking that
#       Σ propagates as Φ Σ Φᵀ consistently.
# ---------------------------------------------------------------------------

@testset "PyCall / Brahe matrix orientation" begin
    local brahe_ok = false
    local pymod = nothing
    try
        include(joinpath(@__DIR__, "..", "utils", "genConjunctions.jl"))
        pymod = get_brahe()
        brahe_ok = true
    catch e
        @warn "Skipping Brahe/PyCall orientation checks — Brahe not importable: $e"
    end

    if brahe_ok
        using PyCall
        np = pyimport("numpy")

        # (a) Non-symmetric matrix round-trip Julia -> numpy -> Julia.
        # If PyCall transposed at either boundary, A_back would equal Aᵀ.
        A = reshape(collect(1.0:9.0), 3, 3)          # column-major fill
        A[1, 2] = 99.0                                # break symmetry unmistakably
        A_np = np.array(A)
        A_back = convert(Matrix{Float64}, A_np)
        @test A_back == A
        @test A_back[1, 2] == 99.0 && A_back[2, 1] != 99.0
        @info "  Julia↔numpy round trip preserves orientation (A[1,2]=99 stays at [1,2])"

        # Build a short propagation with STM + covariance to exercise Brahe.
        pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false,
                                  conjunctionType = "crossing",
                                  rMag = 500.0, vMag = 500.0)
        sc_eci, debris_eci = generate_tca_relative(pomdp)

        epoch0 = pomdp.epochTCA
        Σ0 = pomdp.P0_debris
        prop, bh_epoch = eci2orb_brahe(debris_eci, epoch0, pomdp.debrisParams,
                                       pomdp.forceModel; initial_covariance = Σ0)
        epoch_later = bh_epoch + 3600.0    # +1 hour
        prop.propagate_to(epoch_later)

        # (b) covariance_rtn must come back symmetric and PSD-ish.
        P_rtn = convert(Matrix{Float64}, prop.covariance_rtn(epoch_later))
        @test size(P_rtn) == (6, 6)
        asym = maximum(abs.(P_rtn .- transpose(P_rtn)))
        @test asym < 1e-6 * maximum(abs.(P_rtn))
        @test all(diag(P_rtn) .>= 0.0)
        @info "  Brahe covariance_rtn is symmetric (max|P-Pᵀ| = $(asym)); diag ≥ 0"

        # (c) STM orientation: Σ propagated by hand as Φ Σ0 Φᵀ must match Brahe's
        # own propagated ECI covariance. A transposed Φ would make these differ.
        # Brahe's stm_at returns Φ from the initial epoch to the requested epoch.
        Φ = convert(Matrix{Float64}, prop.stm_at(epoch_later))
        @test size(Φ) == (6, 6)
        Σ_hand = Φ * Σ0 * transpose(Φ)
        P_eci = convert(Matrix{Float64}, prop.covariance(epoch_later))
        rel_mat = maximum(abs.(Σ_hand .- P_eci)) / maximum(abs.(P_eci))
        @test rel_mat < 1e-6
        @info "  STM orientation consistent: max rel diff Φ Σ Φᵀ vs Brahe covariance = $(rel_mat)"
    end
end
