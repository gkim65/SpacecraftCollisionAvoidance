# curvilinearUtils.jl — covariance + parabola-fit utilities for the curvilinear
# usage-violation analysis. Faithful Julia ports of NASA CARA helpers:
#   CovRemEigValClip.m   (ProbabilityOfCollision/Utils) — NPD eigenvalue clipping
#   cov_make_symmetric.m (Utils/AugmentedMath)          — symmetrize
#   TimeParabolaFit.m    (ProbabilityOfCollision/Utils) — 3-point min parabola
#
# References: Hall, Hejduk & Johnson (2017), AAS-17-567 (eigenvalue clipping).

using LinearAlgebra

"""
    cov_make_symmetric(C) -> Matrix

Symmetrize a square covariance. Port of `cov_make_symmetric.m`: if already
symmetric return as-is, else average with the transpose and force exact
symmetry. (`0.5(C+Cᵀ)` is already exactly symmetric to machine precision, so the
extra `triu` reflection in the MATLAB is belt-and-suspenders; we replicate the
average, which is what matters numerically.)
"""
function cov_make_symmetric(C::AbstractMatrix)
    Ct = transpose(C)
    C == Ct && return Matrix(C)
    return Matrix((C .+ Ct) ./ 2)
end

"""
    CovRemResult

Result of eigenvalue-clip NPD remediation (subset of CARA's outputs actually
used by the curvilinear analysis): remediated eigenvalues `Lrem`, raw
eigenvectors `Vraw`, PD status (−1 NPD / 0 PSD / +1 PD), clip status, and the
remediated determinant / inverse / matrix.
"""
struct CovRemResult
    Lrem::Vector{Float64}
    Lraw::Vector{Float64}
    Vraw::Matrix{Float64}
    pos_def_status::Int
    clip_status::Bool
    Adet::Float64
    Ainv::Matrix{Float64}
    Arem::Matrix{Float64}
end

"""
    cov_rem_eigval_clip(Araw; Lclip=0.0) -> CovRemResult

Detect and remediate a non-positive-definite covariance by eigenvalue clipping
(Hall et al. 2017). Any eigenvalue below `Lclip` is raised to `Lclip`; the
remediated determinant/inverse/matrix are rebuilt from the RAW eigenvectors
(matching `CovRemEigValClip.m` exactly). Recommended `Lclip = (1e-4·HBR)^2` for
position covariances used in Mahalanobis/Pc calculations.
"""
function cov_rem_eigval_clip(Araw::AbstractMatrix; Lclip::Real = 0.0)
    Lclip < 0 && throw(ArgumentError("Lclip cannot be negative"))

    F = eigen(Symmetric(Matrix(Araw)))
    Lraw = collect(Float64, F.values)
    Vraw = Matrix{Float64}(F.vectors)

    pos_def_status = Int(sign(minimum(Lraw)))

    Lrem = copy(Lraw)
    clip_status = minimum(Lraw) < Lclip
    if clip_status
        Lrem[Lraw .< Lclip] .= Lclip
    end

    Adet = prod(Lrem)
    Ainv = Vraw * Diagonal(1.0 ./ Lrem) * Vraw'
    Arem = clip_status ? Vraw * Diagonal(Lrem) * Vraw' : Matrix(Araw)

    return CovRemResult(Lrem, Lraw, Vraw, pos_def_status, clip_status,
                        Adet, Ainv, Arem)
end

"""
    time_parabola_fit(t, F) -> (c, tinc, Finc, rankAmat)

Best-fit parabola `y = c[1]·x² + c[2]·x + c[3]` through the (t, F) points nearest
the minimum-F point, using the fewest points needed for a full-rank fit (≥3).
Port of `TimeParabolaFit.m`. Solves in centered/normalized time for conditioning,
then converts the coefficients back to un-normalized time.
"""
function time_parabola_fit(t::AbstractVector, F::AbstractVector)
    # Restrict to unique times (avoid singular design matrix).
    tu = Float64[]; Fu = Float64[]
    seen = Set{Float64}()
    for (ti, Fi) in zip(t, F)
        tif = Float64(ti)
        if !(tif in seen)
            push!(seen, tif); push!(tu, tif); push!(Fu, Float64(Fi))
        end
    end
    Nt = length(tu)
    Nt < 3 && error("Minimum of three unique points required for parabolic fit")

    # Sort by ascending F.
    srt = sortperm(Fu)
    tvec = tu[srt]; Fvec = Fu[srt]

    inc = falses(Nt); inc[1:3] .= true

    function build(inc)
        tinc = tvec[inc]
        tmin = minimum(tinc); tmax = maximum(tinc)
        tdel = 0.5 * (tmax - tmin); tmid = 0.5 * (tmax + tmin)
        z = (tinc .- tmid) ./ tdel
        Amat = hcat(z .^ 2, z, ones(length(z)))
        return tinc, tmid, tdel, z, Amat
    end

    tinc, tmid, tdel, z, Amat = build(inc)
    rankAtol = 1000 * maximum(size(Amat)) * eps(norm(Amat))
    rankAmat = rank(Amat; atol = rankAtol)

    while rankAmat < 3 && any(.!inc)
        # Add up to (3 - #included) more of the not-yet-included points.
        need = max(1, 3 - count(inc))
        added = 0
        for i in 1:Nt
            (!inc[i]) || continue
            inc[i] = true; added += 1
            added >= need && break
        end
        tinc, tmid, tdel, z, Amat = build(inc)
        rankAtol = 1000 * maximum(size(Amat)) * eps(norm(Amat))
        rankAmat = rank(Amat; atol = rankAtol)
    end

    Finc = Fvec[inc]
    x = pinv(Amat) * Finc                       # coefficients in normalized z

    # Convert z-coefficients to t-coefficients.
    trat = tmid / tdel
    tratx1 = trat * x[1]
    c = [x[1] / tdel^2,
         (x[2] - 2 * tratx1) / tdel,
         trat * (tratx1 - x[2]) + x[3]]

    return (c, tinc, Finc, rankAmat)
end