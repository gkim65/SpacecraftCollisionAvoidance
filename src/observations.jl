using Distributions
using LinearAlgebra


function POMDPs.observation(pomdp::SpacecraftCAPOMDP, a::CAAction, sp::CAState)
    μ_o = vcat(sp.sc_eci, sp.debris_eci)
    # Block-diagonal 12×12 R = [R_sc 0; 0 R_debris]. These are the full 6×6
    # per-object measurement-noise covariances (class-tiered + anisotropic; see
    # sensorTiers.jl). For the default synthetic path they equal σ²·I₆, so this is
    # identical to the old diag([σ_sc²×6, σ_debris²×6]).
    Σ_o = zeros(12, 12)
    Σ_o[1:6, 1:6]   = pomdp.R_sc
    Σ_o[7:12, 7:12] = pomdp.R_debris
    return MvNormal(μ_o, Symmetric(Σ_o))
end
POMDPs.obstype(::SpacecraftCAPOMDP) = Vector{Float64}