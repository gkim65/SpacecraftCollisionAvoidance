using Distributions
using LinearAlgebra


function POMDPs.observation(pomdp::SpacecraftCAPOMDP, a::CAAction, sp::CAState)
    μ_o = vcat(sp.sc_eci, sp.debris_eci)
    Σ_o = diagm(vcat(fill(pomdp.σ_sc^2, 6), fill(pomdp.σ_debris^2, 6)))
    return MvNormal(μ_o, Σ_o)
end
POMDPs.obstype(::SpacecraftCAPOMDP) = Vector{Float64}