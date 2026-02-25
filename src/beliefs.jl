struct CABelief
    μ_sc::Vector{Float64}     # 6-element mean spacecraft state in ECI
    Σ_sc::Matrix{Float64}     # 6x6 spacecraft covariance in ECI
    μ_debris::Vector{Float64} # 6-element mean debris state in ECI
    Σ_debris::Matrix{Float64} # 6x6 debris covariance in ECI
end