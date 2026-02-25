using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("../SpacecraftCollisionAvoidance.jl")
include("genConjunctions_test_funcs.jl")

using POMDPTools
pomdp = SpacecraftCAPOMDP(
    randAdd = false,
    rMag = 100.0,
    vMag = 0.1,
    TCA_max = 600.0
)
has_consistent_distributions(pomdp)


fig = test_all_conjunctions(pomdp)
display(fig)


# Run it
pomdp_base = SpacecraftCAPOMDP(
    randAdd = false,
    rMag = 500.0,
    vMag = 100.0,
    forceModel = false,
    seed = 42,
)

fig = plot_conjunctions_3d_rtn(pomdp_base)
save("conjunction_3d_rtn.png", fig, px_per_unit=2)
