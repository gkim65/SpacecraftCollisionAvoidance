using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("../SpacecraftCollisionAvoidance.jl")

using POMCPOW
using POMDPs
using ParticleFilters
using POMDPTools
using Distributions
using LinearAlgebra

# -------------------------------------------------------
# Particle initialization
# -------------------------------------------------------

function initialize_particles(pomdp::SpacecraftCAPOMDP; N::Int=500)
    s0 = rand(initialstate(pomdp))
    
    particles = map(1:N) do _
        # Add noise to each object's absolute ECI state based on initial covariances
        sc_eci_noisy     = s0.sc_eci     + sqrt.(diag(pomdp.P0_sc))     .* randn(6)
        debris_eci_noisy = s0.debris_eci + sqrt.(diag(pomdp.P0_debris)) .* randn(6)
        return CAState(sc_eci_noisy, debris_eci_noisy, s0.t)
    end
    
    return ParticleCollection(particles)
end
# -------------------------------------------------------
# Solver setup
# -------------------------------------------------------

function setup_solver(;
    tree_queries  = 100,
    max_depth     = 20,
    k_observation = 4.0,
    alpha_observation = 0.1,
    k_action      = 4.0,
    alpha_action  = 0.1,
    criterion     = MaxUCB(1.0)
)
    return POMCPOWSolver(
        tree_queries      = tree_queries,
        max_depth         = max_depth,
        k_observation     = k_observation,
        alpha_observation = alpha_observation,
        k_action          = k_action,
        alpha_action      = alpha_action,
        criterion         = criterion,
        estimate_value    = (pomdp, s, h, steps) -> begin
            miss_dist  = norm(get_x_rel(s)[1:3])
            R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
            return 100.0 * (miss_dist / R_combined)
        end
    )
end

# -------------------------------------------------------
# Run simulation
# -------------------------------------------------------


function run_simulation(pomdp::SpacecraftCAPOMDP; 
                        N_particles::Int = 500,
                        tree_queries::Int = 100,
                        max_steps::Int = 100)

    bh = get_brahe()
    solver  = setup_solver(tree_queries=tree_queries)
    planner = solve(solver, pomdp)
    b       = initialize_particles(pomdp, N=N_particles)
    s       = rand(initialstate(pomdp))

    up = BootstrapFilter(pomdp, N_particles)

    total_reward = 0.0
    step         = 0
    history      = []
    println("sc_eci position norm: $(norm(s.sc_eci[1:3])) — should be ~6.8e6 m")
    println("sc_eci velocity norm: $(norm(s.sc_eci[4:6])) — should be ~7500 m/s")
    println("debris_eci position norm: $(norm(s.debris_eci[1:3])) — should be ~6.8e6 m")
    println("debris_eci velocity norm: $(norm(s.debris_eci[4:6])) — should be ~7500 m/s")
    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - s.t
    pc_foster = compute_pc(pomdp, s.sc_eci, s.debris_eci, epoch_current, epoch_tca, method=:foster)
    println("  Foster Pc: $(round(pc_foster, sigdigits=4))")

    while !isterminal(pomdp, s) && step < max_steps
        # Get action from planner
        a = action(planner, b)
        # a = WAIT

        # Step the true state
        sp_dist = transition(pomdp, s, a)
        sp      = rand(sp_dist)

        # Get observation
        o = rand(observation(pomdp, a, sp))

        # Get reward
        r = reward(pomdp, s, a)

        # Update belief
        b = update(up, b, a, o)

        # Log
        push!(history, (
            s           = s,
            a           = a,
            o           = o,
            r           = r,
            miss_dist   = norm(get_x_rel(sp)[1:3]),
            t_remaining = s.t / 60  # in minutes
        ))

        total_reward += r
        s             = sp
        step         += 1

        println("Step $step | Action: $a | Miss dist: $(round(norm(get_x_rel(sp)[1:3]), digits=1))m | t_remaining: $(round(s.t/60, digits=1))min | r: $r")
        println("epoch_current: $(s.t/3600) hours before TCA")
        bh = get_brahe()
        println("miss dist ECI: $(norm(s.debris_eci[1:3] - s.sc_eci[1:3])) m")
        epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
        epoch_current = epoch_tca - s.t
        pc_foster = compute_pc(pomdp, s.sc_eci, s.debris_eci, epoch_current, epoch_tca, method=:foster)
        println("  Foster Pc: $(round(pc_foster, sigdigits=4))")

    end
    # Compute terminal reward for final state
    r_terminal = reward(pomdp, s, WAIT)
    total_reward += r_terminal

    println("\n=== Simulation Complete ===")
    println("Total reward: $total_reward")
    println("Final miss distance: $(round(norm(get_x_rel(s)[1:3]), digits=1)) m")
    println("Steps taken: $step")
    println("Maneuvers: $(count(h -> h.a == MANEUVER, history))")

    return history, total_reward
end



# Create POMDP
pomdp = SpacecraftCAPOMDP(
    seed = 42,
    randAdd = false,
    conjunctionType = "crossing",
    rMag = 100.0,
    vMag = 100.0,
    TCA_max = 10*60*60,
    dt = 30*60
)

# Test 1 - initial state
println("\nTesting initialstate...")
s0 = rand(initialstate(pomdp))
println("  x_rel: $(round.(get_x_rel(s0)[1:3], digits=2))")
println("  t: $(s0.t / 60) minutes")
println("  terminal: $(s0.terminal)")

# Test 2 - transition
println("\nTesting transition...")
sp_dist = transition(pomdp, s0, WAIT)
sp = rand(sp_dist)
println("  new t: $(sp.t / 60) minutes")
println("  miss dist: $(norm(get_x_rel(sp)[1:3])) m")

# Test observation
println("\nTesting observation...")
sp = rand(transition(pomdp, s0, WAIT))
o = rand(observation(pomdp, WAIT, sp))
println("  observation length: $(length(o))")
println("  first 3 elements (sc position): $(round.(o[1:3], digits=1))")
println("  last 3 elements (debris velocity): $(round.(o[10:12], digits=1))")

println("Testing reward...")
r_wait     = reward(pomdp, s0, WAIT)
r_maneuver = reward(pomdp, s0, MANEUVER)
println("  reward wait:     $r_wait")
println("  reward maneuver: $r_maneuver")

# Also test terminal reward
println("\nTesting terminal reward...")
# Create a fake terminal state at TCA with small miss distance
s_terminal_close = CAState(
    [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 50m miss distance
    s0.sc_eci, 0.0, true
)
s_terminal_far = CAState(
    [500.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 500m miss distance
    s0.sc_eci, 0.0, true
)
s_collision = CAState(
    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # inside hard body
    s0.sc_eci, 0.0, true
)
println("  reward terminal close (50m):    $(reward(pomdp, s_terminal_close, WAIT))")
println("  reward terminal far (500m):     $(reward(pomdp, s_terminal_far, WAIT))")
println("  reward collision (5m):          $(reward(pomdp, s_collision, WAIT))")

# Test 5 - particles
println("\nTesting particle initialization...")
b0 = initialize_particles(pomdp, N=10)
println("  n_particles: $(length(b0._particles))")

println("POMDP created successfully")
println("TCA_max: $(pomdp.TCA_max / 3600) hours")
println("dt: $(pomdp.dt / 60) minutes")
println("Steps: $(Int(pomdp.TCA_max / pomdp.dt))")

println("\nTesting full solver...")

pomdp = SpacecraftCAPOMDP(
    seed = 42,
    randAdd = false,
    forceModel = false,  # two-body only, much faster
    conjunctionType = "crossing",
    rMag = 100.0,
    vMag = 100.0,
    TCA_max = 10*60*60,
    dt = 30*60,
    σ_sc = 1000.0,
    σ_debris = 5000.0
)

# planner = solve(solver, pomdp)
# b0      = initialize_particles(pomdp, N=100)

# println("Planner created successfully")
# println("Getting first action...")
# a = action(planner, b0)
# println("First action: $a")

# Run simulation
history, total_reward = run_simulation(pomdp, 
                                        N_particles  = 500,
                                        tree_queries = 100,
                                        max_steps    = Int(pomdp.TCA_max / pomdp.dt))