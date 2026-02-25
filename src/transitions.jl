# function POMDPs.transition(pomdp::SpacecraftCAPOMDP, s::CAState, a::CAAction)
#     # Pass through terminal states
#     if isterminal(pomdp, s)
#         return Deterministic(s)
#     end

#     bh = get_brahe()

#     # Recover current and next absolute epochs
#     epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
#     epoch_current = epoch_tca - s.t
#     epoch_next    = epoch_tca - (s.t - pomdp.dt)

#     # Recover debris absolute ECI from relative RTN state
#     debris_eci = collect(bh.state_rtn_to_eci(s.sc_eci, s.x_rel))

#     # Create temporary propagators from current true states
#     epoch_current_tuple = epoch_to_tuple(epoch_current)
#     prop_sc, _     = eci2orb_brahe(s.sc_eci, epoch_current_tuple,
#                                     pomdp.satParams, pomdp.forceModel)
#     prop_debris, _ = eci2orb_brahe(debris_eci, epoch_current_tuple,
#                                     pomdp.debrisParams, pomdp.forceModel)

#     # Apply impulsive maneuver to spacecraft if action is MANEUVER
#     if a == MANEUVER
#         np = pyimport("numpy")
        
#         function maneuver_callback(event_epoch, event_state)
#             println("  Maneuver callback triggered!")
#             println("  State before: $event_state")
#             new_state = np.copy(event_state)
#             v         = new_state[4:6]
#             v_hat     = v / norm(v)
#             new_state[4:6] += pomdp.Δv * v_hat
#                 println("  State after: $new_state")

#             # Return as Python tuple with numpy array
#             bh = get_brahe()
#             return (np.array(new_state), bh.EventAction.CONTINUE)
#         end

#         maneuver_event = bh.TimeEvent(epoch_current + 1.0, "Maneuver").with_callback(maneuver_callback)
#         prop_sc.add_event_detector(maneuver_event)
#     end

#     # Propagate both to next timestep
#     prop_sc.propagate_to(epoch_next)
#     prop_debris.propagate_to(epoch_next)

#     # Get new absolute ECI states
#     sc_eci_next     = collect(prop_sc.current_state()[1:6])
#     debris_eci_next = collect(prop_debris.current_state()[1:6])


#     println("sc_eci_next: $sc_eci_next")
#     println("debris_eci_next: $debris_eci_next")
#     println("any NaN in sc: $(any(isnan.(sc_eci_next)))")
#     println("any NaN in debris: $(any(isnan.(debris_eci_next)))")

#     println("epoch_current: $(epoch_to_tuple(epoch_current))")
#     println("epoch_next: $(epoch_to_tuple(epoch_next))")
#     println("maneuver event time: $(epoch_to_tuple(epoch_current + 1.0))")
#     println("time delta: $(pomdp.dt) seconds")
#     # Convert to RTN relative state
#     x_rel_next = collect(bh.state_eci_to_rtn(sc_eci_next, debris_eci_next))

#     # New time remaining
#     t_next = s.t - pomdp.dt

#     # Terminal if time is up or collision occurred
#     R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
#     terminal   = t_next <= 0.0 || norm(x_rel_next[1:3]) < R_combined

#     # Check terminal condition — carry forward pc for now
#     # terminal = t_next <= 0.0 || s.pc < pomdp.pc_threshold

#     sp = CAState(x_rel_next, sc_eci_next, t_next, terminal)

#     return Deterministic(sp)
# end


function POMDPs.transition(pomdp::SpacecraftCAPOMDP, s::CAState, a::CAAction)
    if isterminal(pomdp, s)
        return Deterministic(s)
    end

    bh = get_brahe()

    epoch_tca     = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
    epoch_current = epoch_tca - s.t
    epoch_next    = epoch_tca - (s.t - pomdp.dt)

    # Apply maneuver directly to spacecraft state before propagating
    sc_eci_start = copy(s.sc_eci)
    if a == MANEUVER
        v     = sc_eci_start[4:6]
        v_hat = v / norm(v)
        sc_eci_start[4:6] += pomdp.Δv * v_hat
    end

    epoch_current_tuple = epoch_to_tuple(epoch_current)
    prop_sc, _     = eci2orb_brahe(sc_eci_start, epoch_current_tuple,
                                    pomdp.satParams, pomdp.forceModel)
    prop_debris, _ = eci2orb_brahe(s.debris_eci, epoch_current_tuple,
                                    pomdp.debrisParams, pomdp.forceModel)

    prop_sc.propagate_to(epoch_next)
    prop_debris.propagate_to(epoch_next)

    sc_eci_next     = collect(prop_sc.current_state()[1:6])
    debris_eci_next = collect(prop_debris.current_state()[1:6])

    t_next     = s.t - pomdp.dt
    R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    
    # Only compute RTN for terminal check — valid near TCA when objects are close
    x_rel_next = collect(bh.state_eci_to_rtn(sc_eci_next, debris_eci_next))
    terminal   = t_next <= 0.0 || norm(x_rel_next[1:3]) < R_combined

    sp = CAState(sc_eci_next, debris_eci_next, t_next, terminal)
    return Deterministic(sp)
end