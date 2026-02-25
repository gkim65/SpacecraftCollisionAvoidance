
function get_x_rel(s::CAState)
    bh = get_brahe()
    return collect(bh.state_eci_to_rtn(s.sc_eci, s.debris_eci))
end

function POMDPs.isterminal(pomdp::SpacecraftCAPOMDP, s::CAState)
    R_combined = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    return s.t <= 0.0 || norm(get_x_rel(s)[1:3]) < R_combined || s.terminal
end

# TODO: revisit so we have more random states

function POMDPs.initialstate(pomdp::SpacecraftCAPOMDP)
    eci_spacecraft, _, prop_spacecraft, prop_debris, epoch_tca, _, x_rel_rtn, epoch_start = generate_conjunction(pomdp)

    sc_eci_t0     = collect(prop_spacecraft.current_state()[1:6])
    debris_eci_t0 = collect(prop_debris.current_state()[1:6])

    # Initial belief TODO: do we want to randomize this?
    # b0 = CABelief(spacecraft_eci_t0, pomdp.P0_sc, debris_eci_t0, pomdp.P0_debris)

    # pc0 = compute_pc(pomdp, sc_eci_t0, debris_eci_t0, epoch_start, epoch_tca)
    s0 = CAState(sc_eci_t0, debris_eci_t0,pomdp.TCA_max)
    return Deterministic(s0)
end