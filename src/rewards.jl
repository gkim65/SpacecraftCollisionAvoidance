# function POMDPs.reward(pomdp::SpacecraftCAPOMDP, s::CAState, a::CAAction)
#     # Successfully reduced Pc below threshold mid-episode
#     if s.terminal
#         return 100.0
#     end

#     # At TCA — penalize if Pc still above threshold
#     if s.t <= 0.0
#         return s.pc > pomdp.pc_threshold ? -1000.0 : 100.0
#     end

#     # Maneuver cost during episode
#     r = 0.0
#     if a == MANEUVER
#         r -= pomdp.maneuver_cost
#     end

#     return r
# end


function POMDPs.reward(pomdp::SpacecraftCAPOMDP, s::CAState, a::CAAction)
    if isterminal(pomdp, s)
        miss_distance = norm(get_x_rel(s)[1:3])
        R_combined    = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
        if miss_distance < R_combined || miss_distance <= pomdp.rMag
            return -1000.0
        else
            return 0 #* (miss_distance / R_combined)
        end
    end
    r = 0.0
    if a == MANEUVER
        r -= pomdp.maneuver_cost
    end
    return r
end