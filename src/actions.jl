
POMDPs.actions(pomdp::SpacecraftCAPOMDP) = [WAIT, MANEUVER]
POMDPs.actionindex(pomdp::SpacecraftCAPOMDP, a::CAAction) = Int(a)