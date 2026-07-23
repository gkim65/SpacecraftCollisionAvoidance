# =========================================================================
# phase6_validation_data.jl — data for the Phase 6 validation-figure pass
# (built on the CORRECTED accumulated-Σ Pc: node_pc_at_tca uses each node's OWN
# tracked belief Σ propagated to TCA, NOT a fresh P0).
#
# Julia is the SOURCE OF TRUTH: this runs the actual beliefMCTS.jl / beliefTracker.jl
# and dumps JSON that the matplotlib plotters render. Kept OUT of the pinned
# brahe/numpy venv (plotting is a separate `uv run --with matplotlib` step).
#
# Run from the repo root:
#   julia --project=. figureScripts/phase6_validation_data.jl
# writes figureScripts/phase6_validation_data.json
#
# Produces the data for two figures (no MCTS tree — both are direct belief
# propagation, so this is cheap despite ~172 ms/Pc):
#
#   Fig 6a — Pc vs. time-remaining. The planner's actual node_pc_at_tca swept
#            over τ for a WAIT-only belief: a coarse 1-hr grid (dots — what the
#            planner samples) + a fine sub-orbit grid (line — resolves the
#            once-per-orbit R/N ripple, Phase 3 finding). Shows the corrected
#            Pc is finite/sensible, contrasts at-TCA vs hours-out, and makes the
#            "single-instant Pc is fragile" (~10× hour-to-hour swing) explicit.
#
#   Fig 6b — Σ-at-TCA is ~branch-invariant. Grow matched WAIT vs MANEUVER
#            branch beliefs to TCA (_grow_belief_to_tca) and compare Σ-at-TCA
#            (identical to many digits, because Σ is z-/maneuver-independent
#            today, §8) against the mean separation at TCA (which DOES move with
#            the maneuver). The visual "Σ fixed, mean moves" argument that the
#            constraint acts through the mean.
#
# Fixture: SpacecraftCAPOMDP(seed=42, randAdd=false), Phase-3 feasible co-orbital
# cross-track conjunction (miss 500 m, v_rel 15 m/s).
# =========================================================================
using LinearAlgebra
using Random
using PyCall
using POMDPs
using POMDPTools

# Minimal JSON writer (avoid adding JSON to the project's direct deps).
_json(x::Bool) = x ? "true" : "false"
_json(x::Real) = isfinite(x) ? string(x) : "null"
_json(x::AbstractString) = "\"$x\""
_json(v::AbstractVector) = "[" * join(_json.(v), ",") * "]"
_json(m::AbstractMatrix) = "[" * join([_json(collect(m[i, :])) for i in 1:size(m, 1)], ",") * "]"
_json(d::AbstractDict) = "{" * join(["\"$k\":" * _json(v) for (k, v) in d], ",") * "}"

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(REPO, "src", "utils", "genConjunctions.jl"))
include(joinpath(REPO, "src", "utils", "computePc.jl"))
include(joinpath(REPO, "src", "utils", "covarianceTable.jl"))
include(joinpath(REPO, "src", "states.jl"))
include(joinpath(REPO, "src", "actions.jl"))
include(joinpath(REPO, "src", "observations.jl"))
include(joinpath(REPO, "src", "transitions.jl"))
include(joinpath(REPO, "src", "utils", "beliefTracker.jl"))
include(joinpath(REPO, "src", "utils", "beliefMCTS.jl"))

pomdp = SpacecraftCAPOMDP(seed = 42, randAdd = false)
sc_tca, db_tca = generate_conjunction_geometry(pomdp;
    geometry = :cross_track, miss_m = 500.0, v_rel = 15.0)

# Anchor the conjunction: place both objects at TCA at the requested miss, then
# propagate BOTH back so we have a state at any time-remaining τ. (Same recipe as
# make_conjunction_state in test_belief_mcts.jl.)
bh = get_brahe()
epoch_tca = bh.Epoch.from_datetime(pomdp.epochTCA..., bh.TimeSystem.UTC)
et = epoch_to_tuple(epoch_tca)
prop_sc_ref, ep0 = eci2orb_brahe(sc_tca, et, pomdp.satParams, pomdp.forceModel)
prop_db_ref, _   = eci2orb_brahe(db_tca, et, pomdp.debrisParams, pomdp.forceModel)

"""State (sc_eci, debris_eci) at time-remaining `t` (s), propagated back from TCA."""
function state_at_tau(t::Real)
    prop_sc_ref.propagate_to(ep0 - Float64(t))
    prop_db_ref.propagate_to(ep0 - Float64(t))
    sc = collect(prop_sc_ref.current_state()[1:6])
    db = collect(prop_db_ref.current_state()[1:6])
    return sc, db
end

pos_sig(Σ) = sqrt(maximum(diag(Matrix(Σ))[1:3]))

# =========================================================================
# FIG 6a — Pc vs. time-remaining (the planner's own node_pc_at_tca).
#
# At each τ we build a belief anchored at the true state at τ with P0 (the belief
# the planner holds when it stops measuring at τ and coasts — "coast Pc at TCA"),
# and evaluate node_pc_at_tca. Coarse grid = the 1-hr grid the planner samples;
# fine grid resolves the once-per-orbit ripple.
#
# NB (why P0-anchored, not a full predict/correct rollout to τ): the corrected
# model is "coast from now" — a node's Pc grows the belief Σ it HOLDS to TCA. Far
# out, a WAIT rollout that measured every step lands at a Σ close to P0-scale (the
# measurement holds it near P0, then it coasts) — and the ripple we want to show
# is the orbital-phase dependence of Φ(τ→TCA)·Σ·Φᵀ, which the P0-anchored coast
# isolates cleanly. This is exactly node_pc_at_tca on a freshly-anchored node.
# =========================================================================
orbit_period_min = 92.6   # ~LEO period at this altitude (Phase 3 figure caption)

# coarse: hourly, 24 h -> 1 h remaining
tau_coarse_hr = collect(24.0:-1.0:1.0)
pc_coarse = Float64[]
for th in tau_coarse_hr
    sc, db = state_at_tau(th * 3600.0)
    node = BeliefNode(belief_from_pomdp(pomdp, sc, db, th * 3600.0),
                      CAState(sc, db, th * 3600.0), false)
    push!(pc_coarse, node_pc_at_tca(pomdp, node))
end

# fine: every ~11 min (well under the ~92.6 min orbit) to resolve the R/N ripple
dt_fine_hr = (orbit_period_min / 8) / 60      # ~8 samples per orbit
tau_fine_hr = collect(24.0:-dt_fine_hr:0.5)
pc_fine = Float64[]
sig_sc_fine = Float64[]; sig_db_fine = Float64[]
for th in tau_fine_hr
    sc, db = state_at_tau(th * 3600.0)
    node = BeliefNode(belief_from_pomdp(pomdp, sc, db, th * 3600.0),
                      CAState(sc, db, th * 3600.0), false)
    push!(pc_fine, node_pc_at_tca(pomdp, node))
    # Σ-at-TCA position 1σ along the way (shows the breathing that drives the ripple)
    μsc, Σsc = _grow_belief_to_tca(pomdp, node.belief.sc.μ, node.belief.sc.Σ,
                                   pomdp.satParams, th * 3600.0)
    μdb, Σdb = _grow_belief_to_tca(pomdp, node.belief.debris.μ, node.belief.debris.Σ,
                                   pomdp.debrisParams, th * 3600.0)
    push!(sig_sc_fine, pos_sig(Σsc)); push!(sig_db_fine, pos_sig(Σdb))
end

# at-TCA Pc (no growth): belief held at t≈0 -> node_pc_at_tca uses (μ,Σ) directly
sc0, db0 = state_at_tau(0.0)
pc_at_tca = node_pc_at_tca(pomdp,
    BeliefNode(belief_from_pomdp(pomdp, sc0, db0, 0.0),
               CAState(sc0, db0, 0.0), false))

# =========================================================================
# FIG 6b — Σ-at-TCA is ~branch-invariant across WAIT vs MANEUVER.
#
# Build a belief at a representative τ, take one WAIT vs one MANEUVER predict
# (Σ is propagated IDENTICALLY, only μ differs — the maneuver kicks the SC mean),
# grow each to TCA, and compare Σ-at-TCA (should match to many digits) vs the
# mean separation at TCA (should differ — the burn moved the mean). Do this at a
# few depths to show it holds along a branch, not just at one step.
#
# Use Δv = 5 m/s (as the end-to-end fixture) so a burn visibly moves the mean.
# =========================================================================
pomdp_mv = SpacecraftCAPOMDP(seed = 42, randAdd = false, Δv = 5.0)

"""Grow a belief to TCA, return (sc Σ pos 1σ, db Σ pos 1σ, sc-db mean sep at TCA, full Σ pos-1σ per-branch)."""
function branch_at_tca(b::Belief)
    μsc, Σsc = _grow_belief_to_tca(pomdp_mv, b.sc.μ, b.sc.Σ, pomdp_mv.satParams, b.t)
    μdb, Σdb = _grow_belief_to_tca(pomdp_mv, b.debris.μ, b.debris.Σ, pomdp_mv.debrisParams, b.t)
    sep = norm(μsc[1:3] .- μdb[1:3])
    return pos_sig(Σsc), pos_sig(Σdb), sep, Σsc, Σdb
end

# depths to probe (hours remaining at which we branch)
branch_depths_hr = [6.0, 4.0, 2.0]
b6 = let (sc, db) = state_at_tau(6.0 * 3600.0)
    belief_from_pomdp(pomdp_mv, sc, db, 6.0 * 3600.0)
end

# Walk down a WAIT trunk, and at each probed depth branch off one WAIT vs one
# MANEUVER predict, grow BOTH to TCA, compare. Wrapped in a function so the
# trunk belief `b` is a proper local (avoids soft-scope shadowing of globals).
function branch_invariance_sweep(b_start::Belief, depths_hr)
    sig_sc_wait = Float64[]; sig_db_wait = Float64[]; sep_wait = Float64[]
    sig_sc_mvr  = Float64[]; sig_db_mvr  = Float64[]; sep_mvr  = Float64[]
    relΣ_sc = Float64[]; relΣ_db = Float64[]
    b = b_start
    for th in depths_hr
        # advance the trunk (WAIT predict) to reach th if needed
        while b.t / 3600.0 > th + 1e-6
            b = predict(pomdp_mv, b, WAIT; dt = 3600.0)
        end
        bw = predict(pomdp_mv, b, WAIT;     dt = 3600.0)
        bm = predict(pomdp_mv, b, MANEUVER; dt = 3600.0)
        sw_sc, sw_db, sepw, Σw_sc, Σw_db = branch_at_tca(bw)
        sm_sc, sm_db, sepm, Σm_sc, Σm_db = branch_at_tca(bm)
        push!(sig_sc_wait, sw_sc); push!(sig_db_wait, sw_db); push!(sep_wait, sepw)
        push!(sig_sc_mvr,  sm_sc); push!(sig_db_mvr,  sm_db); push!(sep_mvr,  sepm)
        # max relative Σ-at-TCA difference (full matrix) WAIT vs MANEUVER
        push!(relΣ_sc, maximum(abs.(Σw_sc .- Σm_sc)) / max(maximum(abs.(Σw_sc)), eps()))
        push!(relΣ_db, maximum(abs.(Σw_db .- Σm_db)) / max(maximum(abs.(Σw_db)), eps()))
        b = bw   # continue the trunk along WAIT
    end
    return (sig_sc_wait, sig_db_wait, sep_wait,
            sig_sc_mvr, sig_db_mvr, sep_mvr, relΣ_sc, relΣ_db)
end

sig_sc_wait, sig_db_wait, sep_wait, sig_sc_mvr, sig_db_mvr, sep_mvr, relΣ_sc, relΣ_db =
    branch_invariance_sweep(b6, branch_depths_hr)

# =========================================================================
data = Dict(
    "orbit_period_min" => orbit_period_min,
    # Fig 6a
    "tau_coarse_hr" => tau_coarse_hr, "pc_coarse" => pc_coarse,
    "tau_fine_hr" => tau_fine_hr, "pc_fine" => pc_fine,
    "sig_sc_fine" => sig_sc_fine, "sig_db_fine" => sig_db_fine,
    "pc_at_tca" => pc_at_tca,
    "pc_threshold" => pomdp.pc_threshold,
    "miss_m" => 500.0, "v_rel" => 15.0,
    # Fig 6b
    "branch_depths_hr" => branch_depths_hr,
    "sig_sc_wait" => sig_sc_wait, "sig_db_wait" => sig_db_wait, "sep_wait" => sep_wait,
    "sig_sc_mvr"  => sig_sc_mvr,  "sig_db_mvr"  => sig_db_mvr,  "sep_mvr"  => sep_mvr,
    "relSigma_sc" => relΣ_sc, "relSigma_db" => relΣ_db,
    "dv_ms" => pomdp_mv.Δv,
)

out = joinpath(@__DIR__, "phase6_validation_data.json")
open(out, "w") do io; write(io, _json(data)); end
println("wrote $out")
println("Fig 6a: Pc at TCA (no growth) = ", pc_at_tca,
        " ; Pc @2h remaining = ", pc_coarse[end-1])
println("Fig 6a: coarse hour-to-hour Pc swing factor (max/min over last 6 h) = ",
        round(maximum(pc_coarse[end-5:end]) / minimum(pc_coarse[end-5:end]); digits = 1))
println("Fig 6b: max relative Σ-at-TCA diff WAIT vs MANEUVER (sc / debris) = ",
        maximum(relΣ_sc), " / ", maximum(relΣ_db))
println("Fig 6b: mean separation at TCA WAIT vs MANEUVER (last depth) = ",
        round(sep_wait[end]; digits = 1), " m / ", round(sep_mvr[end]; digits = 1), " m")
