# =========================================================================
# baselines.jl — F3 baseline DECISION POLICIES (paper_readiness_audit.md
# "SEQUENCE UPDATE 2026-08-09 night", baseline set — REVISED with Grace 2026-08-09).
#
# Each baseline is a STANDALONE, importable POLICY function with the SAME contract
# the MCTS default (`mcts_policy`, beliefExecutor.jl) obeys:
#
#     policy(pomdp, root, rng; planner, t_remaining, grid, pc_threshold,
#            delta_v, step) -> CAAction
#
# so it drops straight into the ONE decision point of `run_episode` (the `plan`
# call it replaces). Slotting into the same loop means every baseline gets the
# IDENTICAL belief update (incl. the noisy sensor drift), decision grid, metrics
# dict, observation draws, and wandb path as the planner — the comparison isolates
# the DECISION RULE and nothing else. That fairness is the whole point.
#
# THE BASELINE SET (Grace, 2026-08-09 — after the omniscient-bound framing collapsed):
#
#   There is NO fixed ground-truth covariance in this problem — uncertainty IS the
#   measurements. A bound that freezes Σ at the CDM-at-TCA value asks an artificial
#   question, and a measured-Σ bound just reproduces the WAIT-spine. So the omniscient
#   single-burn bound was DROPPED. The honest baselines are two DECISION RULES:
#
#   (1) DELAY-CLOCK gate — `delay_gate_policy(T_act)`.  A single fixed-threshold
#       family (δ = pc_threshold = 1e-5 for every sweep point). The ONLY knob is a
#       WAIT clock: force-defer until time-to-TCA < T_act, THEN maneuver iff the
#       belief Pc exceeds δ. Swept over T_act (act-immediately … act-late). This is
#       the naive "screen on a threshold, but on some clock" policy — burns too early
#       (wasted Δv) at large T_act, risks the fixable window at small T_act. Reads Pc
#       off the SAME (possibly drifted) belief MCTS sees. Latches to a SINGLE burn.
#
#   (2) WAIT-FEASIBILITY oracle — `wait_feasibility_policy()`.  The "optimization"
#       baseline: it mirrors the FIRST thing MCTS asks — will measurements shrink Σ
#       enough to drive Pc < δ before TCA? It computes the full-horizon WAIT+measure
#       spine (`wait_spine_pc`) from the CURRENT belief to TCA and DEFERS while that
#       spine goes durably < δ; if the spine NEVER drops below δ it MANEUVERs now.
#       IDEALIZED: the spine uses ZERO-INNOVATION measurements (z = μ⁻) so Σ only
#       SHRINKS and the mean stays pinned to truth (NO drift) — the clean feasibility
#       "true answer per measurement scenario", NOT what the noisy random-measurement
#       MCTS/executor belief sees. Frame it honestly as a near-oracle upper reference
#       (it sees the no-drift future MCTS cannot). This is exactly the curve the
#       debris_wait_cadence figures plot.
#
# THE CRUX OF FAIRNESS: the DELAY gate reads Pc off the SAME (possibly noisy /
# drifted) belief the MCTS planner sees at that step — `node_pc_at_tca(pomdp, root)`
# — never the truth. The WAIT-FEASIBILITY oracle DELIBERATELY reads the clean spine
# instead (that is the point — it is the idealized reference), and is labeled as such.
#
# Both gate factories take their policy PARAMETER and return a policy closure bound
# to it, so a sweep varies the param by building a fresh policy. The returned
# closures ignore `planner` (no MCTS).
# =========================================================================

"""
    delay_gate_policy(T_act; pc_threshold=nothing) -> policy

**Baseline (1): DELAY-CLOCK gate.** Returns a SINGLE-burn policy with a fixed Pc
threshold `δ` (= `pomdp.pc_threshold`, i.e. 1e-5 — the same chance constraint for
every sweep point) whose ONLY knob is the act-time `T_act` (s):

    if already maneuvered once            ⇒ WAIT       (latch: single burn)
    elseif t_remaining ≥ T_act            ⇒ WAIT       (forced defer — clock not reached)
    elseif node_pc_at_tca(root) > δ       ⇒ MANEUVER   (in the act window AND unsafe)
    else                                  ⇒ WAIT

`T_act` is the policy parameter the sweep varies (e.g. {28, 24, …, 3 h}); `T_act ≥
horizon` = act-on-first-sight (maneuver as soon as belief-Pc>δ), small `T_act` =
forced late action. This collapses the earlier θ-gate + timing-gate into ONE family
(threshold fixed, only the clock moves). It reads Pc off `root`'s CURRENT belief
(the same possibly-drifted belief MCTS sees), never the truth. It LATCHES: after
the first burn it always WAITs, so it is a single-burn policy (honest Δv). Whether
that single burn actually MITIGATED is checked downstream in the metrics dict
(`maneuver_mitigated`) regardless of the latch — Grace's ask.

`pc_threshold` override pins δ explicitly; `nothing` (default) reads
`pomdp.pc_threshold` at decision time.
"""
function delay_gate_policy(T_act::Real; pc_threshold::Union{Real,Nothing} = nothing)
    Tf   = Float64(T_act)
    thrp = pc_threshold === nothing ? nothing : Float64(pc_threshold)
    burned = Ref(false)                     # latch state (this policy instance)
    return function delay_gate(pomdp::SpacecraftCAPOMDP, root::BeliefNode,
                               rng::AbstractRNG;
                               planner = nothing, t_remaining::Real = root.belief.t,
                               grid = nothing,
                               pc_threshold::Real = pomdp.pc_threshold,
                               delta_v::Real = pomdp.Δv, step::Int = 0, kwargs...)
        burned[] && return WAIT              # latch: one burn per episode
        Float64(t_remaining) >= Tf && return WAIT   # forced defer — clock not reached
        thr = thrp === nothing ? Float64(pc_threshold) : thrp
        if node_pc_at_tca(pomdp, root) > thr
            burned[] = true
            return MANEUVER
        end
        return WAIT
    end
end

"""
    wait_feasibility_policy(; pc_threshold=nothing, exclude_last=true) -> policy

**Baseline (2): WAIT-FEASIBILITY oracle (IDEALIZED, precompute-once fixed plan).**
Mirrors the first question MCTS asks — *will continuing to WAIT + measure drive Pc
below `δ` before TCA?* — but answered from the CLEAN, drift-free feasibility truth
and executed as a FIXED schedule.

WHY PRECOMPUTE-ONCE (Grace, 2026-08-09): the clean WAIT+measure spine does NOT
change over the episode — it is a property of the case + cadence + sensor precision,
not of the realized (noisy) belief. `wait_spine_pc` uses ZERO-INNOVATION
measurements (z = μ⁻): the correction only SHRINKS Σ and the mean stays pinned to
truth (NO drift). So the oracle computes the spine ONCE from the root belief (which
at step 1 IS the clean detection-epoch b0), extracts the ideal schedule, and follows
it — it does NOT re-read the executor's DRIFTED belief on later steps (that drift is
exactly the noise this idealized reference is meant to be free of). The ideal plan:

    clean spine durably < δ (last trusted epoch)  ⇒ DEFER the whole episode (never burn)
    else                                          ⇒ MANEUVER at the LAST feasible epoch
                                                    (or the first step if never feasible —
                                                     Grace: "if pc never goes down, maneuver")

This is a NEAR-ORACLE upper reference: it sees the no-drift measurement future that
the noisy MCTS/executor belief cannot. Report it as IDEALIZED, not an achievable
online policy. Its comparability to MCTS: both run on the SAME true state; at good
tracking (:best) MCTS's belief ≈ truth so it should MATCH the oracle, and at poor
tracking (:median/:worst) MCTS's drifted belief makes it DIVERGE — the gap = the
cost of measurement noise. It LATCHES to a single burn; whether that burn actually
mitigated is checked downstream (`maneuver_mitigated`), regardless of the latch.

"Durably < δ" drops the very-last epoch by default (`exclude_last=true`, the
forward-growth endpoint artifact — same convention as the metrics dict + debris
findings). "Last feasible epoch" = the last epoch (nearest TCA) at which a burn
would still be initiated in the ideal plan; here, since a 0.1 m/s burn clears δ at
essentially any lead on well-conditioned cases (maneuver_effectiveness_findings), a
never-durably-safe case simply burns at the FIRST decision step (the earliest, most
conservative time), which the metrics' mitigation check then evaluates.
"""
function wait_feasibility_policy(; pc_threshold::Union{Real,Nothing} = nothing,
                                 exclude_last::Bool = true)
    thrp = pc_threshold === nothing ? nothing : Float64(pc_threshold)
    plan_ready = Ref(false)      # has the fixed plan been computed yet?
    plan_defer = Ref(true)       # DEFER the whole episode (clean spine durably safe)?
    burned     = Ref(false)      # single-burn latch (for the non-defer plan)
    return function wait_feas(pomdp::SpacecraftCAPOMDP, root::BeliefNode,
                              rng::AbstractRNG;
                              planner = nothing, t_remaining::Real = root.belief.t,
                              grid = nothing,
                              pc_threshold::Real = pomdp.pc_threshold,
                              delta_v::Real = pomdp.Δv, step::Int = 0, kwargs...)
        thr = thrp === nothing ? Float64(pc_threshold) : thrp

        # FIRST call: compute the clean spine ONCE from the (clean, un-drifted) root
        # belief and fix the ideal plan. All later calls follow the fixed plan and do
        # NOT re-read the executor's drifted belief.
        if !plan_ready[]
            epochs = grid === nothing ? Float64[Float64(t_remaining), 0.0] :
                     collect(Float64.(grid.t_epochs))
            spine  = wait_spine_pc(pomdp, root.belief, root.s_true, epochs)
            didx   = exclude_last ? max(1, length(spine) - 1) : length(spine)
            plan_defer[] = spine[didx] <= thr
            plan_ready[] = true
        end

        # DEFER plan: never burn.
        plan_defer[] && return WAIT
        # MANEUVER plan: single burn at the FIRST decision step, then WAIT (latch).
        # (The clean spine never gets durably safe → a burn is needed; the earliest
        # step is the most conservative feasible time on well-conditioned cases.)
        if !burned[]
            burned[] = true
            return MANEUVER
        end
        return WAIT
    end
end

"""
    make_policy(spec) -> policy

Build a policy from a serializable `spec` (a `Dict` / config sub-dict, so a wandb
sweep can name a policy + its parameter in the episode CONFIG). Recognized specs:

  - `Dict("kind"=>"mcts")`                          → the default MCTS planner policy
  - `Dict("kind"=>"delay_gate",  "T_act_h"=>T)`     → baseline (1), `delay_gate_policy(T·3600)`
    (`T_act_h` in HOURS for config readability; converted to seconds here.)
  - `Dict("kind"=>"wait_feasibility")`              → baseline (2), `wait_feasibility_policy()`

Returns `mcts_policy` for the mcts kind (or a `nothing`/absent spec) so the default
`run_episode` path is unchanged. Unknown kinds error loudly (a sweep typo should
not silently fall back to MCTS).
"""
function make_policy(spec::Union{AbstractDict,Nothing})
    spec === nothing && return mcts_policy
    kind = get(spec, "kind", "mcts")
    if kind == "mcts"
        return mcts_policy
    elseif kind == "delay_gate"
        haskey(spec, "T_act_h") || error("make_policy: delay_gate needs \"T_act_h\".")
        return delay_gate_policy(Float64(spec["T_act_h"]) * 3600)
    elseif kind == "wait_feasibility"
        return wait_feasibility_policy()
    else
        error("make_policy: unknown policy kind \"$kind\" " *
              "(expected \"mcts\", \"delay_gate\", or \"wait_feasibility\").")
    end
end

"""
    policy_variant_spec(variant) -> (policy_kind::String, policy_params::Union{Dict,Nothing})

Map a FLAT sweep-variant string to the `(policy, policy_params)` pair
`episode_config` takes. This is the single-axis encoding Grace wants for the wandb
sweep: ONE `policy_variant` axis holds every baseline (+ mcts) as a plain string, so
a grid sweep has no invalid combinations (no `mcts × T_act` cells to waste) and all
baselines live in ONE sweep. Recognized variants:

  - `"mcts"`                → the MCTS planner
  - `"wait_feasibility"`    → baseline (2), the idealized wait-feasibility oracle
  - `"delay_<N>h"`          → baseline (1), `delay_gate` with `T_act_h = N`
                              (e.g. `"delay_28h"`, `"delay_3h"`; N may be a decimal
                              like `"delay_1.5h"`).

A sweep sets `policy_variant`; `run_episode_entry.jl` calls this to fill `policy` /
`policy_params`. Unknown variants error loudly (a typo should not silently fall back
to mcts).
"""
function policy_variant_spec(variant::AbstractString)
    v = String(variant)
    if v == "mcts"
        return "mcts", nothing
    elseif v == "wait_feasibility"
        return "wait_feasibility", nothing
    elseif startswith(v, "delay_") && endswith(v, "h")
        num = v[(length("delay_") + 1):(end - 1)]        # between "delay_" and trailing "h"
        T = tryparse(Float64, num)
        T === nothing && error("policy_variant_spec: bad delay variant \"$v\" " *
                               "(expected \"delay_<hours>h\", e.g. \"delay_12h\").")
        return "delay_gate", Dict{String,Any}("T_act_h" => T)
    else
        error("policy_variant_spec: unknown policy_variant \"$v\" " *
              "(expected \"mcts\", \"wait_feasibility\", or \"delay_<hours>h\").")
    end
end
