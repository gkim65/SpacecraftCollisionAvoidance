# =========================================================================
# cdmScenario.jl — turn a parsed NASA CARA CDM into a POMDP scenario the
# belief-space MCTS planner can run on (pipeline assembly, step 1).
#
# GOAL: run the planner on a REAL NASA conjunction instead of a synthetic
# fixture. This is the loader half — it produces a `SpacecraftCAPOMDP` + an
# initial `Belief` + a true `CAState` + a horizon, all sourced from one CDM.
#
# DESIGN (agreed with Grace — this session is deliberately UNBLOCKED and does
# NOT do detection→TCA back-propagation; that is a later "covariance fix"
# session):
#
#   • NEAR-TCA scenario. The CDM's TCA covariance is used DIRECTLY as the belief
#     (P0_sc = cov1_eci, P0_debris = cov2_eci). This is operationally defensible:
#     an operator decides using the latest CDM's covariance, and each vendored
#     CDM is a single TCA snapshot (see notes/cara_cdm_deepdive_findings.md §a).
#     There is NO Σ(τ) growth model and NO seeding back to detection here — the
#     CDM covariance IS the belief at the root.
#
#   • HORIZON = the CDM's own lead time (CREATION_DATE → TCA), i.e. how far ahead
#     of TCA this CDM actually existed. That is the honest "how far ahead is it"
#     answer for a single-snapshot CDM (the deepdive found no per-event time
#     series). For the JILIN payload-payload case that is ~33.1 h.
#
#   • REAL HBR preserved from the CDM (COMMENT HBR). The combined hard-body radius
#     is the only radius the literature / the Pc integral uses (CCSDS CDMs carry a
#     single combined HBR; Chan/Foster/Elrod integrate over one disk of that
#     combined radius). We therefore put the whole CDM HBR on R_hard_body_sc and
#     0 on R_hard_body_debris — the sum (all that any Pc call uses) is exactly the
#     CDM's HBR, with no invented per-object decomposition.
#
#   • SECONDARY CLASS tagged via `classify_secondary` (debris / rocket_body /
#     payload / unknown) for later stratification.
#
#   • VALIDITY FLAG from the 53/53-exact usage-violation detector
#     (usageViolationCurvilinear.jl). FLAG ONLY this session — margining / handling
#     of a flagged case is a later decision (audit B3).
#
# SCOPE: get ONE clean PAYLOAD-VS-PAYLOAD case loadable + runnable. Not a sweep,
# not all 53, not baselines.
# =========================================================================

using Dates

# The CDM parser + the usage-violation detector live outside the main module's
# include chain, so pull them in here (idempotent — a second include is a no-op
# to the definitions). `parse_cdm` / `classify_secondary` come from cdmParser.jl;
# `usage_violation_pc2d_curvilinear` / `CurvilinearUVResult` from the curvilinear
# tier (which itself pulls in the rectilinear tier + constants).
if !isdefined(@__MODULE__, :parse_cdm)
    include(joinpath(@__DIR__, "..", "tests", "cdmParser.jl"))
end
if !isdefined(@__MODULE__, :usage_violation_pc2d_curvilinear)
    include(joinpath(@__DIR__, "usageViolationCurvilinear.jl"))
end

"""
    _parse_cdm_epoch(s::AbstractString) -> (y, mo, d, h, mi, sec_float, ns)

Parse a CCSDS CDM ISO-8601 UTC timestamp (`YYYY-MM-DDThh:mm:ss.sss`) into the
7-tuple `SpacecraftCAPOMDP.epochTCA` / brahe's `from_datetime` expect: integer
year/month/day/hour/minute, a Float64 second (fractional seconds folded in), and
a Float64 nanosecond field (kept 0.0 — the fractional second already carries
sub-second precision). Both `TCA` and `CREATION_DATE` use this format.
"""
function _parse_cdm_epoch(s::AbstractString)
    m = match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", strip(s))
    m === nothing && error("Unparseable CDM timestamp: $(repr(s))")
    y  = parse(Int, m.captures[1]); mo = parse(Int, m.captures[2])
    d  = parse(Int, m.captures[3]); h  = parse(Int, m.captures[4])
    mi = parse(Int, m.captures[5]); sec = parse(Float64, m.captures[6])
    return (y, mo, d, h, mi, sec, 0.0)
end

"""
    _epoch_seconds(t) -> Float64

Absolute UTC seconds (from an arbitrary fixed calendar origin) for a
`(y, mo, d, h, mi, sec, ns)` epoch tuple, so the lead-time subtraction below is
calendar-correct across day / month / year boundaries (the JILIN case straddles a
day boundary). Uses `Dates` for the whole-second calendar arithmetic and folds
the fractional second back in — CDM `TCA` / `CREATION_DATE` are plain UTC
calendar timestamps, so no leap-second / time-system handling is needed for a
difference of two such stamps.
"""
function _epoch_seconds(t::Tuple)
    whole = floor(Int, t[6])
    frac  = t[6] - whole
    dt = Dates.DateTime(t[1], t[2], t[3], t[4], t[5], whole)
    return Dates.datetime2unix(dt) + frac
end

"""
    CDMScenario

A loaded CDM turned into everything the planner consumes, plus provenance for
the test / later stratification. Fields:

- `pomdp`      : `SpacecraftCAPOMDP` with the CDM's TCA epoch, the CDM covariances
                 as P0 (the belief), and the real combined HBR on `R_hard_body_sc`.
- `b0`         : initial `Belief` at time-remaining `t_horizon`. With
                 `backprop=true` (default) it is the DETECTION-epoch seed that
                 forward-grows to the CDM's TCA belief (covariance-fix, step 2b);
                 with `backprop=false` it anchors the CDM TCA covariance directly
                 (the old near-TCA design — kept for input-fidelity checks).
- `s_true`     : true `CAState` at `t_horizon`. Means are back-propagated to the
                 detection epoch when `backprop=true` (so the truth and the
                 belief share the same detection→TCA trajectory), else the CDM
                 TCA states directly.
- `b_tca`      : the CDM's TCA-anchored belief (time-remaining 0.5 s) — the real
                 endpoint the back-prop seed is built to reproduce; used by tests
                 to check elrod-at-TCA == CARA (the untouched anchor).
- `t_horizon`  : root time-to-TCA (s) = the CDM's lead time (creation → TCA).
- `hbr`        : combined hard-body radius (m), straight from the CDM.
- `sec_class`  : `classify_secondary(name2)` — :debris / :rocket_body / :payload /
                 :unknown.
- `validity`   : the full `CurvilinearUVResult`; `valid` is its negated
                 `any_violation` (2D-Pc method holds for this geometry?).
- `valid`      : `!validity.any_violation` — convenience flag (flag only; no
                 handling this session).
- `pc_cdm`     : CARA's own operational COLLISION_PROBABILITY from the file.
- `name1`, `name2`, `id1`, `id2`, `miss_distance`, `relative_speed`, `tca` : from
                 the CDM, for labelling / sanity checks.
"""
struct CDMScenario
    pomdp::SpacecraftCAPOMDP
    b0::Belief
    s_true::CAState
    b_tca::Belief
    t_horizon::Float64
    hbr::Float64
    sec_class::Symbol
    validity::CurvilinearUVResult
    valid::Bool
    pc_cdm::Float64
    name1::String
    name2::String
    id1::String
    id2::String
    miss_distance::Float64
    relative_speed::Float64
    tca::String
end

"""
    load_cdm_scenario(path; dt=60*60, t_horizon=nothing, base_pomdp_kwargs...)
        -> CDMScenario

Load the CDM at `path` into a near-TCA POMDP scenario (see the module header for
the design). Uses the CDM's real TCA covariance as the belief, the real HBR, and
a horizon equal to the CDM's lead time (creation → TCA) unless `t_horizon` (s) is
given explicitly. `dt` is the planner/POMDP step (default 1 h).

Measurement noise R and cadence are CLASS-TIERED (sensorTiers.jl): the primary
(own asset) gets the GPS-grade isotropic R with a continuous cadence (= `dt`);
the secondary gets ITS class's R + cadence — SSN-radar anisotropic for
debris/rocket_body/unknown, GPS-grade for an active payload. `sensor_quality`
(`:best`/`:median`/`:worst`) selects the SSN radar grade percentile (a sweepable
measurement-quality knob). `sec_class_override` forces the secondary class
(else `classify_secondary(name2)`).

Any extra keyword overrides are forwarded to the `SpacecraftCAPOMDP` constructor
(splatted LAST, so they win over the tier defaults — e.g. `Δv`, `pc_threshold`,
`R_debris_mat`, `cadence_*`).

The returned `pomdp` has `randAdd = false` (the CDM fixes the geometry — no
element jitter), `forceModel = true`, the CDM's TCA epoch as `epochTCA`, and the
CDM covariances as `P0_sc` / `P0_debris`. The initial belief `b0` and true state
`s_true` are both anchored at the CDM ECI states at time-remaining `t_horizon`.
"""
function load_cdm_scenario(path::AbstractString;
                           dt::Real = 60 * 60,
                           t_horizon::Union{Real,Nothing} = nothing,
                           backprop::Bool = true,
                           sensor_quality::Symbol = :median,
                           sec_class_override::Union{Symbol,Nothing} = nothing,
                           base_pomdp_kwargs...)
    cdm = parse_cdm(path)
    cdm.hbr === nothing && error("CDM at $path has no COMMENT HBR — the real " *
        "hard-body radius must be present (design: do NOT fall back to a default).")
    hbr = Float64(cdm.hbr)

    tca_tuple = _parse_cdm_epoch(cdm.tca)

    # Horizon: the CDM's own lead time (creation → TCA), unless overridden.
    horizon = if t_horizon === nothing
        creation_tuple = _parse_cdm_epoch(cdm.creation_date)
        lead = _epoch_seconds(tca_tuple) - _epoch_seconds(creation_tuple)
        lead > 0 || error("CDM lead time (creation → TCA) is non-positive " *
            "($(round(lead, digits=1)) s) — cannot use it as a planning horizon.")
        lead
    else
        Float64(t_horizon)
    end

    # --- Class-tiered, SOURCED measurement noise R + cadence (sensorTiers.jl) ---
    # The secondary's R and cadence come from ITS object class (not a hardcoded
    # debris sensor). The PRIMARY is the own asset → GPS-grade with a CONTINUOUS
    # cadence (≈ every step). `sec_class_override` lets a caller force a class;
    # `sensor_quality` (:best/:median/:worst) sweeps the SSN radar grade.
    sec_class = sec_class_override === nothing ?
                classify_secondary(cdm.name2) : sec_class_override
    # R is a FIXED linear matrix rotated once at the CDM state (documented Option-A
    # approximation — the real per-pass anisotropy rotation is future work).
    R_sc_mat     = sensor_R_eci(:payload, cdm.state1)                       # own-asset GPS
    R_debris_mat = sensor_R_eci(sec_class, cdm.state2; quality = sensor_quality)
    cadence_sc_tier     = Float64(dt)                # own-asset GPS ≈ continuous → every step
    cadence_debris_tier = tier_cadence(sec_class)    # class ground-pass cadence

    # The combined HBR is the only radius any Pc call uses; put it all on the sc
    # side (0 on debris) so the SUM is exactly the CDM's HBR — no invented split.
    # Tier R + cadence are defaults here; base_pomdp_kwargs may still override them
    # (splatted last).
    pomdp = SpacecraftCAPOMDP(;
        epochTCA           = tca_tuple,
        randAdd            = false,
        forceModel         = true,
        P0_sc              = Matrix{Float64}(cdm.cov1_eci),
        P0_debris          = Matrix{Float64}(cdm.cov2_eci),
        R_sc_mat           = R_sc_mat,
        R_debris_mat       = R_debris_mat,
        cadence_sc         = cadence_sc_tier,
        cadence_debris     = cadence_debris_tier,
        R_hard_body_sc     = hbr,
        R_hard_body_debris = 0.0,
        dt                 = Float64(dt),
        base_pomdp_kwargs...)

    # The CDM's TCA-anchored belief (time-remaining ~0 s). This is the REAL
    # endpoint — elrod on it reproduces CARA's Pc (the untouched anchor). The
    # back-prop seed below is built to forward-grow back to exactly this.
    b_tca = belief_from_pomdp(pomdp, cdm.state1, cdm.state2, 0.5)

    if backprop
        # Detection-epoch seed: back-propagate the CDM TCA belief (mean + cov) so
        # that forward-growing it to TCA reproduces the CDM endpoint (covariance
        # fix, audit F2/F3 step 2b). Means AND covariances are back-propagated so
        # the truth and belief share the same detection→TCA trajectory; growing
        # the loader root FORWARD to TCA then lands on the conjunction geometry
        # (the old direct-anchor root grew the TCA geometry 33 h PAST TCA → Pc→0).
        b0 = backprop_belief_to_detection(pomdp,
                 collect(float.(cdm.state1)), Matrix{Float64}(cdm.cov1_eci),
                 collect(float.(cdm.state2)), Matrix{Float64}(cdm.cov2_eci), horizon)
        s_true = CAState(copy(b0.sc.μ), copy(b0.debris.μ), horizon)
    else
        # Old near-TCA design: anchor the CDM TCA covariance directly as the belief
        # at time-remaining `horizon` (no back-propagation). Kept for the loader's
        # input-fidelity checks; NOT physically the detection→TCA scenario.
        b0     = belief_from_pomdp(pomdp, cdm.state1, cdm.state2, horizon)
        s_true = CAState(collect(float.(cdm.state1)), collect(float.(cdm.state2)), horizon)
    end

    # Validity flag (flag only — no handling this session). Runs the 53/53-exact
    # usage-violation detector on the CDM's OWN states + covariances at TCA.
    validity = usage_violation_pc2d_curvilinear(
        cdm.state1[1:3], cdm.state1[4:6], cdm.cov1_eci,
        cdm.state2[1:3], cdm.state2[4:6], cdm.cov2_eci, hbr)

    return CDMScenario(pomdp, b0, s_true, b_tca, horizon, hbr,
                       sec_class, validity,
                       !validity.any_violation, cdm.pc_cdm,
                       cdm.name1, cdm.name2, cdm.id1, cdm.id2,
                       cdm.miss_distance, cdm.relative_speed, cdm.tca)
end
