# debris_wait_cadence_data.jl — DEBRIS WAIT-feasibility × cadence sweep (belief-level,
# NO MCTS). For each real debris-secondary CARA CDM in the operational window (lead
# ≤ 30 h), walk a WAIT-only belief (predict → cadence-due correct → repeat) to TCA and
# record Pc-at-TCA (elrod, exact node_pc_at_tca) at every step. Sweep the debris
# measurement CADENCE and the SSN quality grade. Answers "how fresh must tracking be for
# a debris case to safely defer a burn?"
#
# Output: figureScripts/data/debris_wait_cadence.json
# Usage:  julia --project=. figureScripts/debris_wait_cadence_data.jl
#
# READ-ONLY: no src/ changes, no planner/reward code. Uses the loader's class-generic
# debris handling (P0 = real CDM cov2, R + cadence keyed off classify_secondary), the
# back-prop detection seed, and the EXACT Pc path.

using LinearAlgebra, Printf, Dates, Random, PyCall, POMDPs, POMDPTools, Distributions

include(joinpath(@__DIR__, "..", "src", "SpacecraftCAPOMDP.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "sensorTiers.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "genConjunctions.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "computePc.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "covarianceTable.jl"))
include(joinpath(@__DIR__, "..", "src", "states.jl"))
include(joinpath(@__DIR__, "..", "src", "actions.jl"))
include(joinpath(@__DIR__, "..", "src", "rewards.jl"))
include(joinpath(@__DIR__, "..", "src", "observations.jl"))
include(joinpath(@__DIR__, "..", "src", "transitions.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefTracker.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "beliefMCTS.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "cdmScenario.jl"))

const CDM_DIR = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms"))
const OUT     = joinpath(@__DIR__, "data", "debris_wait_cadence.json")

# --- COHORT: the 8 debris-secondary CDMs with lead ≤ 30 h (debris_cohort_probe.jl) ---
# ordered by lead time (21.8 h … 27.7 h).
const COHORT_ALL = [
    "000040059_conj_000035921_20220326_194122_20220325_215435.cdm",
    "000025994_conj_000037558_20210324_151047_20210323_154356.cdm",
    "000038771_conj_000030802_20201216_182131_20201215_171306.cdm",
    "000029108_conj_000034995_20220706_165058_20220705_143113.cdm",
    "000037849_conj_000013512_20210612_084905_20210611_062043.cdm",
    "000033591_conj_000042216_20211203_183431_20211202_153618.cdm",
    "000028654_conj_000041835_20220106_193032_20220105_161142.cdm",
    "000040115_conj_000030660_20230721_100115_20230720_061903.cdm",
]
# DWC_NCASES limits how many cohort cases to run (default 1 → validate one case first,
# then bump to 8 once timing looks right). No code edit to scale up.
const NCASES = parse(Int, get(ENV, "DWC_NCASES", "1"))
const COHORT = COHORT_ALL[1:clamp(NCASES, 1, length(COHORT_ALL))]

const CADENCES_H = [2.0, 4.0, 8.0, 24.0]          # debris measurement cadence sweep (8 h = default)
const QUALITIES  = [:best, :median, :worst]        # SSN radar grade sweep
const DT_H       = 1.0                              # belief-walk step (h) — divides every cadence

# σ (m) of the position block of a 6×6 covariance, along principal axes.
_pos_sigmas(M) = round.(sqrt.(eigvals(Symmetric(Matrix(M)[1:3, 1:3]))), digits=1)

# Is a covariance a valid (finite, symmetric, PD) belief covariance?
function _cov_ok(M)
    A = Matrix(M)
    all(isfinite, A) || return false
    ev = eigvals(Symmetric(A))
    return minimum(ev) > 0
end

# Stop evaluating Pc within the last hour of TCA. `node_pc_at_tca` grows the belief
# forward to TCA; if the walk steps to/past TCA it reports a spurious Pc→0 endpoint (means
# crossed; NOT a physical crossing — diagnostic debris_wait_pc_diag.jl, Grace's catch).
# The last hour is also physically untrustworthy (along-track impulsive model + CW ripple
# break down within a few orbits of TCA). So the walk records only down to TCA_STOP_S.
const TCA_STOP_S = 3600.0    # 1 h — closest-to-TCA epoch we record/trust

"""
Walk the WAIT spine toward TCA at step `dt`, correcting each object when its cadence is
due (zero-innovation, mean-tracks-truth so the correct only shrinks Σ). Records Pc-at-TCA
(exact, grown forward) at every belief epoch with t > TCA_STOP_S; STOPS before stepping to
t ≤ TCA_STOP_S so the forward-growth endpoint artifact (Pc→0 as means cross) is never
recorded. Returns per-step time-to-TCA (h), Pc-at-TCA, and cumulative debris-fix count.
`cadence_debris`/`cadence_sc` come from the loaded pomdp (overridden per call).
"""
function wait_pc_walk(pomdp, b0; dt_s)
    b = b0
    since_sc = 0.0; since_db = 0.0; nfix_db = 0
    ts_h = Float64[]; pcs = Float64[]; fixes = Int[]
    guard = 0
    while b.t > TCA_STOP_S && guard < 100_000
        guard += 1
        st = CAState(copy(b.sc.μ), copy(b.debris.μ), b.t)
        pc = node_pc_at_tca(pomdp, BeliefNode(b, st, false))
        push!(ts_h, b.t / 3600); push!(pcs, pc); push!(fixes, nfix_db)
        # Would the next predict step over TCA? If so, stop here (last trusted epoch).
        b.t - dt_s <= TCA_STOP_S && break
        b = predict(pomdp, b, WAIT; dt = dt_s)
        since_sc += dt_s; since_db += dt_s
        z = vcat(b.sc.μ, b.debris.μ)            # zero-innovation obs at the propagated mean
        if since_sc >= pomdp.cadence_sc
            b = correct_linear_sc(pomdp, b, z); since_sc = 0.0
        end
        if since_db >= pomdp.cadence_debris
            b = correct_linear_debris(pomdp, b, z); since_db = 0.0; nfix_db += 1
        end
    end
    return ts_h, pcs, fixes
end

# Hand-rolled JSON writer (project convention — JSON is only a transitive dep under
# --project=., so figureScripts serialize by hand). Handles nested Dict/Vector.
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::Real) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_j(x::Nothing) = "null"
_j(::Missing) = "null"
_j(x::Symbol) = _j(string(x))
_j(v::AbstractVector) = "[" * join(_j.(v), ",") * "]"
_j(d::AbstractDict) = "{" * join([_j(string(k)) * ":" * _j(v) for (k, v) in d], ",") * "}"

# Write the current results incrementally so partial output is usable mid-run.
function write_json(results, ncohort)
    meta = Dict{String,Any}(
        "generated_utc" => string(Dates.now(Dates.UTC)),
        "cohort_desc" => "debris-secondary CARA CDMs, lead (creation->TCA) <= 30 h",
        "n_cases_done" => length(results), "n_cases_planned" => ncohort,
        "cadences_h" => CADENCES_H, "qualities" => string.(QUALITIES),
        "dt_h" => DT_H, "pc_threshold" => 1e-5,
    )
    open(OUT, "w") do io
        write(io, _j(Dict("meta" => meta, "cases" => results)))
    end
end

println("="^78)
println("DEBRIS WAIT-feasibility × cadence sweep — ", length(COHORT), " cases × ",
        length(CADENCES_H), " cadences × ", length(QUALITIES), " grades")
println("cadences (h) = ", CADENCES_H, "   grades = ", QUALITIES, "   dt = ", DT_H, " h")
println("threshold = 1e-5 (pc_threshold)")
println("="^78)
flush(stdout)

results = Dict{String,Any}[]
t_start = time()

for (ci, fname) in enumerate(COHORT)
    path = joinpath(CDM_DIR, fname)
    println()
    println("-"^78)
    @printf("[case %d/%d] %s\n", ci, length(COHORT), first(fname, 60))
    flush(stdout)

    # Load once at :median just to read provenance (class, lead, miss, Pc, backprop health).
    sc0 = load_cdm_scenario(path; dt = DT_H*3600, sensor_quality = :median)
    lead_h = sc0.t_horizon / 3600
    pc_tca_anchor = node_pc_at_tca(sc0.pomdp, BeliefNode(sc0.b_tca, sc0.s_true, false))
    seed_sc_ok = _cov_ok(sc0.b0.sc.Σ)
    seed_db_ok = _cov_ok(sc0.b0.debris.Σ)
    seed_sc_sig = _pos_sigmas(sc0.b0.debris.Σ)   # debris seed σ (the anisotropic one to watch)
    @printf("  class=%s  lead=%.1f h  miss=%.0f m  CARA Pc=%.2e  anchor Pc(TCA)=%.2e\n",
            string(sc0.sec_class), lead_h, sc0.miss_distance, sc0.pc_cdm, pc_tca_anchor)
    @printf("  debris seed Σ pos σ = %s m  | seed PD? sc=%s db=%s\n",
            string(seed_sc_sig), seed_sc_ok, seed_db_ok)
    flush(stdout)

    case = Dict{String,Any}(
        "file" => fname,
        "class" => string(sc0.sec_class),
        "lead_h" => lead_h,
        "miss_m" => sc0.miss_distance,
        "relspeed_ms" => sc0.relative_speed,
        "hbr_m" => sc0.hbr,
        "pc_cara" => sc0.pc_cdm,
        "pc_tca_anchor" => pc_tca_anchor,
        "valid_2d" => sc0.valid,
        "seed_sc_pd" => seed_sc_ok,
        "seed_db_pd" => seed_db_ok,
        "seed_db_sigma_m" => collect(seed_sc_sig),
        "curves" => Dict{String,Any}(),
    )

    for q in QUALITIES
        for cad in CADENCES_H
            key = string(q, "_", Int(cad), "h")
            # Load with this quality + cadence override. b0 (back-prop seed) is
            # cadence-independent (q_rtn=0) and quality only rotates R_debris, which
            # does NOT enter back-prop → the seed is the SAME across the sweep, per
            # "same case, same seed, vary cadence".
            sc = load_cdm_scenario(path; dt = DT_H*3600, sensor_quality = q,
                                   cadence_debris = cad*3600)
            t0 = time()
            ts, pcs, fixes = wait_pc_walk(sc.pomdp, sc.b0; dt_s = DT_H*3600)
            thr = sc.pomdp.pc_threshold
            # THE QUESTION: as measurements come in, does Pc drop DURABLY below threshold
            # before the 1 h wall — i.e. do we find out we're OK and don't need to maneuver?
            # "Durable" = Pc stays below thr from some epoch all the way to the last recorded
            # (closest-to-TCA, ~1.8 h) point, robust to the cadence sawtooth / CW ripple
            # bumping Pc back up between fixes (a single dip is not "resolved").
            bel = pcs .< thr
            safe = bel[end]                          # below at the last trusted point?
            cross_t = nothing; fixes_at_cross = nothing
            if safe
                i = length(pcs)
                while i > 1 && bel[i-1]              # extend the durable-below run backward
                    i -= 1
                end
                cross_t = ts[i]                      # earliest epoch (largest t) of the run
                fixes_at_cross = fixes[i]            # # debris fixes landed by then
            end
            ever_below = any(bel)                    # dipped below at some point (may bump back)
            pc_min = minimum(pcs)
            case["curves"][key] = Dict(
                "quality" => string(q), "cadence_h" => cad,
                "t_h" => ts, "pc" => pcs, "fixes_db" => fixes,
                "safe" => safe,                      # durably below thr before the 1 h wall?
                "cross_time_h" => cross_t,           # when it becomes durably safe (h to TCA)
                "fixes_at_cross" => fixes_at_cross,  # # debris fixes needed to get there
                "ever_below" => ever_below,
                "pc_first" => pcs[1], "pc_final" => pcs[end], "pc_min" => pc_min,
                "n_db_fixes_total" => fixes[end],
            )
            elapsed = time() - t0
            cm = cross_t === nothing ? "no" : @sprintf("%.0fh/%df", cross_t, fixes_at_cross)
            @printf("    %-8s cad=%2dh : Pc %8.2e→%8.2e (min %8.2e)  safe=%-5s @ %-8s fixes=%d  (%.1fs)\n",
                    string(q), Int(cad), pcs[1], pcs[end], pc_min, safe, cm, fixes[end], elapsed)
            flush(stdout)
        end
    end
    push!(results, case)
    write_json(results, length(COHORT))         # incremental dump — partial output usable now
    @printf("  [case %d/%d done, JSON updated, total elapsed %.0fs]\n",
            ci, length(COHORT), time()-t_start)
    flush(stdout)
end

write_json(results, length(COHORT))             # final dump
println()
println("="^78)
@printf("WROTE %s  (%d cases, %.0fs total)\n", OUT, length(results), time()-t_start)
println("="^78)
