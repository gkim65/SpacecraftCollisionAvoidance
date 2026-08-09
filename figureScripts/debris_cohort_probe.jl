# debris_cohort_probe.jl — classify all 53 CARA CDMs and report the DEBRIS-secondary
# cohort with lead time (creation → TCA) within ~30 h. This is the COHORT SELECTION
# probe for the debris WAIT-feasibility × cadence sweep (read-only; no POMDP/brahe).

using Dates, Printf
include(joinpath(@__DIR__, "..", "src", "tests", "cdmParser.jl"))

const CDM_DIR = normpath(joinpath(@__DIR__, "..", "data", "cara_cdms"))

# Parse a CCSDS ISO-8601 UTC stamp to absolute seconds (calendar-correct).
function epoch_seconds(s::AbstractString)
    m = match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", strip(s))
    m === nothing && error("Unparseable: $(repr(s))")
    y  = parse(Int, m.captures[1]); mo = parse(Int, m.captures[2])
    d  = parse(Int, m.captures[3]); h  = parse(Int, m.captures[4])
    mi = parse(Int, m.captures[5]); sec = parse(Float64, m.captures[6])
    whole = floor(Int, sec); frac = sec - whole
    return Dates.datetime2unix(Dates.DateTime(y, mo, d, h, mi, whole)) + frac
end

files = sort(filter(f -> endswith(f, ".cdm"), readdir(CDM_DIR)))
println("Total CDMs: ", length(files))
println()

rows = NamedTuple[]
for f in files
    cdm = parse_cdm(joinpath(CDM_DIR, f))
    lead_h = (epoch_seconds(cdm.tca) - epoch_seconds(cdm.creation_date)) / 3600
    cls = classify_secondary(cdm.name2)
    push!(rows, (file=f, cls=cls, lead_h=lead_h, name2=cdm.name2,
                 pc=cdm.pc_cdm, miss=cdm.miss_distance, relspd=cdm.relative_speed,
                 hbr=cdm.hbr))
end

# Class breakdown
println("Secondary class breakdown (all 53):")
for c in (:debris, :rocket_body, :payload, :unknown)
    n = count(r -> r.cls == c, rows)
    println(@sprintf("  %-12s %d", string(c), n))
end
println()

# Debris cohort within 30 h
debris = filter(r -> r.cls == :debris, rows)
println("DEBRIS secondaries: ", length(debris))
sort!(debris, by = r -> r.lead_h)
println(@sprintf("%-58s %8s %8s %10s %8s", "file", "lead_h", "miss_m", "pc", "relkm/s"))
for r in debris
    within = r.lead_h <= 30 ? " *" : ""
    println(@sprintf("%-58s %8.1f %8.0f %10.2e %8.2f%s",
        first(r.file, 58), r.lead_h, r.miss, r.pc, r.relspd/1000, within))
end
println()
cohort = filter(r -> r.lead_h <= 30, debris)
println("=> DEBRIS cohort with lead time <= 30 h: ", length(cohort), " cases (marked *)")
println("   lead-time range in cohort: ",
        @sprintf("%.1f – %.1f h", minimum(r->r.lead_h, cohort), maximum(r->r.lead_h, cohort)))
