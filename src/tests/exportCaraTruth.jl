# exportCaraTruth.jl — export the NASA CARA reference Pc table to CSV
#
# Reads `DataFiles/PcTestCaseCDMs/CARA_PcMethod_Test_Conjunctions.xlsx` from the
# NASA CARA Analysis Tools SDK and writes `cara_truth.csv` next to this file.
# That CSV is the reference table `test_cara_validation.jl` compares against.
#
# An .xlsx is a zip of XML parts, so this reads it with the system `unzip` and a
# small hand-rolled XML scan — no extra Julia package dependency for what is a
# one-off data export.
#
# Two XML parts matter:
#   xl/sharedStrings.xml — the string pool; cells with t="s" index into it
#   xl/worksheets/sheet1.xml — the cells themselves, as <c r="A1" t="s"><v>3</v></c>
#
# Usage:
#   julia --project=. src/tests/exportCaraTruth.jl [path/to/CARA_Analysis_Tools]

const DEFAULT_CARA_ROOT = joinpath(
    homedir(), "Documents", "School_Everything_and_LEARNING", "Stanford",
    "Githubs", "CARA_Analysis_Tools",
)

const XLSX_RELPATH = joinpath(
    "DataFiles", "PcTestCaseCDMs", "CARA_PcMethod_Test_Conjunctions.xlsx")

"Read one member of a zip archive as a String, via the system `unzip`."
function read_zip_member(archive::AbstractString, member::AbstractString)
    isfile(archive) || error("Archive not found: $archive")
    out = try
        read(`unzip -p $archive $member`, String)
    catch err
        error("Failed to extract '$member' from $archive.\n" *
              "Is `unzip` on PATH? Underlying error: $err")
    end
    isempty(out) && error("Empty or missing zip member: $member")
    return out
end

"Undo the five XML predefined entities. Sufficient for spreadsheet text."
function unescape_xml(s::AbstractString)
    s = replace(s, "&lt;" => "<", "&gt;" => ">", "&quot;" => "\"",
                   "&apos;" => "'", "&amp;" => "&")
    return s
end

"""
    parse_shared_strings(xml) -> Vector{String}

The shared-string pool. Each `<si>` is one string, but it may be split across
several `<t>` runs (rich text), so concatenate the runs within each `<si>`.
"""
function parse_shared_strings(xml::AbstractString)
    pool = String[]
    for si in eachmatch(r"<si\b[^>]*>(.*?)</si>"s, xml)
        buf = IOBuffer()
        for t in eachmatch(r"<t\b[^>]*>(.*?)</t>"s, si.captures[1])
            print(buf, t.captures[1])
        end
        push!(pool, unescape_xml(String(take!(buf))))
    end
    return pool
end

"Split a cell reference like \"AB12\" into (\"AB\", 12)."
function split_ref(ref::AbstractString)
    m = match(r"^([A-Z]+)(\d+)$", ref)
    m === nothing && return ("", 0)
    return (String(m.captures[1]), parse(Int, m.captures[2]))
end

"Column letters -> 1-based index. \"A\"->1, \"Z\"->26, \"AA\"->27."
function col_index(letters::AbstractString)
    n = 0
    for ch in letters
        n = 26n + (Int(ch) - Int('A') + 1)
    end
    return n
end

"""
    parse_sheet(xml, pool) -> Vector{Dict{Int,String}}

Parse worksheet cells into one Dict per row, keyed by 1-based column index.
Only inline numbers and shared strings appear in this workbook.
"""
function parse_sheet(xml::AbstractString, pool::Vector{String})
    rows = Vector{Dict{Int,String}}()
    for rowm in eachmatch(r"<row\b[^>]*>(.*?)</row>"s, xml)
        cells = Dict{Int,String}()
        for cm in eachmatch(r"<c\b([^>]*)>(.*?)</c>"s, rowm.captures[1])
            attrs, body = cm.captures[1], cm.captures[2]

            refm = match(r"r=\"([A-Z]+\d+)\"", attrs)
            refm === nothing && continue
            letters, _ = split_ref(refm.captures[1])
            isempty(letters) && continue

            vm = match(r"<v\b[^>]*>(.*?)</v>"s, body)
            vm === nothing && continue
            val = unescape_xml(vm.captures[1])

            # t="s" means the value is an index into the shared-string pool.
            tm = match(r"t=\"([^\"]+)\"", attrs)
            if tm !== nothing && tm.captures[1] == "s"
                idx = parse(Int, val) + 1   # xlsx is 0-based
                val = checkbounds(Bool, pool, idx) ? pool[idx] : ""
            end

            cells[col_index(letters)] = val
        end
        isempty(cells) || push!(rows, cells)
    end
    return rows
end

"Quote a CSV field only when it contains a comma, quote, or newline."
function csv_escape(s::AbstractString)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return '"' * replace(s, '"' => "\"\"") * '"'
    end
    return String(s)
end

function export_truth(cara_root::AbstractString = DEFAULT_CARA_ROOT;
                      outpath::AbstractString = joinpath(@__DIR__, "cara_truth.csv"))

    # Prefer the workbook vendored into this repo; fall back to an SDK checkout.
    vendored = normpath(joinpath(@__DIR__, "..", "..", "data", "cara_cdms",
                                 "CARA_PcMethod_Test_Conjunctions.xlsx"))
    xlsx = isfile(vendored) ? vendored : joinpath(cara_root, XLSX_RELPATH)
    isfile(xlsx) || error(
        "CARA reference workbook not found:\n  $xlsx\n" *
        "Pass the SDK root as the first argument, e.g.\n" *
        "  julia --project=. src/tests/exportCaraTruth.jl /path/to/CARA_Analysis_Tools")

    pool = parse_shared_strings(read_zip_member(xlsx, "xl/sharedStrings.xml"))
    rows = parse_sheet(read_zip_member(xlsx, "xl/worksheets/sheet1.xml"), pool)

    isempty(rows) && error("No rows parsed from worksheet.")

    header = rows[1]
    ncols = maximum(keys(header))
    colnames = [get(header, i, "col$i") for i in 1:ncols]

    open(outpath, "w") do io
        println(io, join(csv_escape.(colnames), ','))
        nwritten = 0
        for r in rows[2:end]
            # Skip blank rows and any trailing notes without a conjunction ID.
            isempty(get(r, 1, "")) && continue
            println(io, join((csv_escape(get(r, i, "")) for i in 1:ncols), ','))
            nwritten += 1
        end
        @info "Wrote CARA reference table" outpath rows=nwritten cols=ncols
    end

    return outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CARA_ROOT
    export_truth(root)
end
