# cdmParser.jl — minimal CCSDS CDM reader for Pc validation
#
# Reads a NASA CARA CDM (Conjunction Data Message) and returns the two objects'
# ECI states and ECI position-velocity covariances at TCA, in the units
# `chan_pc` expects (metres, metres/second).
#
# CDM conventions (per CCSDS 508.0-B-1, as written by CARA):
#   - X, Y, Z, X_DOT, ... are in km and km/s, in the EME2000 (ECI) frame.
#   - The covariance block CR_R ... CNDOT_NDOT is the LOWER TRIANGLE of a 6x6
#     matrix expressed in the RTN (radial / in-track / cross-track) frame,
#     in m^2, m^2/s and m^2/s^2. It must be rotated into ECI before the two
#     objects' covariances can be summed, since each object has its own RTN
#     frame.
#   - `COMMENT HBR = <x> [m]` carries the combined hard-body radius.

using LinearAlgebra

# The 21 lower-triangular covariance keys, in CDM order. Index (i,j), i>=j.
const CDM_COV_KEYS = [
    ("CR_R", 1, 1),
    ("CT_R", 2, 1), ("CT_T", 2, 2),
    ("CN_R", 3, 1), ("CN_T", 3, 2), ("CN_N", 3, 3),
    ("CRDOT_R", 4, 1), ("CRDOT_T", 4, 2), ("CRDOT_N", 4, 3), ("CRDOT_RDOT", 4, 4),
    ("CTDOT_R", 5, 1), ("CTDOT_T", 5, 2), ("CTDOT_N", 5, 3), ("CTDOT_RDOT", 5, 4),
        ("CTDOT_TDOT", 5, 5),
    ("CNDOT_R", 6, 1), ("CNDOT_T", 6, 2), ("CNDOT_N", 6, 3), ("CNDOT_RDOT", 6, 4),
        ("CNDOT_TDOT", 6, 5), ("CNDOT_NDOT", 6, 6),
]

"""
    parse_cdm(path) -> NamedTuple

Parse a CDM file. Returns a NamedTuple with:
- `state1`, `state2` : 6-element ECI states at TCA (m, m/s)
- `cov1_eci`, `cov2_eci` : 6x6 ECI covariances (m^2, ...)
- `cov1_rtn`, `cov2_rtn` : 6x6 RTN covariances as given in the file
- `hbr` : combined hard-body radius (m), or `nothing` if absent
- `pc_cdm` : CARA's own operational COLLISION_PROBABILITY from the file
- `name1`, `name2`, `id1`, `id2`, `miss_distance`, `relative_speed`, `tca`
"""
function parse_cdm(path::AbstractString)
    # Each object's block repeats the same keys, so collect values per key in
    # file order and take [1] for OBJECT1, [2] for OBJECT2.
    vals = Dict{String,Vector{String}}()
    hbr = nothing

    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue

        # `COMMENT HBR = 15 [m]` — the only COMMENT we need.
        if startswith(line, "COMMENT")
            m = match(r"^COMMENT\s+HBR\s*=\s*([-\d.eE+]+)", line)
            m !== nothing && (hbr = parse(Float64, m.captures[1]))
            continue
        end

        parts = split(line, '=', limit = 2)
        length(parts) == 2 || continue
        key = strip(parts[1])
        # Strip the trailing unit annotation, e.g. "108 [m]" -> "108".
        val = strip(replace(parts[2], r"\[.*\]" => ""))
        push!(get!(vals, key, String[]), val)
    end

    getnum(key, idx) = parse(Float64, vals[key][idx])
    getstr(key, idx) = get(vals, key, [""])[min(idx, length(get(vals, key, [""])))]

    # --- states: km -> m, km/s -> m/s ---
    function state(idx)
        1000.0 .* [getnum("X", idx), getnum("Y", idx), getnum("Z", idx),
                   getnum("X_DOT", idx), getnum("Y_DOT", idx), getnum("Z_DOT", idx)]
    end

    # --- covariance: fill lower triangle, mirror to upper. Already in m^2. ---
    function cov_rtn(idx)
        C = zeros(6, 6)
        for (key, i, j) in CDM_COV_KEYS
            haskey(vals, key) || continue
            length(vals[key]) >= idx || continue
            C[i, j] = parse(Float64, vals[key][idx])
            C[j, i] = C[i, j]
        end
        return C
    end

    s1, s2 = state(1), state(2)
    C1_rtn, C2_rtn = cov_rtn(1), cov_rtn(2)

    pc_cdm = haskey(vals, "COLLISION_PROBABILITY") ?
        parse(Float64, vals["COLLISION_PROBABILITY"][1]) : NaN

    return (
        state1 = s1, state2 = s2,
        cov1_rtn = C1_rtn, cov2_rtn = C2_rtn,
        cov1_eci = rtn_to_eci_cov(C1_rtn, s1),
        cov2_eci = rtn_to_eci_cov(C2_rtn, s2),
        hbr = hbr,
        pc_cdm = pc_cdm,
        name1 = getstr("OBJECT_NAME", 1), name2 = getstr("OBJECT_NAME", 2),
        id1 = getstr("OBJECT_DESIGNATOR", 1), id2 = getstr("OBJECT_DESIGNATOR", 2),
        miss_distance = haskey(vals, "MISS_DISTANCE") ? getnum("MISS_DISTANCE", 1) : NaN,
        relative_speed = haskey(vals, "RELATIVE_SPEED") ? getnum("RELATIVE_SPEED", 1) : NaN,
        tca = getstr("TCA", 1),
        creation_date = getstr("CREATION_DATE", 1),
    )
end

"""
    rtn_rotation(state) -> 3x3

Rotation matrix whose COLUMNS are the RTN basis vectors expressed in ECI.
Maps an RTN vector to ECI: `v_eci = R * v_rtn`.

R = radial (along position), N = cross-track (along orbital angular momentum),
T = in-track (completes the right-handed set, N x R).
"""
function rtn_rotation(state::AbstractVector)
    r = state[1:3]
    v = state[4:6]

    r_hat = r ./ norm(r)
    h = cross(r, v)
    n_hat = h ./ norm(h)
    t_hat = cross(n_hat, r_hat)

    return hcat(r_hat, t_hat, n_hat)   # columns: R, T, N
end

"""
    rtn_to_eci_cov(C_rtn, state) -> 6x6

Rotate a 6x6 RTN position-velocity covariance into ECI. The position and
velocity 3x3 blocks are each rotated by the same instantaneous RTN->ECI
rotation (block-diagonal similarity transform).

Note: this is the standard rotation-only convention CARA uses for CDM
covariances. It ignores the frame's rotation rate, which is the correct
treatment here because the covariance is defined in the instantaneous RTN
axes at TCA, not in a rotating-frame sense.
"""
function rtn_to_eci_cov(C_rtn::AbstractMatrix, state::AbstractVector)
    R = rtn_rotation(state)
    A = zeros(6, 6)
    A[1:3, 1:3] = R
    A[4:6, 4:6] = R
    return A * C_rtn * transpose(A)
end

"""
    classify_secondary(name) -> Symbol

Classify the secondary object from its OBJECT_NAME. CDMs carry no OBJECT_TYPE
field, so the name string is the only in-file signal.

Returns `:debris`, `:rocket_body`, `:payload`, or `:unknown`.
"""
function classify_secondary(name::AbstractString)
    n = uppercase(strip(name))
    occursin(r"\bDEB\b|\bDEBRIS\b|DEB\s*\(", n) && return :debris
    occursin(r"R/B", n)                          && return :rocket_body
    (isempty(n) || n == "UNKNOWN")               && return :unknown
    return :payload
end
