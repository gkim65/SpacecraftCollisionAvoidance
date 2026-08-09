# =========================================================================
# sensorTiers.jl — class-tiered, SOURCED measurement-noise (R) model.
#
# WHY: the observation model was isotropic (R = σ²·I₆, one scalar on all six
# axes) for BOTH objects, and the CDM loader hardcoded the secondary to the
# DEBRIS grade regardless of its object class. Two realism gaps (audit C1):
#   1. A real single measurement is ~10× anisotropic (range-tight,
#      cross-range-loose), not isotropic.
#   2. A well-tracked ACTIVE PAYLOAD should not be measured with a debris/TLE
#      sensor — its class determines its sensor grade.
#
# FIX (Grace, locked 2026-08-09 — "Option A"): an ANISOTROPIC LINEAR R (H = I
# stays; the linear filter stays Elrod-Pc-compatible), with ONE sourced R per
# object CLASS, keyed off `classify_secondary`. The nonlinear az/el/range EKF is
# future work — it would add pass-geometry-dependent anisotropy ROTATION, which
# a fixed linear R cannot capture (stated below).
#
# TIERS:
#   • debris / rocket_body → SSN radar. R sourced from brahe 1.7.0's calibrated
#     SSN sensors (`ssn_sensors.load()` → `SimpleSSNSensor.from_locations_
#     calibrated`; 13 calibrated radar sites). Each site's radar measurement
#     model gives a real range σ and az/el σ. We reduce the 13 sites to a
#     grade (best / median / worst percentile) so measurement QUALITY is a
#     sweepable knob, and project to a fixed RTN-oriented anisotropic position R
#     (see `_ssn_radar_R_rtn`).
#   • payload (active) → GPS-grade, near-isotropic ~10 m (Hauschild & Montenbruck
#     NAVIGATION 2021).
#   • unknown → conservative fallback = debris grade.
#   • primary own-asset → GPS-grade (same R as payload) with a CONTINUOUS cadence
#     (≈ every step) — an own asset carries GNSS, not a discrete ground-contact.
#
# RTN-PROJECTION APPROXIMATION (documented): a single radar measurement is
# tight along the line of sight (range) and loose across it (angular × slant).
# We orient the tight axis to RADIAL and the two loose angular axes to
# along-track / cross-track — a representative near-overhead-pass proxy. The REAL
# anisotropy rotates with the object–station geometry every pass; a FIXED linear
# R does NOT capture that rotation. This is the deliberate Option-A simplification
# (the rotation is second-order for a planner paper); the nonlinear SSN model
# (future work) is what would restore it. The along-track DOMINANCE seen in real
# OD covariance is produced by the dynamics BETWEEN measurements (Q / STM
# propagation), NOT by this per-measurement R — so R is radial-tight, which is
# the sensor-honest shape.
#
# All magnitudes are SOURCED — see CONSTANTS.md "Measurement noise (class-tiered
# R)". Extraction script + numbers: this file's `SSN_RADAR_*` consts.
# =========================================================================

using LinearAlgebra

# --- Brahe calibrated SSN radar noise, reduced to grade percentiles ----------
# Extracted 2026-08-09 from brahe 1.7.0:
#   locs    = brahe.ssn_sensors.load()
#   sensors = brahe.SimpleSSNSensor.from_locations_calibrated(locs)  # 16 sites
#   radar   = [s for s in sensors if s.sensor_type == AzElRange]      # 13 sites
#   σ       = sqrt.(diag(s.measurement_model().noise_covariance()))   # [az°, el°, range m]
# Angles are DEGREES (verified: `SimpleSSNSensor.measure` returns [az_deg, el_deg,
# range_m]). Range σ across the 13 radar sites: 2.9–163 m; combined az/el σ:
# 0.009–0.079°. Grade = percentile of (range σ, angular σ) taken independently
# (per-site range and angular σ do NOT correlate — e.g. Shemya has tiny range but
# large angular — so a percentile-of-each "grade" is cleaner than any single site).
#
#   grade    range σ (m)   angular σ (deg)
#   :best      26.0          0.0115          (10th pct — a well-calibrated site)
#   :median    50.0          0.0224          (50th pct — representative)
#   :worst    140.3          0.0477          (90th pct — a poorly-calibrated site)
const SSN_RADAR_RANGE_SIGMA_M = Dict(:best => 26.0, :median => 50.0, :worst => 140.3)
const SSN_RADAR_ANGULAR_SIGMA_DEG = Dict(:best => 0.0115, :median => 0.0224, :worst => 0.0477)

# Representative LEO tracked-pass slant range (m) used to turn an angular σ (rad)
# into a cross-range POSITION σ: σ_cross = angular_σ · slant. 1200 km ≈ a target
# at ~550 km altitude seen at ~30–45° elevation (a typical mid-pass tracking
# geometry). Documented representative value — the real slant sweeps ~550 km
# (overhead) to ~2500 km (horizon) across a pass, which the fixed R averages out.
const SSN_SLANT_RANGE_M = 1200.0e3

# Debris/RB velocity 1σ (m/s), one axis, isotropic. SSN radar range-rate is not
# in brahe's calibrated set; keep the conservative SGP4-OD near-epoch value used
# for P0_debris (CONSTANTS.md): ~1–3 cm/s radial growing over days → 0.1 m/s
# covers degradation. Applied on all three velocity axes.
const SSN_VEL_SIGMA_MS = 0.1

# --- GPS (own-asset / active-payload) ----------------------------------------
# Near-isotropic state-level GNSS fix. Real-time LEO onboard positioning is
# ~0.5–1 m (3D) — Hauschild & Montenbruck, NAVIGATION 68(2), 2021. 10 m is a
# CONSERVATIVE upper bound (defensible as-is; could tighten to ~1 m).
const GPS_POS_SIGMA_M = 10.0
const GPS_VEL_SIGMA_MS = 0.01

"""
    _rtn_to_eci_R(R_rtn, state) -> 6×6

Rotate a 6×6 RTN position/velocity noise covariance into ECI, using the RTN
basis at `state` (a 6-vector ECI [r; v], m & m/s). Block-diagonal similarity
transform (position and velocity blocks share the instantaneous RTN→ECI
rotation) — the same rotation-only convention `cdmParser.rtn_to_eci_cov` uses
for CDM covariances. Reproduced here (not shared) so this file does not depend
on cdmParser's include order.
"""
function _rtn_to_eci_R(R_rtn::AbstractMatrix, state::AbstractVector)
    r = state[1:3]
    v = state[4:6]
    r_hat = r ./ norm(r)
    h = cross(r, v)
    n_hat = h ./ norm(h)
    t_hat = cross(n_hat, r_hat)
    Rot = hcat(r_hat, t_hat, n_hat)   # columns: R, T, N (RTN→ECI)
    A = zeros(6, 6)
    A[1:3, 1:3] = Rot
    A[4:6, 4:6] = Rot
    return A * Matrix(R_rtn) * transpose(A)
end

"""
    ssn_radar_R_rtn(; quality=:median) -> 6×6 (RTN frame)

The debris / rocket-body single-measurement position/velocity noise R in the
RTN frame, sourced from brahe's calibrated SSN radar (see the file header).
RADIAL is the tight (range) axis; along-track and cross-track are the loose
(angular × slant) axes; velocity is isotropic `SSN_VEL_SIGMA_MS`. `quality` ∈
`(:best, :median, :worst)` selects the grade percentile (a sweepable
measurement-quality knob, Grace's ask).
"""
function ssn_radar_R_rtn(; quality::Symbol = :median)
    haskey(SSN_RADAR_RANGE_SIGMA_M, quality) ||
        error("Unknown SSN radar quality $(repr(quality)); expected :best/:median/:worst.")
    σ_radial = SSN_RADAR_RANGE_SIGMA_M[quality]                       # m, range → radial (tight)
    σ_cross  = deg2rad(SSN_RADAR_ANGULAR_SIGMA_DEG[quality]) * SSN_SLANT_RANGE_M  # m, angular → cross-range (loose)
    return diagm([σ_radial^2, σ_cross^2, σ_cross^2,
                  SSN_VEL_SIGMA_MS^2, SSN_VEL_SIGMA_MS^2, SSN_VEL_SIGMA_MS^2])
end

"""
    gps_R() -> 6×6 (frame-independent; isotropic)

The GPS-grade (own-asset / active-payload) near-isotropic R: `GPS_POS_SIGMA_M`
on the three position axes, `GPS_VEL_SIGMA_MS` on the three velocity axes. Being
isotropic in position it is rotation-invariant, so it needs no state.
"""
gps_R() = diagm([GPS_POS_SIGMA_M^2, GPS_POS_SIGMA_M^2, GPS_POS_SIGMA_M^2,
                 GPS_VEL_SIGMA_MS^2, GPS_VEL_SIGMA_MS^2, GPS_VEL_SIGMA_MS^2])

"""
    sensor_R_eci(class, state; quality=:median) -> 6×6 (ECI frame)

The measurement-noise R for an object of `class` (`:debris`, `:rocket_body`,
`:payload`, `:unknown`), expressed in ECI at `state` (ECI 6-vector, used only to
rotate the anisotropic radar R from RTN → ECI once; GPS is isotropic so `state`
is ignored for payloads). `quality` selects the SSN radar grade for the
radar-tracked classes.

Tiers:
  • `:debris`, `:rocket_body` → `ssn_radar_R_rtn(quality)` rotated to ECI.
  • `:payload`               → `gps_R()` (GPS-grade, isotropic).
  • `:unknown`               → conservative fallback = debris grade.

This is a FIXED linear R (rotated once at the reference `state`); the real
per-pass anisotropy rotation is future work (nonlinear SSN) — see the header.
"""
function sensor_R_eci(class::Symbol, state::AbstractVector; quality::Symbol = :median)
    if class === :payload
        return gps_R()
    elseif class === :debris || class === :rocket_body || class === :unknown
        return _rtn_to_eci_R(ssn_radar_R_rtn(quality = quality), state)
    else
        error("Unknown object class $(repr(class)); expected " *
              ":debris/:rocket_body/:payload/:unknown.")
    end
end

"""
    tier_cadence(class) -> Float64  (seconds)

Representative measurement cadence for an object of `class`. Radar-tracked
classes refresh on the SSN/TLE ground-pass cadence (~8 h, CONSTANTS.md); an
active payload is assumed cooperatively tracked more often (~2 h GPS
operator-contact/downlink cadence). The PRIMARY own-asset uses a continuous
cadence set separately by the loader (≈ every step). Swappable per call.
"""
function tier_cadence(class::Symbol)
    if class === :payload
        return 2 * 60 * 60.0          # 2 h — cooperative / GPS-downlink cadence
    else                               # debris / rocket_body / unknown
        return 8 * 60 * 60.0          # 8 h — SSN/TLE ground-pass cadence
    end
end
