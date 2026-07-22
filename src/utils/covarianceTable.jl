# =========================================================================
# covarianceTable.jl — Phase 3: precomputed covariance-vs-time-remaining table
#
# Offline sweep computing Σ(τ) for τ = dt, 2·dt, …, TCA_window remaining until
# TCA, for BOTH the spacecraft and the debris, starting from P0_sc / P0_debris
# (SpacecraftCAPOMDP.jl). Σ is propagated with the accurate numerical force model
# (drag/SRP) via Brahe's STM machinery (eci2orb_brahe with initial_covariance=,
# which enables .with_stm().with_stm_history()); the propagated covariance is
# read straight back with prop.covariance_gcrf(epoch) (ECI) and
# prop.covariance_rtn(epoch) (RTN). Phase 1 proved brahe's own covariance_gcrf
# equals the hand-computed Φ Σ₀ Φᵀ to 0.0 relative diff, so we read it directly
# rather than hand-rolling STM propagation.
#
# WHY THIS TABLE IS VALID (architecture doc §5): while maneuvers are noiseless,
# Σ's growth from any point forward to TCA depends only on HOW MUCH TIME REMAINS,
# not on which branch of the search tree you are in — so Σ(τ) is a pure
# lookup-by-time-remaining, precomputable once.
#
# VALIDITY CAVEAT (architecture doc §5 / §8): this lookup-by-time-remaining is
# ONLY valid under the noiseless-maneuver assumption. Once maneuver execution
# uncertainty is added (Phase 8), a maneuver injects real process noise into Σ,
# Σ then depends on the action sequence, and this table can no longer be a pure
# lookup — it must be tracked per-node instead. Revisit before Phase 8.
#
# TIMESTEP: the τ grid is parameterized by `dt` (defaults to pomdp.dt). The
# code's pomdp.dt (30 min) vs. the architecture doc's hourly steps is an open
# reconciliation (CONSTANTS.md timing row); we keep dt an explicit, swappable
# argument so both grids can be built and compared rather than baking one in.
# =========================================================================

using LinearAlgebra

# Numerical drift threshold: if brahe's returned covariance is asymmetric by
# more than this (relative to its largest entry), symmetrize Σ = (Σ+Σᵀ)/2 and
# flag it. Purely a numerical-hygiene tolerance, not a physical constant.
const COVTABLE_SYM_RTOL = 1e-9

"""
    _symmetrize_check(Σ; rtol=COVTABLE_SYM_RTOL) -> (Σ_sym, was_symmetrized, max_asym)

Enforce symmetry and check positive-definiteness of a propagated covariance.
`max_asym` is max|Σ-Σᵀ| relative to max|Σ|. If it exceeds `rtol`, Σ is replaced
by (Σ+Σᵀ)/2 and `was_symmetrized` is true. Always returns a symmetric matrix.
Positive-definiteness is checked (via Cholesky on the symmetrized matrix) and a
warning is emitted if it fails — we do not throw, so the sweep can report every
τ rather than aborting on the first drift.
"""
function _symmetrize_check(Σ::AbstractMatrix; rtol::Real = COVTABLE_SYM_RTOL)
    scale = maximum(abs.(Σ))
    max_asym = scale == 0.0 ? 0.0 : maximum(abs.(Σ .- transpose(Σ))) / scale
    # Always snap to exactly symmetric so downstream eigen/Cholesky are clean;
    # flag it only when the pre-symmetrization drift exceeded the tolerance.
    Σsym = (Matrix(Σ) .+ transpose(Matrix(Σ))) ./ 2
    was_symmetrized = max_asym > rtol

    is_pd = true
    try
        cholesky(Symmetric(Σsym))
    catch
        is_pd = false
    end
    if !is_pd
        @warn "Propagated covariance is not positive-definite after symmetrization" max_asym
    end
    return Σsym, was_symmetrized, max_asym, is_pd
end

"""
    build_covariance_table(pomdp, sc_eci, debris_eci;
                           dt=pomdp.dt, tca_window=nothing, verbose=true)
        -> NamedTuple

Offline sweep of Σ(τ) for τ = dt, 2·dt, …, tca_window remaining until TCA, for
BOTH objects, starting from `pomdp.P0_sc` / `pomdp.P0_debris`.

`sc_eci` / `debris_eci` are the two objects' ECI states **at TCA** (e.g. from
`generate_conjunction_geometry`). Each object is anchored with its initial
covariance at TCA and propagated forward by τ; because Σ growth is symmetric in
±time under the STM, propagating forward by τ from TCA gives the same covariance
magnitude as sitting τ before TCA — which is exactly Σ(τ remaining). (The mean
state's forward/back distinction does not matter here: the table stores only Σ,
which the noiseless-maneuver argument makes a pure function of |elapsed time|.)

Args:
- `dt`         : timestep (s). τ grid is dt, 2·dt, …, tca_window. Swappable so
                 the 30-min and 1-hr grids can both be built and compared.
- `tca_window` : total detection-to-TCA window (s). Defaults to 24 hr.
- `verbose`    : print per-τ diagnostics.

Returns a NamedTuple with:
- `τ_s`         : Vector of τ values (s), ascending
- `Σ_sc_rtn`    : Vector of 6×6 RTN covariances for the spacecraft, one per τ
- `Σ_debris_rtn`: Vector of 6×6 RTN covariances for the debris, one per τ
- `Σ_sc_eci`    : Vector of 6×6 ECI (GCRF) covariances for the spacecraft
- `Σ_debris_eci`: Vector of 6×6 ECI (GCRF) covariances for the debris
- `dt`, `tca_window`, `n_steps`
- `all_pd`, `any_symmetrized` : global health flags across the sweep

VALIDITY: noiseless-maneuver only (see module header / architecture doc §5,§8).
"""
function build_covariance_table(pomdp::SpacecraftCAPOMDP,
                                sc_eci::AbstractVector,
                                debris_eci::AbstractVector;
                                dt::Real = pomdp.dt,
                                tca_window::Union{Real,Nothing} = nothing,
                                verbose::Bool = true)

    window = tca_window === nothing ? 24 * 60 * 60.0 : Float64(tca_window)
    n_steps = Int(round(window / dt))
    τ_s = collect(dt .* (1:n_steps))

    # Anchor each object at TCA with its initial covariance, STM enabled.
    prop_sc, ep0 = eci2orb_brahe(collect(sc_eci), pomdp.epochTCA, pomdp.satParams,
                                 pomdp.forceModel; initial_covariance = pomdp.P0_sc)
    prop_debris, _ = eci2orb_brahe(collect(debris_eci), pomdp.epochTCA, pomdp.debrisParams,
                                   pomdp.forceModel; initial_covariance = pomdp.P0_debris)

    Σ_sc_rtn = Vector{Matrix{Float64}}(undef, n_steps)
    Σ_debris_rtn = Vector{Matrix{Float64}}(undef, n_steps)
    Σ_sc_eci = Vector{Matrix{Float64}}(undef, n_steps)
    Σ_debris_eci = Vector{Matrix{Float64}}(undef, n_steps)

    all_pd = true
    any_symmetrized = false

    if verbose
        println("Building Σ(τ) table: dt = $(dt/60) min, window = $(window/3600) hr, " *
                "$(n_steps) steps")
    end

    for (k, τ) in enumerate(τ_s)
        ep = ep0 + Float64(τ)
        prop_sc.propagate_to(ep)
        prop_debris.propagate_to(ep)

        for (prop, store_rtn, store_eci) in (
                (prop_sc, Σ_sc_rtn, Σ_sc_eci),
                (prop_debris, Σ_debris_rtn, Σ_debris_eci))
            P_rtn = collect(prop.covariance_rtn(ep))
            P_eci = collect(prop.covariance_gcrf(ep))
            P_rtn_s, sym_r, _, pd_r = _symmetrize_check(P_rtn)
            P_eci_s, sym_e, _, pd_e = _symmetrize_check(P_eci)
            store_rtn[k] = P_rtn_s
            store_eci[k] = P_eci_s
            all_pd &= pd_r & pd_e
            any_symmetrized |= sym_r | sym_e
        end

        if verbose
            σ_along_sc = sqrt(Σ_sc_rtn[k][2, 2])       # RTN axis 2 = transverse = along-track
            σ_along_deb = sqrt(Σ_debris_rtn[k][2, 2])
            println("  τ = $(lpad(round(τ/3600, digits=2), 6)) hr  |  " *
                    "along-track 1σ: sc = $(round(σ_along_sc, digits=1)) m, " *
                    "debris = $(round(σ_along_deb, digits=1)) m")
        end
    end

    return (τ_s = τ_s,
            Σ_sc_rtn = Σ_sc_rtn, Σ_debris_rtn = Σ_debris_rtn,
            Σ_sc_eci = Σ_sc_eci, Σ_debris_eci = Σ_debris_eci,
            dt = Float64(dt), tca_window = window, n_steps = n_steps,
            all_pd = all_pd, any_symmetrized = any_symmetrized)
end

"""
    pc_through_table(pomdp, sc_eci, debris_eci, table) -> Vector{Float64}

Pc-trust check (Phase 3, folded in from Phase 2): evaluate Chan Pc at each τ in
`table`, using the propagated ECI covariances Σ(τ) for both objects, for a fixed
conjunction whose two objects sit at `sc_eci` / `debris_eci` at TCA.

Returns Pc[k] = chan_pc at τ = table.τ_s[k] remaining. As τ → 0 (approaching
TCA) the covariance grows toward its TCA magnitude, so Pc should evolve smoothly
and settle to the at-TCA value — this is the "are the numbers believable" check.

Note: `chan_pc` combines the two objects' ECI covariances in the ECI frame and
projects onto the encounter plane, so it needs ECI (GCRF) covariances — we pass
`Σ_sc_eci` / `Σ_debris_eci`, not the RTN ones.
"""
function pc_through_table(pomdp::SpacecraftCAPOMDP,
                          sc_eci::AbstractVector,
                          debris_eci::AbstractVector,
                          table)
    hbr = pomdp.R_hard_body_sc + pomdp.R_hard_body_debris
    pcs = Vector{Float64}(undef, table.n_steps)
    for k in 1:table.n_steps
        pcs[k] = chan_pc(collect(sc_eci), collect(debris_eci),
                         table.Σ_sc_eci[k], table.Σ_debris_eci[k], hbr)
    end
    return pcs
end
