using CairoMakie
using LinearAlgebra
using Colors

function with_conjunction_type(pomdp::SpacecraftCAPOMDP, conj_type::String)
    pomdp_type = SpacecraftCAPOMDP(
        satParams = pomdp.satParams,
        debrisParams = pomdp.debrisParams,
        epochTCA = pomdp.epochTCA,
        forceModel = pomdp.forceModel,
        R_alt = pomdp.R_alt,
        e = pomdp.e,
        i = pomdp.i,
        Ω = pomdp.Ω,
        ω = pomdp.ω,
        M = pomdp.M,
        seed = pomdp.seed,
        randAdd = pomdp.randAdd,
        conjunctionType = conj_type,
        rMag = pomdp.rMag,
        vMag = pomdp.vMag,
        TCA_max = pomdp.TCA_max
    )
    return pomdp_type
end

function test_all_conjunctions(pomdp::SpacecraftCAPOMDP)

    types = ["head-on", "overtaking", "crossing"]

    fig = Figure(size=(900,600))
    ax = Axis3(fig[1,1], xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

    for conj_type in types

        pomdp_type = with_conjunction_type(pomdp, conj_type)

        eci_sc_tca, eci_debris_tca, prop_sc, prop_debris,
        epoch_sc_tca, epoch_debris_tca, pomdp_state_t0 =
            generate_conjunction(pomdp_type)

        # Forward propagate back to TCA
        prop_sc.propagate_to(epoch_sc_tca)
        prop_debris.propagate_to(epoch_debris_tca)

        r_rel_tca = prop_debris.current_state()[1:3] -
                    prop_sc.current_state()[1:3]

        v_rel_tca = prop_debris.current_state()[4:6] -
                    prop_sc.current_state()[4:6]
        rv_dot = dot(r_rel_tca, v_rel_tca)
        sc_final = prop_sc.current_state()[1:3]
        debris_final = prop_debris.current_state()[1:3]
        round_trip_error = norm((eci_sc_tca[1:3] - sc_final) + (eci_debris_tca[1:3] - debris_final))

        println("=== $conj_type ===")
        println("‖r_rel‖ = ", norm(r_rel_tca))
        println("‖v_rel‖ = ", norm(v_rel_tca))
        println("r · v at TCA = $rv_dot")
        println("Round-trip position error = $round_trip_error m\n")
        println()
        

        lines!(ax,
               [0, r_rel_tca[1]],
               [0, r_rel_tca[2]],
               [0, r_rel_tca[3]],
               linewidth=2,
               label=conj_type)

        scatter!(ax,
                 [r_rel_tca[1]],
                 [r_rel_tca[2]],
                 [r_rel_tca[3]],
                 markersize=10)
    end

    axislegend(ax)
    fig
end




function plot_conjunctions_3d_rtn(pomdp_base)
    conjunction_types = ["head-on", "overtaking", "crossing"]
    colors = [:dodgerblue, :orangered, :mediumseagreen]

    fig = Figure(size=(1400, 580), backgroundcolor=:gray10)

    Label(fig[0, 1:3],
        "Conjunction Geometry — Relative Motion in RTN Frame (Spacecraft at Origin)";
        color=:white, fontsize=16, font=:bold)

    Label(fig[2, 1:3],
        "Dashed line = debris trajectory ±60s around TCA  |  Star = debris position at TCA  |  Yellow arrow = relative velocity direction  |  Drop lines show 3D position";
        color=:gray50, fontsize=11)

    for (j, (ctype, col)) in enumerate(zip(conjunction_types, colors))
        pomdp = with_conjunction_type(pomdp_base, ctype)

        spacecraft_eci, debris_eci, _, _, _, _, _, _ = generate_conjunction(pomdp)

        # RTN basis at TCA
        r_sc = spacecraft_eci[1:3]
        v_sc = spacecraft_eci[4:6]
        R_hat = r_sc / norm(r_sc)
        N_hat = cross(r_sc, v_sc) / norm(cross(r_sc, v_sc))
        T_hat = cross(N_hat, R_hat)

        # Propagate ±60s around TCA in 3s steps
        prop_sc,  ep_sc  = eci2orb_brahe(spacecraft_eci, pomdp.epochTCA, pomdp.satParams,  pomdp.forceModel)
        prop_deb, ep_deb = eci2orb_brahe(debris_eci,    pomdp.epochTCA, pomdp.debrisParams, pomdp.forceModel)

        dt = 3.0
        n_steps = 20
        rel_rtn = zeros(3, 2*n_steps+1)

        for (k, step) in enumerate(-n_steps:n_steps)
            t = ep_sc + step * dt
            prop_sc.propagate_to(t)
            prop_deb.propagate_to(t)
            dr = prop_deb.current_state()[1:3] - prop_sc.current_state()[1:3]
            rel_rtn[1, k] = dot(dr, R_hat)
            rel_rtn[2, k] = dot(dr, T_hat)
            rel_rtn[3, k] = dot(dr, N_hat)
        end

        # TCA relative state
        r_rel = debris_eci[1:3] - r_sc
        v_rel = debris_eci[4:6] - v_sc
        rtn_pos = [dot(r_rel, R_hat), dot(r_rel, T_hat), dot(r_rel, N_hat)]
        rtn_vel = [dot(v_rel, R_hat), dot(v_rel, T_hat), dot(v_rel, N_hat)]

        # Print diagnostic
        println("=== $ctype ===")
        println("  |r_rel| = $(round(norm(r_rel), digits=3)) m  (expected rMag = $(pomdp_base.rMag))")
        println("  |v_rel| = $(round(norm(v_rel), digits=3)) m/s (expected vMag = $(pomdp_base.vMag))")
        println("  RTN pos: R=$(round(rtn_pos[1],digits=2))  T=$(round(rtn_pos[2],digits=2))  N=$(round(rtn_pos[3],digits=2))")
        println("  dominant axis: $(["R","T","N"][argmax(abs.(rtn_pos))])")

        ax = Axis3(fig[1, j];
            xlabel="R (Radial) [m]",
            ylabel="T (Along-track) [m]",
            zlabel="N (Cross-track) [m]",
            title=uppercase(ctype),
            titlecolor=col,
            titlesize=15,
            xlabelcolor=:gray70, ylabelcolor=:gray70, zlabelcolor=:gray70,
            xticklabelcolor=:gray55, yticklabelcolor=:gray55, zticklabelcolor=:gray55,
            xspinecolor_1=:gray30, xspinecolor_2=:gray30, xspinecolor_3=:gray30,
            yspinecolor_1=:gray30, yspinecolor_2=:gray30, yspinecolor_3=:gray30,
            zspinecolor_1=:gray30, zspinecolor_2=:gray30, zspinecolor_3=:gray30,
            backgroundcolor=:gray12,
            xgridcolor=(:white, 0.05), ygridcolor=(:white, 0.05), zgridcolor=(:white, 0.05),
        )

        # Trajectory colored dark→bright (past→future)
        n_pts = size(rel_rtn, 2)
        for k in 1:n_pts-1
            frac = (k-1) / (n_pts-2)
            c = RGBAf(
                red(to_color(col))   * (0.25 + 0.75*frac),
                green(to_color(col)) * (0.25 + 0.75*frac),
                blue(to_color(col))  * (0.25 + 0.75*frac),
                1.0)
            lines!(ax,
                rel_rtn[1, k:k+1],
                rel_rtn[2, k:k+1],
                rel_rtn[3, k:k+1];
                color=c, linewidth=3, linestyle=:dash)
        end
        

        # Spacecraft at origin
        scatter!(ax, [0.0], [0.0], [0.0];
            color=:white, markersize=18, marker=:circle, label="Spacecraft")

        # Debris star at TCA
        scatter!(ax, [rtn_pos[1]], [rtn_pos[2]], [rtn_pos[3]];
            color=col, markersize=18, marker=:star5, label="Debris @ TCA")

        # Velocity arrow — scaled to 25% of rMag for visibility
        vscale = pomdp_base.rMag * 0.25 / max(norm(rtn_vel), 1e-6)
        arrows3d!(ax,
            [rtn_pos[1]], [rtn_pos[2]], [rtn_pos[3]],
            [rtn_vel[1]*vscale], [rtn_vel[2]*vscale], [rtn_vel[3]*vscale];
            color=:yellow,
            tipradius=20,    # replaces arrowsize x/y
            tiplength=45,    # replaces arrowsize z
            label="v_rel direction")

        # Drop lines from debris to each axis plane — critical for 3D depth reading
        x, y, z = rtn_pos
        lines!(ax, [x, x], [y, y], [z, 0.0]; color=(:white, 0.25), linewidth=1, linestyle=:dot)
        lines!(ax, [x, x], [y, 0.0], [z, z]; color=(:white, 0.25), linewidth=1, linestyle=:dot)
        lines!(ax, [x, 0.0], [y, y], [z, z]; color=(:white, 0.25), linewidth=1, linestyle=:dot)

        # Colored axis reference lines through origin
        lim = pomdp_base.rMag * 1.4
        lines!(ax, [-lim, lim], [0.0, 0.0], [0.0, 0.0]; color=(:red,   0.5), linewidth=1.5)
        lines!(ax, [0.0, 0.0], [-lim, lim], [0.0, 0.0]; color=(:green, 0.5), linewidth=1.5)
        lines!(ax, [0.0, 0.0], [0.0, 0.0], [-lim, lim]; color=(:orange,  0.5), linewidth=1.5)

        # Axis labels at tips
        text!(ax,  lim*1.05, 0.0, 0.0; text="R", color=:red,   fontsize=13)
        text!(ax, 0.0,  lim*1.05, 0.0; text="T", color=:green, fontsize=13)
        text!(ax, 0.0, 0.0,  lim*1.05; text="N", color=:orange,  fontsize=13)

        axislegend(ax;
            backgroundcolor=:gray20, labelcolor=:white,
            framecolor=:gray40, labelsize=11, position=:lt)
    end

    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 20)

    return fig
end
