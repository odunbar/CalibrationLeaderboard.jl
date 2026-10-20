# Lorenz 96 forward map, visualised (no fitting algorithm involved).
#
#   julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l96_forward_map.jl [case ...]
#
# `case` is any of const-force, vec-force, flux-force (default: all three).
# Shows the forcing F(i)  ->  trajectory (Hovmöller, statistics window start marked)  ->  G(θ) = window mean & std per site,
# for three runs:
#   truth      : truth forcing, spun-up initial condition      (G = the observation y)
#   different θ: a different forcing (one prior draw), same initial condition
#   different x₀: truth forcing, perturbed initial condition   (the chaos "noise" in G)
# Writes output/l96_<case>_forward_map.png

include(joinpath(@__DIR__, "l96_problem.jl"))

function plot_case(case::AbstractString)
    @info "Lorenz 96 forward map: $(case)"
    prob = l96_setup(case)
    sc = run_scenarios(prob; seed = 1, n_ic_cloud = 30)
    nx, dt = prob.nx, prob.dt
    idx = stats_window_indices(prob.config, prob.obs)
    t0, t1 = window_times(idx, dt)
    t = time_axis(sc.truth.states, dt)
    sites = 1:nx

    runs = [
        (sc.truth, prob.phi_truth, INK_COLOR, "truth"),
        (sc.alt_param, sc.phi_alt, ALT_PARAM_COLOR, "different θ"),
        (sc.alt_ic, prob.phi_truth, ALT_IC_COLOR, "different x₀"),
    ]

    # --- input: forcing profiles (the legend of this panel names the runs and the noise layers) ---
    F_truth = forcing_profile(prob.phi_truth, prob.x0)
    F_alt = forcing_profile(sc.phi_alt, prob.x0)
    p_force = plot(; xlabel = "site i", ylabel = "F(i)", title = "Forcing θ", legend = :topright)
    plot!(p_force, sites, F_truth; color = INK_COLOR, linewidth = lwidth(3), label = "truth")
    plot!(p_force, sites, F_alt; color = ALT_PARAM_COLOR, linewidth = lwidth(2.5), label = "different θ")
    plot!(p_force, sites, F_truth; color = ALT_IC_COLOR, linewidth = lwidth(2), linestyle = :dash, label = "different x₀")
    plot!(p_force, [NaN], [NaN]; color = MUTED_COLOR, alpha = 0.5, label = "other x₀")
    plot!(p_force, [NaN], [NaN]; color = INK_COLOR, ribbon = [1.0], fillalpha = 0.2, label = "y ± σ")
    all_F = vcat(F_truth, F_alt)
    ylims!(p_force, min(0.0, minimum(all_F)), 2.3 * maximum(all_F))  # head-room for the in-panel legend

    # --- output: window mean and std per site ---------------------------------------------
    σ = sqrt.(diag(prob.R))
    function output_panel(rows, name, title)
        p = plot(; xlabel = "site i", ylabel = name, title = title, legend = false)
        for k in 1:size(sc.cloud_G, 2)
            plot!(p, sites, sc.cloud_G[rows, k]; color = MUTED_COLOR, alpha = 0.3, linewidth = lwidth(0.8), label = false)
        end
        plot!(p, sites, prob.y[rows]; ribbon = σ[rows], fillalpha = 0.2, color = INK_COLOR, linewidth = lwidth(2.5), label = false)
        plot!(p, sites, sc.alt_param.G[rows]; color = ALT_PARAM_COLOR, linewidth = lwidth(2), label = false)
        plot!(p, sites, sc.alt_ic.G[rows]; color = ALT_IC_COLOR, linewidth = lwidth(2), label = false)
        p
    end
    p_mean = output_panel(1:nx, "window mean", "G(θ): window mean")
    p_std = output_panel((nx + 1):(2 * nx), "window std", "G(θ): window std")

    # --- space-time view of each run, window marked ------------------------------------------
    clim = maximum(abs, sc.truth.states)
    hov_panels = map(enumerate(runs)) do (i, (r, _, c, l))
        p = heatmap(t, sites, r.states; color = :balance, clims = (-clim, clim), xlabel = "time", ylabel = i == 1 ? "site i" : "",
            title = l, colorbar = false, xticks = 0:(prob.T > 20 ? 20 : 5):prob.T, titlefontcolor = c == INK_COLOR ? :black : c)
        vline!(p, [t0]; color = :black, linestyle = :dash, linewidth = lwidth(2), label = false)
        p
    end
    annotate!(hov_panels[1], t0, nx, text(" window →", 9 * FONT_SCALE, :black, :left, :top))

    layout = @layout [grid(1, 3){0.45h}; grid(1, 3)]
    fig = plot(
        p_force, p_mean, p_std, hov_panels...;
        layout = layout, size = figsize(1800, 1000),
        plot_title = "Lorenz 96 ($(case)):  θ → trajectory → G(θ)",
    )
    save_figure(fig, "l96_$(case)_forward_map.png")
end

function main()
    cases = isempty(ARGS) ? L96_CASES : ARGS
    foreach(plot_case, cases)
end

main()
