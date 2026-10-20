# Lorenz 63 prior and its pushforward through the forward map (no fitting algorithm involved).
#
#   julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l63_prior_pushforward.jl
#   N_PRIOR=200 julia --project=... plot_l63_prior_pushforward.jl      # number of simulated prior samples
#
# Top row   : prior density of ρ and β (shaded) and the x–z trajectories of a few prior samples.
# Bottom    : pushforward of the prior onto each of the 9 window statistics (histograms), against the
#             observation y and its noise level ±σ = sqrt(diag R).
# Writes output/l63_prior_pushforward.png

include(joinpath(@__DIR__, "l63_problem.jl"))

function main()
    prob = l63_setup()
    n_prior = parse(Int, get(ENV, "N_PRIOR", "100"))
    pf = prior_pushforward(prob; n = n_prior, n_traj = 8, seed = 2)
    idx = stats_window_indices(prob.config, prob.obs)
    truth = run_forward(prob, prob.phi_truth, prob.x0)

    # --- prior density of each parameter (many cheap draws, no simulation needed) -----------
    θ_dense = sample_prior_constrained(MersenneTwister(3), prob.prior, 5000)
    marginal_panels = map(1:2) do k
        p = density(θ_dense[k, :]; fill = (0, 0.3, PRIOR_COLOR), color = PRIOR_COLOR, linewidth = lwidth(2), trim = true, label = "prior density",
            xlabel = prob.param_names[k], ylabel = "density", title = "Prior on $(prob.param_names[k])", legend = :topright)
        vline!(p, [prob.truth_θ[k]]; color = INK_COLOR, linestyle = :dash, linewidth = lwidth(2), label = "truth")
        p
    end

    # --- pushforward onto trajectory space --------------------------------------------------
    p_traj = plot(; xlabel = "x", ylabel = "z", title = "Prior trajectories", legend = :topright)
    for (j, xn) in enumerate(pf.trajs)
        window_segments!(p_traj, xn, idx, (1, 3); color = PRIOR_COLOR, label = false, lw = 1.0)
    end
    window_segments!(p_traj, truth.states, idx, (1, 3); color = INK_COLOR, label = "truth")

    # --- pushforward onto each output statistic --------------------------------------------
    σ = sqrt.(diag(prob.R))
    out_panels = map(1:prob.ny) do k
        p = histogram(pf.G[k, :]; bins = 25, normalize = :pdf, color = PRIOR_COLOR, alpha = 0.55, linewidth = 0, label = k == 1 ? "prior pushforward" : false,
            title = prob.output_names[k], yticks = false, legend = k == 1 ? :topright : false)
        vspan!(p, [prob.y[k] - σ[k], prob.y[k] + σ[k]]; color = MUTED_COLOR, alpha = 0.25, linewidth = 0, label = k == 1 ? "y ± σ" : false)
        vline!(p, [prob.y[k]]; color = INK_COLOR, linewidth = lwidth(2), label = k == 1 ? "y" : false)
        p
    end

    layout = @layout [grid(1, 3){0.3h}; grid(3, 3)]
    fig = plot(
        marginal_panels..., p_traj, out_panels...;
        layout = layout, size = figsize(1800, 1450),
        plot_title = "Lorenz 63: prior and pushforward onto G(θ)",
    )
    save_figure(fig, "l63_prior_pushforward.png")
end

main()
