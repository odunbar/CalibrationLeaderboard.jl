# Lorenz 96 prior and its pushforward through the forward map (no fitting algorithm involved).
#
#   julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l96_prior_pushforward.jl [case ...]
#   N_PRIOR=200 julia --project=... plot_l96_prior_pushforward.jl const-force     # number of simulated prior samples
#
# `case` is any of const-force, vec-force, flux-force (default: all three).
# Panels: prior on the forcing (shaded density for the constant forcing, shaded quantile bands of the
# forcing profile otherwise), its pushforward onto the window mean and window std per site (shaded bands) against the
# observation y ± noise σ = sqrt(diag R).
# Writes output/l96_<case>_prior_pushforward.png

include(joinpath(@__DIR__, "l96_problem.jl"))

function plot_case(case::AbstractString)
    @info "Lorenz 96 prior pushforward: $(case)"
    prob = l96_setup(case)
    n_prior = parse(Int, get(ENV, "N_PRIOR", "100"))
    pf = prior_pushforward(prob; n = n_prior, n_traj = 0, seed = 2)
    nx, dt = prob.nx, prob.dt
    sites = 1:nx

    # --- prior on the forcing ---------------------------------------------------------------
    if case == "const-force"
        θ_dense = sample_prior_constrained(MersenneTwister(3), prob.prior, 5000)
        p_force = density(θ_dense[1, :]; fill = (0, 0.3, PRIOR_COLOR), color = PRIOR_COLOR, linewidth = lwidth(2), trim = true, label = "prior density",
            xlabel = "forcing φ", ylabel = "density", title = "Prior on φ", legend = :topright)
        vline!(p_force, [prob.truth_θ[1]]; color = INK_COLOR, linestyle = :dash, linewidth = lwidth(2), label = "truth")
    else
        F = reduce(hcat, forcing_profile(phi, prob.x0) for phi in pf.phis)
        p_force = plot(; xlabel = "site i", ylabel = "forcing F(i)", title = "Prior on F(i)", legend = :topright)
        for j in 1:min(10, size(F, 2))
            plot!(p_force, sites, F[:, j]; color = PRIOR_COLOR, alpha = 0.25, linewidth = lwidth(0.8), label = j == 1 ? "prior samples" : false)
        end
        band!(p_force, sites, F; color = PRIOR_COLOR, label = "prior")
        plot!(p_force, sites, forcing_profile(prob.phi_truth, prob.x0); color = INK_COLOR, linewidth = lwidth(2.5), label = "truth")
        pad_top!(p_force; frac = 0.35)
    end

    # --- pushforward onto window mean / std per site ---------------------------------------
    σ = sqrt.(diag(prob.R))
    function output_panel(rows, name, title)
        first_panel = name == "window mean"
        p = plot(; xlabel = "site i", ylabel = name, title = title, legend = first_panel ? :topright : false)
        band!(p, sites, pf.G[rows, :]; color = PRIOR_COLOR, label = "prior")
        plot!(p, sites, prob.y[rows]; ribbon = σ[rows], fillalpha = 0.2, color = INK_COLOR, linewidth = lwidth(2.5), label = first_panel ? "y ± σ" : false)
        first_panel && pad_top!(p)
        p
    end
    p_mean = output_panel(1:nx, "window mean", "Pushforward: window mean")
    p_std = output_panel((nx + 1):(2 * nx), "window std", "Pushforward: window std")

    fig = plot(
        p_force, p_mean, p_std;
        layout = (1, 3), size = figsize(1800, 760), bottom_margin = 26Plots.mm,
        plot_title = "Lorenz 96 ($(case)): prior and pushforward onto G(θ)",
    )
    save_figure(fig, "l96_$(case)_prior_pushforward.png")
end

function main()
    cases = isempty(ARGS) ? L96_CASES : ARGS
    foreach(plot_case, cases)
end

main()
