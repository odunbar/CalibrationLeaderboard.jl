# Lorenz 63 forward map, visualised (no fitting algorithm involved).
#
#   julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l63_forward_map.jl
#
# Shows θ = (ρ, β)  ->  G(θ) (the 9 window statistics, against the observation y ± σ)  ->  the x–z phase portrait
# with the statistics window in bold,
# for three runs:
#   black  : truth parameter, spun-up initial condition        (G = the observation y)
#   orange : a different parameter (a prior draw with β < true β), same initial condition
#   green  : truth parameter, a different initial condition    (the chaos "noise" in G)
# Writes output/l63_forward_map.png

include(joinpath(@__DIR__, "l63_problem.jl"))

function main()
    prob = l63_setup()
    # the "different θ" is a prior draw with β below the true β
    sc = run_scenarios(prob; seed = 1, n_ic_cloud = 0, alt_accept = θ -> θ[2] < prob.truth_θ[2])
    idx = stats_window_indices(prob.config, prob.obs)

    runs = [
        (sc.truth, INK_COLOR, "truth"),
        (sc.alt_param, ALT_PARAM_COLOR, "different θ"),
        (sc.alt_ic, ALT_IC_COLOR, "different x₀"),
    ]

    # --- input: the two parameter values, on axes spanning the bulk (1-99%) of the prior ----------
    θ_ref = sample_prior_constrained(MersenneTwister(0), prob.prior, 2000)
    function span(k)
        lo, hi = extrema(vcat(quantile(θ_ref[k, :], [0.01, 0.99]), prob.truth_θ[k], sc.θ_alt[k]))
        pad = 0.05 * (hi - lo)
        return (lo - pad, hi + pad)
    end
    p_theta = plot(; xlabel = "ρ", ylabel = "β", title = "Parameters θ", legend = :topright, xlims = span(1), ylims = span(2))
    scatter!(p_theta, [prob.truth_θ[1]], [prob.truth_θ[2]]; marker = :star5, markersize = 18, color = INK_COLOR, label = "truth θ†")
    scatter!(p_theta, [sc.θ_alt[1]], [sc.θ_alt[2]]; marker = :diamond, markersize = 15, color = ALT_PARAM_COLOR, label = "different θ")

    # --- phase portrait: bold inside the statistics window, faint before it --------------
    p_phase = plot(; xlabel = "x", ylabel = "z", title = "Phase portrait (bold: window)", legend = :topright)
    for (r, c, l) in runs
        window_segments!(p_phase, r.states, idx, (1, 3); color = c, label = l)
    end

    # --- output: the 9 window statistics G(θ) --------------------------------------------
    σ = sqrt.(diag(prob.R))
    xs = 1:prob.ny
    p_G = plot(;
        xticks = (xs, prob.output_names), xrotation = 35,
        title = "G(θ): window statistics", ylabel = "statistic value", legend = :topleft,
    )
    # error bars drawn explicitly (thick bar + wide caps) for full control of their width
    cap = 0.25
    for k in xs
        plot!(p_G, [k, k], [prob.y[k] - σ[k], prob.y[k] + σ[k]]; color = INK_COLOR, linewidth = lwidth(3), label = false)
        for e in (prob.y[k] - σ[k], prob.y[k] + σ[k])
            plot!(p_G, [k - cap, k + cap], [e, e]; color = INK_COLOR, linewidth = lwidth(3), label = false)
        end
    end
    scatter!(p_G, xs, prob.y; color = INK_COLOR, markersize = 10, label = "y ± σ")
    scatter!(p_G, xs .+ 0.15, sc.alt_param.G; color = ALT_PARAM_COLOR, marker = :diamond, markersize = 10, label = "different θ")
    scatter!(p_G, xs .- 0.15, sc.alt_ic.G; color = ALT_IC_COLOR, marker = :utriangle, markersize = 10, label = "different x₀")

    fig = plot(
        p_theta, p_G, p_phase;
        layout = (1, 3), size = figsize(1800, 800), bottom_margin = 26Plots.mm,
        plot_title = "Lorenz 63:  θ → G(θ) → trajectory",
    )
    save_figure(fig, "l63_forward_map.png")
end

main()
