# Explore the Lorenz-96 simulator used by calibrate_l96.jl, driven with the
# true forcing, for 200 time units. Produces a 4-panel figure: the true
# forcing profile, a Hovmöller diagram (time vs. state, colored by state
# value) with the statistics window overlaid, and the mean/std profiles that
# the forward model's summary statistics are computed from over that window.
#
# Local: EXPERIMENT=l96_const julia --project=. explore_l96_example.jl
# (EXPERIMENT defaults to the toggle in experiment_config.jl if unset.)

using LinearAlgebra
using Random
using Statistics
using Flux
using Plots

include(joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

# Mirrors the force-case forcing construction in l96_preliminaries.jl /
# calibrate_l96.jl — nx/phi/stats-window bounds are all that's needed here
# (stats_T_start/stats_T_end match the ObservationConfig(4.0, T) used there).
function true_forcing_setup(force_case::AbstractString)
    stats_T_start = 4.0
    if force_case == "const-force"
        nx = 40
        phi = ConstantEMC(8.0)
        stats_T_end = 14.0
    elseif force_case == "vec-force"
        nx = 40
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        stats_T_end = 54.0
    elseif force_case == "flux-force"
        nx = 100
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * pi * x) / 10)
        x_train = collect(-5.0:0.01:5.0)
        Random.seed!(20260529)
        y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, _ = train_network(deepcopy(phi_structure), x_train, y_train)
        sample_range = Float32.(collect(-5.0:0.1:4.9))
        phi = FluxEMC(true_model, sample_range)
        stats_T_end = 54.0
    else
        throw(ArgumentError("Unknown force_case: $force_case"))
    end
    return (nx = nx, phi = phi, stats_T_start = stats_T_start, stats_T_end = stats_T_end)
end

function main()
    exp = l96_experiment()
    @assert exp in (:l96_const, :l96_vec, :l96_flux) "EXPERIMENT must be :l96_const, :l96_vec, or :l96_flux (got $exp)"
    cfg = experiment_config(exp)

    setup = true_forcing_setup(cfg.force_case)
    nx, phi = setup.nx, setup.phi
    stats_T_start, stats_T_end = setup.stats_T_start, setup.stats_T_end
    dt = 0.01
    T = 200.0

    # Spin up onto the attractor (same recipe as l96_preliminaries.jl).
    rng_i = MersenneTwister(11)
    x_initial = rand(rng_i, nx) .* 2 .- 1
    x_spun_up = lorenz_solve(phi, x_initial, LorenzConfig(dt, 1000.0))
    x0 = x_spun_up[:, end]

    # Run the simulator with the true forcing for T time units.
    xn = lorenz_solve(phi, x0, LorenzConfig(dt, T))
    t_axis = range(0, T, length = size(xn, 2))

    @info "Ran Lorenz-96 ($(cfg.force_case), nx=$nx) for T=$T with true forcing"
    @info "State range: [$(minimum(xn)), $(maximum(xn))]"

    # Forward-model summary statistics (mean/std) over the same window used to
    # build the synthetic observations in l96_preliminaries.jl.
    gt = stats(xn, LorenzConfig(dt, T), ObservationConfig(stats_T_start, stats_T_end))
    state_mean = gt[1:nx]
    state_std = gt[(nx + 1):(2 * nx)]
    true_forcing_profile = forcing(phi, x0)

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    # Subsample rows for plotting only (keeps the simulation at full dt
    # resolution) — one plotted sample per 0.25 time units, i.e. every 25
    # timesteps at dt=0.01 — otherwise the heatmap backend downsamples ~2*10^4
    # rows into a much shorter pixel height and produces banding artifacts.
    plot_stride = max(1, round(Int, 0.25 / dt))
    plot_rows = 1:plot_stride:length(t_axis)

    gr(
        size = (2400, 1000),
        guidefontsize = 22, tickfontsize = 20, colorbar_titlefontsize = 20,
        legendfontsize = 20,
        margin = 8Plots.mm,
    )
    line_width = 3
    bottom_margin = 14Plots.mm
    left_margin = 16Plots.mm

    forcing_panel = plot(
        true_forcing_profile, 1:nx;
        xlabel = "Forcing", ylabel = "State index i",
        xticks = false, grid = false, linewidth = line_width,
        ylims = (1, nx), legend = false,
        left_margin = left_margin, bottom_margin = bottom_margin,
    )
    hovmoller = heatmap(
        t_axis[plot_rows], 1:nx, xn[:, plot_rows];
        xlabel = "Time", ylabel = "",
        yticks = false, grid = false,
        c = :balance,
        colorbar = false,
        bottom_margin = bottom_margin,
    )
    vspan!(hovmoller, [stats_T_start, stats_T_end]; color = :green, alpha = 0.15, label = "Observed window")
    mean_panel = plot(
        state_mean, 1:nx;
        xlabel = "Observed mean", ylabel = "",
        xticks = false, yticks = false, grid = false, linewidth = line_width,
        ylims = (1, nx), legend = false,
        bottom_margin = bottom_margin,
    )
    std_panel = plot(
        state_std, 1:nx;
        xlabel = "Observed std.", ylabel = "",
        xticks = false, yticks = false, grid = false, linewidth = line_width,
        ylims = (1, nx), legend = false,
        bottom_margin = bottom_margin,
    )

    l = @layout [a{0.14w} b{0.58w} c{0.14w} d{0.14w}]
    fig = plot(forcing_panel, hovmoller, mean_panel, std_panel; layout = l, link = :y)

    fig_path = joinpath(output_dir, "l96_hovmoller_$(cfg.force_case).png")
    savefig(fig, fig_path)
    @info "Saved Hovmöller diagram to $fig_path"
end

main()
