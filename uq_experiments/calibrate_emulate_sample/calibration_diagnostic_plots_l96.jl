# load packages
# CES

using Random
using JLD2
using Statistics
using LinearAlgebra
using Dates

using Plots
using Plots.Measures

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

include("experiment_config.jl")
include(joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz96.jl"))  # provides Flux.destructure for flux-force truth_params extraction

#### CHOOSE YOUR CASE:
exp = l96_experiment()
@assert exp in (:l96_const, :l96_vec, :l96_flux) "For plot_l96_forcing.jl, set EXPERIMENT to :l96_const, :l96_vec, or :l96_flux in experiment_config.jl"
cfg        = experiment_config(exp)
method     = method_cases[1]
calib_dir  = calib_directory(method, cfg)
force_cases = [cfg.force_case]
N_enss     = cfg.N_ens_sizes
rng_idxs   = collect(1:cfg.n_repeats)

homedir = @__DIR__ # not pwd(), so SLURM jobs submitted from hpc-variant/ find the same output/ as calibrate

# Discover runs that have calibrate output (ekp + results files).
valid_file_items = []
valid_files = []
for force_case in force_cases
    for N_ens in N_enss
        for rng_idx in rng_idxs
            data_save_directory = joinpath(homedir, "output", calib_dir)
            ekp_file = joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx))
            res_file = joinpath(data_save_directory, results_filename(cfg, N_ens, rng_idx))
            if isfile(ekp_file) && isfile(res_file)
                push!(valid_files, case_suffix(cfg, N_ens, rng_idx))
                push!(valid_file_items, (force_case, N_ens, rng_idx))
            end
        end
    end
end
@info "Plotting valid files:"
display(valid_files)
println(" ")

# Load x0 from preliminaries (needed for computing panel-ii forcing).
prelim_file = joinpath(homedir, "output", "l96_computed_preliminaries_$(cfg.force_case).jld2")
if !isfile(prelim_file)
    error("Preliminaries file not found: $(prelim_file). Run calibrate_l96.jl first.")
end
prelim = load_preliminaries(prelim_file)
x0                     = prelim.x0
lorenz_config_settings = prelim.lorenz_config_settings
observation_config     = prelim.observation_config

for ((force_case, N_ens, rng_idx), calib_filename_suffix) in zip(valid_file_items, valid_files)

    @info("Plotting for L96: \n method: $(calib_dir) \n experiment: $(calib_filename_suffix)")

    data_save_directory   = joinpath(homedir, "output", calib_dir)
    figure_save_directory = data_save_directory

    # --- load calibrate outputs ---
    ekp_file = joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx))
    res_file = joinpath(data_save_directory, results_filename(cfg, N_ens, rng_idx))
    pri_file = joinpath(data_save_directory, prior_filename(cfg))

    ekpobj = load(ekp_file)["ekpobj"]
    prior  = load(pri_file)["prior"]

    (truth_params_obj, _) = load(res_file)["truth_params_structure"]
    (truth_params_constrained, structure, sample_range) = if force_case == "const-force"
        ([truth_params_obj.val], nothing, nothing)
    elseif force_case == "vec-force"
        (truth_params_obj.val, nothing, nothing)
    elseif force_case == "flux-force"
        tp, _ = Flux.destructure(truth_params_obj.model)
        (tp, truth_params_obj.model, truth_params_obj.sample_range)
    end

    final_params_constrained = get_ϕ_mean_final(prior, ekpobj)

    N_ens = get_N_ens(ekpobj)
    n_par = length(truth_params_constrained)
    nx    = length(x0)
    y     = get_obs(ekpobj)
    ny    = length(y)

    # --- compute forcing quantities ---
    is_const      = force_case == "const-force"
    truth_emc     = build_forcing(truth_params_obj, truth_params_constrained, structure, sample_range)
    truth_forcing = forcing(truth_emc, x0)
    xaxis_forcing = isnothing(sample_range) ? range(0, length(truth_forcing) - 1, step = 1) : sample_range
    final_emc     = build_forcing(truth_params_obj, final_params_constrained, structure, sample_range)
    final_forcing = forcing(final_emc, x0)

    # --- data plot ---
    gr(size = (3 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)

    p1 = if is_const
        hline(
            [truth_params_constrained[1]],
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Parameter (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    else
        plot(
            range(0, n_par - 1, step = 1),
            truth_params_constrained,
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Parameter (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    end

    p_mid = if is_const
        hline(
            [truth_forcing[1]],
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Forcing (transformed-input)",
            ylims = (0, 16),
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    else
        plot(
            xaxis_forcing,
            truth_forcing,
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Forcing (transformed-input)",
            ylims = (0, 16),
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    end

    p2 = plot(
        1:ny,
        y,
        ribbon = sqrt.(diag(get_obs_noise_cov(ekpobj))),
        label = "data",
        color = :black,
        linewidth = 4,
        xlabel = "Spatial index",
        ylabel = "State mean/std (output)",
        left_margin = 15mm,
        bottom_margin = 15mm,
    )

    plt = plot(p1, p_mid, p2, layout = @layout [a b c])
    savefig(plt, joinpath(figure_save_directory, "data_$(calib_filename_suffix).png"))
    savefig(plt, joinpath(figure_save_directory, "data_$(calib_filename_suffix).pdf"))

    # --- solution plots (mean ensemble) ---
    gr(size = (3 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)

    p1 = if is_const
        hline(
            [0.0],
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Parameter difference (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    else
        plot(
            range(0, n_par - 1, step = 1),
            zeros(n_par),
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Parameter difference (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    end

    p_mid = if is_const
        hline(
            [truth_forcing[1]],
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Forcing (transformed-input)",
            ylims = (0, 16),
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    else
        plot(
            xaxis_forcing,
            truth_forcing,
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Forcing (transformed-input)",
            ylims = (0, 16),
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
    end

    p2 = plot(
        1:ny,
        y,
        ribbon = sqrt.(diag(get_obs_noise_cov(ekpobj))),
        label = "data",
        color = :black,
        linewidth = 4,
        xlabel = "Spatial index",
        ylabel = "State mean/std (output)",
        left_margin = 15mm,
        bottom_margin = 15mm,
    )

    p3     = deepcopy(p1)
    p_mid3 = deepcopy(p_mid)
    p4     = deepcopy(p2)

    if is_const
        hline!(p1, [final_params_constrained[1] - truth_params_constrained[1]], label = "mean ensemble input", color = :lightgreen, linewidth = 4)
        hline!(p_mid, [final_forcing[1]], label = "mean ensemble forcing", color = :lightgreen, linewidth = 4)
    else
        plot!(p1, range(0, n_par - 1, step = 1), final_params_constrained .- truth_params_constrained, label = "mean ensemble input", color = :lightgreen, linewidth = 4)
        plot!(p_mid, xaxis_forcing, final_forcing, label = "mean ensemble forcing", color = :lightgreen, linewidth = 4)
    end
    plot!(p2, 1:length(y), get_g_mean_final(ekpobj), label = "mean ensemble output", color = :lightgreen, linewidth = 4)

    plt = plot(p1, p_mid, p2, layout = @layout [a b c])
    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix).png"))
    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix).pdf"))

    # --- full ensemble plot ---
    ens_final_params   = get_ϕ_final(prior, ekpobj)
    ens_init_params    = get_ϕ(prior, ekpobj, 1)
    ens_final_forcings = hcat([forcing(build_forcing(truth_params_obj, ens_final_params[:, j], structure, sample_range), x0) for j in axes(ens_final_params, 2)]...)
    ens_init_forcings  = hcat([forcing(build_forcing(truth_params_obj, ens_init_params[:, j], structure, sample_range), x0) for j in axes(ens_init_params, 2)]...)

    if is_const
        hline!(p3, [ens_final_params[1, 1] - truth_params_constrained[1]], label = "ensemble inputs", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        hline!(p3, ens_final_params[1, 2:end] .- truth_params_constrained[1], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        hline!(p3, ens_init_params[1, :] .- truth_params_constrained[1], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)

        hline!(p_mid3, [ens_final_forcings[1, 1]], label = "ensemble forcings", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        hline!(p_mid3, ens_final_forcings[1, 2:end], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        hline!(p_mid3, ens_init_forcings[1, :], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
    else
        plot!(p3, range(0, n_par - 1, step = 1), ens_final_params[:, 1] .- truth_params_constrained, label = "ensemble inputs", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p3, range(0, n_par - 1, step = 1), ens_final_params[:, 2:end] .- truth_params_constrained, label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p3, range(0, n_par - 1, step = 1), ens_init_params[:, 1] .- truth_params_constrained, label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p3, range(0, n_par - 1, step = 1), ens_init_params[:, 2:end] .- truth_params_constrained, label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)

        plot!(p_mid3, xaxis_forcing, ens_final_forcings[:, 1], label = "ensemble forcings", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p_mid3, xaxis_forcing, ens_final_forcings[:, 2:end], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p_mid3, xaxis_forcing, ens_init_forcings[:, 1], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p_mid3, xaxis_forcing, ens_init_forcings[:, 2:end], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
    end

    plot!(p4, 1:length(y), get_g_final(ekpobj)[:, 1], color = :lightgreen, label = "ensemble outputs", linewidth = 4, linealpha = 0.1)
    plot!(p4, 1:length(y), get_g_final(ekpobj)[:, 2:end], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)
    plot!(p4, 1:length(y), get_g(ekpobj, 1)[:, 1], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)
    plot!(p4, 1:length(y), get_g(ekpobj, 1)[:, 2:end], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)

    plt = plot(p3, p_mid3, p4, layout = @layout [a b c])
    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix)_full_ens.png"))
    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix)_full_ens.pdf"))

    # --- Hovmöller diagram (state vs. time) with initial/final ensembles overlaid
    # on the forcing/mean/std panels (very transparent = initial, opaque = final) ---
    dt_hov = lorenz_config_settings.dt
    T_hov  = 200.0
    xn_hov = lorenz_solve(truth_emc, x0, LorenzConfig(dt_hov, T_hov))
    t_axis_hov = range(0, T_hov, length = size(xn_hov, 2))
    plot_rows_hov = 1:max(1, round(Int, 1.0 / dt_hov)):length(t_axis_hov)

    mean_truth = y[1:nx]
    std_truth  = y[(nx + 1):(2 * nx)]
    g_init  = get_g(ekpobj, 1)
    g_final = get_g_final(ekpobj)
    mean_init_ens  = g_init[1:nx, :]
    std_init_ens   = g_init[(nx + 1):(2 * nx), :]
    mean_final_ens = g_final[1:nx, :]
    std_final_ens  = g_final[(nx + 1):(2 * nx), :]

    init_color, init_alpha   = :gray, 0.25
    final_color, final_alpha = :green, 0.4
    line_width_hov = 5
    ensemble_line_width = 1
    gr(
        size = (2400, 1000),
        guidefontsize = 22, tickfontsize = 20, colorbar_titlefontsize = 20,
        legendfontsize = 20,
        margin = 8mm,
    )

    forcing_panel = plot(
        ens_init_forcings, 1:nx;
        xlabel = "Forcing", ylabel = "State",
        xticks = false, grid = false, linewidth = ensemble_line_width, color = init_color, linealpha = init_alpha,
        ylims = (1, nx), label = "",
        left_margin = 16mm, bottom_margin = 14mm,
    )
    plot!(forcing_panel, ens_final_forcings, 1:nx; color = final_color, linealpha = final_alpha, linewidth = ensemble_line_width, label = "")
    plot!(forcing_panel, truth_forcing, 1:nx; color = :black, linewidth = line_width_hov, label = "truth")

    hovmoller = heatmap(
        t_axis_hov[plot_rows_hov], 1:nx, xn_hov[:, plot_rows_hov];
        xlabel = "Time", ylabel = "",
        yticks = false, grid = false,
        c = :balance, colorbar = false,
        bottom_margin = 14mm,
    )
    vspan!(hovmoller, [observation_config.T_start, observation_config.T_end]; color = :green, alpha = 0.15, label = "Observed window")

    mean_panel = plot(
        mean_init_ens, 1:nx;
        xlabel = "Observed mean", ylabel = "",
        xticks = false, yticks = false, grid = false, linewidth = ensemble_line_width, color = init_color, linealpha = init_alpha,
        ylims = (1, nx), legend = false,
        bottom_margin = 14mm,
    )
    plot!(mean_panel, mean_final_ens, 1:nx; color = final_color, linealpha = final_alpha, linewidth = ensemble_line_width, label = "")
    plot!(mean_panel, mean_truth, 1:nx; color = :black, linewidth = line_width_hov, label = "")

    std_panel = plot(
        std_init_ens, 1:nx;
        xlabel = "Observed std.", ylabel = "",
        xticks = false, yticks = false, grid = false, linewidth = ensemble_line_width, color = init_color, linealpha = init_alpha,
        ylims = (1, nx), legend = false,
        bottom_margin = 14mm,
    )
    plot!(std_panel, std_final_ens, 1:nx; color = final_color, linealpha = final_alpha, linewidth = ensemble_line_width, label = "")
    plot!(std_panel, std_truth, 1:nx; color = :black, linewidth = line_width_hov, label = "")

    l_hov = @layout [a{0.14w} b{0.58w} c{0.14w} d{0.14w}]
    fig_hov = plot(forcing_panel, hovmoller, mean_panel, std_panel; layout = l_hov, link = :y)
    savefig(fig_hov, joinpath(figure_save_directory, "hovmoller_$(calib_filename_suffix).png"))
    savefig(fig_hov, joinpath(figure_save_directory, "hovmoller_$(calib_filename_suffix).pdf"))

end
