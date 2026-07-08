# Import modules
using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Flux
using BSON
using Dates

# CES
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

include(joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

verbose_flag = false
save_all_ekp = true

########################################################################
############### Deterministic problem setup ############################
########################################################################

function build_setup(cfg, output_dir)
    case = cfg.force_case
    N_iter      = cfg.N_iter
    N_ens_sizes = cfg.N_ens_sizes
    terminate_at = cfg.terminate_at
    # Seeded so every array task sees the same rng_seeds list.
    rng_seeds = randperm(MersenneTwister(20260529), 1_000_000)[1:cfg.n_repeats]

    if case == "const-force"
        nx = 40
        nu = 1
        phi = ConstantEMC(8.0)
        phi_structure = nothing
        sample_range = nothing
        prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)

    elseif case == "vec-force"
        nx = 40
        nu = nx
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        phi_structure = nothing
        sample_range = nothing
        pl, psig = 2.0, 3.0
        prior_cov = zeros(nx, nx)
        for ii in 1:nx, jj in 1:nx
            prior_cov[ii, jj] = psig^2 * exp(-abs(ii - jj) / pl)
        end
        prior_mean = 8.0 * ones(nx)
        distribution = Parameterized(MvNormal(prior_mean, prior_cov))
        constraint = repeat([no_constraint()], 40)
        prior = ParameterDistribution(distribution, constraint, "ml96_prior")

    elseif case == "flux-force"
        nx = 100
        nu = 61
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * pi * x) / 10)
        x_train = collect(-5.0:0.01:5.0)
        Random.seed!(20260529)
        y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, _ = train_network(deepcopy(phi_structure), x_train, y_train)
        sample_range = Float32.(collect(-5.0:0.1:4.9))
        phi = FluxEMC(true_model, sample_range)
        prior_sinusoid(x) = 8.02 .+ 6.5 * sin.(1.02 * (4 * pi * x) / 10 + 0.2)
        prior_train = prior_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        prior_model, prior_mean = train_network(deepcopy(phi_structure), x_train, prior_train)
        #prior_cov = (0.1^2) * I(length(prior_mean)) # original cov
        prior_cov = (0.1^2) * Diagonal(prior_mean.^2) # scaled to mean
        distribution = Parameterized(MvNormal(prior_mean, prior_cov))
        constraint = repeat([no_constraint()], 61)
        prior = ParameterDistribution(distribution, constraint, "l96_nn_prior")
    else
        throw(ArgumentError("Unknown force_case: $case"))
    end

    ny = nx * 2

    # Truth data (spin-up, synthetic observations, noise/IC covariances) is
    # computed once by l96_preliminaries.jl and shared by every array task.
    prelim_file = joinpath(output_dir, prelim_filename(cfg))
    isfile(prelim_file) || error("Prelim file not found: $(prelim_file)\nRun l96_preliminaries.jl first.")
    prelim = load_preliminaries(prelim_file)
    x0                     = prelim.x0
    y                      = prelim.y
    ic_cov_sqrt            = prelim.ic_cov_sqrt
    R                      = prelim.R
    R_inv_var              = prelim.R_inv_var
    lorenz_config_settings = prelim.lorenz_config_settings
    observation_config     = prelim.observation_config

    configuration = Dict(
        "case"         => case,
        "N_iter"       => N_iter,
        "N_ens_sizes"  => N_ens_sizes,
        "rng_seeds"    => rng_seeds,
        "terminate_at" => terminate_at,
    )

    return (;
        nx, nu, ny,
        phi, phi_structure, sample_range, prior,
        x0, y, R, R_inv_var,
        lorenz_config_settings, observation_config, ic_cov_sqrt,
        rng_seeds, configuration,
        N_iter, terminate_at,
    )
end

# Write prior files for all 4 method dirs, idempotent.
function write_priors(cfg, setup, output_dir)
    for method_name in method_cases
        per_method_dir = joinpath(output_dir, calib_directory(method_name, cfg))
        mkpath(per_method_dir)
        pf = joinpath(per_method_dir, prior_filename(cfg))
        if !isfile(pf)
            pf_tmp = splitext(pf)[1] * ".tmp.$(getpid()).jld2"
            JLD2.save(pf_tmp, "prior", setup.prior)
            try
                mv(pf_tmp, pf)
            catch
                rm(pf_tmp; force = true)
            end
        end
    end
end

########################################################################
############### Per-cell calibration (all 4 methods) ##################
########################################################################

function calibrate_one(cfg, setup, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(setup.rng_seeds[rng_idx])
    @info "Calibrating (N_ens=$(N_ens), rng_idx=$(rng_idx))"

    initial_params = construct_initial_ensemble(rng, setup.prior, N_ens)
    methods = [
        Inversion(),
#        TransformInversion(),
#        GaussNewtonInversion(setup.prior),
#        Unscented(setup.prior),
    ]

    conv_cell   = fill(NaN, 4)
    params_cell = zeros(4, setup.nu)
    output_cell = zeros(4, setup.ny)

    for (kk, method) in enumerate(methods)
        @info "Method: $(nameof(typeof(method)))"
        if isa(method, Unscented)
            ekpobj = EKP.EnsembleKalmanProcess(
                setup.y,
                setup.R,
                deepcopy(method);
                rng = copy(rng),
                verbose = verbose_flag,
                localization_method = NoLocalization(),
                scheduler = DataMisfitController(terminate_at = setup.terminate_at),
            )
        else
            ekpobj = EKP.EnsembleKalmanProcess(
                initial_params,
                setup.y,
                setup.R,
                deepcopy(method);
                rng = copy(rng),
                verbose = verbose_flag,
                localization_method = NoLocalization(),
                scheduler = DataMisfitController(terminate_at = setup.terminate_at),
            )
        end
        Ne = get_N_ens(ekpobj)

        ens_mean_final = zeros(setup.nu)
        G_ens_mean_final = zeros(setup.ny)
        for i in 1:setup.N_iter
            params_i = get_ϕ_final(setup.prior, ekpobj)
            ens_mean = mean(params_i, dims = 2)[:]
            forcing = build_forcing(setup.phi, ens_mean, setup.phi_structure, setup.sample_range)
            G_ens_mean = lorenz_forward(
                forcing,
                setup.x0 .+ setup.ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), setup.nx, 1),
                setup.lorenz_config_settings,
                setup.observation_config,
            )
            RMSE_e = norm(setup.R_inv_var * (setup.y - G_ens_mean[:])) / sqrt(setup.ny)
            @info "RMSE (at G(u_mean)): $(RMSE_e)"
            ens_mean_final = ens_mean
            G_ens_mean_final = G_ens_mean[:]
            G_ens = hcat(
                [
                    lorenz_forward(
                        build_forcing(setup.phi, params_i[:, j], setup.phi_structure, setup.sample_range),
                        (setup.x0 .+ setup.ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), setup.nx, Ne))[:, j],
                        setup.lorenz_config_settings,
                        setup.observation_config,
                    ) for j in 1:Ne
                ]...,
            )
            terminated = EKP.update_ensemble!(ekpobj, G_ens)
            if !isnothing(terminated)
                conv_cell[kk] = i * Ne
                break
            end
        end
        params_cell[kk, :] = ens_mean_final
        output_cell[kk, :] = G_ens_mean_final
        if isnan(conv_cell[kk])
            conv_cell[kk] = setup.N_iter * Ne
        end

        if save_all_ekp
            per_method_dir = joinpath(output_dir, calib_directory(nameof(typeof(method)), cfg))
            JLD2.save(
                joinpath(per_method_dir, ekp_filename(cfg, N_ens, rng_idx)),
                "N_ens", N_ens,
                "method", method,
                "ekpobj", ekpobj,
            )
            u_stored = get_u(ekpobj, return_array = false)
            g_stored = get_g(ekpobj, return_array = false)
            JLD2.save(
                joinpath(per_method_dir, results_filename(cfg, N_ens, rng_idx)),
                "y", setup.y,
                "R", setup.R,
                "inputs", u_stored,
                "outputs", g_stored,
                "truth_params_structure", (setup.phi, setup.phi_structure),
            )
        end
    end

    return (conv_cell, params_cell, output_cell)
end

########################################################################
############### Summary helpers ########################################
########################################################################

function save_combined_summary(cfg, setup, output_dir, conv, pars, outs)
    data_filename = joinpath(output_dir, summary_filename(cfg))
    JLD2.save(
        data_filename,
        "configuration",      setup.configuration,
        "method_names",       method_names,
        "conv_alg_iters",     conv,
        "final_parameters",   pars,
        "final_model_output", outs,
    )
    @info "Saved summary to $(data_filename)"
end

########################################################################
############### Main dispatcher ########################################
########################################################################

function main()
    exp = l96_experiment()
    @assert(
        exp in (:l96_const, :l96_vec, :l96_flux),
        "EXPERIMENT must be :l96_const, :l96_vec, or :l96_flux (got $exp)",
    )

    cfg        = experiment_config(exp)
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    setup = build_setup(cfg, output_dir)
    write_priors(cfg, setup, output_dir)

    tasks = flat_tasks(cfg)
    idx   = task_index_from_args()

    if isnothing(idx)
        # Serial mode: run every cell, assemble and write combined summary.
        n_ens = length(cfg.N_ens_sizes)
        n_rep = cfg.n_repeats
        conv  = fill(NaN, 4, n_ens, n_rep)
        pars  = zeros(4, n_ens, n_rep, setup.nu)
        outs  = zeros(4, n_ens, n_rep, setup.ny)
        for (t, (N_ens, rng_idx)) in enumerate(tasks)
            ee = (t - 1) ÷ n_rep + 1
            rr = (t - 1) % n_rep + 1
            c, p, o = calibrate_one(cfg, setup, N_ens, rng_idx, output_dir)
            conv[:, ee, rr]    = c
            pars[:, ee, rr, :] = p
            outs[:, ee, rr, :] = o
        end
        save_combined_summary(cfg, setup, output_dir, conv, pars, outs)
    else
        # Array mode: run one cell; ekp/results files are written inside calibrate_one.
        if idx < 1 || idx > length(tasks)
            error("SLURM_ARRAY_TASK_ID $(idx) out of range 1:$(length(tasks))")
        end
        N_ens, rng_idx = tasks[idx]
        calibrate_one(cfg, setup, N_ens, rng_idx, output_dir)
    end
end

main()
