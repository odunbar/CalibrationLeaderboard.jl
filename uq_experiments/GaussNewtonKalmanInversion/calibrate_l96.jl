# GaussNewtonKalmanInversion — calibrate stage, L96 (const / vec / flux forcing)
#
# Runs GaussNewtonInversion (GNKI) to (approximate) convergence for one
# (N_ens, rng_idx) cell, storing the ensemble at every iteration. There is no
# separate emulate_sample stage: the ensemble at each iteration IS the
# UQ estimate of the posterior, and is pushed forward directly in
# pushforward_from_posterior_l96.jl.
#
# Local (all cells):  EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
# Local (one cell):   EXPERIMENT=l96_const julia --project=. calibrate_l96.jl <task_index>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l96.jl

using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Flux

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
const EKP = EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

verbose_flag = false

########################################################################

function force_case_setup(force_case::AbstractString)
    if force_case == "const-force"
        nx = 40
        phi = ConstantEMC(8.0)
        phi_structure = nothing
        sample_range = nothing
        prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
        T = 14.0
        inff = 2.0
    elseif force_case == "vec-force"
        nx = 40
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        phi_structure = nothing
        sample_range = nothing
        pl, psig = 2.0, 3.0
        prior_cov = [psig^2 * exp(-abs(ii - jj) / pl) for ii in 1:nx, jj in 1:nx]
        prior_mean = 8.0 * ones(nx)
        prior = ParameterDistribution(
            Parameterized(MvNormal(prior_mean, prior_cov)),
            repeat([no_constraint()], nx),
            "l96_vec_prior",
        )
        T = 54.0
        inff = 2.0
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

        prior_sinusoid(x) = 8.02 .+ 6.5 * sin.(1.02 * (4 * pi * x) / 10 + 0.2)
        prior_train = prior_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        prior_model, prior_mean = train_network(deepcopy(phi_structure), x_train, prior_train)

        prior_cov = (0.1^2) * Diagonal(prior_mean .^ 2)
        prior = ParameterDistribution(
            Parameterized(MvNormal(prior_mean, prior_cov)),
            repeat([no_constraint()], length(prior_mean)),
            "l96_nn_prior",
        )
        T = 54.0
        inff = 2.5
    else
        throw(ArgumentError("Unknown force_case: $force_case"))
    end
    return (nx = nx, phi = phi, phi_structure = phi_structure, sample_range = sample_range, prior = prior, T = T, inff = inff)
end

function calibrate_one(cfg, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)
    force_case = cfg.force_case
    setup = force_case_setup(force_case)
    nx, phi, phi_structure, sample_range, prior = setup.nx, setup.phi, setup.phi_structure, setup.sample_range, setup.prior

    prelim_file = joinpath(@__DIR__, "output", prelim_filename(cfg))
    isfile(prelim_file) || error("Prelim file not found: $(prelim_file)\nRun l96_preliminaries.jl first.")
    prelim = load_preliminaries(prelim_file)
    x0                     = prelim.x0
    y                      = prelim.y
    lorenz_config_settings = prelim.lorenz_config_settings
    observation_config     = prelim.observation_config
    R                      = prelim.R
    ic_cov_sqrt            = prelim.ic_cov_sqrt

    # ── GNKI calibration ────────────────────────────────────────────────
    initial_params = construct_initial_ensemble(rng, prior, N_ens)
    process = GaussNewtonInversion(prior)

    ekpobj = EKP.EnsembleKalmanProcess(
        initial_params, y, R, deepcopy(process);
        rng = copy(rng),
        verbose = verbose_flag,
        localization_method = NoLocalization(),
        scheduler = DefaultScheduler(),
    )

    for i in 1:cfg.N_iter
        params_i = get_ϕ_final(prior, ekpobj)
        G_ens = hcat(
            [
                lorenz_forward(
                    build_forcing(phi, params_i[:, j], phi_structure, sample_range),
                    x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx),
                    lorenz_config_settings,
                    observation_config,
                ) for j in 1:N_ens
            ]...,
        )
        terminated = EKP.update_ensemble!(ekpobj, G_ens)
        isnothing(terminated) || break
    end

    ϕ_stored = get_ϕ(prior, ekpobj)  # [iteration][param, ens_member]; ϕ_stored[1] = prior draw

    JLD2.save(
        joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
        "phi_stored", ϕ_stored,
        "prior", prior,
        "y", y,
        "R", R,
        "x0", x0,
        "ic_cov_sqrt", ic_cov_sqrt,
        "lorenz_config_settings", lorenz_config_settings,
        "observation_config", observation_config,
        "truth_phi", phi,
        "phi_structure", phi_structure,
        "sample_range", sample_range,
    )
    @info "Calibrate done: N_ens=$N_ens, rng_idx=$rng_idx ($(length(ϕ_stored) - 1) EKI iterations stored)"
end

function main()
    experiment = l96_experiment()
    @assert experiment in (:l96_const, :l96_vec, :l96_flux) "calibrate_l96.jl requires EXPERIMENT to be :l96_const, :l96_vec, or :l96_flux (got $experiment)"
    cfg = experiment_config(experiment)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))
    mkpath(output_dir)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        calibrate_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
