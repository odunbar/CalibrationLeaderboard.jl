# BayesianOptimalExperimentalDesign — calibrate stage (the GBOED loop), L96
# (const / vec / flux forcing)
#
# Runs the GBOED loop for one (N_ens, rng_idx) cell, for one of the three L96
# force cases. See calibrate_l63.jl's header for the general algorithm
# description; this file additionally builds the per-case forcing
# object/prior, reusing the exact same physical setup already established in
# uq_experiments/HistoryMatching/calibrate_l96.jl's `l96_case_setup` (this
# method has no EKP dependency of its own, so priors are expressed via
# Distributions.jl linear algebra rather than
# EnsembleKalmanProcesses.ParameterDistributions).
#
# Every case whitens the GP's output space against R and its input space
# against the prior covariance (both fixed for the whole cell — see
# boed_core.jl's header). l96_flux's theta is the flattened weight vector of
# a small NN (Flux.destructure); unlike History Matching, GBOED does not add
# a wave-local ensemble PCA on top of the prior whitening for this case (that
# mechanism exists specifically to cope with NROY rejection sampling
# fragmenting under weight-permutation symmetry — GBOED's gradient-based
# acquisition does not rejection-sample, so this is not currently needed; a
# candidate follow-up if flux-force GBOED convergence is poor).
#
# Local (all cells):  EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
# Local (one cell):   EXPERIMENT=l96_const julia --project=. calibrate_l96.jl <task_index>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l96.jl

using Distributions
using LinearAlgebra
using Random
using Statistics
using JLD2
using Flux

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")
include("boed_core.jl")

########################################################################
###############  Per-force-case setup  #################################
########################################################################
function l96_case_setup(force_case::AbstractString)
    if force_case == "const-force"
        nx, nu = 40, 1
        phi = ConstantEMC(8.0)
        phi_structure = nothing
        sample_range = nothing
        mu_log, sig_log = lognormal_params_from_moments(10.0, 4.0)
        prior_mean = [mu_log]
        prior_cov_sqrt = reshape([sig_log], 1, 1)
        prior_cov = reshape([sig_log^2], 1, 1)
        constraint_transform = exp
        inverse_transform = log

    elseif force_case == "vec-force"
        nx, nu = 40, 40
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        phi_structure = nothing
        sample_range = nothing
        pl, psig = 2.0, 3.0
        prior_cov = [psig^2 * exp(-abs(ii - jj) / pl) for ii in 1:nx, jj in 1:nx]
        prior_mean = 8.0 * ones(nx)
        prior_cov_sqrt = Matrix(cholesky(Symmetric(prior_cov)).L)
        constraint_transform = identity
        inverse_transform = identity

    elseif force_case == "flux-force"
        nx, nu = 100, 61
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
        _, prior_mean = train_network(deepcopy(phi_structure), x_train, prior_train)
        prior_cov_sqrt = Diagonal(0.1 .* abs.(prior_mean))
        prior_cov = Diagonal((0.1 .* abs.(prior_mean)) .^ 2)
        constraint_transform = identity
        inverse_transform = identity

    else
        throw(ArgumentError("Unknown force_case: $force_case"))
    end
    return (; nx, nu, phi, phi_structure, sample_range, prior_mean, prior_cov_sqrt, prior_cov, constraint_transform, inverse_transform)
end

########################################################################
###############  Per-cell GBOED loop  ###################################
########################################################################

function boed_one(cfg, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)
    setup = l96_case_setup(cfg.force_case)
    nx, nu = setup.nx, setup.nu

    prelim_file = joinpath(@__DIR__, "output", prelim_filename(cfg))
    isfile(prelim_file) || error("Prelim file not found: $(prelim_file)\nRun l96_preliminaries.jl first.")
    prelim = load_preliminaries(prelim_file)
    x0 = prelim.x0
    y = prelim.y
    R = prelim.R
    ic_cov_sqrt = prelim.ic_cov_sqrt
    lorenz_cfg = prelim.lorenz_config_settings
    obs_cfg = prelim.observation_config
    n_out = length(y)

    # Fixed for the whole cell: whiten the GP's output space against R and its
    # input space against the prior covariance (see boed_core.jl's header).
    prob = make_boed_problem(
        y, R, setup.prior_mean, Matrix(setup.prior_cov), setup.inverse_transform, setup.constraint_transform;
        retain_var_output = cfg.retain_var, retain_var_input = cfg.retain_var_input,
    )
    k_R_prior = prob.prior_basis.k_R

    # NOTE: the batch accumulator here must NOT be named `results` — `results`
    # is also assigned in this function's outer scope (the cumulative dataset
    # below), and Julia closures share (rather than shadow) an enclosing
    # local of the same name, so reusing that name would silently clobber the
    # cumulative dataset on every call.
    forward_eval_batch(theta_batch) = begin
        M = size(theta_batch, 2)
        batch_results = zeros(M, n_out)
        for j in 1:M
            t0 = time()
            forcing_j = build_forcing(setup.phi, theta_batch[:, j], setup.phi_structure, setup.sample_range)
            batch_results[j, :] = lorenz_forward(forcing_j, x0 .+ ic_cov_sqrt * randn(rng, nx), lorenz_cfg, obs_cfg)
            @info "forward_eval_batch: member $j/$M done" elapsed_s = round(time() - t0; digits = 2)
        end
        return batch_results
    end

    # Iteration 1: initial LHS design in the truncated whitened prior space,
    # decoded to raw theta, forward-evaluated, and used for the first GP fit
    # + first ST-MCMC posterior. No acquisition yet.
    Z = lhs_standard_normal_sample(k_R_prior, N_ens, rng)
    theta = from_prior_whitened(prob, Z)
    results = timed_stage(() -> forward_eval_batch(theta), "forward_eval_batch (iter 1, N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))")
    gps = timed_stage(() -> fit_boed_gps(prob, Z, results), "fit_boed_gps (iter 1, rng_idx=$rng_idx, case=$(cfg.force_case))")
    X_post = timed_stage(
        () -> run_tmcmc(prob, gps, cfg.n_posterior_samples, rng; burnin = cfg.tmcmc_burnin, thin = cfg.tmcmc_thin),
        "run_tmcmc (iter 1, rng_idx=$rng_idx, case=$(cfg.force_case))",
    )

    gps_by_k = Dict{Int, BOEDGPs}(1 => gps)
    posteriors_by_k = Dict{Int, Matrix{Float64}}(1 => from_prior_whitened(prob, X_post))
    n_iters_completed = 1
    @info "GBOED iteration 1/$(cfg.max_iters) done (N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))"

    for k in 2:cfg.max_iters
        hyperparams = extract_hyperparams(gps)
        X_cand0 = init_candidate_batch(X_post, N_ens, rng; strategy = cfg.batch_init_strategy)
        X_cand = timed_stage(
            () -> optimize_batch(
                X_cand0, X_post, hyperparams, k_R_prior, N_ens;
                bound_std = cfg.eig_bounds_std, iters = cfg.eig_optim_iters, outer_iters = cfg.eig_outer_iters,
                jitter = cfg.eig_jitter,
            ),
            "optimize_batch (iter $k, rng_idx=$rng_idx, case=$(cfg.force_case))",
        )
        theta_cand = from_prior_whitened(prob, X_cand)
        results_cand = timed_stage(
            () -> forward_eval_batch(theta_cand),
            "forward_eval_batch (iter $k, N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))",
        )

        Z = hcat(Z, X_cand)
        results = vcat(results, results_cand)
        gps = timed_stage(() -> fit_boed_gps(prob, Z, results), "fit_boed_gps (iter $k, rng_idx=$rng_idx, case=$(cfg.force_case))")
        X_post = timed_stage(
            () -> run_tmcmc(prob, gps, cfg.n_posterior_samples, rng; burnin = cfg.tmcmc_burnin, thin = cfg.tmcmc_thin),
            "run_tmcmc (iter $k, rng_idx=$rng_idx, case=$(cfg.force_case))",
        )

        gps_by_k[k] = gps
        posteriors_by_k[k] = from_prior_whitened(prob, X_post)
        n_iters_completed = k
        @info "GBOED iteration $k/$(cfg.max_iters) done (N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))"
    end

    JLD2.save(
        joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
        "gps_by_k", gps_by_k,
        "posteriors_by_k", posteriors_by_k,
        "k_values", collect(1:n_iters_completed),
        "n_iters_completed", n_iters_completed,
        "y", y, "R", R, "x0", x0, "ic_cov_sqrt", ic_cov_sqrt,
        "lorenz_cfg", lorenz_cfg, "obs_cfg", obs_cfg,
        "truth_phi", setup.phi, "phi_structure", setup.phi_structure, "sample_range", setup.sample_range,
    )
    @info "Calibrate (GBOED) done: N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case), iters_completed=$n_iters_completed"
end

function main()
    experiment = l96_experiment()
    @assert experiment in (:l96_const, :l96_vec, :l96_flux) "calibrate_l96.jl requires EXPERIMENT to be :l96_const, :l96_vec, or :l96_flux (got $experiment)"
    cfg = experiment_config(experiment)
    tasks = flat_tasks(cfg)
    tidx = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))
    mkpath(output_dir)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        boed_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
