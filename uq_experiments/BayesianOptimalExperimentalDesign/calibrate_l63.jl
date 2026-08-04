# BayesianOptimalExperimentalDesign — calibrate stage (the GBOED loop), L63
#
# Runs the Goal-Oriented Bayesian Optimal Experimental Design loop (paper
# Algorithm 2 — see boed_core.jl's header) for one (N_ens, rng_idx) cell:
# an initial Latin-hypercube design in the truncated whitened prior space,
# forward-evaluated and used to fit one independent GP per whitened output
# statistic; then, for each subsequent iteration, ST-MCMC posterior sampling
# from the current GP fit, a joint-batch L-BFGS-B optimization of a new
# candidate design to maximize the closed-form EIG against those posterior
# samples, a forward evaluation of the optimized batch, and a refit of the GP
# on the CUMULATIVE (ever-growing) dataset.
#
# Unlike History Matching, there is no separate "emulate" or "sample" stage:
# the GP fit happens inside every iteration here, and ST-MCMC sampling is
# itself part of the per-iteration loop (both to target the EIG acquisition
# and, at the final iteration, to produce the stored "posterior"). Goes
# straight from here to pushforward_from_posterior_l63.jl (see README.md;
# the same reasoning is why uq_experiments/HistoryMatching and
# uq_experiments/GaussNewtonKalmanInversion also merge stages).
#
# Local (all cells):  julia --project=. calibrate_l63.jl
# Local (one cell):   julia --project=. calibrate_l63.jl <task_index>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l63.jl

using Distributions
using LinearAlgebra
using Random
using JLD2

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")
include("boed_core.jl")

# Independent log-normal prior on (rho, beta); sigma is fixed at 10 inside the
# forward map itself (common/forward_maps/Lorenz63.jl's `f`), matching both
# HistoryMatching's and the repo's other L63 experiments' 2-parameter setup.
const PRIOR_MEAN_LOG = [3.3, 1.2]
const PRIOR_STD_LOG = [0.5, 0.15]

function boed_one(cfg, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)
    nx = 3

    prelim_file = joinpath(@__DIR__, "output", prelim_filename(cfg))
    isfile(prelim_file) || error("Prelim file not found: $(prelim_file)\nRun l63_preliminaries.jl first.")
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
        y, R, PRIOR_MEAN_LOG, Matrix(Diagonal(PRIOR_STD_LOG .^ 2)), log, exp;
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
            batch_results[j, :] = lorenz_forward(
                EnsembleMemberConfig(theta_batch[:, j]),
                x0 .+ ic_cov_sqrt * randn(rng, nx),
                lorenz_cfg, obs_cfg,
            )
            @info "forward_eval_batch: member $j/$M done" elapsed_s = round(time() - t0; digits = 2)
        end
        return batch_results
    end

    # Initial LHS design in the truncated whitened prior space,
    Z = lhs_standard_normal_sample(k_R_prior, N_ens, rng)
    # decoded to raw theta, forward-evaluated
    theta = from_prior_whitened(prob, Z)
    results = timed_stage(() -> forward_eval_batch(theta), "forward_eval_batch (iter 1, N_ens=$N_ens, rng_idx=$rng_idx)")
    # Fit the first GP
    gps = timed_stage(() -> fit_boed_gps(prob, Z, results), "fit_boed_gps (iter 1, rng_idx=$rng_idx)")
    # Use ST-MCMC (sequential parallel sampler for the posterior
    X_post = timed_stage(
        () -> run_tmcmc(prob, gps, cfg.n_posterior_samples, rng; burnin = cfg.tmcmc_burnin, thin = cfg.tmcmc_thin),
        "run_tmcmc (iter 1, rng_idx=$rng_idx)",
    )

    gps_by_k = Dict{Int, BOEDGPs}(1 => gps)
    posteriors_by_k = Dict{Int, Matrix{Float64}}(1 => from_prior_whitened(prob, X_post))
    n_iters_completed = 1
    @info "GBOED iteration 1/$(cfg.max_iters) done (N_ens=$N_ens, rng_idx=$rng_idx)"

    # Calibration loop: each adds one EIG-optimized acquisition batch,
    for k in 2:cfg.max_iters
        hyperparams = extract_hyperparams(gps)
        # sample candidates from X_post
        X_cand0 = init_candidate_batch(X_post, N_ens, rng; strategy = cfg.batch_init_strategy)

        # Optimizes EIG at the posterior samples
        X_cand = timed_stage(
            () -> optimize_batch(
                X_cand0, X_post, hyperparams, k_R_prior, N_ens;
                bound_std = cfg.eig_bounds_std, iters = cfg.eig_optim_iters, outer_iters = cfg.eig_outer_iters,
                jitter = cfg.eig_jitter, g_tol = cfg.eig_g_tol, f_reltol = cfg.eig_f_reltol, call_limit = cfg.eig_call_limit,
            ),
            "optimize_batch (iter $k, rng_idx=$rng_idx)",
        )
        # decode and foward evaluate again
        theta_cand = from_prior_whitened(prob, X_cand)
        results_cand = timed_stage(() -> forward_eval_batch(theta_cand), "forward_eval_batch (iter $k, N_ens=$N_ens, rng_idx=$rng_idx)")

        Z = hcat(Z, X_cand)
        #  augment dataset, and refit GP
        results = vcat(results, results_cand)
        gps = timed_stage(() -> fit_boed_gps(prob, Z, results), "fit_boed_gps (iter $k, rng_idx=$rng_idx)")

        # re-samples the ST-MCMC posterior, for the next iteration
        X_post = timed_stage(
            () -> run_tmcmc(prob, gps, cfg.n_posterior_samples, rng; burnin = cfg.tmcmc_burnin, thin = cfg.tmcmc_thin),
            "run_tmcmc (iter $k, rng_idx=$rng_idx)",
        )
        gps_by_k[k] = gps
        posteriors_by_k[k] = from_prior_whitened(prob, X_post)
        n_iters_completed = k
        @info "GBOED iteration $k/$(cfg.max_iters) done (N_ens=$N_ens, rng_idx=$rng_idx)"
    end

    JLD2.save(
        joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
        "gps_by_k", gps_by_k,
        "posteriors_by_k", posteriors_by_k,
        "k_values", collect(1:n_iters_completed),
        "n_iters_completed", n_iters_completed,
        "y", y, "R", R, "x0", x0, "ic_cov_sqrt", ic_cov_sqrt,
        "lorenz_cfg", lorenz_cfg, "obs_cfg", obs_cfg,
    )
    @info "Calibrate (GBOED) done: N_ens=$N_ens, rng_idx=$rng_idx, iters_completed=$n_iters_completed"
end

function main()
    cfg = experiment_config(:l63)
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
