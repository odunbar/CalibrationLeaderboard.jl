# HistoryMatching — calibrate stage (the wave loop), L63
#
# Runs the Bayesian History Matching wave loop for one (N_ens, rng_idx) cell:
# draw an ensemble (Latin-hypercube from the prior for wave 1, NROY rejection
# sampling for later waves), forward-evaluate it, fit one independent GP per
# (whitened) output statistic, and use the accumulated waves' implausibility
# to define the next wave's NROY region. The GP's output space is whitened
# against R and its input space against the prior covariance — see
# history_matching_core.jl's header for why both are available before any
# wave runs.
#
# There is no separate "emulate" or "sample" stage: emulation (the GP fit)
# happens inside every wave here, not in a distinct step, so a file named
# "emulate_sample" would do no emulation of its own. The "posterior at wave
# k" and "training ensemble for wave k+1" are also the same rejection-sampled
# draw (both are just NROY samples conditioned on waves 1..k) — so each wave
# draws ONE ~1000-sample batch, keeps the whole thing as that wave's stored
# posterior, and reuses its first N_ens columns as the next wave's training
# ensemble, rather than sampling twice. Goes straight from here to
# pushforward_from_posterior_l63.jl (see README.md; the same reasoning is why
# uq_experiments/GaussNewtonKalmanInversion also merges stages).
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
include("history_matching_core.jl")

# Independent log-normal prior on (rho, beta); sigma is fixed at 10 inside the
# forward map itself (common/forward_maps/Lorenz63.jl's `f`), matching both
# calib_race_hm_l63.py and the repo's other L63 experiments' 2-parameter setup.
const PRIOR_MEAN_LOG = [3.3, 1.2]
const PRIOR_STD_LOG  = [0.5, 0.15]

# Size of the NROY sample set stored as each wave's "posterior" and pushed
# forward by pushforward_from_posterior_l63.jl — matches the fixed sample
# count used elsewhere on the leaderboard (calibrate_emulate_sample's
# n_pushforward_samples).
const n_posterior_samples = 1000

function history_matching_one(cfg, N_ens, rng_idx, output_dir)
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

    prior_cov_sqrt = Diagonal(PRIOR_STD_LOG)
    prior_sampler = (rng_, n) -> exp.(PRIOR_MEAN_LOG .+ prior_cov_sqrt * randn(rng_, 2, n))

    # Fixed for the whole cell: whiten the GP's output space against R and its
    # input space against the prior covariance, both known before any wave
    # runs (see history_matching_core.jl's header for why this matters).
    prob = make_hm_problem(
        y, R, PRIOR_MEAN_LOG, Matrix(Diagonal(PRIOR_STD_LOG .^ 2)), log;
        retain_var_output = cfg.retain_var, retain_var_input = cfg.retain_var_input,
    )
    threshold = quantile(Chisq(prob.output_basis.k_R), cfg.confidence)

    waves = WaveGPs[]
    posteriors_by_k = Dict{Int, Matrix{Float64}}()
    n_waves_completed = 0
    theta_ens = lhs_prior_sample(PRIOR_MEAN_LOG, prior_cov_sqrt, N_ens, rng; constraint_transform = exp)

    for wave in 1:cfg.max_waves
        results = zeros(N_ens, n_out)
        for j in 1:N_ens
            results[j, :] = lorenz_forward(
                EnsembleMemberConfig(theta_ens[:, j]),
                x0 .+ ic_cov_sqrt * randn(rng, nx),
                lorenz_cfg, obs_cfg,
            )
        end

        wave_gps = fit_wave_gps(prob, theta_ens, results)
        push!(waves, wave_gps)
        n_waves_completed = wave

        # One NROY draw serves both roles: it IS this wave's posterior, and
        # its first N_ens columns become the next wave's training ensemble.
        posterior, ok = rejection_sample_nroy(
            prob, prior_sampler, 2, waves, threshold,
            n_posterior_samples, cfg.max_rejection_samples, cfg.n_candidate_batch, rng,
        )
        posteriors_by_k[wave] = posterior
        ok || @warn "NROY draw at wave $wave only found $(size(posterior, 2))/$(n_posterior_samples) samples within max_rejection_samples (N_ens=$N_ens, rng_idx=$rng_idx)."
        @info "History matching wave $wave/$(cfg.max_waves) done (N_ens=$N_ens, rng_idx=$rng_idx)"

        wave == cfg.max_waves && break

        if size(posterior, 2) < N_ens
            @warn "Fewer than N_ens=$N_ens NROY samples found; stopping the wave loop early at wave $wave (N_ens=$N_ens, rng_idx=$rng_idx)."
            break
        end
        theta_ens = posterior[:, 1:N_ens]
    end

    JLD2.save(
        joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
        "waves", waves,
        "posteriors_by_k", posteriors_by_k,
        "k_values", collect(1:n_waves_completed),
        "n_waves_completed", n_waves_completed,
        "y", y, "R", R, "x0", x0, "ic_cov_sqrt", ic_cov_sqrt,
        "lorenz_cfg", lorenz_cfg, "obs_cfg", obs_cfg,
    )
    @info "Calibrate (history matching) done: N_ens=$N_ens, rng_idx=$rng_idx, waves_completed=$n_waves_completed"
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
        history_matching_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
