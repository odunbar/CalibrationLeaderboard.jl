# HistoryMatching — calibrate stage (the wave loop), L96 (const / vec / flux forcing)
#
# Runs the Bayesian History Matching wave loop for one (N_ens, rng_idx) cell,
# for one of the three L96 force cases. See calibrate_l63.jl's header for the
# general algorithm description (in particular why there is no separate
# "emulate_sample" stage — emulation happens inside every wave here, and each
# wave's NROY draw serves as both that wave's stored posterior and the next
# wave's training ensemble); this file additionally builds the per-case
# forcing object/prior, mirroring
# uq_experiments/calibrate_emulate_sample/calibrate_l96.jl's `build_setup`.
#
# Every case whitens the GP's output space against R and its input space
# against the prior covariance (both fixed for the whole cell, known before
# any wave runs — see history_matching_core.jl's header). l96_flux's theta is
# additionally the flattened weight vector of a small NN (Flux.destructure),
# which is non-identifiable under hidden-unit permutation and tanh sign-flip
# symmetries — the prior alone can't reveal that (it's a property of the
# forward map, not the prior), so `fit_wave_gps` is called with
# `ensemble_retain_var = cfg.retain_var_input` for flux-force only, adding a
# wave-local PCA (from that wave's own ensemble) on top of the prior-whitened
# coordinates.
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
include("history_matching_core.jl")

# Size of the NROY sample set stored as each wave's "posterior" and pushed
# forward by pushforward_from_posterior_l96.jl — matches the fixed sample
# count used elsewhere on the leaderboard (calibrate_emulate_sample's
# n_pushforward_samples).
const n_posterior_samples = 1000

########################################################################
###############  Per-force-case setup  #################################
########################################################################
# Priors reuse the exact same physical mean/std/covariance already
# established in uq_experiments/calibrate_emulate_sample/calibrate_l96.jl,
# just expressed via Distributions.jl linear algebra instead of
# EnsembleKalmanProcesses.ParameterDistributions (this method has no EKP
# dependency of its own).
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
###############  Per-cell wave loop  ####################################
########################################################################

function history_matching_one(cfg, N_ens, rng_idx, output_dir)
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

    ensemble_retain_var = cfg.force_case == "flux-force" ? cfg.retain_var_input : nothing
    prior_sampler = (rng_, n) -> setup.constraint_transform.(setup.prior_mean .+ setup.prior_cov_sqrt * randn(rng_, nu, n))

    # Fixed for the whole cell: whiten the GP's output space against R and its
    # input space against the prior covariance, both known before any wave
    # runs (see history_matching_core.jl's header for why this matters).
    prob = make_hm_problem(
        y, R, setup.prior_mean, Matrix(setup.prior_cov), setup.inverse_transform;
        retain_var_output = cfg.retain_var, retain_var_input = cfg.retain_var_input,
    )
    threshold = quantile(Chisq(prob.output_basis.k_R), cfg.confidence)

    waves = WaveGPs[]
    posteriors_by_k = Dict{Int, Matrix{Float64}}()
    n_waves_completed = 0
    theta_ens = lhs_prior_sample(setup.prior_mean, setup.prior_cov_sqrt, N_ens, rng; constraint_transform = setup.constraint_transform)

    for wave in 1:cfg.max_waves
        results = zeros(N_ens, n_out)
        for j in 1:N_ens
            forcing_j = build_forcing(setup.phi, theta_ens[:, j], setup.phi_structure, setup.sample_range)
            results[j, :] = lorenz_forward(forcing_j, x0 .+ ic_cov_sqrt * randn(rng, nx), lorenz_cfg, obs_cfg)
        end

        wave_gps = fit_wave_gps(prob, theta_ens, results; ensemble_retain_var = ensemble_retain_var)
        push!(waves, wave_gps)
        n_waves_completed = wave

        # One NROY draw serves both roles: it IS this wave's posterior, and
        # its first N_ens columns become the next wave's training ensemble.
        posterior, ok = rejection_sample_nroy(
            prob, prior_sampler, nu, waves, threshold,
            n_posterior_samples, cfg.max_rejection_samples, cfg.n_candidate_batch, rng,
        )
        posteriors_by_k[wave] = posterior
        ok || @warn "NROY draw at wave $wave only found $(size(posterior, 2))/$(n_posterior_samples) samples within max_rejection_samples (N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))."
        @info "History matching wave $wave/$(cfg.max_waves) done (N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))"

        wave == cfg.max_waves && break

        if size(posterior, 2) < N_ens
            @warn "Fewer than N_ens=$N_ens NROY samples found; stopping the wave loop early at wave $wave (N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case))."
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
        "truth_phi", setup.phi, "phi_structure", setup.phi_structure, "sample_range", setup.sample_range,
    )
    @info "Calibrate (history matching) done: N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case), waves_completed=$n_waves_completed"
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
        history_matching_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
