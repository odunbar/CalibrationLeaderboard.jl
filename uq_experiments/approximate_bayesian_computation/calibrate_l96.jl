# Approximate Bayesian Computation — calibrate stage, L96 (const / vec / flux forcing)
#
# ABC rejection sampling: draw i.i.d. candidates from the prior, forward-
# evaluate each through Lorenz96, and accept iff the (2R)-whitened
# implausibility-squared against `y` is below a Chi-squared(ny) threshold.
#
# Per rng_idx, draw one flat stream of N_ens_max*N_iter candidates; for each
# N_ens, phi_stored[k] is the accepted pool among the first N_ens*k draws.
# The candidate loop runs on Threads.nthreads() threads (set JULIA_NUM_THREADS).
#
# Local (all cells):  EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
# Local (one cell):   EXPERIMENT=l96_const julia --project=. calibrate_l96.jl <rng_idx>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l96.jl

using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Flux

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

########################################################################

function force_case_setup(force_case::AbstractString)
    if force_case == "const-force"
        nx = 40
        nu = 1
        phi = ConstantEMC(8.0)
        phi_structure = nothing
        sample_range = nothing
        prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
    elseif force_case == "vec-force"
        nx = 40
        nu = nx
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
    elseif force_case == "flux-force"
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

        prior_cov = (0.1^2) * Diagonal(prior_mean .^ 2)
        prior = ParameterDistribution(
            Parameterized(MvNormal(prior_mean, prior_cov)),
            repeat([no_constraint()], length(prior_mean)),
            "l96_nn_prior",
        )
    else
        throw(ArgumentError("Unknown force_case: $force_case"))
    end
    return (nx = nx, nu = nu, phi = phi, phi_structure = phi_structure, sample_range = sample_range, prior = prior)
end

function calibrate_one(cfg, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)
    force_case = cfg.force_case
    setup = force_case_setup(force_case)
    nx, nu, phi, phi_structure, sample_range, prior =
        setup.nx, setup.nu, setup.phi, setup.phi_structure, setup.sample_range, setup.prior
    ny = 2 * nx

    prelim_file = joinpath(@__DIR__, "output", prelim_filename(cfg))
    isfile(prelim_file) || error("Prelim file not found: $(prelim_file)\nRun l96_preliminaries.jl first.")
    prelim = load_preliminaries(prelim_file)
    x0                     = prelim.x0
    y                      = prelim.y
    lorenz_config_settings = prelim.lorenz_config_settings
    observation_config     = prelim.observation_config
    R                      = prelim.R
    ic_cov_sqrt            = prelim.ic_cov_sqrt

    # ── ABC rejection criterion ─────────────────────────────────────────
    Σ_abc     = Symmetric(2 .* R)
    threshold = quantile(Chisq(ny), 1 - cfg.alpha_reject)

    # ── Flat i.i.d. ABC draw stream (one batch at the largest budget needed) ─
    N_ens_max = maximum(cfg.N_ens_sizes)
    M_max     = N_ens_max * cfg.N_iter

    u_candidates = construct_initial_ensemble(rng, prior, M_max)                      # nu x M_max, unconstrained
    θ_candidates = transform_unconstrained_to_constrained(prior, u_candidates)        # nu x M_max, constrained

    # Each m writes only accepted_mask[m], so this is thread-safe as-is; the
    # per-draw RNG keeps IC perturbations independent of thread scheduling.
    accepted_mask = falses(M_max)
    Threads.@threads for m in 1:M_max
        θ_m = θ_candidates[:, m]
        forcing = build_forcing(phi, θ_m, phi_structure, sample_range)
        ic_rng = MersenneTwister(hash((rng_idx, m)))
        G_m = lorenz_forward(
            forcing,
            x0 .+ ic_cov_sqrt * rand(ic_rng, Normal(0.0, 1.0), nx),
            lorenz_config_settings,
            observation_config,
        )
        Δ = G_m - y
        implaus_sq = dot(Δ, Σ_abc \ Δ)
        accepted_mask[m] = implaus_sq <= threshold
    end
    accepted_idx    = findall(accepted_mask)      # increasing, within 1:M_max
    accepted_params = θ_candidates[:, accepted_idx]

    # ── Per N_ens, phi_stored[k] = accepted pool among the first N_ens*k draws ──
    for N_ens in cfg.N_ens_sizes
        phi_stored = Vector{Matrix{Float64}}(undef, cfg.N_iter)  # phi_stored[k] = nu x pool_size(k), accepted pool through round k
        pool_sizes = zeros(Int, cfg.N_iter)
        for k in 1:cfg.N_iter
            n_acc         = searchsortedlast(accepted_idx, N_ens * k)
            phi_stored[k] = accepted_params[:, 1:n_acc]
            pool_sizes[k] = n_acc
        end

        JLD2.save(
            joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
            "phi_stored", phi_stored,
            "pool_sizes", pool_sizes,
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
            "threshold", threshold,
        )
        @info "Calibrate done: N_ens=$N_ens, rng_idx=$rng_idx (final pool size $(pool_sizes[end]) after $(cfg.N_iter) rounds)"
    end
end

function main()
    experiment = l96_experiment()
    @assert experiment in (:l96_const, :l96_vec, :l96_flux) "calibrate_l96.jl requires EXPERIMENT to be :l96_const, :l96_vec, or :l96_flux (got $experiment)"
    cfg = experiment_config(experiment)
    tidx = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))
    mkpath(output_dir)

    run_cells = tidx === nothing ? (1:cfg.n_repeats) : [tidx]
    for rng_idx in run_cells
        calibrate_one(cfg, rng_idx, output_dir)
    end
end

main()
