# Approximate Bayesian Computation — calibrate stage, L63
#
# ABC rejection sampling: draw i.i.d. candidates from the prior, forward-
# evaluate each through Lorenz63, and accept iff the (2R)-whitened
# implausibility-squared against `y` is below a Chi-squared(ny) threshold.
#
# Per rng_idx, draw one flat stream of N_ens_max*N_iter candidates; for each
# N_ens, phi_stored[k] is the accepted pool among the first N_ens*k draws.
# The candidate loop runs on Threads.nthreads() threads (set JULIA_NUM_THREADS).
#
# Local (all cells):  julia --project=. calibrate_l63.jl
# Local (one cell):   julia --project=. calibrate_l63.jl <rng_idx>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l63.jl

using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

########################################################################

function calibrate_one(cfg, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)

    # ── Problem setup (shared across all cells of this case) ───────────
    nx = 3   # state dimension
    nu = 2   # parameter dimension (rho, beta)
    ny = 9   # summary-statistic dimension
    truth_params = EnsembleMemberConfig([28.0, 8.0 / 3.0])

    prelim_file = joinpath(@__DIR__, "output", prelim_filename(cfg))
    isfile(prelim_file) || error("Prelim file not found: $(prelim_file)\nRun l63_preliminaries.jl first.")
    prelim = load_preliminaries(prelim_file)
    x0                     = prelim.x0
    y                      = prelim.y
    lorenz_config_settings = prelim.lorenz_config_settings
    observation_config     = prelim.observation_config
    R                      = prelim.R
    ic_cov_sqrt            = prelim.ic_cov_sqrt

    # ── Prior (shared with GaussNewtonKalmanInversion / calibrate_emulate_sample) ──
    prior_r = constrained_gaussian("rho", exp(3.3), 4.153, 0, Inf)
    prior_b = constrained_gaussian("beta", exp(1.2), 2.016, 0, Inf)
    prior = combine_distributions([prior_r, prior_b])

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
        ic_rng = MersenneTwister(hash((rng_idx, m)))
        G_m = lorenz_forward(
            EnsembleMemberConfig(θ_m),
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
            "truth_params_constrained", truth_params.u,
            "threshold", threshold,
        )
        @info "Calibrate done: N_ens=$N_ens, rng_idx=$rng_idx (final pool size $(pool_sizes[end]) after $(cfg.N_iter) rounds)"
    end
end

function main()
    cfg = experiment_config(:l63)
    tidx = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))
    mkpath(output_dir)

    run_cells = tidx === nothing ? (1:cfg.n_repeats) : [tidx]
    for rng_idx in run_cells
        calibrate_one(cfg, rng_idx, output_dir)
    end
end

main()
