# Approximate Bayesian Computation — pushforward stage, L96 (const / vec / flux forcing)
#
# For each calibrate cell and each round k = k0, ..., K (k0 = first round
# whose accumulated accepted-sample pool has at least `nu + 2` members; K =
# min(rounds stored, cfg.max_iter)): fit a Gaussian to the raw accepted pool
# (in unconstrained space, so the fit respects parameter constraints) and push
# forward n_pushforward_samples resampled from it through the Lorenz-96
# forward map (fresh IC perturbation per sample). The raw pool size is often
# small relative to the forcing parameterization (e.g. flux-force has 61
# params) and grows unevenly (ABC rejection sampling), which makes coverage
# estimated directly from it noisy; resampling from the Gaussian implied by
# the pool's own mean/cov gives a much larger, fixed-size sample set (matching
# calibrate_emulate_sample's n_pushforward_samples = 1000) and so less
# quantile-estimation noise in the coverage metric. It does NOT reduce the
# underlying estimation error of the mean/cov themselves, which is still
# limited by the raw pool size, and it assumes the pool is well-approximated
# by a Gaussian in unconstrained space. Rounds before k0 (pool too small for a
# non-degenerate covariance estimate) are skipped entirely — since the pool
# only grows, once k0 is reached every later round also qualifies.
#
# Local (all cells):  EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
# Local (one cell):   EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl <task_index>
# SLURM:              invoked via pushforward_from_posterior.sbatch

using Distributions
using LinearAlgebra
using Random
using Statistics
using JLD2
using Flux
using EnsembleKalmanProcesses.ParameterDistributions

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

const n_pushforward_samples = 1000

function pushforward_one(cfg, N_ens, rng_idx, output_dir)
    fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
    if !isfile(fn)
        @warn "No calibrate results for $(case_suffix(cfg, N_ens, rng_idx)); skipping."
        return
    end
    if JLD2.jldopen(f -> haskey(f, "pushforward_output_samples"), fn, "r")
        @info "Pushforward already present in $(fn); skipping."
        return
    end
    phi_stored, pool_sizes, prior, x0, ic_cov_sqrt, lorenz_config_settings, observation_config, truth_phi, phi_structure, sample_range =
        JLD2.jldopen(fn, "r") do f
            (
                f["phi_stored"], f["pool_sizes"], f["prior"], f["x0"], f["ic_cov_sqrt"], f["lorenz_config_settings"], f["observation_config"],
                f["truth_phi"], f["phi_structure"], f["sample_range"],
            )
        end
    nx = length(x0)
    nu = size(phi_stored[end], 1)
    n_output = 2 * nx

    K_total  = min(length(phi_stored), cfg.max_iter)
    min_pool = nu + 2
    k0       = findfirst(k -> pool_sizes[k] >= min_pool, 1:K_total)
    if k0 === nothing
        @warn "Accepted-sample pool never reached minimum size ($(min_pool)) within $(K_total) rounds for N_ens=$(N_ens), rng_idx=$(rng_idx); skipping pushforward."
        return
    end
    k_values = collect(k0:K_total)
    K = length(k_values)
    output_arr = Array{Float64}(undef, n_pushforward_samples, n_output, K)

    rng = MersenneTwister(rng_idx + 1_000_000)
    for (ki, k) in enumerate(k_values)
        ensemble = phi_stored[k]  # nu x pool_size(k) (constrained space)

        # ── Gaussian resample (in unconstrained space) of the accepted pool ──
        u_ens = transform_constrained_to_unconstrained(prior, ensemble)
        μ = vec(mean(u_ens, dims = 2))
        Σ = Symmetric(cov(u_ens, dims = 2) + 1e-10 * I)
        u_samples = rand(rng, MvNormal(μ, Σ), n_pushforward_samples)
        φ_samples = transform_unconstrained_to_constrained(prior, u_samples)

        @info "Pushforward k=$(k), N_ens=$(N_ens), rng_idx=$(rng_idx): $(n_pushforward_samples) Lorenz96 evals (Gaussian-resampled, pool size $(pool_sizes[k]))"
        for s in 1:n_pushforward_samples
            emc = build_forcing(truth_phi, φ_samples[:, s], phi_structure, sample_range)
            output_arr[s, :, ki] = lorenz_forward(
                emc,
                x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx),
                lorenz_config_settings,
                observation_config,
            )
        end
    end

    JLD2.jldopen(fn, "r+") do f
        f["pushforward_output_samples"] = output_arr    # (n_pushforward_samples, n_output, K)
        f["pushforward_k_values"]       = k_values
        f["pushforward_n_samples"]      = n_pushforward_samples
    end
    @info "Pushforward done: N_ens=$N_ens, rng_idx=$rng_idx"
end

function main()
    experiment = l96_experiment()
    @assert experiment in (:l96_const, :l96_vec, :l96_flux) "pushforward_from_posterior_l96.jl requires EXPERIMENT to be :l96_const, :l96_vec, or :l96_flux (got $experiment)"
    cfg = experiment_config(experiment)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        pushforward_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
