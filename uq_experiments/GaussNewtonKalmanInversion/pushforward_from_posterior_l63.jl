# GaussNewtonKalmanInversion — pushforward stage, L63
#
# For each calibrate cell and each stored iteration k = 1, ..., K: fit a
# Gaussian to the raw N_ens ensemble (in unconstrained space, so the fit
# respects parameter constraints) and push forward n_pushforward_samples
# resampled from it through the Lorenz-63 forward map (fresh IC perturbation
# per sample). N_ens is often small (as low as 4 for L63), which makes
# coverage estimated directly from it noisy; resampling from the Gaussian
# implied by the ensemble's own mean/cov — the quantity GNKI actually reports
# as its posterior approximation — gives a much larger, fixed-size sample set
# (matching calibrate_emulate_sample's n_pushforward_samples = 1000) and so
# less quantile-estimation noise in the coverage metric. It does NOT reduce
# the underlying estimation error of the mean/cov themselves, which is still
# limited by N_ens, and it assumes the ensemble is well-approximated by a
# Gaussian in unconstrained space.
#
# Local (all cells):  julia --project=. pushforward_from_posterior_l63.jl
# Local (one cell):   julia --project=. pushforward_from_posterior_l63.jl <task_index>
# SLURM:              invoked via pushforward_from_posterior.sbatch

using Distributions
using LinearAlgebra
using Random
using Statistics
using JLD2
using EnsembleKalmanProcesses.ParameterDistributions

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
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
    ϕ_stored, prior, x0, ic_cov_sqrt, lorenz_config_settings, observation_config = JLD2.jldopen(fn, "r") do f
        f["phi_stored"], f["prior"], f["x0"], f["ic_cov_sqrt"], f["lorenz_config_settings"], f["observation_config"]
    end
    nx = length(x0)
    n_output = 9  # 3 means + 3 variances + 3 covariances

    K = min(length(ϕ_stored) - 1, cfg.max_iter)
    output_arr = Array{Float64}(undef, n_pushforward_samples, n_output, K)

    rng = MersenneTwister(rng_idx + 1_000_000)
    for k in 1:K
        ensemble = ϕ_stored[k + 1]  # n_params × N_ens (constrained space)

        # ── Gaussian resample (in unconstrained space) of the ensemble ──
        u_ens = transform_constrained_to_unconstrained(prior, ensemble)
        μ = vec(mean(u_ens, dims = 2))
        Σ = Symmetric(cov(u_ens, dims = 2) + 1e-10 * I)
        u_samples = rand(rng, MvNormal(μ, Σ), n_pushforward_samples)
        φ_samples = transform_unconstrained_to_constrained(prior, u_samples)

        @info "Pushforward k=$(k), N_ens=$(N_ens), rng_idx=$(rng_idx): $(n_pushforward_samples) Lorenz63 evals (Gaussian-resampled)"
        for s in 1:n_pushforward_samples
            output_arr[s, :, k] = lorenz_forward(
                EnsembleMemberConfig(φ_samples[:, s]),
                x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx),
                lorenz_config_settings,
                observation_config,
            )
        end
    end

    JLD2.jldopen(fn, "r+") do f
        f["pushforward_output_samples"] = output_arr    # (n_pushforward_samples, n_output, K)
        f["pushforward_k_values"]       = collect(1:K)
        f["pushforward_n_samples"]      = n_pushforward_samples
    end
    @info "Pushforward done: N_ens=$N_ens, rng_idx=$rng_idx"
end

function main()
    cfg = experiment_config(:l63)
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
