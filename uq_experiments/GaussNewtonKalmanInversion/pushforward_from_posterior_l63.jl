# GaussNewtonKalmanInversion — pushforward stage, L63
#
# For each calibrate cell: push the raw ensemble at each stored iteration
# k = 1, ..., K through the Lorenz-63 forward map (with a fresh IC
# perturbation per member). The ensemble itself is the UQ sample set — there
# is no emulator/MCMC upsampling, so the sample count at iteration k is just
# N_ens. Samples are padded with NaN up to max(N_ens_sizes) so all cells share
# one netcdf "pushforward_sample" dimension in the leaderboard writer.
#
# Local (all cells):  julia --project=. pushforward_from_posterior_l63.jl
# Local (one cell):   julia --project=. pushforward_from_posterior_l63.jl <task_index>
# SLURM:              invoked via pushforward_from_posterior.sbatch

using Distributions
using LinearAlgebra
using Random
using JLD2

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

function pushforward_one(cfg, N_ens, rng_idx, output_dir, max_n_ens)
    fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
    if !isfile(fn)
        @warn "No calibrate results for $(case_suffix(cfg, N_ens, rng_idx)); skipping."
        return
    end
    if JLD2.jldopen(f -> haskey(f, "pushforward_output_samples"), fn, "r")
        @info "Pushforward already present in $(fn); skipping."
        return
    end
    ϕ_stored, x0, ic_cov_sqrt, lorenz_config_settings, observation_config = JLD2.jldopen(fn, "r") do f
        f["phi_stored"], f["x0"], f["ic_cov_sqrt"], f["lorenz_config_settings"], f["observation_config"]
    end
    nx = length(x0)
    n_output = 9  # 3 means + 3 variances + 3 covariances

    K = min(length(ϕ_stored) - 1, cfg.max_iter)
    output_arr = fill(NaN, max_n_ens, n_output, K)

    rng = MersenneTwister(rng_idx + 1_000_000)
    for k in 1:K
        ensemble = ϕ_stored[k + 1]  # nx × N_ens (constrained space)
        @info "Pushforward k=$(k), N_ens=$(N_ens), rng_idx=$(rng_idx): $(N_ens) Lorenz63 evals"
        for j in 1:N_ens
            output_arr[j, :, k] = lorenz_forward(
                EnsembleMemberConfig(ensemble[:, j]),
                x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx),
                lorenz_config_settings,
                observation_config,
            )
        end
    end

    JLD2.jldopen(fn, "r+") do f
        f["pushforward_output_samples"] = output_arr    # (max_n_ens, n_output, K); NaN-padded past N_ens
        f["pushforward_k_values"]       = collect(1:K)
        f["pushforward_n_ens"]          = N_ens
    end
    @info "Pushforward done: N_ens=$N_ens, rng_idx=$rng_idx"
end

function main()
    cfg = experiment_config(:l63)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))
    max_n_ens = maximum(cfg.N_ens_sizes)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        pushforward_one(cfg, N_ens, rng_idx, output_dir, max_n_ens)
    end
end

main()
