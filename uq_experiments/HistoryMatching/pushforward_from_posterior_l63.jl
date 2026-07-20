# HistoryMatching — pushforward stage, L63
#
# For each calibrate cell and each completed wave k: push the NROY sample set
# drawn by calibrate_l63.jl's wave loop through the Lorenz-63 forward map
# (fresh IC perturbation per sample). Appends output samples into the same
# results JLD2 calibrate_l63.jl wrote.
#
# Local (all cells):  julia --project=. pushforward_from_posterior_l63.jl
# Local (one cell):   julia --project=. pushforward_from_posterior_l63.jl <task_index>
# SLURM:              invoked via pushforward_from_posterior.sbatch

using Distributions
using LinearAlgebra
using Random
using JLD2
# `results_filename` also holds `waves` (GaussianProcesses.GPE objects, from
# calibrate_l63.jl). This script never touches that key, but JLD2 resolves a
# file's whole committed-datatype table on open regardless of which keys are
# later read — without this `using`, it can't find the real GPE type and
# reconstructs a placeholder (harmless here, since we never read `waves`, but
# noisy and unnecessary).
using GaussianProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

const n_pushforward_samples = 1000

function pushforward_one(cfg, N_ens, rng_idx, output_dir)
    fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
    if !isfile(fn)
        @warn "No calibrate results file for $(case_suffix(cfg, N_ens, rng_idx)); skipping."
        return
    end
    if JLD2.jldopen(f -> haskey(f, "pushforward_output_samples"), fn, "r")
        @info "Pushforward already present in $(fn); skipping."
        return
    end
    # Load only the keys actually needed here (avoids deserializing `waves`,
    # which can be large and isn't used by this stage).
    posteriors_by_k, k_values, x0, ic_cov_sqrt, lorenz_cfg, obs_cfg = JLD2.jldopen(fn, "r") do f
        f["posteriors_by_k"], f["k_values"], f["x0"], f["ic_cov_sqrt"], f["lorenz_cfg"], f["obs_cfg"]
    end

    nx = length(x0)
    n_output = 9   # 3 means + 3 variances + 3 covariances
    K = length(k_values)
    output_arr = zeros(n_pushforward_samples, n_output, K)

    rng = MersenneTwister(rng_idx + 1_000_000)
    for (ki, k) in enumerate(k_values)
        theta_samples = posteriors_by_k[k]   # 2 x n_samples_k (n_samples_k may be < n_pushforward_samples if rejection sampling fell short)
        n_use = min(size(theta_samples, 2), n_pushforward_samples)
        for s in 1:n_use
            output_arr[s, :, ki] = lorenz_forward(
                EnsembleMemberConfig(theta_samples[:, s]),
                x0 .+ ic_cov_sqrt * randn(rng, nx),
                lorenz_cfg, obs_cfg,
            )
        end
        n_use < n_pushforward_samples && (output_arr[(n_use + 1):end, :, ki] .= NaN)
    end

    JLD2.jldopen(fn, "r+") do f
        f["pushforward_output_samples"] = output_arr    # (n_pushforward_samples, n_output, K)
        f["pushforward_k_values"]       = collect(k_values)
        f["pushforward_n_samples"]      = n_pushforward_samples
    end
    @info "Pushforward done: N_ens=$N_ens, rng_idx=$rng_idx"
end

function main()
    cfg = experiment_config(:l63)
    tasks = flat_tasks(cfg)
    tidx = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        pushforward_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
