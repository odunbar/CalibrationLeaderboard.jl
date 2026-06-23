# <METHOD_NAME> — pushforward stage, L63
# For each posterior cell: draw samples from the posterior and push through the
# Lorenz-63 forward map. Saves output samples back into the posterior JLD2.
#
# Local (all cells):  julia --project=. pushforward_from_posterior_l63.jl
# Local (one cell):   julia --project=. pushforward_from_posterior_l63.jl <task_index>
# SLURM:              invoked via pushforward_from_posterior.sbatch

using Distributions
using LinearAlgebra
using Random
using JLD2

const _COMMON = joinpath(@__DIR__, "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

n_pushforward_samples = 1000

function pushforward_one(cfg, N_ens, rng_idx, output_dir)
    fn = joinpath(output_dir, posterior_filename(cfg, N_ens, rng_idx))
    d  = JLD2.load(fn)
    posteriors_by_k = d["posteriors_by_k"]
    x0 = d["x0"]; ic_cov_sqrt = d["ic_cov_sqrt"]
    lorenz_cfg = d["lorenz_cfg"]; obs_cfg = d["obs_cfg"]

    rng = MersenneTwister(rng_idx + 1000)
    nx = 3
    K = length(posteriors_by_k)
    ny = length(lorenz_forward(EnsembleMemberConfig([28.0, 8.0/3.0]), x0, lorenz_cfg, obs_cfg))

    pushforward_output_samples = zeros(n_pushforward_samples, ny, K)

    for k in 1:K
        chain = posteriors_by_k[k]
        # Sample from posterior (assumes chain provides get_samples or similar)
        # ╔══════════════════════════════════════════════════════════════╗
        # ║  REPLACE: extract posterior samples from your chain format   ║
        # ╚══════════════════════════════════════════════════════════════╝
        posterior_samples = chain[:, end-n_pushforward_samples+1:end]  # placeholder

        for s in 1:n_pushforward_samples
            param = exp.(posterior_samples[:, s])
            x0_perturbed = x0 .+ ic_cov_sqrt * randn(rng, nx)
            pushforward_output_samples[s, :, k] = lorenz_forward(
                EnsembleMemberConfig(param), x0_perturbed, lorenz_cfg, obs_cfg)
        end
    end

    JLD2.jldopen(fn, "r+") do f
        f["pushforward_output_samples"] = pushforward_output_samples
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
