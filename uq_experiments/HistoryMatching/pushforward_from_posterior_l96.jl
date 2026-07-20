# HistoryMatching — pushforward stage, L96 (const / vec / flux forcing)
#
# For each calibrate cell and each completed wave k: push the NROY sample set
# drawn by calibrate_l96.jl's wave loop through the Lorenz-96 forward map
# (fresh IC perturbation per sample). Also records forcing-space samples
# (matching
# uq_experiments/calibrate_emulate_sample/pushforward_from_posterior_l96.jl's
# forcing_arr diagnostic), for parity with other L96 methods on the
# leaderboard. Appends output samples into the same results JLD2
# calibrate_l96.jl wrote.
#
# Local (all cells):  EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
# Local (one cell):   EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl <task_index>
# SLURM:              invoked via pushforward_from_posterior.sbatch

using Distributions
using LinearAlgebra
using Random
using JLD2
using Flux
# `results_filename` also holds `waves` (GaussianProcesses.GPE objects, from
# calibrate_l96.jl). This script never touches that key, but JLD2 resolves a
# file's whole committed-datatype table on open regardless of which keys are
# later read — without this `using`, it can't find the real GPE type and
# reconstructs a placeholder (harmless here, since we never read `waves`, but
# noisy and unnecessary).
using GaussianProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
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
    posteriors_by_k, k_values, x0, ic_cov_sqrt, lorenz_cfg, obs_cfg, truth_phi, phi_structure, sample_range =
        JLD2.jldopen(fn, "r") do f
            (
                f["posteriors_by_k"], f["k_values"], f["x0"], f["ic_cov_sqrt"], f["lorenz_cfg"], f["obs_cfg"],
                f["truth_phi"], f["phi_structure"], f["sample_range"],
            )
        end

    nx = length(x0)
    n_output = 2 * nx
    n_forcing = cfg.force_case == "flux-force" ? length(sample_range) : nx
    K = length(k_values)

    output_arr = zeros(n_pushforward_samples, n_output, K)
    forcing_arr = zeros(n_pushforward_samples, n_forcing, K)

    rng = MersenneTwister(rng_idx + 1_000_000)
    for (ki, k) in enumerate(k_values)
        theta_samples = posteriors_by_k[k]   # nu x n_samples_k (n_samples_k may be < n_pushforward_samples if rejection sampling fell short)
        n_use = min(size(theta_samples, 2), n_pushforward_samples)
        for s in 1:n_use
            emc = build_forcing(truth_phi, theta_samples[:, s], phi_structure, sample_range)
            output_arr[s, :, ki] = lorenz_forward(emc, x0 .+ ic_cov_sqrt * randn(rng, nx), lorenz_cfg, obs_cfg)
            forcing_arr[s, :, ki] = forcing(emc, x0)
        end
        if n_use < n_pushforward_samples
            output_arr[(n_use + 1):end, :, ki] .= NaN
            forcing_arr[(n_use + 1):end, :, ki] .= NaN
        end
    end

    JLD2.jldopen(fn, "r+") do f
        f["pushforward_output_samples"]  = output_arr
        f["pushforward_forcing_samples"] = forcing_arr
        f["pushforward_k_values"]        = collect(k_values)
        f["pushforward_n_samples"]       = n_pushforward_samples
    end
    @info "Pushforward done: N_ens=$N_ens, rng_idx=$rng_idx, case=$(cfg.force_case)"
end

function main()
    experiment = l96_experiment()
    @assert experiment in (:l96_const, :l96_vec, :l96_flux) "pushforward_from_posterior_l96.jl requires EXPERIMENT to be :l96_const, :l96_vec, or :l96_flux (got $experiment)"
    cfg = experiment_config(experiment)
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
