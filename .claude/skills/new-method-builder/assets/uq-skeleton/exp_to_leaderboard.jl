# <METHOD_NAME> — leaderboard netcdf writer
# Loads pushforward output samples from all posterior JLD2 files, computes
# coverage metrics, and writes the leaderboard netcdf.
# Run serially after all pushforward cells have completed.
#
# Local: julia --project=. exp_to_leaderboard.jl
# SLURM: invoked via exp_to_leaderboard.sbatch (single job)

using NCDatasets
using Dates
using JLD2
using Statistics
using Distributions
using LinearAlgebra

const _COMMON = joinpath(@__DIR__, "..", "common")
# include(joinpath(_COMMON, "uq_metrics", "coverage_metrics.jl"))  # uncomment when populated
# include(joinpath(_COMMON, "uq_metrics", "write_uq_nc.jl"))       # uncomment when populated

include("experiment_config.jl")

if haskey(ENV, "EXPERIMENT")
    EXPERIMENT = Symbol(ENV["EXPERIMENT"])
end

########################################################################
###############  Metric parameters  ###################################
########################################################################
marginal_coverage_quantiles   = collect(0.05:0.05:0.95)
budget_target_scalings        = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
n_marginal_coverage_quantiles = length(marginal_coverage_quantiles)
n_target_scalings             = length(budget_target_scalings)

########################################################################
###############  Coverage computation  ################################
########################################################################

function compute_coverage(output_samples::AbstractMatrix, quantile_probs)
    # output_samples: (n_samples, n_output)
    n_output = size(output_samples, 2)
    [mean(output_samples[:, d] .<= quantile(output_samples[:, d], q) for d in 1:n_output)
     for q in quantile_probs]
end

function compute_budget_to_target(output_coverage_by_k, N_ens, quantile_probs, scalings)
    n_output_est = 100  # approximate for the calibration test (set to actual n_output if known)
    K = length(output_coverage_by_k)
    budget = fill(NaN, length(scalings))
    iters  = fill(NaN, length(scalings))
    for (ci, c) in enumerate(scalings)
        for k in 1:K
            cov_k = output_coverage_by_k[k]
            if all(abs.(cov_k .- quantile_probs) .<= c .* sqrt.(quantile_probs .* (1 .- quantile_probs) ./ n_output_est))
                budget[ci] = N_ens * k
                iters[ci]  = k
                break
            end
        end
    end
    return budget, iters
end

########################################################################
###############  Main  ################################################
########################################################################

function main()
    experiment = l96_experiment()
    cfg = experiment_config(experiment)
    tasks = flat_tasks(cfg)

    output_dir   = joinpath(@__DIR__, "output", calib_directory(cfg))
    nc_save_path = joinpath(@__DIR__, "output", nc_filename(cfg))

    # Collect results
    n_ens   = length(cfg.N_ens_sizes)
    n_reps  = cfg.n_repeats
    K       = cfg.max_iter
    n_q     = n_marginal_coverage_quantiles
    n_scale = n_target_scalings

    # These will be filled in the loop; adjust dims as needed for your output size
    coverage_arr      = fill(NaN, n_ens, n_reps, K, n_q)
    budget_arr        = fill(NaN, n_ens, n_reps, n_scale)
    iters_arr         = fill(NaN, n_ens, n_reps, n_scale)

    for (N_ens, rng_idx) in tasks
        fn = joinpath(output_dir, posterior_filename(cfg, N_ens, rng_idx))
        !isfile(fn) && (@warn "Missing: $fn"; continue)
        d = JLD2.load(fn)
        !haskey(d, "pushforward_output_samples") && (@warn "No pushforward in: $fn"; continue)

        ee = findfirst(==(N_ens), cfg.N_ens_sizes)
        samples_by_k = d["pushforward_output_samples"]  # (n_samples, n_output, K)

        cov_by_k = []
        for k in 1:K
            cov_k = compute_coverage(samples_by_k[:, :, k], marginal_coverage_quantiles)
            coverage_arr[ee, rng_idx, k, :] = cov_k
            push!(cov_by_k, cov_k)
        end

        budget, iters = compute_budget_to_target(cov_by_k, N_ens, marginal_coverage_quantiles, budget_target_scalings)
        budget_arr[ee, rng_idx, :] = budget
        iters_arr[ee, rng_idx, :]  = iters
    end

    # Write netcdf
    ds = NCDataset(nc_save_path, "c")

    defDim(ds, "ensemble_size",     n_ens)
    defDim(ds, "random_seed",       n_reps)
    defDim(ds, "k_iter",            K)
    defDim(ds, "coverage_quantile", n_q)
    defDim(ds, "target_scaling",    n_scale)

    v_ens   = defVar(ds, "ensemble_size",     Float64, ("ensemble_size",))
    v_seed  = defVar(ds, "random_seed",       Int64,   ("random_seed",))
    v_k     = defVar(ds, "k_iter",            Int64,   ("k_iter",))
    v_q     = defVar(ds, "coverage_quantile", Float64, ("coverage_quantile",))
    v_scale = defVar(ds, "target_scaling",    Float64, ("target_scaling",))

    v_ens[:]   = Float64.(cfg.N_ens_sizes)
    v_seed[:]  = collect(1:n_reps)
    v_k[:]     = collect(1:K)
    v_q[:]     = marginal_coverage_quantiles
    v_scale[:] = budget_target_scalings

    v_cov = defVar(ds, "output_coverage",         Float64, ("ensemble_size","random_seed","k_iter","coverage_quantile"), fillvalue=NaN)
    v_bud = defVar(ds, "output_budget_to_target",  Float64, ("ensemble_size","random_seed","target_scaling"), fillvalue=NaN)
    v_itr = defVar(ds, "output_iters_to_target",   Float64, ("ensemble_size","random_seed","target_scaling"), fillvalue=NaN)

    v_cov.attrib["description"] = "Marginal coverage fraction at each quantile"
    v_bud.attrib["description"] = "N_ens × iterations to reach calibrated coverage target"
    v_itr.attrib["description"] = "Iterations to reach calibrated coverage target"

    v_cov[:] = coverage_arr
    v_bud[:] = budget_arr
    v_itr[:] = iters_arr

    close(ds)
    @info "Leaderboard netcdf written: $nc_save_path"
end

main()
