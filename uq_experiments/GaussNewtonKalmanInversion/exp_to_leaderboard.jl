# GaussNewtonKalmanInversion — leaderboard netcdf writer
#
# Loads the stored ensemble history and pushforward output samples from every
# calibrate cell, computes per-iteration ensemble mean/covariance (parameter
# space) and marginal output-space coverage + budget-to-target, and writes the
# leaderboard netcdf. Run serially after all pushforward cells have completed.
#
# Local: julia --project=. exp_to_leaderboard.jl
# SLURM: invoked via exp_to_leaderboard.sbatch (single job)

using NCDatasets
using JLD2
using Statistics
using LinearAlgebra

include("experiment_config.jl")

########################################################################
###############  Metric parameters  ###################################
########################################################################
marginal_coverage_quantiles   = collect(0.05:0.05:0.95)
budget_target_scalings        = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
n_marginal_coverage_quantiles = length(marginal_coverage_quantiles)
n_target_scalings             = length(budget_target_scalings)

########################################################################
###############  Main  #################################################
########################################################################

function main()
    experiment   = l96_experiment()
    cfg          = experiment_config(experiment)
    tasks        = flat_tasks(cfg)
    output_dir   = joinpath(@__DIR__, "output", calib_directory(cfg))
    nc_save_path = joinpath(@__DIR__, "output", nc_filename(cfg))

    N_enss   = cfg.N_ens_sizes
    rng_idxs = collect(1:cfg.n_repeats)
    n_ens    = length(N_enss)
    n_rng    = length(rng_idxs)

    # ── Locate valid cells (calibrate + pushforward both completed) ────
    valid_items = Tuple{Int, Int}[]
    for (N_ens, rng_idx) in tasks
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        has_pushforward = isfile(fn) && JLD2.jldopen(f -> haskey(f, "pushforward_output_samples"), fn, "r")
        has_pushforward && push!(valid_items, (N_ens, rng_idx))
    end
    isempty(valid_items) && error("No cells with pushforward output found in $(output_dir). Run calibrate + pushforward first.")

    # ── Determine dimensions from a first pass over valid cells ────────
    n_k       = 0
    n_params  = 0
    n_output  = 0
    max_n_ens = maximum(N_enss)
    for (N_ens, rng_idx) in valid_items
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        phi1, pf_output, pf_k_values = JLD2.jldopen(fn, "r") do f
            f["phi_stored"][1], f["pushforward_output_samples"], f["pushforward_k_values"]
        end
        n_k      = max(n_k, length(pf_k_values))
        n_params = max(n_params, size(phi1, 1))
        n_output = max(n_output, size(pf_output, 2))
    end

    # ── Pre-allocate ────────────────────────────────────────────────────
    post_mean_arr      = fill(NaN, n_rng, n_ens, n_k, n_params)
    post_cov_arr       = fill(NaN, n_rng, n_ens, n_k, n_params, n_params)
    output_coverage_arr = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)

    # ── Main loop over cells ────────────────────────────────────────────
    for (N_ens, rng_idx) in valid_items
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        ϕ_stored, y, pf_output, pf_k_values, pf_n_ens = JLD2.jldopen(fn, "r") do f
            f["phi_stored"], f["y"], f["pushforward_output_samples"], f["pushforward_k_values"], f["pushforward_n_ens"]
        end
        # ϕ_stored: [iteration][param, ens_member]; index 1 = prior draw
        # pf_output: (max_n_ens, n_output, K)

        ri = findfirst(==(rng_idx), rng_idxs)
        ei = findfirst(==(N_ens), N_enss)

        for (ki, k) in enumerate(pf_k_values)
            ensemble = ϕ_stored[k + 1]  # nx × N_ens, offset by one to skip the prior draw
            post_mean_arr[ri, ei, k, :]    = vec(mean(ensemble, dims = 2))
            post_cov_arr[ri, ei, k, :, :]  = cov(ensemble, dims = 2)

            os = pf_output[1:pf_n_ens, :, ki]  # drop NaN padding past this cell's N_ens
            for (qi, qp) in enumerate(marginal_coverage_quantiles)
                output_coverage_arr[ri, ei, k, qi] = mean(y[d2] <= quantile(os[:, d2], qp) for d2 in 1:size(os, 2))
            end
        end
    end

    # ── Budget-to-target (smallest N_ens*k reaching calibrated coverage) ─
    output_budget_to_target = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    output_iters_to_target  = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    for (si, c) in enumerate(budget_target_scalings)
        tol = c .* sqrt.(marginal_coverage_quantiles .* (1 .- marginal_coverage_quantiles) ./ n_output)
        for ri in 1:n_rng, ei in 1:n_ens, (qi, qp) in enumerate(marginal_coverage_quantiles)
            for k in 1:n_k
                s = output_coverage_arr[ri, ei, k, qi]
                isnan(s) && continue
                if abs(s - qp) <= tol[qi]
                    output_budget_to_target[ri, ei, si, qi] = N_enss[ei] * k
                    output_iters_to_target[ri, ei, si, qi]  = k
                    break
                end
            end
        end
    end

    # ── Write netcdf ─────────────────────────────────────────────────────
    ds = NCDataset(nc_save_path, "c")

    defDim(ds, "random_seed",       n_rng)
    defDim(ds, "ensemble_size",     n_ens)
    defDim(ds, "k_iter",            n_k)
    defDim(ds, "param_dim",         n_params)
    defDim(ds, "param_dim_2",       n_params)
    defDim(ds, "output_dim",        n_output)
    defDim(ds, "coverage_quantile", n_marginal_coverage_quantiles)
    defDim(ds, "target_scaling",    n_target_scalings)

    rng_var = defVar(ds, "random_seed", Int64, ("random_seed",))
    rng_var[:] = rng_idxs

    ens_var = defVar(ds, "ensemble_size", Int64, ("ensemble_size",))
    ens_var.attrib["description"] = "Number of ensemble members (also the pushforward sample count at each iteration)"
    ens_var[:] = N_enss

    k_var = defVar(ds, "k_iter", Int64, ("k_iter",))
    k_var.attrib["description"] = "Number of GNKI iterations (1-indexed)"
    k_var[:] = collect(1:n_k)

    cov_q_var = defVar(ds, "coverage_quantile", Float64, ("coverage_quantile",))
    cov_q_var.attrib["description"] = "Quantile levels used for marginal coverage fraction metrics"
    cov_q_var[:] = marginal_coverage_quantiles

    ts_var = defVar(ds, "target_scaling", Float64, ("target_scaling",))
    ts_var.attrib["description"] = "Scaling c in α_c(q) = c·√(q(1−q)/N_y); tolerance for budget_to_target / iters_to_target"
    ts_var[:] = budget_target_scalings

    post_mean_v = defVar(ds, "post_mean", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim"); fillvalue = NaN)
    post_mean_v.attrib["description"] = "mean(ensemble) in parameter (constrained) space at each GNKI iteration"
    post_mean_v[:, :, :, :] = post_mean_arr

    post_cov_v = defVar(ds, "post_cov", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim", "param_dim_2"); fillvalue = NaN)
    post_cov_v.attrib["description"] = "cov(ensemble) in parameter (constrained) space at each GNKI iteration"
    post_cov_v[:, :, :, :, :] = post_cov_arr

    output_coverage_v = defVar(ds, "output_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue = NaN)
    output_coverage_v.attrib["description"] = "Marginal coverage in output space: fraction of output dims where y[d] ≤ q_p of the N_ens ensemble-pushforward samples."
    output_coverage_v[:, :, :, :] = output_coverage_arr

    output_budget_v = defVar(ds, "output_budget_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue = NaN)
    output_budget_v.attrib["description"] = "Budget (N_ens·k_iter) to first reach |S(q)−q| ≤ c·√(q(1−q)/N_y) per quantile q in output space. NaN = not reached."
    output_budget_v[:, :, :, :] = output_budget_to_target

    output_iters_v = defVar(ds, "output_iters_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue = NaN)
    output_iters_v.attrib["description"] = "Iterations k_iter to first reach coverage target per quantile in output space. NaN = not reached."
    output_iters_v[:, :, :, :] = output_iters_to_target

    close(ds)
    @info "Leaderboard netcdf written: $nc_save_path"
end

main()
