# HistoryMatching — leaderboard netcdf writer
#
# Loads each cell's NROY posterior sample sets (drawn by calibrate_<MODEL>.jl's
# wave loop) and pushforward output samples (written by
# pushforward_from_posterior_<MODEL>.jl), computes per-wave parameter-space
# mean/cov and R-whitened PCA output
# coverage + budget-to-target, and writes the leaderboard netcdf (full +
# minimal). k_iter here indexes History Matching WAVES, not EKI/RF training
# iterations — output_budget_to_target = N_ens * wave. Run serially after all
# pushforward cells have completed.
#
# Output-space coverage is R-whitened PCA coverage (same metric definition as
# uq_experiments/GaussNewtonKalmanInversion/exp_to_leaderboard.jl and
# uq_experiments/calibrate_emulate_sample/exp_to_leaderboard.jl): decorrelate
# via R's eigenbasis, retain the top modes needed for `retain_var` of R's
# variance, and compute marginal coverage of the whitened truth against
# whitened pushforward samples.
#
# Local: julia --project=. exp_to_leaderboard.jl
# SLURM: invoked via exp_to_leaderboard.sbatch (single job)

using NCDatasets
using JLD2
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "common", "uq_metrics", "coverage_metrics.jl"))
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
    experiment      = l96_experiment()
    cfg             = experiment_config(experiment)
    tasks           = flat_tasks(cfg)
    output_dir      = joinpath(@__DIR__, "output", calib_directory(cfg))
    nc_save_path    = joinpath(@__DIR__, "output", nc_filename(cfg))
    nc_minimal_path = joinpath(@__DIR__, "output", replace(nc_filename(cfg), r"\.nc$" => "_minimal.nc"))

    N_enss     = cfg.N_ens_sizes
    rng_idxs   = collect(1:cfg.n_repeats)
    n_ens      = length(N_enss)
    n_rng      = length(rng_idxs)
    n_k        = cfg.max_waves
    retain_var = cfg.retain_var

    # ── Locate valid cells (calibrate + pushforward both completed) ────
    valid_items = Tuple{Int, Int}[]
    for (N_ens, rng_idx) in tasks
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        has_pushforward = isfile(fn) && JLD2.jldopen(f -> haskey(f, "pushforward_output_samples"), fn, "r")
        has_pushforward && push!(valid_items, (N_ens, rng_idx))
    end
    isempty(valid_items) && error("No cells with pushforward output found in $(output_dir). Run calibrate + pushforward first.")

    # ── Determine n_params / n_output from a first pass over valid cells ──
    n_params = 0
    n_output = 0
    for (N_ens, rng_idx) in valid_items
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        theta1, pf_output = JLD2.jldopen(fn, "r") do f
            f["posteriors_by_k"][1], f["pushforward_output_samples"]
        end
        n_params = max(n_params, size(theta1, 1))
        n_output = max(n_output, size(pf_output, 2))
    end

    # ── R-whitened PCA basis (y, R shared across all cells of this experiment) ──
    fn1 = joinpath(output_dir, results_filename(cfg, valid_items[1]...))
    y, R_obs = JLD2.jldopen(f -> (f["y"], f["R"]), fn1, "r")

    basis = whitened_pca_basis(R_obs, retain_var)
    k_R   = basis.k_R
    yw    = whiten_vector(basis, y)

    @info "R-whitened PCA: retaining $(k_R)/$(n_output) modes ($(round(100 * basis.cum_var[k_R]; digits = 2))% variance)"

    # ── Pre-allocate ────────────────────────────────────────────────────
    post_mean_arr       = fill(NaN, n_rng, n_ens, n_k, n_params)
    post_cov_arr        = fill(NaN, n_rng, n_ens, n_k, n_params, n_params)
    output_coverage_arr = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)

    # ── Main loop over cells ────────────────────────────────────────────
    for (N_ens, rng_idx) in valid_items
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        posteriors_by_k, pf_output, pf_k_values = JLD2.jldopen(fn, "r") do f
            f["posteriors_by_k"], f["pushforward_output_samples"], f["pushforward_k_values"]
        end

        ri = findfirst(==(rng_idx), rng_idxs)
        ei = findfirst(==(N_ens), N_enss)

        for (ki, k) in enumerate(pf_k_values)
            theta_k = posteriors_by_k[k]   # n_params x n_samples (NROY draw at wave k)
            # A wave's NROY draw can come up short of n_posterior_samples (or
            # even empty) if rejection sampling hit max_rejection_samples —
            # leave this (cell, k) as NaN rather than erroring on an
            # underdetermined mean/cov or an empty quantile.
            size(theta_k, 2) < 2 && continue
            post_mean_arr[ri, ei, k, :]   = vec(mean(theta_k, dims = 2))
            post_cov_arr[ri, ei, k, :, :] = cov(theta_k, dims = 2)

            os = pf_output[:, :, ki]   # (n_pushforward_samples, n_output); may contain trailing NaN rows
            valid_rows = .!vec(any(isnan, os, dims = 2))
            os_valid = os[valid_rows, :]
            size(os_valid, 1) < 2 && continue
            sw = whiten_samples(basis, os_valid)   # (n_valid_samples, k_R)
            output_coverage_arr[ri, ei, k, :] = marginal_coverage(sw, yw, marginal_coverage_quantiles)
        end
    end

    # ── Budget-to-target (smallest N_ens*wave reaching calibrated coverage) ─
    output_budget_to_target = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    output_iters_to_target  = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    for ri in 1:n_rng, ei in 1:n_ens
        coverage_by_k = [output_coverage_arr[ri, ei, k, :] for k in 1:n_k]
        budget, iters = budget_to_target(coverage_by_k, N_enss[ei], marginal_coverage_quantiles, budget_target_scalings, k_R)
        output_budget_to_target[ri, ei, :, :] = budget
        output_iters_to_target[ri, ei, :, :]  = iters
    end

    output_coverage_description = "R-whitened PCA marginal coverage: fraction of whitened output dims d where ỹ[d] ≤ q_p of whitened NROY-pushforward samples at History Matching wave k_iter. Whitening: x̃_d = (Vᵀx)_d / √λ_d where R = VΛVᵀ. Retained $(k_R)/$(n_output) R-eigenmodes ($(round(100 * basis.cum_var[k_R]; digits = 1))% variance, threshold $(retain_var))."
    output_budget_description   = "Budget (N_ens·k_iter, k_iter = History Matching wave) to first reach |S(q)−q| ≤ c·√(q(1−q)/N_y) per quantile q using R-whitened PCA coverage (N_y = $(k_R) effective whitened dims). NaN = not reached."
    output_iters_description    = "History Matching wave k_iter to first reach R-whitened PCA coverage target per quantile. NaN = not reached."
    ens_description              = "Ensemble size N_ens used per History Matching wave"
    k_description                = "History Matching wave index (1-indexed)"
    cov_q_description            = "Quantile levels used for marginal coverage fraction metrics"
    ts_description                = "Scaling c in α_c(q) = c·√(q(1−q)/N_y); tolerance for budget_to_target / iters_to_target"

    # ── Write full netcdf (param-space mean/cov + coverage) ─────────────
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
    ens_var.attrib["description"] = ens_description
    ens_var[:] = N_enss

    k_var = defVar(ds, "k_iter", Int64, ("k_iter",))
    k_var.attrib["description"] = k_description
    k_var[:] = collect(1:n_k)

    cov_q_var = defVar(ds, "coverage_quantile", Float64, ("coverage_quantile",))
    cov_q_var.attrib["description"] = cov_q_description
    cov_q_var[:] = marginal_coverage_quantiles

    ts_var = defVar(ds, "target_scaling", Float64, ("target_scaling",))
    ts_var.attrib["description"] = ts_description
    ts_var[:] = budget_target_scalings

    post_mean_v = defVar(ds, "post_mean", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim"); fillvalue = NaN)
    post_mean_v.attrib["description"] = "mean(NROY posterior sample set) in parameter (constrained) space at each History Matching wave"
    post_mean_v[:, :, :, :] = post_mean_arr

    post_cov_v = defVar(ds, "post_cov", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim", "param_dim_2"); fillvalue = NaN)
    post_cov_v.attrib["description"] = "cov(NROY posterior sample set) in parameter (constrained) space at each History Matching wave"
    post_cov_v[:, :, :, :, :] = post_cov_arr

    output_coverage_v = defVar(ds, "output_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue = NaN)
    output_coverage_v.attrib["description"] = output_coverage_description
    output_coverage_v[:, :, :, :] = output_coverage_arr

    output_budget_v = defVar(ds, "output_budget_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue = NaN)
    output_budget_v.attrib["description"] = output_budget_description
    output_budget_v[:, :, :, :] = output_budget_to_target

    output_iters_v = defVar(ds, "output_iters_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue = NaN)
    output_iters_v.attrib["description"] = output_iters_description
    output_iters_v[:, :, :, :] = output_iters_to_target

    close(ds)
    @info "Leaderboard netcdf written: $nc_save_path"

    # ── Write minimal netcdf (coverage-derived fields only, no post_mean/post_cov) ─
    ds_min = NCDataset(nc_minimal_path, "c")

    defDim(ds_min, "random_seed",       n_rng)
    defDim(ds_min, "ensemble_size",     n_ens)
    defDim(ds_min, "k_iter",            n_k)
    defDim(ds_min, "coverage_quantile", n_marginal_coverage_quantiles)
    defDim(ds_min, "target_scaling",    n_target_scalings)
    defDim(ds_min, "output_dim",        k_R)   # effective whitened dimension (not full output_dim = n_output)

    rng_var_min = defVar(ds_min, "random_seed", Int64, ("random_seed",))
    rng_var_min[:] = rng_idxs

    ens_var_min = defVar(ds_min, "ensemble_size", Int64, ("ensemble_size",))
    ens_var_min.attrib["description"] = ens_description
    ens_var_min[:] = N_enss

    k_var_min = defVar(ds_min, "k_iter", Int64, ("k_iter",))
    k_var_min.attrib["description"] = k_description
    k_var_min[:] = collect(1:n_k)

    cov_q_var_min = defVar(ds_min, "coverage_quantile", Float64, ("coverage_quantile",))
    cov_q_var_min.attrib["description"] = cov_q_description
    cov_q_var_min[:] = marginal_coverage_quantiles

    ts_var_min = defVar(ds_min, "target_scaling", Float64, ("target_scaling",))
    ts_var_min.attrib["description"] = ts_description
    ts_var_min[:] = budget_target_scalings

    output_coverage_v_min = defVar(ds_min, "output_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue = NaN)
    output_coverage_v_min.attrib["description"] = output_coverage_description
    output_coverage_v_min[:, :, :, :] = output_coverage_arr

    output_budget_v_min = defVar(ds_min, "output_budget_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue = NaN)
    output_budget_v_min.attrib["description"] = output_budget_description
    output_budget_v_min[:, :, :, :] = output_budget_to_target

    output_iters_v_min = defVar(ds_min, "output_iters_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue = NaN)
    output_iters_v_min.attrib["description"] = output_iters_description
    output_iters_v_min[:, :, :, :] = output_iters_to_target

    close(ds_min)
    @info "Minimal leaderboard netcdf written: $nc_minimal_path"
end

main()
