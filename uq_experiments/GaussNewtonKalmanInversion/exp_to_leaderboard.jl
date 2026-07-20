# GaussNewtonKalmanInversion — leaderboard netcdf writer
#
# Loads the stored ensemble history and pushforward output samples from every
# calibrate cell, computes per-iteration ensemble mean/covariance (parameter
# space, from the raw N_ens ensemble) and output-space coverage + budget-to-
# target (from the Gaussian-resampled pushforward samples written by
# pushforward_from_posterior_l*.jl), and writes two leaderboard netcdfs: the
# full one (param-space mean/cov included) and a minimal one (coverage-derived
# fields only). Run serially after all pushforward cells have completed.
#
# Output-space coverage is R-whitened PCA coverage, not raw per-dimension
# marginal coverage: treating each output dimension as an independent trial
# double-counts correlated components whenever the observation-noise
# covariance R has off-diagonal structure. We decorrelate via R's eigenbasis
# (R = VΛVᵀ), keep the top modes needed to retain R_variance_retain of R's
# variance, and compute coverage of the whitened truth ỹ = Vᵀy/√λ against the
# whitened pushforward samples. Matches the metric in
# calibrate_emulate_sample/hpc-variant/minimal_leaderboard_nc.jl.
#
# Local: julia --project=. exp_to_leaderboard.jl
# SLURM: invoked via exp_to_leaderboard.sbatch (single job)

using NCDatasets
using JLD2
using Statistics

include(joinpath(@__DIR__, "..", "..", "common", "uq_metrics", "coverage_metrics.jl"))
include("experiment_config.jl")

########################################################################
###############  Metric parameters  ###################################
########################################################################
R_variance_retain             = 0.99   # fraction of R eigenvalue variance retained for whitened PCA coverage
marginal_coverage_quantiles   = collect(0.05:0.05:0.95)
budget_target_scalings        = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
n_marginal_coverage_quantiles = length(marginal_coverage_quantiles)
n_target_scalings             = length(budget_target_scalings)

########################################################################
###############  Main  #################################################
########################################################################

function main()
    experiment       = l96_experiment()
    cfg              = experiment_config(experiment)
    tasks            = flat_tasks(cfg)
    output_dir       = joinpath(@__DIR__, "output", calib_directory(cfg))
    nc_save_path     = joinpath(@__DIR__, "output", nc_filename(cfg))
    nc_minimal_path  = joinpath(@__DIR__, "output", replace(nc_filename(cfg), r"\.nc$" => "_minimal.nc"))

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
    n_k      = 0
    n_params = 0
    n_output = 0
    for (N_ens, rng_idx) in valid_items
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        phi1, pf_output, pf_k_values = JLD2.jldopen(fn, "r") do f
            f["phi_stored"][1], f["pushforward_output_samples"], f["pushforward_k_values"]
        end
        n_k      = max(n_k, length(pf_k_values))
        n_params = max(n_params, size(phi1, 1))
        n_output = max(n_output, size(pf_output, 2))
    end

    # ── R-whitened PCA basis (y, R are shared across all cells of this experiment) ─
    fn1 = joinpath(output_dir, results_filename(cfg, valid_items[1]...))
    y, R_obs = JLD2.jldopen(f -> (f["y"], f["R"]), fn1, "r")

    basis = whitened_pca_basis(R_obs, R_variance_retain)
    k_R   = basis.k_R
    yw    = whiten_vector(basis, y)

    @info "R-whitened PCA: retaining $(k_R)/$(n_output) modes ($(round(100*basis.cum_var[k_R]; digits=2))% variance)"

    # ── Pre-allocate ────────────────────────────────────────────────────
    post_mean_arr       = fill(NaN, n_rng, n_ens, n_k, n_params)
    post_cov_arr        = fill(NaN, n_rng, n_ens, n_k, n_params, n_params)
    output_coverage_arr = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)

    # ── Main loop over cells ────────────────────────────────────────────
    for (N_ens, rng_idx) in valid_items
        fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
        ϕ_stored, pf_output, pf_k_values = JLD2.jldopen(fn, "r") do f
            f["phi_stored"], f["pushforward_output_samples"], f["pushforward_k_values"]
        end
        # ϕ_stored: [iteration][param, ens_member]; index 1 = prior draw
        # pf_output: (n_pushforward_samples, n_output, K); Gaussian-resampled from the ensemble
        # (see pushforward_from_posterior_l*.jl), so its sample count is fixed across all cells

        ri = findfirst(==(rng_idx), rng_idxs)
        ei = findfirst(==(N_ens), N_enss)

        for (ki, k) in enumerate(pf_k_values)
            ensemble = ϕ_stored[k + 1]  # nx × N_ens, offset by one to skip the prior draw
            post_mean_arr[ri, ei, k, :]    = vec(mean(ensemble, dims = 2))
            post_cov_arr[ri, ei, k, :, :]  = cov(ensemble, dims = 2)

            os = pf_output[:, :, ki]   # (n_pushforward_samples, n_output)
            sw = whiten_samples(basis, os)   # (n_pushforward_samples, k_R), R-whitened PCA
            output_coverage_arr[ri, ei, k, :] = marginal_coverage(sw, yw, marginal_coverage_quantiles)
        end
    end

    # ── Budget-to-target (smallest N_ens*k reaching calibrated coverage) ─
    output_budget_to_target = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    output_iters_to_target  = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    for ri in 1:n_rng, ei in 1:n_ens
        coverage_by_k = [output_coverage_arr[ri, ei, k, :] for k in 1:n_k]
        budget, iters = budget_to_target(coverage_by_k, N_enss[ei], marginal_coverage_quantiles, budget_target_scalings, k_R)
        output_budget_to_target[ri, ei, :, :] = budget
        output_iters_to_target[ri, ei, :, :]  = iters
    end

    output_coverage_description = "R-whitened PCA marginal coverage: fraction of whitened output dims d where ỹ[d] ≤ q_p of whitened ensemble-pushforward samples. Whitening: x̃_d = (Vᵀx)_d / √λ_d where R = VΛVᵀ. Retained $(k_R)/$(n_output) R-eigenmodes ($(round(100*basis.cum_var[k_R]; digits=1))% variance, threshold $(R_variance_retain))."
    output_budget_description   = "Budget (N_ens·k_iter) to first reach |S(q)−q| ≤ c·√(q(1−q)/N_y) per quantile q using R-whitened PCA coverage (N_y = $(k_R) effective whitened dims). NaN = not reached."
    output_iters_description    = "Iterations k_iter to first reach R-whitened PCA coverage target per quantile. NaN = not reached."
    ens_description             = "Number of ensemble members used to calibrate (post_mean/post_cov); output_coverage is instead computed from a fixed-size Gaussian resample of this ensemble (see pushforward_from_posterior_l*.jl)"
    k_description               = "Number of GNKI iterations (1-indexed)"
    cov_q_description           = "Quantile levels used for marginal coverage fraction metrics"
    ts_description               = "Scaling c in α_c(q) = c·√(q(1−q)/N_y); tolerance for budget_to_target / iters_to_target"

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
    post_mean_v.attrib["description"] = "mean(ensemble) in parameter (constrained) space at each GNKI iteration"
    post_mean_v[:, :, :, :] = post_mean_arr

    post_cov_v = defVar(ds, "post_cov", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim", "param_dim_2"); fillvalue = NaN)
    post_cov_v.attrib["description"] = "cov(ensemble) in parameter (constrained) space at each GNKI iteration"
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
