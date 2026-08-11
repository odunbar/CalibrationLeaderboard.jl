# score_based_lm — leaderboard netcdf writer
# Reads per-cell JLD2 result files and writes one leaderboard netcdf per arm.
# Run after all run_l63 / run_l96 cells for that arm have completed.
#
# Unlike levenberg_marquardt, the ensemble_size axis is REAL here (N_ens is the
# shared budget knob for both modes) rather than a singleton [1.0].
#
# Local: julia --project=. run_to_leaderboard.jl
#        EXPERIMENT=l96_const SCORE_KIND=dsm BUDGET_MODE=parallel julia --project=. run_to_leaderboard.jl

using JLD2
using Dates
using Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")

function main()
    experiment = l96_experiment()
    cfg        = experiment_config(experiment)
    tasks      = flat_tasks(cfg)
    output_dir = joinpath(@__DIR__, "output")

    n_ens_ax  = cfg.N_ens_sizes
    n_rmse_ax = cfg.rmse_targets

    conv_scores = fill(NaN, cfg.n_repeats, length(n_ens_ax), length(n_rmse_ax))
    fwd_actual  = fill(NaN, cfg.n_repeats, length(n_ens_ax), length(n_rmse_ax))
    n_missing = 0

    for (N_ens, rmse_target, rng_idx) in tasks
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rmse_target, rng_idx))
        if !isfile(fn)
            n_missing += 1
            continue
        end
        d  = JLD2.load(fn)
        ee = findfirst(==(N_ens), n_ens_ax)
        rr = findfirst(==(rmse_target), n_rmse_ax)
        conv_scores[rng_idx, ee, rr] = d["conv_score"]
        fwd_actual[rng_idx, ee, rr]  = get(d, "n_fwd_actual", NaN)
    end

    n_missing > 0 && @warn "$n_missing of $(length(tasks)) result files missing; those cells stay NaN"

    nc_path = joinpath(output_dir, nc_filename(cfg))
    write_results_nc(
        nc_path;
        random_seed    = collect(1:cfg.n_repeats),
        ensemble_size  = Float64.(n_ens_ax),
        rmse_target    = Float64.(n_rmse_ax),
        algorithm_type = [algorithm_type(cfg)],
        metric         = reshape(conv_scores,
                                 cfg.n_repeats, length(n_ens_ax), length(n_rmse_ax), 1),
    )
    @info "Leaderboard written: $nc_path  (algorithm_type = $(algorithm_type(cfg)))"

    # Convergence summary plus the audit counter.  conv_score is the headline
    # (comparable to every other entry); n_fwd_actual measures integration time in
    # base-run equivalents and is LOWER than the charge in :serial/:parallel mode,
    # which documents that both modes are conservatively charged.
    for (ee, N_ens) in enumerate(n_ens_ax), (rr, tgt) in enumerate(n_rmse_ax)
        col = @view conv_scores[:, ee, rr]
        nconv = count(!isnan, col)
        med = nconv > 0 ? median(filter(!isnan, col)) : NaN
        af = @view fwd_actual[:, ee, rr]
        medf = any(!isnan, af) ? median(filter(!isnan, af)) : NaN
        @info "N_ens=$N_ens rmse=$tgt: converged $nconv/$(cfg.n_repeats), median conv_score=$med, median n_fwd_actual=$(round(medf; sigdigits=4))"
    end
end

main()
