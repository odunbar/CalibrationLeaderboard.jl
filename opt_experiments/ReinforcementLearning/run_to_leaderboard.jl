# ReinforcementLearning — leaderboard netcdf writer.
# Reads per-cell JLD2 result files and writes one leaderboard netcdf per (algorithm, update_frequency, case);
# algorithm_type is rl-{ddpg,td3}-{short,long}.
# Run after all run_{l63,l96}_ddpg.jl cells for that configuration have completed.
#
# Local: ALGORITHM=td3 UPDATE_FREQUENCY=short julia --project=. run_to_leaderboard.jl
#        ALGORITHM=ddpg UPDATE_FREQUENCY=long EXPERIMENT=l96_const julia --project=. run_to_leaderboard.jl

using JLD2
using Dates

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")

function main()
    experiment = l96_experiment()
    cfg        = experiment_config(experiment, rl_update_frequency(), rl_algorithm())
    tasks      = flat_tasks(cfg)
    output_dir = joinpath(@__DIR__, "output")

    conv_scores = fill(NaN, cfg.n_repeats, length(cfg.rmse_targets))

    for (rmse_target, rng_idx) in tasks
        fn = joinpath(output_dir, result_filename(cfg, rmse_target, rng_idx))
        if !isfile(fn)
            @warn "Missing: $fn"
            continue
        end
        d  = JLD2.load(fn)
        rr = findfirst(==(rmse_target), cfg.rmse_targets)
        conv_scores[rng_idx, rr] = d["conv_score"]
    end

    nc_path = joinpath(output_dir, nc_filename(cfg))
    write_results_nc(
        nc_path;
        random_seed    = collect(1:cfg.n_repeats),
        ensemble_size  = [1.0],
        rmse_target    = Float64.(cfg.rmse_targets),
        algorithm_type = [algorithm_name(cfg)],
        metric         = reshape(conv_scores, cfg.n_repeats, 1, length(cfg.rmse_targets), 1),
    )
    @info "Leaderboard written: $nc_path"
end

main()
