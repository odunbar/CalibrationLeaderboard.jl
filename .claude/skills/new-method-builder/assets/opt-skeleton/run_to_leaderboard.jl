# <METHOD_NAME> — leaderboard netcdf writer
# Reads all per-cell JLD2 result files and writes a single leaderboard netcdf.
# Run after all run_l63/l96 cells have completed.
#
# Local: julia --project=. run_to_leaderboard.jl
# SLURM: invoked via leaderboard.sbatch (single job, after run_array completes)

using JLD2
using Dates

const _COMMON = joinpath(@__DIR__, "..", "common")
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))

include("experiment_config.jl")

# Allow EXPERIMENT env var to override the toggle in experiment_config.jl
if haskey(ENV, "EXPERIMENT")
    EXPERIMENT = Symbol(ENV["EXPERIMENT"])
end

function main()
    experiment = l96_experiment()   # for l63, this returns :l63 (from the toggle)
    cfg = experiment_config(experiment)
    tasks = flat_tasks(cfg)

    output_dir = joinpath(@__DIR__, "output")

    # Collect results across all cells
    conv_scores   = fill(NaN, length(cfg.N_ens_sizes), cfg.n_repeats)
    final_params  = []
    final_outputs = []

    for (N_ens, rng_idx) in tasks
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rng_idx))
        if !isfile(fn)
            @warn "Missing result file: $fn  (cell not yet run?)"
            continue
        end
        d = JLD2.load(fn)
        ee = findfirst(==(N_ens), cfg.N_ens_sizes)
        conv_scores[ee, rng_idx] = d["conv_score"]
    end

    nc_path = joinpath(output_dir, nc_filename(cfg))
    write_results_nc(
        nc_path;
        random_seed    = collect(1:cfg.n_repeats),
        ensemble_size  = Float64.(cfg.N_ens_sizes),
        rmse_target    = [cfg.target_rmse],
        algorithm_type = ["<METHOD_NAME>"],
        metric         = reshape(conv_scores, length(cfg.N_ens_sizes), cfg.n_repeats, 1, 1),
    )
    @info "Leaderboard written: $nc_path"
end

main()
