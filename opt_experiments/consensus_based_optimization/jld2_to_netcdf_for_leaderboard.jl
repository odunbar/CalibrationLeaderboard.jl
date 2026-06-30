# CBO — leaderboard netcdf writer
# Reads per-cell JLD2 result files and writes one leaderboard netcdf.
# Run after all run_l63 / run_l96 cells have completed.
#
# Local: julia --project=. jld2_to_netcdf_for_leaderboard.jl
#        EXPERIMENT=l96_const julia --project=. jld2_to_netcdf_for_leaderboard.jl

using JLD2

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")

function main()
    experiment = l96_experiment()
    cfg        = experiment_config(experiment)
    tasks      = flat_tasks(cfg)
    output_dir = joinpath(@__DIR__, "output")

    n_rng  = cfg.n_repeats
    n_ens  = length(cfg.N_ens_sizes)
    n_rmse = length(cfg.rmse_targets)

    conv_scores = fill(NaN, n_rng, n_ens, n_rmse)

    for (N_ens, rmse_target, rng_idx) in tasks
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rmse_target, rng_idx))
        if !isfile(fn)
            @warn "Missing: $fn"
            continue
        end
        d  = JLD2.load(fn)
        ee = findfirst(==(N_ens), cfg.N_ens_sizes)
        rr = findfirst(==(rmse_target), cfg.rmse_targets)
        conv_scores[rng_idx, ee, rr] = d["conv_score"]
    end

    nc_path = joinpath(output_dir, nc_filename(cfg))
    write_results_nc(
        nc_path;
        random_seed    = collect(1:n_rng),
        ensemble_size  = Float64.(cfg.N_ens_sizes),
        rmse_target    = Float64.(cfg.rmse_targets),
        algorithm_type = [method_tag(cfg.cbo_method)],
        metric         = reshape(conv_scores, n_rng, n_ens, n_rmse, 1),
    )
    @info "Leaderboard written: $nc_path"
end

main()
