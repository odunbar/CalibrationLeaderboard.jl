using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[3]

# Pinned at submission time via RUN_DATE env var (set by submit_*.sh).
# Falls back to today() for local runs.
run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
function experiment_config(case::Symbol)
    n_repeats   = 100
    rmse_targets = [1.0, 1.1, 1.2]
    N_iter      = 10000   # Adam needs more iterations than LM

    if case == :l63
        return (
            model        = "l63",
            force_case   = nothing,
            rmse_targets = rmse_targets,
            n_repeats    = n_repeats,
            N_iter       = N_iter,
            run_date     = run_date,
        )
    elseif case == :l96_const
        return (
            model        = "l96",
            force_case   = "const-force",
            rmse_targets = rmse_targets,
            n_repeats    = n_repeats,
            N_iter       = N_iter,
            run_date     = run_date,
        )
    elseif case == :l96_vec
        return (
            model        = "l96",
            force_case   = "vec-force",
            rmse_targets = rmse_targets,
            n_repeats    = n_repeats,
            N_iter       = N_iter,
            run_date     = run_date,
        )
    elseif case == :l96_flux
        return (
            model        = "l96",
            force_case   = "flux-force",
            rmse_targets = rmse_targets,
            n_repeats    = n_repeats,
            N_iter       = N_iter,
            run_date     = run_date,
        )
    else
        throw(ArgumentError("Unknown experiment: $case"))
    end
end

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
function case_suffix(cfg, rmse_target, rng_idx)
    tgt = replace(string(rmse_target), "." => "p")
    cfg.force_case === nothing ? "$(tgt)_$(rng_idx)" : "$(cfg.force_case)_$(tgt)_$(rng_idx)"
end

function result_filename(cfg, rmse_target, rng_idx)
    "$(cfg.model)_adam_result_$(case_suffix(cfg, rmse_target, rng_idx))_$(cfg.run_date).jld2"
end

function nc_filename(cfg)
    if cfg.force_case === nothing
        return "leaderboard_adam_$(cfg.model)_$(cfg.run_date).nc"
    else
        return "leaderboard_adam_$(cfg.model)_$(cfg.force_case)_$(cfg.run_date).nc"
    end
end

########################################################################
###############  ARRAY-JOB HELPERS  ###################################
########################################################################
flat_tasks(cfg) =
    [(rmse_target, rng_idx) for rmse_target in cfg.rmse_targets for rng_idx in 1:cfg.n_repeats]

function task_index_from_args()
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        return parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    elseif !isempty(ARGS) && !isempty(ARGS[1])
        return parse(Int, ARGS[1])
    else
        return nothing
    end
end

function l96_experiment()
    if haskey(ENV, "EXPERIMENT")
        return Symbol(ENV["EXPERIMENT"])
    elseif length(ARGS) >= 2 && !isempty(ARGS[2])
        return Symbol(ARGS[2])
    else
        return EXPERIMENT
    end
end
