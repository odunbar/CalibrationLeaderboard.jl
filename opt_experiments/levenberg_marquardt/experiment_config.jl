using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[3]

# PIN this before submitting an array job:
# run_date = Date("YYYY-MM-DD", "yyyy-mm-dd")
run_date = today()

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
function experiment_config(case::Symbol)
    n_repeats   = 20
    target_rmse = 1.2
    N_iter      = 50   # LM converges fast; increase if needed

    if case == :l63
        return (
            model       = "l63",
            force_case  = nothing,
            N_ens_sizes = [1],
            n_repeats   = n_repeats,
            N_iter      = N_iter,
            target_rmse = target_rmse,
            run_date    = run_date,
        )
    elseif case == :l96_const
        return (
            model       = "l96",
            force_case  = "const-force",
            N_ens_sizes = [1],
            n_repeats   = n_repeats,
            N_iter      = N_iter,
            target_rmse = target_rmse,
            run_date    = run_date,
        )
    elseif case == :l96_vec
        return (
            model       = "l96",
            force_case  = "vec-force",
            N_ens_sizes = [1],
            n_repeats   = n_repeats,
            N_iter      = N_iter,
            target_rmse = target_rmse,
            run_date    = run_date,
        )
    elseif case == :l96_flux
        return (
            model       = "l96",
            force_case  = "flux-force",
            N_ens_sizes = [1],
            n_repeats   = n_repeats,
            N_iter      = N_iter,
            target_rmse = target_rmse,
            run_date    = run_date,
        )
    else
        throw(ArgumentError("Unknown experiment: $case"))
    end
end

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
function case_suffix(cfg, N_ens, rng_idx)
    cfg.force_case === nothing ? "$(N_ens)_$(rng_idx)" : "$(cfg.force_case)_$(N_ens)_$(rng_idx)"
end

function result_filename(cfg, N_ens, rng_idx)
    "$(cfg.model)_gradient_descent_result_$(case_suffix(cfg, N_ens, rng_idx))_$(cfg.run_date).jld2"
end

function nc_filename(cfg)
    if cfg.force_case === nothing
        return "leaderboard_gradient_descent_$(cfg.model)_$(cfg.run_date).nc"
    else
        return "leaderboard_gradient_descent_$(cfg.model)_$(cfg.force_case)_$(cfg.run_date).nc"
    end
end

########################################################################
###############  ARRAY-JOB HELPERS  ###################################
########################################################################
flat_tasks(cfg) =
    [(N_ens, rng_idx) for N_ens in cfg.N_ens_sizes for rng_idx in 1:cfg.n_repeats]

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
