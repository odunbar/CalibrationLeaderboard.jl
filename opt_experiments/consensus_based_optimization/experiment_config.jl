using Dates

########################################################################
###############  USER TOGGLES  ########################################
########################################################################
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT  = experiments[1]

# CBO method: :CBO1 (first-order) or :CBO2 (second-order)
CBO_METHOD = :CBO1

run_date = today()

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
function experiment_config(case::Symbol)
    rmse_targets = [1.0, 1.1, 1.2]
    n_repeats    = 10
    cbo_method   = CBO_METHOD

    if case == :l63
        return (
            model           = "l63",
            force_case      = nothing,
            rmse_targets    = rmse_targets,
            N_ens_sizes     = [20, 25, 30],
            n_repeats       = n_repeats,
            N_iter          = 20,
            sigma           = 0.2,
            lambda          = 1.0,
            inertia         = 0.2,
            Δt              = 0.3,
            weight_exponent = 20.0,
            cbo_method      = cbo_method,
            run_date        = run_date,
        )
    elseif case == :l96_const
        return (
            model           = "l96",
            force_case      = "const-force",
            rmse_targets    = rmse_targets,
            N_ens_sizes     = [100, 200, 400],
            n_repeats       = n_repeats,
            N_iter          = 50,
            sigma           = 0.01,
            lambda          = 1.0,
            inertia         = 0.2,
            Δt              = 1.0,
            weight_exponent = 1.0,
            cbo_method      = cbo_method,
            run_date        = run_date,
        )
    elseif case == :l96_vec
        return (
            model           = "l96",
            force_case      = "vec-force",
            rmse_targets    = rmse_targets,
            N_ens_sizes     = [100, 200, 400],
            n_repeats       = n_repeats,
            N_iter          = 50,
            sigma           = 0.01,
            lambda          = 1.0,
            inertia         = 0.2,
            Δt              = 1.0,
            weight_exponent = 1.0,
            cbo_method      = cbo_method,
            run_date        = run_date,
        )
    elseif case == :l96_flux
        return (
            model           = "l96",
            force_case      = "flux-force",
            rmse_targets    = rmse_targets,
            N_ens_sizes     = [100, 200, 400],
            n_repeats       = n_repeats,
            N_iter          = 50,
            sigma           = 0.01,
            lambda          = 1.0,
            inertia         = 0.2,
            Δt              = 1.0,
            weight_exponent = 1.0,
            cbo_method      = cbo_method,
            run_date        = run_date,
        )
    else
        throw(ArgumentError("Unknown experiment: $case"))
    end
end

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
function method_tag(cbo_method::Symbol)
    cbo_method == :CBO1 ? "cbo1" : "cbo2"
end

function case_suffix(cfg, N_ens, rmse_target, rng_idx)
    tgt  = replace(string(rmse_target), "." => "p")
    mtag = method_tag(cfg.cbo_method)
    cfg.force_case === nothing ?
        "$(mtag)_$(N_ens)_$(tgt)_$(rng_idx)" :
        "$(cfg.force_case)_$(mtag)_$(N_ens)_$(tgt)_$(rng_idx)"
end

function result_filename(cfg, N_ens, rmse_target, rng_idx)
    "$(cfg.model)_cbo_result_$(case_suffix(cfg, N_ens, rmse_target, rng_idx))_$(cfg.run_date).jld2"
end

function nc_filename(cfg)
    mtag = method_tag(cfg.cbo_method)
    cfg.force_case === nothing ?
        "leaderboard_$(mtag)_$(cfg.model)_$(cfg.run_date).nc" :
        "leaderboard_$(mtag)_$(cfg.model)_$(cfg.force_case)_$(cfg.run_date).nc"
end

function prelim_filename(cfg)
    cfg.force_case === nothing ?
        "$(cfg.model)_computed_preliminaries.jld2" :
        "$(cfg.model)_computed_preliminaries_$(cfg.force_case).jld2"
end

########################################################################
###############  ARRAY-JOB HELPERS  ###################################
########################################################################
flat_tasks(cfg) = [
    (N_ens, rmse_target, rng_idx)
    for N_ens       in cfg.N_ens_sizes
    for rmse_target in cfg.rmse_targets
    for rng_idx     in 1:cfg.n_repeats
]

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
