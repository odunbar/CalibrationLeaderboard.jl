using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
# Set EXPERIMENT to one of: :l63, :l96_const, :l96_vec, :l96_flux
# (Overridden at runtime by EXPERIMENT env var or ARGS[2])
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[1]

# Date identifying this calibration run — PIN before submitting an array job.
#calibrate_date = Date("<YYYY-MM-DD>", "yyyy-mm-dd")
calibrate_date = today()

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
# Leaderboard key for this method — used in netcdf filenames
method_key = "<METHOD_NAME>"   # e.g. "ces-eki-dmc", "ces-uki-dmc"

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
function experiment_config(case::Symbol)
    n_repeats = 20

    if case == :l63
        return (
            model          = "l63",
            force_case     = nothing,
            N_ens_sizes    = collect(4:2:4+8*2),   # adjust ens_step per problem
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 10,
            retain_var     = 0.99,
            n_features     = 100,
            n_features_opt = 60,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_const
        return (
            model          = "l96",
            force_case     = "const-force",
            N_ens_sizes    = collect(4:2:4+8*2),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_vec
        return (
            model          = "l96",
            force_case     = "vec-force",
            N_ens_sizes    = collect(40:5:40+8*5),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_flux
        return (
            model          = "l96",
            force_case     = "flux-force",
            N_ens_sizes    = collect(30:5:30+8*5),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    else
        throw(ArgumentError("Unknown experiment: $case. Expected one of :l63, :l96_const, :l96_vec, :l96_flux"))
    end
end

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
function case_suffix(cfg, N_ens, rng_idx)
    cfg.force_case === nothing ? "$(N_ens)_$(rng_idx)" : "$(cfg.force_case)_$(N_ens)_$(rng_idx)"
end

calib_directory(cfg)                   = "<METHOD_NAME>_$(cfg.calibrate_date)"
prior_filename(cfg)                    = cfg.force_case === nothing ?
    "$(cfg.model)_priors.jld2" : "$(cfg.model)_priors_$(cfg.force_case).jld2"
ekp_filename(cfg, N_ens, rng_idx)     = "$(cfg.model)_ekp_$(case_suffix(cfg, N_ens, rng_idx)).jld2"
results_filename(cfg, N_ens, rng_idx) = "$(cfg.model)_calibrate_results_$(case_suffix(cfg, N_ens, rng_idx)).jld2"
posterior_filename(cfg, N_ens, rng_idx) = "$(cfg.model)_posterior_$(case_suffix(cfg, N_ens, rng_idx)).jld2"

function nc_filename(cfg)
    if cfg.force_case === nothing
        return "$(method_key)_$(cfg.model)_ensemble_results_$(cfg.calibrate_date).nc"
    else
        return "$(method_key)_$(cfg.model)_$(cfg.force_case)_$(cfg.calibrate_date).nc"
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
