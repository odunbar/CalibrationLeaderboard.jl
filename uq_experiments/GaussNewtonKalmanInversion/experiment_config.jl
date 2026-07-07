using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
# Set EXPERIMENT to one of: :l63, :l96_const, :l96_vec, :l96_flux
# (Overridden at runtime by EXPERIMENT env var or ARGS[2])
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[2]

# Date identifying this calibration run — set once per pipeline submission via
# the CALIBRATE_DATE env var (see hpc-variant/submit_*.sh); falls back to
# today() for local runs where CALIBRATE_DATE is unset.
calibrate_date = haskey(ENV, "CALIBRATE_DATE") ? Date(ENV["CALIBRATE_DATE"]) : today()

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
# This experiment runs a single method (GaussNewtonInversion, a.k.a GNKI) —
# there is no method axis, unlike uq_experiments/calibrate_emulate_sample.
method_key = "gnki"   # leaderboard netcdf filename prefix

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    terminate_at: pseudotime to terminate (DataMisfitController). T=1 is
#                  approximately the exact posterior for GNKI; terminate_at
#                  is set a little beyond that as a safety margin.
#    N_iter:       maximum number of EKI iterations allowed (hard cap).
#    max_iter:     number of iterations (k = 1, ..., max_iter) whose ensembles
#                  are pushed forward and scored on the leaderboard.
#    N_ens_sizes:  sweep of ensemble sizes. There is no separate emulate_sample
#                  stage — the ensemble at each iteration IS the posterior
#                  estimate — but pushforward_from_posterior_l*.jl resamples a
#                  fixed, larger number of points from the Gaussian implied by
#                  the ensemble's mean/cov to reduce quantile-estimation noise
#                  in the coverage metric (post_mean/post_cov still come
#                  directly from the raw N_ens ensemble).
function experiment_config(case::Symbol)
    n_repeats = 20

    if case == :l63
        return (
            model          = "l63",
            force_case     = nothing,
            N_ens_sizes    = collect(4:2:4+8*2),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 10,
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
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_vec
        return (
            model          = "l96",
            force_case     = "vec-force",
            N_ens_sizes    = collect(50:5:50+8*5),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_flux
        return (
            model          = "l96",
            force_case     = "flux-force",
            N_ens_sizes    = collect(50:5:50+8*5),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
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

calib_directory(cfg) = "$(method_key)_$(cfg.calibrate_date)"

prior_filename(cfg) = cfg.force_case === nothing ?
    "$(cfg.model)_priors.jld2" : "$(cfg.model)_priors_$(cfg.force_case).jld2"

prelim_filename(cfg) = cfg.force_case === nothing ?
    "$(cfg.model)_computed_preliminaries.jld2" : "$(cfg.model)_computed_preliminaries_$(cfg.force_case).jld2"

# Calibrate writes this file (ensemble history); pushforward reads it and
# appends the pushforward samples into the same file.
results_filename(cfg, N_ens, rng_idx) = "$(cfg.model)_calibrate_results_$(case_suffix(cfg, N_ens, rng_idx)).jld2"

function nc_filename(cfg)
    if cfg.force_case === nothing
        return "leaderboard_$(method_key)_$(cfg.model)_$(cfg.calibrate_date).nc"
    else
        return "leaderboard_$(method_key)_$(cfg.model)_$(cfg.force_case)_$(cfg.calibrate_date).nc"
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
