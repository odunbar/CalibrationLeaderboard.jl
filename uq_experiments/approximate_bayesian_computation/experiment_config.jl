using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
# Set EXPERIMENT to one of: :l63, :l96_const, :l96_vec, :l96_flux
# (Overridden at runtime by EXPERIMENT env var or ARGS[2])
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[2]

# Date identifying this calibration run — set once per pipeline submission via
# the CALIBRATE_DATE env var (see hpc-variant/submit_*.sh, if added); falls
# back to today() for local runs where CALIBRATE_DATE is unset.
calibrate_date = haskey(ENV, "CALIBRATE_DATE") ? Date(ENV["CALIBRATE_DATE"]) : today()

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
# This experiment runs a single method (Approximate Bayesian Computation via
# rejection sampling, "ABC") — there is no method axis, unlike
# uq_experiments/calibrate_emulate_sample.
method_key = "abc"   # leaderboard netcdf filename prefix

# Rejection threshold: accept iff the (2R)-whitened implausibility-squared is
# below the (1-alpha_reject) quantile of a Chi-squared(ny) distribution.
alpha_reject = 0.05

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    N_ens_sizes:  sweep of ABC batch sizes; the leaderboard's "ensemble size" axis.
#    N_iter:       maximum number of ABC rounds allowed (hard cap).
#    max_iter:     rounds (k = 1, ..., max_iter) whose accumulated pool is scored.
# N_ens_sizes match GaussNewtonKalmanInversion's, for a comparable evaluation-
# budget axis (N_ens * k). max_iter = N_iter so every computed round is scored
# (N_ens_max * N_iter ~ 10,000).
function experiment_config(case::Symbol)
    n_repeats = 20

    if case == :l63
        return (
            model          = "l63",
            force_case     = nothing,
            N_ens_sizes    = collect(4:2:4+8*2),
            N_iter         = 500,
            n_repeats      = n_repeats,
            max_iter       = 500,
            alpha_reject   = alpha_reject,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_const
        return (
            model          = "l96",
            force_case     = "const-force",
            N_ens_sizes    = collect(4:2:4+8*2),
            N_iter         = 500,
            n_repeats      = n_repeats,
            max_iter       = 500,
            alpha_reject   = alpha_reject,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_vec
        return (
            model          = "l96",
            force_case     = "vec-force",
            N_ens_sizes    = collect(50:5:50+8*5),
            N_iter         = 111,
            n_repeats      = n_repeats,
            max_iter       = 111,
            alpha_reject   = alpha_reject,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_flux
        return (
            model          = "l96",
            force_case     = "flux-force",
            N_ens_sizes    = collect(50:5:50+8*5),
            N_iter         = 111,
            n_repeats      = n_repeats,
            max_iter       = 111,
            alpha_reject   = alpha_reject,
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

# Calibrate writes this file (accepted-sample-pool history); pushforward reads
# it and appends the pushforward samples into the same file.
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
# Used by pushforward_from_posterior_l*.jl / exp_to_leaderboard.jl (one
# (N_ens, rng_idx) result file at a time). calibrate_l*.jl indexes by
# rng_idx ∈ 1:cfg.n_repeats instead (see their header comments).
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
