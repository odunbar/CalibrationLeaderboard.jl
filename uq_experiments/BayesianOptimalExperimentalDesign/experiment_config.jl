using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
# Set EXPERIMENT to one of: :l63, :l96_const, :l96_vec, :l96_flux
# (Overridden at runtime by EXPERIMENT env var or ARGS[2])
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[1]

# Date identifying this calibration run — PIN before submitting an array job.
calibrate_date = haskey(ENV, "CALIBRATE_DATE") ? Date(ENV["CALIBRATE_DATE"]) : today()

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
# This experiment runs a single method (GBOED) — there is no method axis,
# unlike uq_experiments/calibrate_emulate_sample.
method_key = "boed"

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    max_iters:             number of GBOED iterations; N_ens is both the initial LHS size and the fixed per-iteration acquisition batch size.
#    n_posterior_samples:   ST-MCMC population size |X'| drawn each iteration (paper uses ~8092; much lower here for compute budget).
#    tmcmc_burnin/tmcmc_thin: passed directly to TransitionalMCMC.tmcmc, set below its own 20/3 defaults since they multiply per-stage cost.
#    eig_optim_iters:       Fminbox(LBFGS())'s inner-solve iteration cap (per fixed barrier weight).
#    eig_outer_iters:       Fminbox's outer barrier-loop iteration cap.
#    eig_g_tol:             g_abstol/outer_g_abstol; loosened from Optim's 1e-8 default, which this EIG objective rarely satisfies.
#    eig_f_reltol:          f_reltol/outer_f_reltol; Optim's default is 0.0 (disabled), so this is the only relative-improvement stopping test.
#    eig_call_limit:        hard cap on raw EIG evaluations, enforced directly by `optimize_batch` (Optim's own f_calls_limit/g_calls_limit don't cumulate across Fminbox's outer rounds).
#    eig_bounds_std:        Fminbox box half-width for the candidate batch, in prior-whitened-truncated standard-normal units.
#    eig_jitter:            diagonal jitter (as a multiple of the GP's signal variance) added before eig_objective's solve/logdet calls.
#    batch_init_strategy:   :posterior_subsample (subsample the current ST-MCMC posterior) or :fresh_lhs (a fresh LHS draw).
#    retain_var:            fraction of variance retained by the GP's OUTPUT whitening (against R).
#    retain_var_input:      fraction of variance retained by the GP's INPUT whitening (against the prior); also the GBOED candidate space's dimension, so set per case (0.99 for l63/l96_const's small θ, 0.9 for l96_vec/l96_flux's slowly-decaying prior spectrum).
function experiment_config(case::Symbol)
    n_repeats = 20
    common = (
        n_repeats = n_repeats,
        n_posterior_samples = 50,
        tmcmc_burnin = 5,
        tmcmc_thin = 1,
        eig_optim_iters = 200,
        eig_outer_iters = 20,
        eig_g_tol = 1e-3,
        eig_f_reltol = 1e-6,
        eig_call_limit = 2_000,
        eig_bounds_std = 4.0,
        eig_jitter = 1e-6,
        batch_init_strategy = :posterior_subsample,
        retain_var = 0.99,
        calibrate_date = calibrate_date,
    )

    if case == :l63
        return (
            model = "l63",
            force_case = nothing,
            N_ens_sizes = collect(4:2:4 + 8 * 2),
            max_iters = 10,
            retain_var_input = 0.99,
            common...,
        )
    elseif case == :l96_const
        return (
            model = "l96",
            force_case = "const-force",
            N_ens_sizes = collect(4:2:4 + 8 * 2),
            max_iters = 10,
            retain_var_input = 0.99,
            common...,
        )
    elseif case == :l96_vec
        return (
            model = "l96",
            force_case = "vec-force",
            N_ens_sizes = collect(50:5:50 + 8 * 5),
            max_iters = 8, # it's so slow, so lets start here
            retain_var_input = 0.9,
            common...,
        )
    elseif case == :l96_flux
        return (
            model = "l96",
            force_case = "flux-force",
            N_ens_sizes = collect(50:5:50 + 8 * 5),
            max_iters = 8,
            retain_var_input = 0.9,
            common...,
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

prelim_filename(cfg) = cfg.force_case === nothing ?
    "$(cfg.model)_computed_preliminaries.jld2" : "$(cfg.model)_computed_preliminaries_$(cfg.force_case).jld2"

results_filename(cfg, N_ens, rng_idx) = "$(cfg.model)_calibrate_results_$(case_suffix(cfg, N_ens, rng_idx)).jld2"

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
