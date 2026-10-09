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

# GBOED variant — what samples the posterior from the GP, and what proposes each iteration's N_ens-point batch after the initial LHS design:
#   :eig   -> the full GBOED: ST-MCMC posterior + joint-batch maximization of the closed-form EIG against it (leaderboard key "boed").
#   :tmcmc -> no EIG: ST-MCMC posterior, and the batch is just N_ens draws from it (key "boed-tmcmc").
#   :iekf  -> no EIG, and no ST-MCMC: the posterior is the final ensemble of an IEKF (GaussNewtonInversion, fixed-step DefaultScheduler) run on the GP-mean surrogate (no
#             forward evaluations), and the batch is N_ens draws from it (key "boed-iekf").
# The GP fit, pushforward and leaderboard stages are identical across variants; only calibrate's posterior-sampling and acquisition steps differ. Override via the BOED_VARIANT env var
# (set it for EVERY stage, so pushforward/exp_to_leaderboard read the matching output directory).
boed_variants = [:eig, :tmcmc, :iekf]
BOED_VARIANT = boed_variants[1]
boed_variant = haskey(ENV, "BOED_VARIANT") ? Symbol(ENV["BOED_VARIANT"]) : BOED_VARIANT
boed_variant in boed_variants || throw(ArgumentError("Unknown BOED_VARIANT: $boed_variant. Expected one of $boed_variants"))

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
# This experiment runs a single method (GBOED) — there is no method axis,
# unlike uq_experiments/calibrate_emulate_sample — but three acquisition variants (see above), each with its own output directory/netcdf.
method_key = Dict(:eig => "boed", :tmcmc => "boed-tmcmc", :iekf => "boed-iekf")[boed_variant]

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    max_iters:             number of GBOED iterations; N_ens is both the initial LHS size and the fixed per-iteration acquisition batch size.
#    n_posterior_samples:   ST-MCMC population size drawn each iteration (paper uses ~8092; much lower here for compute budget). 1000 (matches the other methods' n_pushforward_samples; was 500, and 50 before that): with 50 the
#                           posterior was degenerate in the 26-D l96_vec case (GBOED lagged ~3 waves behind a 500-particle run); tmcmc cost is ~linear in this.
#    n_eig_posterior_samples: size of the random posterior subsample X' that the EIG objective (paper Eq. 8) targets. EIG costs ~n^3 per output mode, so this
#                           stays at 50 even though ST-MCMC now draws more; candidate batches are still initialised from all n_posterior_samples draws.
#    tmcmc_burnin/tmcmc_thin: passed directly to TransitionalMCMC.tmcmc, set below its own 20/3 defaults since they multiply per-stage cost.
#    eig_optim_iters:       LBFGS iteration cap: Fminbox's inner-solve cap (per fixed barrier weight) for :box, the whole-solve cap for :ball.
#    eig_outer_iters:       Fminbox's outer barrier-loop iteration cap.
#    eig_g_tol:             g_abstol/outer_g_abstol; loosened from Optim's 1e-8 default, which this EIG objective rarely satisfies.
#    eig_f_reltol:          f_reltol/outer_f_reltol; Optim's default is 0.0 (disabled), so this is the only relative-improvement stopping test.
#    eig_call_limit:        hard cap on raw EIG evaluations, enforced directly by `optimize_batch` (Optim's own f_calls_limit/g_calls_limit don't cumulate across Fminbox's outer rounds).
#    eig_region:            :ball (each candidate inside the χ²_k(eig_ball_quantile) ball of the prior-whitened space; k = truncated input dim) or :box (Fminbox on ±eig_bounds_std per coordinate).
#    eig_ball_quantile:     χ² quantile setting the :ball radius R = sqrt(quantile(Chisq(k), q)).
#    eig_bounds_std:        (:box only) Fminbox box half-width for the candidate batch, in prior-whitened-truncated standard-normal units.
#    eig_jitter:            diagonal jitter (as a multiple of the GP's signal variance) added before eig_objective's solve/logdet calls.
#    gp_optim_iters:        Fminbox inner L-BFGS iteration cap for each GP hyperparameter fit (fit_boed_gps). Optim's default is 1000, which dominated wallclock.
#    gp_outer_iters:        Fminbox outer barrier-loop cap for the GP fit.
#    gp_g_tol, gp_f_reltol: gradient / relative-improvement stopping tolerances for the GP fit.
#    gp_warm_start:         (off by default: in benchmarks it cost 2-11 nats of marginal likelihood for a further ~1.5x speedup in 24-D only) start each iteration's GP fit from the previous iteration's hyperparameters (clamped inside the new bounds) instead of the std(Z)-based guess.
#    batch_init_strategy:   :posterior_subsample (subsample the current ST-MCMC posterior) or :fresh_lhs (a fresh LHS draw).
#    iekf_step, iekf_iters: (:iekf variant only) fixed-step DefaultScheduler size and number of steps for the GP-surrogate IEKF posterior sampler (iekf_step*iekf_iters = 1).
#    retain_var:            fraction of variance retained by the GP's OUTPUT whitening (against R).
#    retain_var_input:      fraction of variance retained by the GP's INPUT whitening (against the prior); also the GBOED candidate space's dimension, so set per case (0.99 for l63/l96_const's small θ, 0.9 for l96_vec/l96_flux's slowly-decaying prior spectrum).
function experiment_config(case::Symbol)
    n_repeats = 20
    common = (
        n_repeats = n_repeats,
        n_posterior_samples = 1000, # matches the n_posterior samples for other approaches.
        n_eig_posterior_samples = 50,
        tmcmc_burnin = 5,
        tmcmc_thin = 1,
        eig_optim_iters = 500,
        eig_outer_iters = 20,
        eig_g_tol = 1e-3,
        eig_f_reltol = 1e-6,
        eig_call_limit = 2_000,
        eig_region = :ball,
        eig_ball_quantile = 0.999,
        eig_bounds_std = 4.0,
        eig_jitter = 1e-6,
        gp_optim_iters = 50,
        gp_outer_iters = 3,
        gp_g_tol = 1e-3,
        gp_f_reltol = 1e-6,
        gp_warm_start = false,
        batch_init_strategy = :posterior_subsample,
        retain_var = 0.99,
        iekf_step = 0.1,   # :iekf variant only: fixed DefaultScheduler step (matches GaussNewtonKalmanInversion / CES's IEKF)
        iekf_iters = 10,   # :iekf variant only: iekf_iters * iekf_step = 1, the finite-time approximate posterior
        variant = boed_variant,
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
