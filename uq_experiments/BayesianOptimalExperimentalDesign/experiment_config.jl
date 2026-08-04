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
#    max_iters:             number of GBOED iterations (k = 1, ..., max_iters)
#                           whose ST-MCMC posterior sample set is pushed
#                           forward and scored. Plays the role of History
#                           Matching's max_waves / "max_iter"/"K" in the other
#                           UQ methods. Iteration 1 is the initial LHS design
#                           + first GP fit + first ST-MCMC posterior (no
#                           acquisition yet, mirroring paper Algorithm 2's
#                           steps 1-3 before the "for k=1..K" loop begins);
#                           iterations 2..max_iters each add ONE EIG-optimized
#                           acquisition batch. N_ens doubles as both the
#                           initial LHS design size (n0 in the paper) and the
#                           acquisition batch size (B in the paper, fixed
#                           across iterations) — this keeps
#                           budget = N_ens * k_iter identical to every other
#                           UQ method's leaderboard convention with zero
#                           changes to common/uq_metrics/coverage_metrics.jl's
#                           budget_to_target, since the paper's own
#                           experiments also fix n0 = B.
#    n_posterior_samples:   ST-MCMC population size |X'| drawn each iteration
#                           (paper uses ~8092; defaulted MUCH lower here, 50,
#                           for compute budget). TransitionalMCMC.jl's
#                           tmcmc() evaluates the GP-based log-likelihood
#                           roughly n_posterior_samples * (tmcmc_burnin +
#                           tmcmc_thin) * 2 times PER tempering stage, via
#                           Distributed.pmap — which does not actually
#                           parallelize unless Julia was started with extra
#                           worker processes (`-p N`/`addprocs()`), so this
#                           cost is effectively serial (with real per-call
#                           scheduling overhead) in a plain `julia
#                           --project=.` run. Bump only if affordable; 1000
#                           at tmcmc's own default burnin/thin (20/3) was
#                           measured to make even a single N_ens=8 cell's
#                           first iteration take many minutes and still not
#                           finish.
#    tmcmc_burnin/tmcmc_thin: passed directly to TransitionalMCMC.tmcmc's
#                           per-particle Metropolis-Hastings mutation step
#                           (library defaults are 20/3 — set much lower here
#                           for the same reason as n_posterior_samples above).
#    eig_optim_iters:       Optim.jl Fminbox(LBFGS()) iteration cap for the
#                           INNER LBFGS solve at each fixed barrier weight μ
#                           (Fminbox's own "outer" barrier-shrinking loop is
#                           separate — see eig_outer_iters).
#    eig_outer_iters:       Fminbox's OUTER barrier-loop iteration cap
#                           (Optim.jl's `outer_iterations` option). Left
#                           unset, this defaults to Optim.jl's own 1000, and
#                           since Fminbox's outer x/f convergence tolerances
#                           also default to 0 (i.e. practically unreachable),
#                           the outer loop's only realistic early-exit is its
#                           outer_g_abstol=1e-8 projected-gradient tolerance —
#                           quite tight for this EIG objective. Without an
#                           explicit cap, a batch that doesn't quickly hit
#                           that tolerance can silently run up to 1000 outer
#                           rounds, each re-running the full eig_optim_iters
#                           inner LBFGS solve — the dominant source of
#                           `optimize_batch` looking hung. Set much lower
#                           (20) since in practice a well-behaved batch
#                           converges within a handful of outer barrier
#                           updates (μ shrinks geometrically by mufactor=1e-3
#                           each round); `fit_boed_gps`/`optimize_batch`'s
#                           `converged`/`objective_evals` diagnostics + the
#                           @warn emitted when this cap is hit tell you when
#                           a batch is being cut off before genuine
#                           convergence, so it can be raised deliberately if
#                           that happens often.
#    eig_g_tol:             g_abstol/outer_g_abstol (gradient-norm convergence
#                           tolerance), inner+outer. Optim's own default
#                           (1e-8) is essentially unreachable for this
#                           objective, so left there `optimize_batch` never
#                           "converges" and always burns through the full
#                           eig_optim_iters * eig_outer_iters budget. 1e-3
#                           reflects that an O(10-100s)-scale EIG doesn't need
#                           gradient-flat-to-1e-8, especially given
#                           eig_jitter already perturbs it by ~1e-6.
#    eig_f_reltol:          f_reltol/outer_f_reltol (relative-improvement
#                           convergence tolerance), inner+outer. Optim's
#                           default is 0.0 (disabled) — enabling this catches
#                           plateaus (negligible further improvement) that
#                           happen well before eig_g_tol is satisfied.
#    eig_call_limit:        Hard cap on raw EIG evaluations. NOT Optim's own
#                           f_calls_limit/g_calls_limit — those are enforced
#                           PER INNER LBFGS SOLVE under Fminbox, not
#                           cumulatively across outer rounds, so they don't
#                           actually bound total cost (verified directly);
#                           `optimize_batch` tracks its own running count and
#                           best-so-far point instead. On a representative
#                           case, quality vs. this cap measured as: 2,000 →
#                           90.0%, 5,000 → 90.2%, 10,000 → 92.0%, 20,000 →
#                           96.5% (all relative to an uncapped run). 5,000
#                           sits in a flat region between 2,000 and 10,000 —
#                           raise toward 10,000-20,000 if quality matters more
#                           than wall-clock here.
#    eig_bounds_std:        Fminbox box half-width for the candidate batch,
#                           in prior-whitened-truncated standard-normal units
#                           (keeps candidates within the GP's trust region).
#    eig_jitter:            diagonal jitter (as a multiple of the GP's own
#                           signal variance τ²) added before the EIG
#                           objective's K_cc \\ ... solve and both logdet
#                           calls, for numerical stability.
#    batch_init_strategy:   how the joint EIG optimization's starting batch
#                           is drawn — :posterior_subsample (default, subsample
#                           B points from the current ST-MCMC posterior) or
#                           :fresh_lhs (a fresh LHS draw in whitened space).
#    retain_var:            fraction of variance retained by the GP's OUTPUT
#                           whitening (against R) — shared with the same
#                           truncation threshold exp_to_leaderboard.jl uses
#                           for the coverage metric.
#    retain_var_input:      fraction of variance retained by the GP's INPUT
#                           whitening (against the prior covariance). This is
#                           ALSO the dimension of the LHS/ST-MCMC/EIG space
#                           GBOED samples and optimizes candidates in — unlike
#                           HistoryMatching (whose candidates are always drawn
#                           in full-rank raw prior space, using the truncated
#                           whitened representation only internally for GP
#                           input/implausibility scoring), GBOED's candidates
#                           are generated IN the truncated space and then
#                           decoded, so a mode dropped here is a raw parameter
#                           direction genuinely held fixed at its prior mean
#                           for every candidate, not just an approximation
#                           used for GP scoring. This is set PER CASE (not
#                           shared) for exactly that reason: l63's prior is a
#                           2-D diagonal Gaussian with a ~11:1 variance ratio
#                           between its two parameters (log ρ, log β) — the
#                           dominant mode alone already captures ~91.7% of
#                           variance, so history_matching_core.jl's original
#                           retain_var_input=0.9 (tuned for l96_vec, see
#                           below) would silently collapse l63/l96_const's GP
#                           input, LHS, and EIG-optimized design entirely onto
#                           1 dimension, holding the other of only 2
#                           parameters fixed at its prior mean for the whole
#                           run. l63 and l96_const (whose theta is small
#                           enough that no real dimension reduction is
#                           needed/possible) use 0.99 to guarantee full
#                           retention; l96_vec and l96_flux keep 0.9,
#                           following HistoryMatching's original reasoning:
#                           l96_vec's prior covariance has a slowly-decaying
#                           (exponential/OU, not squared-exponential)
#                           eigenspectrum, so retain_var itself would keep
#                           ~39/40 modes — no real dimension reduction — while
#                           0.9 keeps ~26/40, a real reduction traded for
#                           treating the discarded low-variance prior
#                           directions as fixed at their prior mean.
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
            max_iters = 15,
            retain_var_input = 0.99,
            common...,
        )
    elseif case == :l96_vec
        return (
            model = "l96",
            force_case = "vec-force",
            N_ens_sizes = collect(50:5:50 + 8 * 5),
            max_iters = 15,
            retain_var_input = 0.9,
            common...,
        )
    elseif case == :l96_flux
        return (
            model = "l96",
            force_case = "flux-force",
            N_ens_sizes = collect(50:5:50 + 8 * 5),
            max_iters = 15,
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
