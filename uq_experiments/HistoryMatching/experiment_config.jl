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
# This experiment runs a single method (Bayesian History Matching) — there is
# no method axis, unlike uq_experiments/calibrate_emulate_sample.
method_key = "history-matching"

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    max_waves:             number of History Matching waves (k = 1, ..., max_waves)
#                           whose NROY sample set is pushed forward and scored.
#                           Plays the role of "max_iter"/"K" in the other UQ methods.
#    confidence:            implausibility acceptance level; the chi-squared
#                           threshold is quantile(Chisq(n_out), confidence).
#    n_candidate_batch:     batch size for rejection-sampling candidate draws
#                           (GP-predict is evaluated on a whole batch at once).
#    max_rejection_samples: guard against an ever-shrinking NROY region. Each
#                           wave draws one NROY batch (serving as both that
#                           wave's stored posterior and the next wave's
#                           training ensemble) — if it can't fill within this
#                           many candidates, the batch is left short and, if
#                           that leaves fewer than N_ens samples, the wave
#                           loop stops early — see calibrate_l*.jl.
#    retain_var:            fraction of variance retained by the GP's OUTPUT
#                           whitening (against R) — shared with the same
#                           truncation threshold exp_to_leaderboard.jl uses
#                           for the coverage metric.
#    retain_var_input:      fraction of variance retained by the GP's INPUT
#                           whitening (against the prior covariance, and —
#                           for l96_flux only — the additional wave-local
#                           ensemble PCA on top of it). Deliberately lower
#                           than retain_var: l96_vec's prior covariance has a
#                           slowly-decaying (exponential/OU, not
#                           squared-exponential) eigenspectrum, so retain_var
#                           itself would keep ~39/40 modes — no real
#                           dimension reduction. retain_var_input=0.9 keeps
#                           ~26/40, a real reduction traded for treating the
#                           discarded low-variance prior directions as fixed
#                           at their prior mean.
function experiment_config(case::Symbol)
    n_repeats = 20
    common = (
        n_repeats = n_repeats,
        confidence = 0.95,
        n_candidate_batch = 2_000,
        max_rejection_samples = 1_000_000,   # matches calib_race_hm_l63.py's own max_samples default
        retain_var = 0.99,
        retain_var_input = 0.9,
        calibrate_date = calibrate_date,
    )

    if case == :l63
        return (
            model = "l63",
            force_case = nothing,
            N_ens_sizes = collect(4:2:4 + 8 * 2),
            max_waves = 10,
            common...,
        )
    elseif case == :l96_const
        return (
            model = "l96",
            force_case = "const-force",
            N_ens_sizes = collect(4:2:4 + 8 * 2),
            max_waves = 15,
            common...,
        )
    elseif case == :l96_vec
        return (
            model = "l96",
            force_case = "vec-force",
            N_ens_sizes = collect(50:5:50 + 8 * 5),
            max_waves = 15,
            common...,
        )
    elseif case == :l96_flux
        return (
            model = "l96",
            force_case = "flux-force",
            N_ens_sizes = collect(50:5:50 + 8 * 5),
            max_waves = 15,
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
