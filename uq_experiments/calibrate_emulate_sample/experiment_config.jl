using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
# Set EXPERIMENT to one of:
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[4]

# Date identifying this calibration run — written by calibrate, read by
# emulate_sample and exp_to_leaderboard (so all stages stay in sync).
# submit_l*.sh pins this via the CALIBRATE_DATE env var so every stage in the
# chain agrees, even if the pipeline runs across a midnight boundary; local
# runs fall back to today().
calibrate_date = haskey(ENV, "CALIBRATE_DATE") ? Date(ENV["CALIBRATE_DATE"]) : today()

# EKI ("Inversion") scheduler variant:
#   :dmc   → adaptive DataMisfitController(terminate_at = cfg.terminate_at)
#   :const → fixed-step DefaultScheduler(0.1)
# Only affects the "Inversion" method (EKI) — other methods keep DataMisfitController.
# Also selects the "ces-eki-dmc" vs "ces-eki-const" leaderboard file-naming key
# below. Override via the EKI_SCHEDULER env var for SLURM runs.
eki_scheduler_variants = [:dmc, :const]
EKI_SCHEDULER = eki_scheduler_variants[1]
eki_scheduler_case = haskey(ENV, "EKI_SCHEDULER") ? Symbol(ENV["EKI_SCHEDULER"]) : EKI_SCHEDULER

# Calibrate-stage update rule:
#   :eki  -> Inversion() (EKI), scheduler chosen via EKI_SCHEDULER above.
#   :iekf -> GaussNewtonInversion(prior) (IEKF), fixed-step DefaultScheduler(0.1),
#            matching the scheduler used in the dedicated
#            uq_experiments/KalmanSampler experiment. Leaderboard
#            file-naming key is "ces-iekf-const" (see method_cases_key below).
#   :eks  -> Sampler(prior) (EKS), EKSStableScheduler(), matching the
#            "sampler" variant of uq_experiments/KalmanSampler. Leaderboard
#            file-naming key is "ces-eks" (see method_cases_key below).
# Determines which entry of method_cases is "active" (method_cases[1]), so
# every downstream stage (emulate_sample, pushforward, exp_to_leaderboard,
# diagnostic plots) that defaults to method_cases[1] picks up the right
# calib_directory / leaderboard key automatically.
# Override via the CALIBRATE_METHOD env var for SLURM runs.
calibrate_method_variants = [:eki, :iekf, :eks]
CALIBRATE_METHOD = calibrate_method_variants[1]
calibrate_method_case = haskey(ENV, "CALIBRATE_METHOD") ? Symbol(ENV["CALIBRATE_METHOD"]) : CALIBRATE_METHOD

# Emulator dimension-reduction (emulate_sample stage only; calibrate is unaffected):
#   :LIS -> original. L96 cases: decorrelate_structure_mat() then likelihood_informed()
#           (retain_info = cfg.retain_var) on input and output. L63: PCA (see :PCA).
#   :PCA -> default. PCA-only, matching the truncated whitening used by GBOED and History
#           Matching: decorrelate_structure_mat(retain_var = cfg.retain_var_input)
#           on the input (prior-covariance PCA) and
#           decorrelate_structure_mat(retain_var = cfg.retain_var_output) on the
#           output (R PCA). Per-case values below mirror
#           uq_experiments/BayesianOptimalExperimentalDesign/experiment_config.jl:
#           99%/99% (l63, l96_const), 90%/99% (l96_vec, l96_flux) input/output.
# L63 already uses PCA under :LIS, so :PCA is identical there and shares its outputs;
# for L96, :PCA writes to separate posterior files and a "-pca" leaderboard key so it
# never overwrites the :LIS results (the calibrate outputs are shared).
# Override via the DIM_REDUCTION env var (PCA or LIS, case-insensitive) for SLURM runs.
dim_reduction_variants = [:PCA, :LIS]
DIM_REDUCTION = dim_reduction_variants[1]
dim_reduction_case = haskey(ENV, "DIM_REDUCTION") ? Symbol(uppercase(ENV["DIM_REDUCTION"])) : DIM_REDUCTION
dim_reduction_case in dim_reduction_variants ||
    throw(ArgumentError("Unknown DIM_REDUCTION: $dim_reduction_case. Expected one of $dim_reduction_variants"))

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
calibrate_method_case in calibrate_method_variants ||
    throw(ArgumentError("Unknown CALIBRATE_METHOD: $calibrate_method_case. Expected one of $calibrate_method_variants"))
method_cases = if calibrate_method_case == :iekf
    ["GaussNewtonInversion", "TransformInversion", "Unscented", "Inversion"]
elseif calibrate_method_case == :eks
    ["Sampler", "TransformInversion", "Unscented", "Inversion"]
else
    ["Inversion", "TransformInversion", "Unscented", "GaussNewtonInversion"]
end

method_names = [
    ("Inversion()", "EKI"),
    ("TransformInversion()", "ETKI"),
    ("GaussNewtonInversion()", "GNKI"),
    ("Unscented(prior)", "UKI"),
]

method_cases_key = Dict(
    "Inversion"            => "ces-eki-$(eki_scheduler_case)",
    "TransformInversion"   => "ces-etki-dmc",
    "Unscented"            => "ces-uki-dmc",
    "GaussNewtonInversion" => "ces-iekf-const",
    "Sampler"              => "ces-eks",
)

forcing_cases_key = Dict(
    "const-force" => "ensemble_results",
    "vec-force"   => "spatial_forcing_ensemble_results",
    "flux-force"  => "nn_forcing_ensemble_results",
)

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    terminate_at: psuedotime to terminate. (T=1 is approx posterior for the finite-time method variants)
#    N_iter: otherwise, maximum iterations
#    N_ens_sizes: Vector of experiments
#    n_repeats: number of rng seeds


function experiment_config(case::Symbol)
    n_ens_step = 8 ## MUST CHANGE SBATCH ARRAY SIZE TO ((N_ENS_STEP + 1) * N_REPEATS)
    n_repeats = 20
    if case == :l63
        ens_step = 2
        return (
            model          = "l63",
            force_case     = nothing,
            N_ens_sizes    = collect(4:ens_step:4+n_ens_step*ens_step),
            N_iter         = 20,
            terminate_at   = 2.0,   # DataMisfitController end time
            n_repeats      = n_repeats,
            max_iter       = 10,
            retain_var     = 0.99,   # :LIS variant (likelihood_informed retain_info)
            retain_var_input  = 0.99,   # :PCA variant, input (prior-cov PCA)
            retain_var_output = 0.99,   # :PCA variant, output (R PCA)
            n_features     = 100,
            n_features_opt = 60,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_const
        ens_step_const = 2
        return (
            model          = "l96",
            force_case     = "const-force",
            N_ens_sizes    = collect(4:ens_step_const:4+n_ens_step*ens_step_const),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,   # :LIS variant (likelihood_informed retain_info)
            retain_var_input  = 0.99,   # :PCA variant, input (prior-cov PCA)
            retain_var_output = 0.99,   # :PCA variant, output (R PCA)
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_vec
        ens_step_vec = 5
        return (
            model          = "l96",
            force_case     = "vec-force",
            N_ens_sizes    = collect(50:ens_step_vec:50+n_ens_step*ens_step_vec),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,   # :LIS variant (likelihood_informed retain_info)
            retain_var_input  = 0.9,   # :PCA variant, input (prior-cov PCA)
            retain_var_output = 0.99,   # :PCA variant, output (R PCA)
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_flux
        ens_step_flux = 5
        return (
            model          = "l96",
            force_case     = "flux-force",
            N_ens_sizes    = collect(50:ens_step_flux:50+n_ens_step*ens_step_flux),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,   # :LIS variant (likelihood_informed retain_info)
            retain_var_input  = 0.9,   # :PCA variant, input (prior-cov PCA)
            retain_var_output = 0.99,   # :PCA variant, output (R PCA)
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
# Tag distinguishing the :PCA emulator variant's outputs from the :LIS ones ("" for
# :LIS, and for L63 where :PCA == :LIS). Appended to emulate_sample-onwards artefacts.
dim_reduction_tag(cfg) = (dim_reduction_case == :PCA && cfg.model != "l63") ? "_pca" : ""

function case_suffix(cfg, N_ens, rng_idx)
    if cfg.force_case === nothing
        return "$(N_ens)_$(rng_idx)"
    else
        return "$(cfg.force_case)_$(N_ens)_$(rng_idx)"
    end
end

calib_directory(method, cfg) = "$(method)_$(cfg.calibrate_date)"

function prior_filename(cfg)
    if cfg.force_case === nothing
        return "$(cfg.model)_priors.jld2"
    else
        return "$(cfg.model)_priors_$(cfg.force_case).jld2"
    end
end

# Written once by l63_preliminaries.jl / l96_preliminaries.jl (before the
# calibrate array starts) and loaded by calibrate_l63.jl / calibrate_l96.jl
# and every downstream stage.
function prelim_filename(cfg)
    if cfg.force_case === nothing
        return "$(cfg.model)_computed_preliminaries.jld2"
    else
        return "$(cfg.model)_computed_preliminaries_$(cfg.force_case).jld2"
    end
end

ekp_filename(cfg, N_ens, rng_idx)      = "$(cfg.model)_ekp_$(case_suffix(cfg, N_ens, rng_idx)).jld2"
results_filename(cfg, N_ens, rng_idx)  = "$(cfg.model)_calibrate_results_$(case_suffix(cfg, N_ens, rng_idx)).jld2"
posterior_filename(cfg, N_ens, rng_idx) = "$(cfg.model)_posterior$(dim_reduction_tag(cfg))_$(case_suffix(cfg, N_ens, rng_idx)).jld2"

function summary_filename(cfg)
    if cfg.force_case === nothing
        return "$(cfg.model)_output_$(cfg.calibrate_date).jld2"
    else
        return "$(cfg.model)_calibrate_$(cfg.force_case)_$(cfg.calibrate_date).jld2"
    end
end

function nc_filename(cfg, method)
    key = method_cases_key[method] * replace(dim_reduction_tag(cfg), "_" => "-")
    if cfg.force_case === nothing
        return "$(key)_$(cfg.model)_ensemble_results_$(cfg.calibrate_date).nc"
    else
        return "$(key)_$(cfg.model)_$(forcing_cases_key[cfg.force_case])_$(cfg.calibrate_date).nc"
    end
end

########################################################################
###############  ARRAY-JOB HELPERS  ###################################
########################################################################

# Flattened (N_ens, rng_idx) task list; index t (1-based) == SLURM_ARRAY_TASK_ID.
# Outer loop = N_ens_sizes, inner = repeats, matching the original serial nesting.
flat_tasks(cfg) =
    [(N_ens, rng_idx) for N_ens in cfg.N_ens_sizes for rng_idx in 1:cfg.n_repeats]

# Task index: prefer SLURM_ARRAY_TASK_ID, then first CLI arg, else nothing (run all).
function task_index_from_args()
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        return parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    elseif !isempty(ARGS) && !isempty(ARGS[1])
        return parse(Int, ARGS[1])
    else
        return nothing
    end
end

# L96 experiment: prefer EXPERIMENT env var, then ARGS[2], else the manual toggle above.
function l96_experiment()
    if haskey(ENV, "EXPERIMENT")
        return Symbol(ENV["EXPERIMENT"])
    elseif length(ARGS) >= 2 && !isempty(ARGS[2])
        return Symbol(ARGS[2])
    else
        return EXPERIMENT
    end
end
