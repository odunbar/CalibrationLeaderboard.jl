using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[1]

# Score estimator:  :dsm      — learned score, denoising score matching (single-
#                               sigma NCSN-style net; see score_gfdt.jl)
#                   :kgmm     — learned score, k-means Gaussian-mixture estimator
#                               (arXiv:2503.18054); avoids DSM's small-sigma /
#                               small-sample blowup at the cost of a clustering
#                               step instead of a training loop
#                   :gaussian — quasi-Gaussian FDT baseline, s(x) = -C^{-1}(x-mu)
SCORE_KIND = :kgmm

# Budget mode (both charged outer_iter * N_ens forward-model evaluations, and
# both integrate T_start once + N_ens*W of total model time):
#   :serial   — ONE continuous run with the statistics window stretched x N_ens
#               (T_start paid once); the whole N_ens*W span is one inherently
#               sequential integration, but it gives the longest usable lags in
#               the correlation integral for a given total cost.
#   :parallel — spin up ONCE per outer iteration, then fork N_ens independent
#               base-length branches from small (ic_cov_sqrt) perturbations of
#               that on-attractor state.  Same total integration cost as
#               :serial, but the N_ens branches are mutually independent and
#               can run concurrently.
# At N_ens == 1 the two modes are identical.
BUDGET_MODE = :serial

# Pinned at submission time via RUN_DATE env var (set by submit_*.sh).
# Falls back to today() for local runs.
run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# GFDT / score hyperparameters (identical defaults to score_based_lm -- the
# Jacobian estimator itself is unchanged, only the update rule differs) plus
# Adam hyperparameters (identical defaults to opt_experiments/adam).
#
# tau_max, tukey_alpha, sigma: see opt_experiments/score_based_lm/experiment_config.jl
# for the measurement basis behind these defaults.
function experiment_config(case::Symbol)
    n_repeats    = 30
    rmse_targets = [1.0, 1.1, 1.2]
    budget_total = 500     # outer_iter * N_ens is capped at this per cell

    common = (
        rmse_targets = rmse_targets,
        n_repeats    = n_repeats,
        budget_total = budget_total,
        run_date     = run_date,
        score_kind   = score_kind_from_env(),
        budget_mode  = budget_mode_from_env(),
        tukey_alpha  = 0.75,
        stein        = true,
        val_frac     = 0.1,
        kgmm_ridge   = 0.1,   # per-cluster covariance shrinkage toward the global
                              # covariance; K itself is chosen adaptively from the
                              # actual window size at fit time (see score_gfdt.jl)
        # Adam hyperparameters -- beta1/beta2/eps kept identical to
        # opt_experiments/adam; adam_alpha is set per-case below (NOT shared),
        # since a single global 0.001 (copied unchanged from opt_experiments/adam,
        # which optimises a differently-scaled residual) leaves Adam unable to
        # traverse even the l63/l96_const prior within its iteration budget --
        # see run_l63_sbadam.jl's failure-rate note. Sized to ~0.25x the case's
        # mean prior std, so a run of consistent-sign steps crosses ~1 prior std
        # in ~4 outer iterations.
        adam_beta1 = 0.9,     # first-moment decay
        adam_beta2 = 0.999,   # second-moment decay
        adam_eps   = 1e-8,    # numerical stability
    )

    if case == :l63
        # prior std = [0.15, 0.5] (log rho, log beta) -> mean 0.325
        return (; model = "l63", force_case = nothing, nx = 3,
                  N_ens_sizes = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
                  tau_max = 5.0, lag_stride = 5,
                  sigma = 0.2, hidden = 128, base = 16,
                  epochs_init = 500, epochs_warm = 50,
                  lr_init = 1e-3, lr_warm = 3e-4, batch = 256, weight_decay = 0.0,
                  adam_alpha = 0.08,
                  common...)
    elseif case == :l96_const
        # prior std = 0.4 (log forcing)
        return (; model = "l96", force_case = "const-force", nx = 40,
                  N_ens_sizes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
                  tau_max = 5.0, lag_stride = 5,
                  sigma = 0.025, hidden = 128, base = 16,
                  epochs_init = 150, epochs_warm = 20,
                  lr_init = 8e-4, lr_warm = 3e-4, batch = 128, weight_decay = 1e-4,
                  adam_alpha = 0.1,
                  common...)
    elseif case == :l96_vec
        # prior std = psig = 3.0 (per-component forcing)
        return (; model = "l96", force_case = "vec-force", nx = 40,
                  N_ens_sizes = [10, 20, 30, 40, 60, 80, 100, 120, 140, 160],
                  tau_max = 8.0, lag_stride = 5,
                  sigma = 0.025, hidden = 128, base = 16,
                  epochs_init = 150, epochs_warm = 20,
                  lr_init = 8e-4, lr_warm = 3e-4, batch = 128, weight_decay = 1e-4,
                  adam_alpha = 0.75,
                  common...)
    elseif case == :l96_flux
        # prior std = 0.1 (NN weight space)
        return (; model = "l96", force_case = "flux-force", nx = 100,
                  N_ens_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  tau_max = 8.0, lag_stride = 10,
                  sigma = 0.025, hidden = 128, base = 16,
                  epochs_init = 150, epochs_warm = 20,
                  lr_init = 8e-4, lr_warm = 3e-4, batch = 128, weight_decay = 1e-4,
                  adam_alpha = 0.025,
                  common...)
    else
        throw(ArgumentError("Unknown experiment: $case"))
    end
end

# Fixed total budget per cell: outer_iter * N_ens <= budget_total, so cells with
# a larger N_ens run fewer, cheaper-Jacobian-per-iteration Adam steps rather
# than a fixed iteration count regardless of N_ens.
n_iter_for(cfg, N_ens::Int) = Int(ceil(cfg.budget_total / N_ens))

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
# Tag encodes both switches so the arms can never overwrite each other.
method_tag(cfg) = string("sbadam_", cfg.score_kind, "_", cfg.budget_mode)

function algorithm_type(cfg)
    base = if cfg.score_kind === :gaussian
        "Quasi-Gaussian FDT Adam"
    elseif cfg.score_kind === :kgmm
        "Score-based Adam (GFDT, KGMM)"
    else
        "Score-based Adam (GFDT, DSM)"
    end
    cfg.budget_mode === :parallel && return base * " (parallel branches)"
    return base * " (extended window)"
end

function case_suffix(cfg, N_ens, rmse_target, rng_idx)
    tgt = replace(string(rmse_target), "." => "p")
    stem = "$(N_ens)_$(tgt)_$(rng_idx)"
    return cfg.force_case === nothing ? stem : "$(cfg.force_case)_$(stem)"
end

function result_filename(cfg, N_ens, rmse_target, rng_idx)
    "$(cfg.model)_$(method_tag(cfg))_result_$(case_suffix(cfg, N_ens, rmse_target, rng_idx))_$(cfg.run_date).jld2"
end

function nc_filename(cfg)
    if cfg.force_case === nothing
        return "leaderboard_$(method_tag(cfg))_$(cfg.model)_$(cfg.run_date).nc"
    else
        return "leaderboard_$(method_tag(cfg))_$(cfg.model)_$(cfg.force_case)_$(cfg.run_date).nc"
    end
end

########################################################################
###############  ARRAY-JOB HELPERS  ###################################
########################################################################
# N_TASKS = length(N_ens_sizes) * length(rmse_targets) * n_repeats = 10 * 3 * 30 = 900 (all cases)
flat_tasks(cfg) = [
    (N_ens, rmse_target, rng_idx)
    for N_ens in cfg.N_ens_sizes
    for rmse_target in cfg.rmse_targets
    for rng_idx in 1:cfg.n_repeats
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

function score_kind_from_env()
    kind = if haskey(ENV, "SCORE_KIND")
        Symbol(ENV["SCORE_KIND"])
    elseif length(ARGS) >= 3 && !isempty(ARGS[3])
        Symbol(ARGS[3])
    else
        SCORE_KIND
    end
    kind in (:dsm, :gaussian, :kgmm) ||
        throw(ArgumentError("SCORE_KIND must be dsm, gaussian, or kgmm, got $kind"))
    return kind
end

function budget_mode_from_env()
    mode = if haskey(ENV, "BUDGET_MODE")
        Symbol(ENV["BUDGET_MODE"])
    elseif length(ARGS) >= 4 && !isempty(ARGS[4])
        Symbol(ARGS[4])
    else
        BUDGET_MODE
    end
    mode in (:serial, :parallel) || throw(ArgumentError("BUDGET_MODE must be serial or parallel, got $mode"))
    return mode
end
