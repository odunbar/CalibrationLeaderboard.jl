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
SCORE_KIND = :dsm

# Budget mode (both charged outer_iter * N_ens forward-model evaluations):
#   :fair     — N_ens independent base-length runs, pooled (the LM convention)
#   :ensemble — ONE run with the statistics window stretched x N_ens, which buys
#               longer usable lags in the correlation integral
# At N_ens == 1 the two modes are identical.
BUDGET_MODE = :fair

# Pinned at submission time via RUN_DATE env var (set by submit_*.sh).
# Falls back to today() for local runs.
run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# GFDT / score hyperparameters.
#
# tau_max: measured on an analytic OU process (validate_gfdt_ou.jl), tau_max is
#   the dominant knob and too SHORT is worse than too long -- at tau_max=2 the
#   truncation bias is 13%, at tau_max=5 it is 4%.  But tau_max also costs
#   n_eff = M - tau_max/dt columns of averaging, which bites hard on the 10-MTU
#   l63 / l96_const windows.  Swept in validate_jacobian_*.jl.
# tukey_alpha: second-order next to tau_max.  0.75 is safe everywhere; a full
#   Hann (alpha=0) biases low badly at small tau_max.
# sigma: the DSM noise level.  This is the mollification scale that makes
#   grad log p exist at all for a singular SRB measure, so it is a physical
#   regulariser and must be swept, not guessed.  Values below are the published
#   starting points (normalised units).
function experiment_config(case::Symbol)
    n_repeats    = 30
    rmse_targets = [1.0, 1.1, 1.2]
    N_ens_sizes  = [1, 5, 10, 20, 30, 40, 50, 60]
    N_iter       = ceil(500/N_ens_sizes)

    common = (
        rmse_targets = rmse_targets,
        N_ens_sizes  = N_ens_sizes,
        n_repeats    = n_repeats,
        N_iter       = N_iter,
        run_date     = run_date,
        score_kind   = score_kind_from_env(),
        budget_mode  = budget_mode_from_env(),
        tukey_alpha  = 0.75,
        stein        = true,
        val_frac     = 0.1,
        kgmm_ridge   = 0.1,   # per-cluster covariance shrinkage toward the global
                              # covariance; K itself is chosen adaptively from the
                              # actual window size at fit time (see score_gfdt.jl)
    )

    if case == :l63
        # sigma = 0.2, not the published 0.05: those values assume ~1e6 samples.
        # Measured on Gaussian data at this window size (n=1001), single-sigma DSM
        # gives score relerr 0.30 / ||Xi|| 0.69 at sigma=0.05 but 0.15 / 0.24 at
        # sigma=0.2 -- too little smoothing leaves no learning signal at all.
        return (; model = "l63", force_case = nothing, nx = 3,
                  tau_max = 5.0, lag_stride = 5,
                  # 300 epochs measured as good as 3000 at sigma=0.2 (relerr
                  # 0.148 vs 0.246), so more training does not buy accuracy here
                  # -- the ceiling is the sample count, not the optimisation.
                  sigma = 0.2, hidden = 128, base = 16,
                  epochs_init = 500, epochs_warm = 50,
                  lr_init = 1e-3, lr_warm = 3e-4, batch = 256, weight_decay = 0.0,
                  common...)
    elseif case == :l96_const
        return (; model = "l96", force_case = "const-force", nx = 40,
                  tau_max = 5.0, lag_stride = 5,
                  sigma = 0.025, hidden = 128, base = 16,
                  epochs_init = 150, epochs_warm = 20,
                  lr_init = 8e-4, lr_warm = 3e-4, batch = 128, weight_decay = 1e-4,
                  common...)
    elseif case == :l96_vec
        return (; model = "l96", force_case = "vec-force", nx = 40,
                  tau_max = 8.0, lag_stride = 5,
                  sigma = 0.025, hidden = 128, base = 16,
                  epochs_init = 150, epochs_warm = 20,
                  lr_init = 8e-4, lr_warm = 3e-4, batch = 128, weight_decay = 1e-4,
                  common...)
    elseif case == :l96_flux
        return (; model = "l96", force_case = "flux-force", nx = 100,
                  tau_max = 8.0, lag_stride = 10,
                  sigma = 0.025, hidden = 128, base = 16,
                  epochs_init = 150, epochs_warm = 20,
                  lr_init = 8e-4, lr_warm = 3e-4, batch = 128, weight_decay = 1e-4,
                  common...)
    else
        throw(ArgumentError("Unknown experiment: $case"))
    end
end

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
# Tag encodes both switches so the four arms can never overwrite each other.
method_tag(cfg) = string("sblm_", cfg.score_kind, "_", cfg.budget_mode)

function algorithm_type(cfg)
    base = if cfg.score_kind === :gaussian
        "Quasi-Gaussian FDT LM"
    elseif cfg.score_kind === :kgmm
        "Score-based LM (GFDT, KGMM)"
    else
        "Score-based LM (GFDT, DSM)"
    end
    return cfg.budget_mode === :ensemble ? base * " (extended window)" : base
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
# N_TASKS = length(N_ens_sizes) * length(rmse_targets) * n_repeats = 3 * 3 * 100 = 900
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
    mode in (:fair, :ensemble) || throw(ArgumentError("BUDGET_MODE must be fair or ensemble, got $mode"))
    return mode
end
