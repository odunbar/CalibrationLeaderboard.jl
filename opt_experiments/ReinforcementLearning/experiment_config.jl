using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[1]   # fallback only; EXPERIMENT env var / ARGS[2] take precedence

# :long  -> RL step == the full statistics window: one RL update per forward evaluation
# :short -> RL step = a fraction (1/steps_per_window) of the window: many RL updates per
#           window-length of simulation, each at fractional cost (sliding statistic window)
update_frequencies = [:short, :long]
UPDATE_FREQUENCY = update_frequencies[1]  # edit here to switch; UPDATE_FREQUENCY env var / ARGS[3] take precedence

# :ddpg -> plain DDPG: one critic, actor/target updated every critic update, no target smoothing
# :td3  -> TD3: twin critics (min target), delayed actor/target updates, target policy smoothing
# (both use Gaussian exploration noise on the applied action)
algorithms = [:ddpg, :td3]
ALGORITHM = algorithms[1]                 # edit here to switch; ALGORITHM env var / ARGS[4] take precedence

# Pinned at submission time via RUN_DATE env var (set by submit_*.sh).
# Falls back to today() for local runs.
run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Cost convention (leaderboard metric):  1 unit = one forward-model evaluation, i.e. a
# simulation of the full LorenzConfig length T (spin-up + statistics window).
#   long : every RL step is one full forward evaluation             -> 1 unit / step
#   short: every RL step advances the sliding window by W/steps_per_window time units
#          -> (W/steps_per_window)/T units / step (plus one full evaluation to initialise)
# `max_evals` is the budget in these units; no Jacobian (nu+1) factor appears since the
# actor gradient comes from the learned critic, not from differentiating the simulator.
function experiment_config(case::Symbol, update_frequency::Symbol, algorithm::Symbol)
    update_frequency in (:short, :long) || throw(ArgumentError("Unknown update_frequency: $update_frequency"))
    algorithm in algorithms || throw(ArgumentError("Unknown algorithm: $algorithm"))

    n_repeats    = 30      # matches score_based_adam / score_based_lm
    rmse_targets = [1.0, 1.1, 1.2]

    # RL hyperparameters shared across cases.  Actor parameters / actions are handled in
    # prior-std units: action = prior_mean + prior_std .* θn.
    rl = (
        # --- Bellman / targets ---
        # gamma=1 makes the critic's F-dependence vanish at the Bellman fixed point for a
        # state-independent policy (the window misfit telescopes away), so gamma < 1.
        gamma             = 0.5,
        polyak            = 0.05,    # target-network rate (1.0 = no target network)
        # --- TD3 switches; these defaults are plain DDPG (TD3 values are set below) ---
        n_critics         = 1,       # >1: bootstrap with the min over critics (TD3)
        policy_delay      = 1,       # actor/target update every policy_delay critic updates
        target_noise      = 0.0,     # TD3 target smoothing noise (prior-std units)
        target_noise_clip = 0.5,
        # --- optimisation ---
        critic_lr         = 1e-3,
        actor_lr          = 0.02,    # Adam step on θn (prior-std units)
        adam_beta1        = 0.9,
        adam_beta2        = 0.999,
        adam_eps          = 1e-8,
        batch_size        = 64,
        updates_per_step  = 1,
        # --- exploration (Gaussian noise on the applied action, prior-std units) ---
        explore_frac      = 0.1,
        warmup_frac       = 0.5,
    )

    # Algorithm-specific overrides: standard TD3 values (Fujimoto et al. 2018)
    if algorithm == :td3
        rl = merge(rl, (n_critics = 2, policy_delay = 2, target_noise = 0.2, target_noise_clip = 0.5))
    end

    # Update-frequency-specific budget / buffer
    if update_frequency == :long
        var_cfg = (steps_per_window = 1,  n_warmup = 20, buffer_size = 2000, max_evals = 10000)
    else  # :short — steps_per_window = 10 (use 100 for a 100x shorter RL step)
        var_cfg = (steps_per_window = 10, n_warmup = 50, buffer_size = 5000, max_evals = 10000)
    end

    base = merge(rl, var_cfg, (algorithm = algorithm, update_frequency = update_frequency,
                               rmse_targets = rmse_targets, n_repeats = n_repeats, run_date = run_date))

    if case == :l63
        # prior std = [0.15, 0.5] (log rho, log beta)
        return merge(base, (model = "l63", force_case = nothing, hidden = 64))
    elseif case == :l96_const
        # prior std = 0.4 (log forcing)
        return merge(base, (model = "l96", force_case = "const-force", hidden = 64))
    elseif case == :l96_vec
        # prior std = 3.0 (per-component forcing, correlated prior)
        return merge(base, (model = "l96", force_case = "vec-force", hidden = 128))
    elseif case == :l96_flux
        # prior std = 0.1 (NN weight space)
        return merge(base, (model = "l96", force_case = "flux-force", hidden = 128))
    else
        throw(ArgumentError("Unknown experiment: $case"))
    end
end

# Algorithm-defaulted views (the preliminaries scripts only need force_case).
experiment_config(case::Symbol, update_frequency::Symbol) = experiment_config(case, update_frequency, :ddpg)
experiment_config(case::Symbol) = experiment_config(case, :long, :ddpg)

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
algorithm_name(cfg) = "rl-$(cfg.algorithm)-$(cfg.update_frequency)"

function case_suffix(cfg, rmse_target, rng_idx)
    tgt = replace(string(rmse_target), "." => "p")
    cfg.force_case === nothing ? "$(tgt)_$(rng_idx)" : "$(cfg.force_case)_$(tgt)_$(rng_idx)"
end

function result_filename(cfg, rmse_target, rng_idx)
    "$(cfg.model)_$(algorithm_name(cfg))_result_$(case_suffix(cfg, rmse_target, rng_idx))_$(cfg.run_date).jld2"
end

function nc_filename(cfg)
    if cfg.force_case === nothing
        return "leaderboard_$(algorithm_name(cfg))_$(cfg.model)_$(cfg.run_date).nc"
    else
        return "leaderboard_$(algorithm_name(cfg))_$(cfg.model)_$(cfg.force_case)_$(cfg.run_date).nc"
    end
end

########################################################################
###############  ARRAY-JOB HELPERS  ###################################
########################################################################
flat_tasks(cfg) =
    [(rmse_target, rng_idx) for rmse_target in cfg.rmse_targets for rng_idx in 1:cfg.n_repeats]

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

# Update-frequency dispatch: UPDATE_FREQUENCY env var, then ARGS[3], then the UPDATE_FREQUENCY toggle above.
function rl_update_frequency()
    if haskey(ENV, "UPDATE_FREQUENCY")
        return Symbol(ENV["UPDATE_FREQUENCY"])
    elseif length(ARGS) >= 3 && !isempty(ARGS[3])
        return Symbol(ARGS[3])
    else
        return UPDATE_FREQUENCY
    end
end

# Algorithm dispatch: ALGORITHM env var, then ARGS[4], then the ALGORITHM toggle above.
function rl_algorithm()
    if haskey(ENV, "ALGORITHM")
        return Symbol(ENV["ALGORITHM"])
    elseif length(ARGS) >= 4 && !isempty(ARGS[4])
        return Symbol(ARGS[4])
    else
        return ALGORITHM
    end
end
