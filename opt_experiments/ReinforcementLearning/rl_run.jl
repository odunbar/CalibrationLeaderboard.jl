# ReinforcementLearning — DDPG training loop for one (rmse_target, rng_idx) cell.
# Shared by run_l63_ddpg.jl and run_l96_ddpg.jl.  Requires ddpg_core.jl and rl_env.jl.
#
# Convergence / cost:
#   RMSE_k = sqrt(Φ_k / ny) of the window just rolled out under the APPLIED action
#   F_k = π_θ + exploration noise (so it is conservative w.r.t. the noise-free policy).
#   conv_score = forward-evaluation-equivalents simulated when RMSE_k < rmse_target
#   (see experiment_config.jl); NaN if not reached within cfg.max_evals.

function run_one(cfg, rmse_target, rng_idx, prob)
    (; nx, ny, nu, prior_mean, prior_cov) = prob
    rng     = MersenneTwister(rng_idx)
    rng_net = MersenneTwister(90_000 + rng_idx)   # critic init / minibatches, independent of the env stream

    prior_dist = MvNormal(prior_mean, Matrix(Symmetric(prior_cov)))
    prior_std  = sqrt.(diag(Matrix(prior_cov)))
    θ0         = rand(rng, prior_dist)

    agent = DDPGAgent(cfg, θ0, prior_mean, prior_std, nx, rng_net)
    buf   = ReplayBuffer(nx, nu, cfg.buffer_size)
    env   = reset_env!(RLEnv(prob.x0), prob, rng)

    cost = 0.0
    step = 0
    n_updates    = 0
    conv_score   = NaN
    final_params = fill(NaN, nu)
    final_output = fill(NaN, ny)
    last_loss    = NaN
    warmed       = false

    while cost < cfg.max_evals
        step += 1
        frac = step <= cfg.n_warmup ? cfg.warmup_frac : cfg.explore_frac
        F    = explore_action(agent, rng, frac)
        out  = env_step!(env, prob, cfg, F)
        cost += out.cost

        if !isfinite(out.Φ)
            # blown-up trajectory (extreme forcing): discard and restart from a fresh IC
            reset_env!(env, prob, rng)
            continue
        end

        push_transition!(buf, out.x, F, out.Φ, out.x_next)

        RMSE = sqrt(out.Φ / ny)
        if RMSE < rmse_target
            conv_score   = cost
            final_params = F
            final_output = out.G
            break
        end

        if step >= cfg.n_warmup && buf.n >= cfg.batch_size
            if !warmed
                init_normalization!(agent, buf)
                warmed = true
            end
            for _ in 1:cfg.updates_per_step
                last_loss, _ = ddpg_update!(agent, buf, rng_net)
                n_updates += 1
            end
        end
    end

    policy_params = policy_action(agent)
    @info "rmse_target=$(rmse_target) rng_idx=$(rng_idx) algorithm=$(cfg.algorithm) update_frequency=$(cfg.update_frequency) conv=$(conv_score) " *
          "steps=$(step) updates=$(n_updates) critic_loss=$(round(last_loss; sigdigits = 3)) " *
          "‖Δθ‖/‖θ₀‖=$(round(norm(policy_params - θ0) / norm(θ0); sigdigits = 4))"
    return (; conv_score, final_params, final_output, policy_params, n_steps = step, n_updates)
end

########################################################################
###############  Main (shared)  ########################################
########################################################################

# `build_problem(cfg, output_dir)` is supplied by the driver.
function rl_main(experiment::Symbol, build_problem)
    update_frequency = rl_update_frequency()
    cfg     = experiment_config(experiment, update_frequency, rl_algorithm())
    tasks   = flat_tasks(cfg)
    tidx    = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prob = build_problem(cfg, output_dir)

    @info "RL algorithm=$(cfg.algorithm) update_frequency=$(update_frequency) experiment=$(experiment) algorithm_type=$(algorithm_name(cfg))"

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        rmse_target, rng_idx = tasks[t]
        @info "Task $t: rmse_target=$rmse_target  rng_idx=$rng_idx"
        result = run_one(cfg, rmse_target, rng_idx, prob)
        fn = joinpath(output_dir, result_filename(cfg, rmse_target, rng_idx))
        JLD2.save(fn,
            "conv_score",    result.conv_score,
            "final_params",  result.final_params,
            "final_output",  result.final_output,
            "policy_params", result.policy_params,
            "n_steps",       result.n_steps,
            "n_updates",     result.n_updates,
            "rmse_target",   rmse_target,
            "rng_idx",       rng_idx,
            "algorithm",              String(cfg.algorithm),
            "update_frequency",       String(update_frequency),
        )
        @info "Saved: $fn"
    end
end
