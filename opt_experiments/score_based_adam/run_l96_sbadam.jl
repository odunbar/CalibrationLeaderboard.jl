# score_based_adam — L96 opt experiment
# Adam on (y - G(θ))' R⁻¹ (y - G(θ)), with the Jacobian from the generalized
# fluctuation-dissipation theorem and a LEARNED score (KGMM / DSM / quasi-Gaussian)
# rather than from ForwardDiff.
#
# Supports three forcing cases: const-force (nu=1), vec-force (nu=40), flux-force (nu=61).
# GFDT gives the whole ny x nu Jacobian from ONE trajectory, so the per-iteration
# cost is independent of nu -- the entire point on vec/flux force, where
# opt_experiments/adam pays 41 and 62 forward evaluations respectively.
#
# Unlike score_based_lm's Levenberg-Marquardt step, Adam has no gain-ratio /
# trust-region signal to react to a bad-iteration score: LM's rho naturally
# shrinks the step when the estimated Jacobian is poor, but a fixed-alpha Adam
# step has no such backstop.  stein_recalibrate's max_defect gate (score_gfdt.jl)
# is the only defense here against a poor score driving a bad step.
#
# Cost metric: outer_iter × N_ens forward-model evaluations (vs adam's × (nu+1)).
#
# Local: EXPERIMENT=l96_const julia --project=. run_l96_sbadam.jl
#        EXPERIMENT=l96_vec   julia --project=. run_l96_sbadam.jl
# One cell: EXPERIMENT=l96_const julia --project=. run_l96_sbadam.jl <task_idx>
# Arms:     SCORE_KIND=dsm|gaussian|kgmm  BUDGET_MODE=serial|parallel

using BSON
using Distributions
using Flux
using ForwardDiff
using JLD2
using LinearAlgebra
using Random
using Statistics

# The n_members forward solves below are independent (see run_one) and threaded
# with Threads.@threads; BLAS threading underneath would oversubscribe on top of
# that, so hand BLAS a single thread whenever Julia itself has more than one.
Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")
include("score_gfdt.jl")
include("score_nets.jl")
include("gfdt_l96.jl")
include("l96_problem.jl")

########################################################################
###############  Budget modes  ########################################
########################################################################

# Both budget modes spend N_ens forward-model evaluations per iteration and
# integrate the same total model time, T_start once + N_ens*W, and differ only
# in how that N_ens*W is arranged:
#   :serial   — one continuous window N_ens times longer than the base window.
#               Inherently sequential (each step depends on the last), but the
#               whole span is one attractor sample, giving the longest usable
#               lags in the GFDT correlation integral for a given cost.
#   :parallel — spin up ONCE (returned as `spinup_cfg`), then N_ens independent
#               base-length branches perturbed off that attractor state.  The
#               branches carry no burn-in of their own and are mutually
#               independent, so they can be integrated concurrently.
# Returns (cfg_k, oc_k, n_members, spinup_cfg); spinup_cfg is `nothing` unless
# a separate once-per-iteration spin-up is needed.
function budget_windows(cfg, prob, N_ens)
    (; lorenz_cfg, obs_cfg) = prob
    W = obs_cfg.T_end - obs_cfg.T_start
    if cfg.budget_mode === :serial
        oc = ObservationConfig(obs_cfg.T_start, obs_cfg.T_start + N_ens * W)
        return (LorenzConfig(lorenz_cfg.dt, oc.T_end), oc, 1, nothing)
    else # :parallel
        branch_cfg = LorenzConfig(lorenz_cfg.dt, W)
        branch_oc  = ObservationConfig(lorenz_cfg.dt, W)
        spinup_cfg = LorenzConfig(lorenz_cfg.dt, obs_cfg.T_start)
        return (branch_cfg, branch_oc, N_ens, spinup_cfg)
    end
end

########################################################################
###############  Run one (N_ens, rmse_target, rng_idx) cell  ##########
########################################################################

function run_one(cfg, N_ens, rmse_target, rng_idx, prob)
    (; x0, y, R_inv_var, ic_cov_sqrt, lorenz_cfg, nx, ny, nu, prior_mean, prior_cov, case) = prob

    # Two independent streams: consuming `rng` inside score training would make the
    # IC draws depend on the epoch count, destroying reproducibility.
    rng     = MersenneTwister(rng_idx)
    rng_net = MersenneTwister(90_000 + rng_idx)

    prior_dist = MvNormal(prior_mean, Matrix(Symmetric(prior_cov)))
    theta      = rand(rng, prior_dist)
    theta_init = copy(theta)

    # Adam hyperparameters (identical to opt_experiments/adam)
    alpha = cfg.adam_alpha
    beta1 = cfg.adam_beta1
    beta2 = cfg.adam_beta2
    epsad = cfg.adam_eps
    m = zeros(nu)   # first moment (mean)
    v = zeros(nu)   # second moment (uncentred variance)

    cfg_k, oc_k, n_members, spinup_cfg = budget_windows(cfg, prob, N_ens)
    win = stats_window_indices(cfg_k, oc_k)
    fwd_unit        = cfg_k.T / lorenz_cfg.T     # integration cost in base-run equivalents
    fwd_unit_spinup = spinup_cfg === nothing ? 0.0 : spinup_cfg.T / lorenz_cfg.T

    score_model = make_score_model(cfg.score_kind, nx, cfg, rng_net)

    conv_score   = NaN
    n_fwd_actual = 0.0
    cost_accum   = 0            # charged forward-model evaluations, accumulated
    ad_iters     = 0            # iterations that fell back to ForwardDiff
    gfdt_iters   = 0
    final_params = fill(NaN, nu)
    final_output = fill(NaN, ny)

    for outer_iter in 1:n_iter_for(cfg, N_ens)
        # :parallel spins up once onto the CURRENT theta's attractor, then forks
        # n_members branches from small perturbations of that shared state;
        # :serial perturbs the (truth-attractor) x0 directly, since cfg_k
        # already carries its own T_start burn-in for every member.
        if spinup_cfg === nothing
            x0p_all = x0 .+ ic_cov_sqrt * randn(rng, nx, n_members)
        else
            x_attr  = lorenz_solve(make_emc(prob, theta), x0, spinup_cfg)[:, end]
            x0p_all = x_attr .+ ic_cov_sqrt * randn(rng, nx, n_members)
            n_fwd_actual += fwd_unit_spinup
        end

        Gs = Vector{Vector{Float64}}(undef, n_members)
        Xs = Vector{Matrix{Float64}}(undef, n_members)
        Threads.@threads for k in 1:n_members
            out = lorenz_forward_with_states(make_emc(prob, theta), x0p_all[:, k], cfg_k, oc_k)
            Gs[k] = out.G
            Xs[k] = out.states[:, win]
        end
        n_fwd_actual += n_members * fwd_unit

        cost_accum += N_ens                      # the residual/state evaluation
        G_bar = mean(Gs)
        rt = R_inv_var * (y - G_bar)
        RMSE = norm(rt) / sqrt(ny)

        if RMSE < rmse_target
            conv_score   = cost_accum
            final_params = theta
            final_output = G_bar
            break
        end

        # Jacobian.  If the window has collapsed onto a fixed point the invariant
        # measure is a point mass, no score exists, and GFDT returns a large
        # meaningless Jacobian -- but ForwardDiff is well-posed there precisely
        # because there is no chaos to blow up the tangent linear.  So: FDT where
        # the measure is smooth, AD where the tangent linear is stable, charged
        # honestly either way.  (Rare for L96 with these priors; common for L63.)
        if any(is_collapsed_window, Xs)
            J_per_member = Vector{Matrix{Float64}}(undef, n_members)
            Threads.@threads for k in 1:n_members
                x0p_k  = x0p_all[:, k]
                G_func = th -> lorenz_forward(make_emc(prob, th), x0p_k, cfg_k, oc_k)
                J_per_member[k] = ForwardDiff.jacobian(G_func, theta)
            end
            n_fwd_actual += n_members * nu * fwd_unit
            Jt = R_inv_var * (sum(J_per_member) / n_members)
            cost_accum += N_ens * nu             # matches adam's (nu+1) convention
            ad_iters += 1
        else
            score_model, _ = refresh_score!(score_model, reduce(hcat, Xs), cfg, rng_net)
            res = gfdt_jacobian(
                Xs, score_model,
                moment_observables_l96,
                (X, S) -> conjugate_variables_l96(case, X, S, theta, prob),
                m -> dphi_dm_l96(m, nx);
                dt = cfg_k.dt, lag_stride = cfg.lag_stride, tau_max = cfg.tau_max,
                tukey_alpha = cfg.tukey_alpha, stein = cfg.stein,
            )
            Jt = R_inv_var * res.J               # no extra forward evaluations
            gfdt_iters += 1
        end

        # Adam step on L(theta) = 1/2 ||rt||^2, with Jt estimated via GFDT/score
        # instead of ForwardDiff (kept verbatim from opt_experiments/adam otherwise):
        #   g       = -Jt' * rt
        #   m_t     = beta1*m_{t-1} + (1-beta1)*g_t
        #   v_t     = beta2*v_{t-1} + (1-beta2)*g_t^2
        #   theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
        g  = -(Jt' * rt)
        m  = beta1 * m + (1 - beta1) * g
        v  = beta2 * v + (1 - beta2) * g .* g
        mhat = m / (1 - beta1^outer_iter)
        vhat = v / (1 - beta2^outer_iter)
        theta = theta - alpha * mhat ./ (sqrt.(vhat) .+ epsad)
    end

    @info "N_ens=$(N_ens) rmse_target=$(rmse_target) rng_idx=$(rng_idx) conv=$(conv_score) " *
          "gfdt_iters=$(gfdt_iters) ad_iters=$(ad_iters) " *
          "norm-step=$(round(norm(theta - theta_init) / norm(theta_init); sigdigits=4))"
    return (; conv_score, n_fwd_actual, ad_iters, gfdt_iters, final_params, final_output)
end

########################################################################
###############  Main  ################################################
########################################################################

function main()
    experiment = l96_experiment()
    cfg        = experiment_config(experiment)
    tasks      = flat_tasks(cfg)
    tidx       = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    case = cfg.force_case
    prob = build_l96_problem(case, output_dir)

    @info "score_kind=$(cfg.score_kind)  budget_mode=$(cfg.budget_mode)  " *
          "algorithm_type=$(algorithm_type(cfg))"

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        N_ens, rmse_target, rng_idx = tasks[t]
        @info "Task $t: case=$case  N_ens=$N_ens  rmse_target=$rmse_target  rng_idx=$rng_idx"
        result = run_one(cfg, N_ens, rmse_target, rng_idx, prob)
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rmse_target, rng_idx))
        JLD2.save(fn,
            "conv_score",   result.conv_score,
            "n_fwd_actual", result.n_fwd_actual,
            "ad_iters",     result.ad_iters,
            "gfdt_iters",   result.gfdt_iters,
            "final_params", result.final_params,
            "final_output", result.final_output,
            "N_ens",        N_ens,
            "rng_idx",      rng_idx,
            "rmse_target",  rmse_target,
            "score_kind",   String(cfg.score_kind),
            "budget_mode",  String(cfg.budget_mode),
        )
        @info "Saved: $fn"
    end
end

main()
