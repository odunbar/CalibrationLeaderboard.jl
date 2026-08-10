# score_based_lm — L96 opt experiment
# Levenberg-Marquardt on (y - G(θ))' R⁻¹ (y - G(θ)), with the Jacobian from the
# generalized fluctuation-dissipation theorem and a LEARNED score rather than
# from ForwardDiff.
#
# Supports three forcing cases: const-force (nu=1), vec-force (nu=40), flux-force (nu=61).
# GFDT gives the whole ny x nu Jacobian from ONE trajectory, so the per-iteration
# cost is independent of nu -- that is the entire point on vec/flux force, where
# levenberg_marquardt pays 41 and 62 forward evaluations respectively.
#
# Cost metric: outer_iter × N_ens forward-model evaluations (vs LM's × (nu+1)).
#
# Local: EXPERIMENT=l96_const julia --project=. run_l96_sblm.jl
#        EXPERIMENT=l96_vec   julia --project=. run_l96_sblm.jl
# One cell: EXPERIMENT=l96_const julia --project=. run_l96_sblm.jl <task_idx>
# Arms:     SCORE_KIND=dsm|gaussian|kgmm  BUDGET_MODE=fair|ensemble

using BSON
using Distributions
using Flux
using ForwardDiff
using JLD2
using LinearAlgebra
using Random
using Statistics

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

# Both modes spend N_ens forward-model evaluations per iteration and differ only
# in how: N_ens base-length windows, or one window N_ens times longer.  T_start is
# never shortened, so the transient onto the current attractor is always
# discarded in full.
function budget_windows(cfg, prob, N_ens)
    (; lorenz_cfg, obs_cfg) = prob
    if cfg.budget_mode === :ensemble
        W  = obs_cfg.T_end - obs_cfg.T_start
        oc = ObservationConfig(obs_cfg.T_start, obs_cfg.T_start + N_ens * W)
        return (LorenzConfig(lorenz_cfg.dt, oc.T_end), oc, 1)
    else
        return (lorenz_cfg, obs_cfg, N_ens)
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
    lambda = 1.0

    cfg_k, oc_k, n_members = budget_windows(cfg, prob, N_ens)
    win = stats_window_indices(cfg_k, oc_k)
    fwd_unit = cfg_k.T / lorenz_cfg.T     # integration cost in base-run equivalents

    score_model = make_score_model(cfg.score_kind, nx, cfg, rng_net)

    conv_score   = NaN
    n_fwd_actual = 0.0
    cost_accum   = 0            # charged forward-model evaluations, accumulated
    ad_iters     = 0            # iterations that fell back to ForwardDiff
    gfdt_iters   = 0
    final_params = fill(NaN, nu)
    final_output = fill(NaN, ny)

    for outer_iter in 1:cfg.N_iter
        x0p_all = x0 .+ ic_cov_sqrt * randn(rng, nx, n_members)

        Gs = Vector{Vector{Float64}}(undef, n_members)
        Xs = Vector{Matrix{Float64}}(undef, n_members)
        for k in 1:n_members
            out = lorenz_forward_with_states(make_emc(prob, theta), x0p_all[:, k], cfg_k, oc_k)
            Gs[k] = out.G
            Xs[k] = out.states[:, win]
            n_fwd_actual += fwd_unit
        end

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
            J_sum = zeros(ny, nu)
            for k in 1:n_members
                x0p_k  = x0p_all[:, k]
                G_func = th -> lorenz_forward(make_emc(prob, th), x0p_k, cfg_k, oc_k)
                J_sum += ForwardDiff.jacobian(G_func, theta)
                n_fwd_actual += nu * fwd_unit
            end
            Jt = R_inv_var * (J_sum / n_members)
            cost_accum += N_ens * nu             # matches LM's (nu+1) convention
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

        # LM step via augmented-system QR, kept verbatim from levenberg_marquardt.
        # With an ESTIMATED Jacobian the gain ratio does double duty: a low rho
        # means either "step too long" or "the score was poor this iteration", and
        # shrinking the trust region is the right response to both.  Column-pivoted
        # QR also keeps conditioning at kappa(J) rather than kappa(J)^2 and turns a
        # non-finite score into lambda-up rather than a crash -- which matters more
        # here than in LM, and matters most on flux-force where J is rank-deficient.
        d     = max.([norm(Jt[:, j]) for j in 1:nu], eps())
        A_aug = vcat(Jt, sqrt(lambda) * Diagonal(d))
        b_aug = vcat(rt, zeros(nu))
        dtheta = qr(A_aug, ColumnNorm()) \ b_aug

        theta_trial = theta + dtheta
        r_trial_sum = zeros(ny)
        for k in 1:n_members
            r_trial_sum += y - lorenz_forward(make_emc(prob, theta_trial), x0p_all[:, k], cfg_k, oc_k)
            n_fwd_actual += fwd_unit
        end
        rt_trial = R_inv_var * (r_trial_sum / n_members)

        rho = (norm(rt)^2 - norm(rt_trial)^2) / (norm(rt)^2 - norm(Jt * dtheta - rt)^2)

        if rho > 0
            theta = theta_trial
        end
        if !isfinite(rho) || rho < 0.25
            lambda = min(lambda * 4.0, 1e8)
        elseif rho > 0.75
            lambda = max(lambda / 3.0, 1e-10)
        end
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
