# score_based_lm — L63 opt experiment
# Levenberg-Marquardt on the quadratic likelihood (y - G(θ))' R⁻¹ (y - G(θ)).
# Jacobians come from the generalized fluctuation-dissipation theorem with a
# LEARNED score, not from ForwardDiff.
#
# Why not ForwardDiff: the L63 tangent-linear solution grows like e^{λ₁t}, so over
# the T=40 run the AD Jacobian has entries ~1e15 with arbitrary sign while the true
# statistical response is O(1)-O(100).  GFDT estimates the response of the invariant
# measure instead, from ONE unperturbed trajectory, at a cost independent of nu.
#
# Cost metric: outer_iter × N_ens forward-model evaluations (vs LM's × (nu+1)).
#
# Local (all cells):  julia --project=. run_l63_sblm.jl
# Local (one cell):   julia --project=. run_l63_sblm.jl <task_index>
# Arms:               SCORE_KIND=dsm|gaussian|kgmm  BUDGET_MODE=fair|ensemble

using Distributions
using Flux
using ForwardDiff          # only for the collapsed-attractor fallback (see run_one)
using JLD2
using LinearAlgebra
using Random
using Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")
include("score_gfdt.jl")
include("score_nets.jl")
include("gfdt_l63.jl")

########################################################################
###############  Problem setup  #######################################
########################################################################

function build_l63_problem(output_dir)
    prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
    isfile(prelim_file) || error("Prelim file not found: $prelim_file\nRun l63_preliminaries.jl first.")
    ld = load_preliminaries(prelim_file)
    @info "Loaded L63 preliminaries from $prelim_file"
    return (; x0 = ld.x0, y = ld.y, R = ld.R, R_inv_var = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config, nx = 3)
end

# Both budget modes spend N_ens forward-model evaluations per iteration and
# differ only in how: N_ens base-length windows, or one window N_ens times longer.
# T_start is never shortened, so the transient onto the current attractor is
# always discarded in full.
function budget_windows(cfg, problem, N_ens)
    (; lorenz_cfg, obs_cfg) = problem
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

function run_one(cfg, N_ens, rmse_target, rng_idx, problem)
    (; x0, y, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx) = problem

    # Two independent streams: consuming `rng` inside score training would make
    # the IC draws depend on the epoch count, destroying reproducibility across
    # hyperparameter changes.
    rng     = MersenneTwister(rng_idx)
    rng_net = MersenneTwister(90_000 + rng_idx)

    # L63: parameters are (log ρ, log β), matching the EKI convention exp.(θ) = [ρ, β]
    nu = 2
    ny = length(y)
    prior_dist = MvNormal([3.3, 1.2], diagm([0.15^2, 0.5^2]))

    θ      = rand(rng, prior_dist)
    θ_init = copy(θ)
    λ = 1.0

    cfg_k, oc_k, n_members = budget_windows(cfg, problem, N_ens)
    win = stats_window_indices(cfg_k, oc_k)
    # Integration-time cost of one solve, in base-run equivalents (audit only).
    fwd_unit = cfg_k.T / lorenz_cfg.T

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
            out = lorenz_forward_with_states(
                EnsembleMemberConfig(exp.(θ)), x0p_all[:, k], cfg_k, oc_k)
            Gs[k] = out.G
            Xs[k] = out.states[:, win]
            n_fwd_actual += fwd_unit
        end

        cost_accum += N_ens                      # the residual/state evaluation
        G_bar = mean(Gs)
        r̃ = R_inv_var * (y - G_bar)
        RMSE = norm(r̃) / sqrt(ny)

        if RMSE < rmse_target
            conv_score   = cost_accum
            final_params = θ
            final_output = G_bar
            break
        end

        # ── Jacobian ────────────────────────────────────────────────────────
        # Roughly 30% of draws from this prior collapse L63 onto a fixed point
        # (large beta kills the chaos).  There the invariant measure is a point
        # mass, no score exists, and GFDT returns a large meaningless Jacobian --
        # measured ||J||=9.7e4 against a true response of ~36.  ForwardDiff is
        # well-posed on exactly those parameters, because no chaos means no
        # tangent-linear blow-up.  So: FDT where the measure is smooth, AD where
        # the tangent linear is stable, charged honestly either way.
        if any(is_collapsed_window, Xs)
            J_sum = zeros(ny, nu)
            for k in 1:n_members
                x0p_k  = x0p_all[:, k]
                G_func = log_θ -> lorenz_forward(
                    EnsembleMemberConfig(exp.(log_θ)), x0p_k, cfg_k, oc_k)
                J_sum += ForwardDiff.jacobian(G_func, θ)
                n_fwd_actual += nu * fwd_unit
            end
            J̃ = R_inv_var * (J_sum / n_members)
            cost_accum += N_ens * nu             # matches LM's (nu+1) convention
            ad_iters += 1
        else
            score_model, _ = refresh_score!(score_model, reduce(hcat, Xs), cfg, rng_net)
            res = gfdt_jacobian(
                Xs, score_model,
                moment_observables_l63,
                (X, S) -> conjugate_variables_l63(X, S, θ),
                dphi_dm_l63;
                dt = cfg_k.dt, lag_stride = cfg.lag_stride, tau_max = cfg.tau_max,
                tukey_alpha = cfg.tukey_alpha, stein = cfg.stein,
            )
            J̃ = R_inv_var * res.J                # no extra forward evaluations
            gfdt_iters += 1
        end

        # ─────────────────────────────────────────────────────────────────────
        # LM step via augmented-system QR (kept verbatim from levenberg_marquardt).
        # The gain ratio does double duty here: with an ESTIMATED Jacobian a low ρ
        # means either "step too long" or "the score was poor this iteration", and
        # either way shrinking the trust region is the right response.  A fixed
        # Tikhonov γ (as in the published Gauss-Newton loop) cannot react at all.
        # Column-pivoted QR also keeps conditioning at κ(J̃) rather than κ(J̃)²,
        # and turns a non-finite score into λ↑ rather than a crash.
        # ─────────────────────────────────────────────────────────────────────
        d     = max.([norm(J̃[:, j]) for j in 1:nu], eps())
        A_aug = vcat(J̃, sqrt(λ) * Diagonal(d))
        b_aug = vcat(r̃, zeros(nu))
        Δθ    = qr(A_aug, ColumnNorm()) \ b_aug

        θ_trial = θ + Δθ
        r_trial_sum = zeros(ny)
        for k in 1:n_members
            r_trial_sum += y - lorenz_forward(
                EnsembleMemberConfig(exp.(θ_trial)), x0p_all[:, k], cfg_k, oc_k)
            n_fwd_actual += fwd_unit
        end
        r̃_trial = R_inv_var * (r_trial_sum / n_members)

        ρ = (norm(r̃)^2 - norm(r̃_trial)^2) / (norm(r̃)^2 - norm(J̃ * Δθ - r̃)^2)

        if ρ > 0
            θ = θ_trial
        end
        if !isfinite(ρ) || ρ < 0.25
            λ = min(λ * 4.0, 1e8)
        elseif ρ > 0.75
            λ = max(λ / 3.0, 1e-10)
        end
    end

    @info "N_ens=$(N_ens) rmse_target=$(rmse_target) rng_idx=$(rng_idx) conv=$(conv_score) " *
          "gfdt_iters=$(gfdt_iters) ad_iters=$(ad_iters) " *
          "‖Δθ‖/‖θ₀‖=$(round(norm(θ - θ_init) / norm(θ_init); sigdigits=4))"
    return (; conv_score, n_fwd_actual, ad_iters, gfdt_iters, final_params, final_output)
end

########################################################################
###############  Main  ################################################
########################################################################

function main()
    experiment = :l63
    cfg   = experiment_config(experiment)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    problem = build_l63_problem(output_dir)

    @info "score_kind=$(cfg.score_kind)  budget_mode=$(cfg.budget_mode)  " *
          "algorithm_type=$(algorithm_type(cfg))"

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        N_ens, rmse_target, rng_idx = tasks[t]
        @info "Task $t: N_ens=$N_ens, rmse_target=$rmse_target, rng_idx=$rng_idx"
        result = run_one(cfg, N_ens, rmse_target, rng_idx, problem)
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
