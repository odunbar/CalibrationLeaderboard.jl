# score_based_adam — L63 opt experiment
# Adam on the quadratic likelihood (y - G(θ))' R⁻¹ (y - G(θ)).
# Jacobians come from the generalized fluctuation-dissipation theorem with a
# LEARNED score (KGMM / DSM / quasi-Gaussian), not from ForwardDiff -- the L63
# tangent-linear solution grows like e^{λ₁t}, so an AD Jacobian over the T=40 run
# has entries ~1e15 with arbitrary sign while the true statistical response is
# O(1)-O(100).  See opt_experiments/score_based_lm/README.md for the measured
# numbers behind this and for the collapsed-attractor AD fallback below.
#
# Gradient: g = -J̃ᵀ r̃  where  J̃ = R_inv_var * J (from GFDT),  r̃ = R_inv_var * (y - G(θ)).
#
# Unlike score_based_lm's Levenberg-Marquardt step, Adam has no gain-ratio /
# trust-region signal to react to a bad-iteration score: LM's ρ naturally
# shrinks the step when the estimated Jacobian is poor, but a fixed-α Adam step
# has no such backstop.  stein_recalibrate's max_defect gate (score_gfdt.jl) is
# the only defense here against a poor score driving a bad step.
#
# Cost metric: outer_iter × N_ens forward-model evaluations (score-based, unlike
#   opt_experiments/adam's × (nu + 1) -- GFDT gives the whole Jacobian from one
#   trajectory, so the per-iteration cost is independent of nu).
#
# Local (all cells):  julia --project=. run_l63_sbadam.jl
# Local (one cell):   julia --project=. run_l63_sbadam.jl <task_index>
# Arms:               SCORE_KIND=dsm|gaussian|kgmm  BUDGET_MODE=serial|parallel

using Distributions
using Flux
using ForwardDiff          # only for the collapsed-attractor fallback (see run_one)
using JLD2
using LinearAlgebra
using Random
using Statistics

# The n_members forward solves below are independent (see run_one) and threaded
# with Threads.@threads; BLAS threading underneath would oversubscribe on top of
# that, so hand BLAS a single thread whenever Julia itself has more than one.
Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)

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

# Identical to score_based_lm's budget_windows -- both modes spend N_ens
# forward-model evaluations per outer iteration and integrate the same total
# model time, T_start once + N_ens*W:
#   :serial   — one continuous window N_ens times longer.  Inherently
#               sequential, but the whole span is one attractor sample, giving
#               the longest usable lags in the GFDT correlation integral.
#   :parallel — spin up ONCE per outer iteration, then fork N_ens independent
#               base-length branches from small perturbations of that shared
#               attractor state.  Mutually independent, so they can be
#               integrated concurrently (Threads.@threads below).
function budget_windows(cfg, problem, N_ens)
    (; lorenz_cfg, obs_cfg) = problem
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

    # Adam hyperparameters (identical to opt_experiments/adam)
    α  = cfg.adam_alpha
    β₁ = cfg.adam_beta1
    β₂ = cfg.adam_beta2
    ε  = cfg.adam_eps
    m  = zeros(nu)   # first moment (mean)
    v  = zeros(nu)   # second moment (uncentred variance)

    cfg_k, oc_k, n_members, spinup_cfg = budget_windows(cfg, problem, N_ens)
    win = stats_window_indices(cfg_k, oc_k)
    # Integration-time cost of one solve, in base-run equivalents (audit only).
    fwd_unit        = cfg_k.T / lorenz_cfg.T
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
            x_attr  = lorenz_solve(EnsembleMemberConfig(exp.(θ)), x0, spinup_cfg)[:, end]
            x0p_all = x_attr .+ ic_cov_sqrt * randn(rng, nx, n_members)
            n_fwd_actual += fwd_unit_spinup
        end

        Gs = Vector{Vector{Float64}}(undef, n_members)
        Xs = Vector{Matrix{Float64}}(undef, n_members)
        Threads.@threads for k in 1:n_members
            out = lorenz_forward_with_states(
                EnsembleMemberConfig(exp.(θ)), x0p_all[:, k], cfg_k, oc_k)
            Gs[k] = out.G
            Xs[k] = out.states[:, win]
        end
        n_fwd_actual += n_members * fwd_unit

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
        # ForwardDiff is well-posed on exactly those parameters, because no
        # chaos means no tangent-linear blow-up.  So: FDT where the measure is
        # smooth, AD where the tangent linear is stable, charged honestly either way.
        if any(is_collapsed_window, Xs)
            J_per_member = Vector{Matrix{Float64}}(undef, n_members)
            Threads.@threads for k in 1:n_members
                x0p_k  = x0p_all[:, k]
                G_func = log_θ -> lorenz_forward(
                    EnsembleMemberConfig(exp.(log_θ)), x0p_k, cfg_k, oc_k)
                J_per_member[k] = ForwardDiff.jacobian(G_func, θ)
            end
            n_fwd_actual += n_members * nu * fwd_unit
            J̃ = R_inv_var * (sum(J_per_member) / n_members)
            cost_accum += N_ens * nu             # matches adam's (nu+1) convention
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

        # ╔══════════════════════════════════════════════════════════════════════════╗
        # ║  Adam step on L(θ) = ½ ‖r̃‖², with J̃ estimated via GFDT/score instead of ║
        # ║  ForwardDiff (kept verbatim from opt_experiments/adam otherwise)          ║
        # ║  Gradient: g = ∇L = -J̃ᵀ r̃                                             ║
        # ║  m_t = β₁ m_{t-1} + (1-β₁) g_t          (first moment)               ║
        # ║  v_t = β₂ v_{t-1} + (1-β₂) g_t²         (second moment)              ║
        # ║  θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)    (bias-corrected update)      ║
        # ╚══════════════════════════════════════════════════════════════════════════╝
        g  = -(J̃' * r̃)
        m  = β₁ * m + (1 - β₁) * g
        v  = β₂ * v + (1 - β₂) * g .* g
        m̂  = m / (1 - β₁^outer_iter)
        v̂  = v / (1 - β₂^outer_iter)
        θ  = θ - α * m̂ ./ (sqrt.(v̂) .+ ε)
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
