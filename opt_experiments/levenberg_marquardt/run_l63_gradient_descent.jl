# gradient_descent — L63 opt experiment
# Levenberg-Marquardt on the quadratic likelihood (y - G(θ))' R⁻¹ (y - G(θ)).
# Jacobians computed via ForwardDiff.jl (forward-mode AD through the Lorenz ODE).
#
# N_ens independent restarts advance one LM step per outer iteration.
# Cost metric: outer_iter × N_ens × (nu + 1) forward-model evaluations,
#   where nu = number of parameters and +1 is the residual evaluation.
#
# Local (all cells):  julia --project=. run_l63_gradient_descent.jl
# Local (one cell):   julia --project=. run_l63_gradient_descent.jl <task_index>

using Distributions
using ForwardDiff
using JLD2
using LinearAlgebra
using Random
using Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")

########################################################################
###############  Problem setup  #######################################
########################################################################

function build_l63_problem(output_dir; rng_seed_init = 11)
    rng_i = MersenneTwister(rng_seed_init)
    t = 0.01; T = 40.0; nx = 3; ny = 9
    u_truth   = EnsembleMemberConfig([28.0, 8.0 / 3.0])
    x_initial = rand(rng_i, Normal(0.0, 1.0), nx)

    prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
    if isfile(prelim_file)
        ld          = load_preliminaries(prelim_file)
        x0          = ld.x0
        y           = ld.y
        R           = ld.R
        R_inv_var   = ld.R_inv_var
        ic_cov_sqrt = ld.ic_cov_sqrt
        lorenz_cfg  = ld.lorenz_config_settings
        obs_cfg     = ld.observation_config
        @info "Loaded precomputed preliminaries from $prelim_file"
    else
        pdc = compute_perfect_data(
            u_truth, nx, ny,
            LorenzConfig(t, 1000.0), x_initial,
            LorenzConfig(t, T), ObservationConfig(30.0, T);
            R_n_samples = 36,
        )
        x0          = pdc.x0
        y           = pdc.y
        R           = pdc.R
        R_inv_var   = pdc.R_inv_var
        ic_cov_sqrt = pdc.ic_cov_sqrt
        lorenz_cfg  = pdc.lorenz_config_settings
        obs_cfg     = pdc.observation_config
        save_preliminaries(pdc, prelim_file)
        @info "Saved computed quantities to $prelim_file"
    end

    return (; x0, y, R, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx)
end

########################################################################
###############  Run one (N_ens, rng_idx) cell  #######################
########################################################################

# Single LM trajectory; N_ens controls how many IC perturbations are averaged
# each step to reduce gradient noise.  N_ens=1 is pure LM with a single IC sample.
# Cost: outer_iter × N_ens × (nu + 1) forward-model evaluations.
function run_one(cfg, N_ens, rng_idx, problem)
    (; x0, y, R, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx) = problem
    rng = MersenneTwister(rng_idx)

    # L63: parameters are (log ρ, log β), matching the EKI convention exp.(θ) = [ρ, β]
    nu = 2
    ny = length(y)
    prior_dist = MvNormal([3.3, 1.2], diagm([0.15^2, 0.5^2]))

    θ      = rand(rng, prior_dist)   # single trajectory starting point
    θ_init = copy(θ)
    λ = 1.0                          # LM damping

    conv_score   = NaN
    final_params = fill(NaN, nu)
    final_output = fill(NaN, ny)

    for outer_iter in 1:cfg.N_iter
        # Draw N_ens IC perturbations for this step.  Fixing them before calling
        # ForwardDiff ensures the Jacobian closure is deterministic (only θ is dual).
        x0p_all = x0 .+ ic_cov_sqrt * randn(rng, nx, N_ens)

        # Accumulate Jacobian and residual over IC samples
        J_sum = zeros(ny, nu)
        r_sum = zeros(ny)
        for k in 1:N_ens
            x0p_k  = x0p_all[:, k]
            G_func = log_θ -> lorenz_forward(
                EnsembleMemberConfig(exp.(log_θ)), x0p_k, lorenz_cfg, obs_cfg)
            G_k    = G_func(θ)
            r_sum += y - G_k
            J_sum += ForwardDiff.jacobian(G_func, θ)
        end
        J̃ = R_inv_var * (J_sum / N_ens)   # whitened Jacobian (ny × nu)
        r̃ = R_inv_var * (r_sum / N_ens)   # whitened residual (ny)
        RMSE = norm(r̃) / sqrt(ny)

        if RMSE < cfg.target_rmse
            conv_score   = outer_iter * N_ens * (nu + 1)
            final_params = θ
            final_output = y - (r_sum / N_ens)
            break
        end

        # ─────────────────────────────────────────────────────────────────────
        # LM step via augmented-system QR (avoids forming J̃'J̃, condition
        # number scales as κ(J̃) not κ(J̃)²):
        #
        #   min_Δθ ‖ [    J̃    ] Δθ - [r̃] ‖²
        #           ‖ [√λ·diag(d)]     [0 ] ‖
        #
        # d = column norms of J̃ (MINPACK scaling; floored to avoid /0).
        # Column-pivoted QR handles rank-deficient J̃ and Jacobian overflow:
        # if J̃ has Inf entries, Δθ = NaN → ρ = NaN → !isfinite(ρ) → λ↑, no crash.
        # Reusing x0p_all for the trial step makes the comparison fair.
        # ─────────────────────────────────────────────────────────────────────
        d     = max.([norm(J̃[:, j]) for j in 1:nu], eps())
        A_aug = vcat(J̃, sqrt(λ) * Diagonal(d))
        b_aug = vcat(r̃, zeros(nu))
        Δθ    = qr(A_aug, ColumnNorm()) \ b_aug

        θ_trial = θ + Δθ
        r_trial_sum = zeros(ny)
        for k in 1:N_ens
            r_trial_sum += y - lorenz_forward(
                EnsembleMemberConfig(exp.(θ_trial)), x0p_all[:, k], lorenz_cfg, obs_cfg)
        end
        r̃_trial = R_inv_var * (r_trial_sum / N_ens)

        # Gain ratio: actual vs. predicted cost reduction
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

    @info "N_ens=$(N_ens)  rng_idx=$(rng_idx)  conv=$(conv_score)  ‖Δθ‖/‖θ₀‖=$(round(norm(θ - θ_init) / norm(θ_init); sigdigits=4))"
    return (; conv_score, final_params, final_output)
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

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        N_ens, rng_idx = tasks[t]
        @info "Task $t: N_ens=$N_ens, rng_idx=$rng_idx"
        result = run_one(cfg, N_ens, rng_idx, problem)
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rng_idx))
        JLD2.save(fn,
            "conv_score",   result.conv_score,
            "final_params", result.final_params,
            "final_output", result.final_output,
            "N_ens",        N_ens,
            "rng_idx",      rng_idx,
            "target_rmse",  cfg.target_rmse,
        )
        @info "Saved: $fn"
    end
end

main()
