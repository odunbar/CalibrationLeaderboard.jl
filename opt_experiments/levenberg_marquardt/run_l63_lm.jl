# lm — L63 opt experiment
# Levenberg-Marquardt on the quadratic likelihood (y - G(θ))' R⁻¹ (y - G(θ)).
# Jacobians computed via ForwardDiff.jl (forward-mode AD through the Lorenz ODE).
#
# N_ens independent restarts advance one LM step per outer iteration.
# Cost metric: outer_iter × N_ens × (nu + 1) forward-model evaluations,
#   where nu = number of parameters and +1 is the residual evaluation.
#
# Local (all cells):  julia --project=. run_l63_lm.jl
# Local (one cell):   julia --project=. run_l63_lm.jl <task_index>

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

########################################################################
###############  Run one (N_ens, rng_idx) cell  #######################
########################################################################

# Single LM trajectory; N_ens controls how many IC perturbations are averaged
# each step to reduce gradient noise.  N_ens=1 is pure LM with a single IC sample.
# Cost: outer_iter × N_ens × (nu + 1) forward-model evaluations.
function run_one(cfg, rmse_target, rng_idx, problem)
    N_ens = 1
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

        if RMSE < rmse_target
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

    @info "rmse_target=$(rmse_target)  rng_idx=$(rng_idx)  conv=$(conv_score)  ‖Δθ‖/‖θ₀‖=$(round(norm(θ - θ_init) / norm(θ_init); sigdigits=4))"
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
        rmse_target, rng_idx = tasks[t]
        @info "Task $t: rmse_target=$rmse_target, rng_idx=$rng_idx"
        result = run_one(cfg, rmse_target, rng_idx, problem)
        fn = joinpath(output_dir, result_filename(cfg, rmse_target, rng_idx))
        JLD2.save(fn,
            "conv_score",   result.conv_score,
            "final_params", result.final_params,
            "final_output", result.final_output,
            "rng_idx",      rng_idx,
            "rmse_target",  rmse_target,
        )
        @info "Saved: $fn"
    end
end

main()
