# gradient_descent — L96 opt experiment
# Levenberg-Marquardt on (y - G(θ))' R⁻¹ (y - G(θ)) with ForwardDiff.jl Jacobians.
#
# Supports three forcing cases: const-force (nu=1), vec-force (nu=40), flux-force (nu=61).
# N_ens independent restarts; each outer iteration all restarts take one LM step.
# Cost: outer_iter × N_ens × (nu + 1) forward-model evaluations.
#
# Local: EXPERIMENT=l96_const julia --project=. run_l96_gradient_descent.jl
#        EXPERIMENT=l96_vec   julia --project=. run_l96_gradient_descent.jl
#        EXPERIMENT=l96_flux  julia --project=. run_l96_gradient_descent.jl
# One cell: EXPERIMENT=l96_const julia --project=. run_l96_gradient_descent.jl <task_idx>

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

########################################################################
###############  Case-specific problem setup  #########################
########################################################################

function build_l96_problem(case::String, output_dir::String; rng_seed_init = 11)
    rng_i = MersenneTwister(rng_seed_init)
    t     = 0.01

    if case == "const-force"
        nx = 40; T = 14.0; T_start = 4.0; inff = 2.0
        phi = ConstantEMC(8.0)
        phi_structure  = nothing
        sample_range   = nothing
        # Prior: φ ~ LogNormal(log(10), 4/10) ≈ N(10, 4²) on [0,∞)
        # Optimise in log-space: θ = [log(φ)], G_func uses exp(θ[1])
        nu = 1
        prior_mean = [log(10.0)]
        prior_cov  = diagm([0.4^2])

    elseif case == "vec-force"
        nx = 40; T = 54.0; T_start = 4.0; inff = 2.0
        pl = 2.0; psig = 3.0
        sinusoid = 8 .+ 6 * sin.((4 * π * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        phi_structure  = nothing
        sample_range   = nothing
        nu = nx
        prior_mean = 8.0 * ones(nx)
        prior_cov  = [psig^2 * exp(-abs(i - j) / pl) for i in 1:nx, j in 1:nx]

    elseif case == "flux-force"
        nx = 100; T = 54.0; T_start = 4.0; inff = 2.5
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * π * x) / 10)
        x_train = collect(-5.0:0.01:5.0)
        y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure   = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, _   = train_network(phi_structure, x_train, y_train)
        sample_range    = Float32.(collect(-5.0:0.1:4.9))
        phi             = FluxEMC(true_model, sample_range)

        prior_sinusoid(x) = 8.02 .+ 6.5 * sin.(1.02 * (4 * π * x) / 10 + 0.2)
        prior_train   = prior_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        prior_model, prior_mean_f32 = train_network(phi_structure, x_train, prior_train)
        prior_mean    = Float64.(prior_mean_f32)
        prior_cov     = (0.1^2) * I(length(prior_mean))
        nu            = length(prior_mean)

    else
        throw(ArgumentError("Unknown L96 case: $case"))
    end

    ny = 2 * nx
    T_long = 1000.0
    lorenz_cfg = LorenzConfig(t, T)
    obs_cfg    = ObservationConfig(T_start, T)

    prelim_file = joinpath(output_dir, "l96_computed_preliminaries_$(case).jld2")
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
        x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
        pdc = compute_perfect_data(
            phi, nx,
            LorenzConfig(t, T_long), x_initial,
            lorenz_cfg, obs_cfg;
            R_inflation = inff,
        )
        x0          = pdc.x0
        y           = pdc.y
        R           = pdc.R
        R_inv_var   = pdc.R_inv_var
        ic_cov_sqrt = pdc.ic_cov_sqrt
        lorenz_cfg  = pdc.lorenz_config_settings
        obs_cfg     = pdc.observation_config
        save_preliminaries(pdc, prelim_file)
        @info "Saved preliminaries to $prelim_file"
    end

    return (; x0, y, R, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx, ny,
              nu, prior_mean, prior_cov, phi, phi_structure, sample_range, case)
end

########################################################################
###############  Forward-map closure for each forcing type  ###########
########################################################################

# Returns a closure G_func(θ) → ℝ^ny that is ForwardDiff-compatible.
# x0p is a pre-fixed (non-dual) perturbed initial condition.
function make_G_func(prob, x0p)
    (; lorenz_cfg, obs_cfg, phi, phi_structure, sample_range, case) = prob

    if case == "const-force"
        # Optimise log(φ): θ ∈ ℝ¹, φ = exp(θ[1])
        return log_θ -> lorenz_forward(
            build_forcing(phi, exp(log_θ[1]), nothing, nothing),
            x0p, lorenz_cfg, obs_cfg)

    elseif case == "vec-force"
        # Optimise φ directly: θ ∈ ℝ^nx
        return θ -> lorenz_forward(
            build_forcing(phi, θ, nothing, nothing),
            x0p, lorenz_cfg, obs_cfg)

    elseif case == "flux-force"
        # Optimise NN weights: θ ∈ ℝ^nu
        # Note: ForwardDiff traces through Flux.Chain via dual-number weights.
        return θ -> lorenz_forward(
            build_forcing(phi, θ, phi_structure, sample_range),
            x0p, lorenz_cfg, obs_cfg)
    end
end

########################################################################
###############  Run one (N_ens, rng_idx) cell  #######################
########################################################################

# Single LM trajectory; N_ens IC perturbations are averaged each step to
# reduce gradient noise.  N_ens=1 is pure LM with a single IC sample.
# Cost: outer_iter × N_ens × (nu + 1) forward-model evaluations.
function run_one(cfg, N_ens, rng_idx, prob)
    (; x0, y, R, R_inv_var, ic_cov_sqrt, nx, ny, nu, prior_mean, prior_cov) = prob
    rng = MersenneTwister(rng_idx)

    prior_dist = MvNormal(prior_mean, Matrix(Symmetric(prior_cov)))

    θ      = rand(rng, prior_dist)   # single trajectory starting point
    θ_init = copy(θ)
    λ = 1.0                          # LM damping

    conv_score   = NaN
    final_params = fill(NaN, nu)
    final_output = fill(NaN, ny)

    for outer_iter in 1:cfg.N_iter
        x0p_all = x0 .+ ic_cov_sqrt * randn(rng, nx, N_ens)

        # Accumulate Jacobian and residual over IC samples
        J_sum = zeros(ny, nu)
        r_sum = zeros(ny)
        for k in 1:N_ens
            x0p_k  = x0p_all[:, k]
            G_func = make_G_func(prob, x0p_k)
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
            r_trial_sum += y - make_G_func(prob, x0p_all[:, k])(θ_trial)
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
    experiment = l96_experiment()
    cfg        = experiment_config(experiment)
    tasks      = flat_tasks(cfg)
    tidx       = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    case = cfg.force_case
    prob = build_l96_problem(case, output_dir)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        N_ens, rng_idx = tasks[t]
        @info "Task $t: case=$case  N_ens=$N_ens  rng_idx=$rng_idx"
        result = run_one(cfg, N_ens, rng_idx, prob)
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
