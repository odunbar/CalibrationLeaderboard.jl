# score_based_lm — shared L96 problem construction.
#
# Extracted so that run_l96_sblm.jl and validate_jacobian_l96.jl cannot drift:
# both the case setup and the forcing constructor live here exactly once.
# Include AFTER common/forward_maps/Lorenz96.jl.

function build_l96_problem(case::String, output_dir::String)
    t = 0.01

    if case == "const-force"
        nx = 40; T = 14.0; T_start = 4.0
        phi = ConstantEMC(8.0)
        phi_structure  = nothing
        sample_range   = nothing
        # Prior: φ ~ LogNormal(log(10), 4/10) ≈ N(10, 4²) on [0,∞)
        # Optimise in log-space: θ = [log(φ)], G_func uses exp(θ[1])
        nu = 1
        prior_mean = [log(10.0)]
        prior_cov  = diagm([0.4^2])

    elseif case == "vec-force"
        nx = 40; T = 54.0; T_start = 4.0
        pl = 2.0; psig = 3.0
        sinusoid = 8 .+ 6 * sin.((4 * π * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        phi_structure  = nothing
        sample_range   = nothing
        nu = nx
        prior_mean = 8.0 * ones(nx)
        prior_cov  = [psig^2 * exp(-abs(i - j) / pl) for i in 1:nx, j in 1:nx]

    elseif case == "flux-force"
        nx = 100; T = 54.0; T_start = 4.0
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

    prelim_file = joinpath(output_dir, "l96_computed_preliminaries_$(case).jld2")
    isfile(prelim_file) || error("Prelim file not found: $prelim_file\nRun l96_preliminaries.jl first.")
    ld = load_preliminaries(prelim_file)
    @info "Loaded L96 ($case) preliminaries from $prelim_file"

    return (; x0 = ld.x0, y = ld.y, R = ld.R, R_inv_var = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config,
              nx, ny, nu, prior_mean, prior_cov, phi, phi_structure, sample_range, case)
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


# Forcing object for a parameter vector, per case (mirrors make_G_func in
# levenberg_marquardt/run_l96_lm.jl, but returns the EMC rather than a closure so
# we can call lorenz_forward_with_states and keep the trajectory).
function make_emc(prob, theta)
    (; phi, phi_structure, sample_range, case) = prob
    if case == "const-force"
        return build_forcing(phi, exp(theta[1]), nothing, nothing)
    elseif case == "vec-force"
        return build_forcing(phi, theta, nothing, nothing)
    elseif case == "flux-force"
        return build_forcing(phi, theta, phi_structure, sample_range)
    else
        throw(ArgumentError("Unknown L96 case: $(case)"))
    end
end

