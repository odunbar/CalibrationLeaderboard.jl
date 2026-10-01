# ReinforcementLearning — L96 problem setup (forcing parameterisations + prior).
# Mirrors opt_experiments/adam/run_l96_adam.jl::build_l96_problem, minus ForwardDiff, and
# returns `make_forcing(F)` (F::Vector -> forcing object) for rl_env.jl instead of a
# ForwardDiff closure.  Requires Lorenz96.jl to be included first.

function build_l96_problem(case::String, output_dir::String)
    if case == "const-force"
        nx = 40
        phi = ConstantEMC(8.0)
        phi_structure = nothing
        sample_range  = nothing
        # Prior: φ ~ LogNormal(log(10), 4/10); optimise in log-space θ = [log φ]
        nu = 1
        prior_mean = [log(10.0)]
        prior_cov  = diagm([0.4^2])
        make_forcing = F -> build_forcing(phi, exp(F[1]), nothing, nothing)

    elseif case == "vec-force"
        nx = 40
        pl = 2.0; psig = 3.0
        sinusoid = 8 .+ 6 * sin.((4 * π * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        phi_structure = nothing
        sample_range  = nothing
        nu = nx
        prior_mean = 8.0 * ones(nx)
        prior_cov  = [psig^2 * exp(-abs(i - j) / pl) for i in 1:nx, j in 1:nx]
        make_forcing = F -> build_forcing(phi, F, nothing, nothing)

    elseif case == "flux-force"
        nx = 100
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
        prior_cov     = Matrix((0.1^2) * I(length(prior_mean)))
        nu            = length(prior_mean)
        make_forcing  = F -> build_forcing(phi, F, phi_structure, sample_range)

    else
        throw(ArgumentError("Unknown L96 case: $case"))
    end

    prelim_file = joinpath(output_dir, "l96_computed_preliminaries_$(case).jld2")
    isfile(prelim_file) || error("Prelim file not found: $prelim_file\nRun l96_preliminaries.jl first.")
    ld = load_preliminaries(prelim_file)
    @info "Loaded L96 ($case) preliminaries from $prelim_file"

    return (; x0 = ld.x0, y = ld.y, R = ld.R, R_inv_var = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config,
              nx, ny = 2 * nx, nu, prior_mean, prior_cov, make_forcing, case)
end
