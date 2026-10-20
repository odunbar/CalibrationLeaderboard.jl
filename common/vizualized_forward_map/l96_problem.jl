# Lorenz 96 problem definitions (const / vec / flux forcing) for the forward-map visualisations.
# Mirrors uq_experiments/*/calibrate_l96.jl `force_case_setup` + l96_preliminaries.jl
# (truth, prior, window). That setup is method-specific by design, so it is duplicated
# here rather than put in common/forward_maps.

get!(ENV, "GKSwstype", "100")
using Distributions
using LinearAlgebra
using Random
using JLD2
using Flux
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

include(joinpath(@__DIR__, "..", "forward_maps", "Lorenz96.jl"))
include(joinpath(@__DIR__, "viz_common.jl"))

const L96_CASES = ["const-force", "vec-force", "flux-force"]

function l96_force_case_setup(force_case::AbstractString)
    if force_case == "const-force"
        nx = 40
        phi = ConstantEMC(8.0)
        truth_θ = [8.0]
        phi_structure = nothing
        sample_range = nothing
        prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
        T = 14.0
        inff = 2.0
    elseif force_case == "vec-force"
        nx = 40
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        truth_θ = collect(Float64, sinusoid)
        phi_structure = nothing
        sample_range = nothing
        pl, psig = 2.0, 3.0
        prior_cov = [psig^2 * exp(-abs(ii - jj) / pl) for ii in 1:nx, jj in 1:nx]
        prior = ParameterDistribution(
            Parameterized(MvNormal(8.0 * ones(nx), prior_cov)),
            repeat([no_constraint()], nx),
            "l96_vec_prior",
        )
        T = 54.0
        inff = 2.0
    elseif force_case == "flux-force"
        nx = 100
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * pi * x) / 10)
        x_train = collect(-5.0:0.01:5.0)
        Random.seed!(20260529)
        y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, truth_params = train_network(deepcopy(phi_structure), x_train, y_train)
        sample_range = Float32.(collect(-5.0:0.1:4.9))
        phi = FluxEMC(true_model, sample_range)
        truth_θ = collect(Float64, truth_params)

        prior_sinusoid(x) = 8.02 .+ 6.5 * sin.(1.02 * (4 * pi * x) / 10 + 0.2)
        prior_train = prior_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        _, prior_mean = train_network(deepcopy(phi_structure), x_train, prior_train)

        prior_cov = (0.1^2) * Diagonal(prior_mean .^ 2)
        prior = ParameterDistribution(
            Parameterized(MvNormal(prior_mean, prior_cov)),
            repeat([no_constraint()], length(prior_mean)),
            "l96_nn_prior",
        )
        T = 54.0
        inff = 2.5
    else
        throw(ArgumentError("Unknown force_case: $force_case (expected one of $(L96_CASES))"))
    end
    return (; nx, phi, truth_θ, phi_structure, sample_range, prior, T, inff)
end

function l96_setup(force_case::AbstractString)
    s = l96_force_case_setup(force_case)
    nx, ny = s.nx, 2 * s.nx
    dt = 0.01
    config = LorenzConfig(dt, s.T)
    obs = ObservationConfig(4.0, s.T)  # statistics window [4, T]

    # Spun-up x0, observation noise covariance R and IC covariance. Computed once, then cached.
    mkpath(OUTPUT_DIR)
    prelim_file = joinpath(OUTPUT_DIR, "l96_computed_preliminaries_$(force_case).jld2")
    if !isfile(prelim_file)
        rng_i = MersenneTwister(11)
        pdc = compute_perfect_data(
            s.phi, nx,
            LorenzConfig(dt, 1000.0), rand(rng_i, Normal(0.0, 1.0), nx),
            config, obs;
            R_inflation = s.inff,
        )
        save_preliminaries(pdc, prelim_file)
        @info "Saved preliminaries to $(prelim_file)"
    end
    prelim = load_preliminaries(prelim_file)

    return (
        name = "l96_$(force_case)",
        force_case = force_case,
        nx = nx,
        ny = ny,
        dt = dt,
        T = s.T,
        config = config,
        obs = obs,
        phi_truth = s.phi,
        truth_θ = s.truth_θ,
        prior = s.prior,
        build = θ -> build_forcing(s.phi, θ, s.phi_structure, s.sample_range),
        x0 = prelim.x0,
        y = prelim.y,
        R = prelim.R,
        ic_cov_sqrt = prelim.ic_cov_sqrt,
        param_names = ["φ"],
        output_names = vcat(["mean x$(i)" for i in 1:nx], ["std x$(i)" for i in 1:nx]),
    )
end

# Forcing as a spatial profile F(i), i = 1..nx, for any of the three forcing types.
forcing_profile(phi, x0) = Float64.(forcing(phi, x0))
