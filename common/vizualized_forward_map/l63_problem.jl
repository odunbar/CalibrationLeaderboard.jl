# Lorenz 63 problem definition for the forward-map visualisations.
# Mirrors uq_experiments/*/calibrate_l63.jl + l63_preliminaries.jl (truth, prior, window).

get!(ENV, "GKSwstype", "100")
using Distributions
using LinearAlgebra
using Random
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

include(joinpath(@__DIR__, "..", "forward_maps", "Lorenz63.jl"))
include(joinpath(@__DIR__, "viz_common.jl"))

function l63_setup()
    nx, ny = 3, 9
    truth_θ = [28.0, 8.0 / 3.0]
    phi_truth = EnsembleMemberConfig(truth_θ)
    dt, T = 0.01, 40.0
    config = LorenzConfig(dt, T)
    obs = ObservationConfig(30.0, T)  # statistics window [30, 40]

    prior = combine_distributions([
        constrained_gaussian("rho", exp(3.3), 4.153, 0, Inf),
        constrained_gaussian("beta", exp(1.2), 2.016, 0, Inf),
    ])

    # Spun-up x0, observation noise covariance R and IC covariance. Computed once, then cached.
    mkpath(OUTPUT_DIR)
    prelim_file = joinpath(OUTPUT_DIR, "l63_computed_preliminaries.jld2")
    if !isfile(prelim_file)
        rng_i = MersenneTwister(11)
        pdc = compute_perfect_data(
            phi_truth, nx, ny,
            LorenzConfig(dt, 1000.0), rand(rng_i, Normal(0.0, 1.0), nx),
            config, obs,
        )
        save_preliminaries(pdc, prelim_file)
        @info "Saved preliminaries to $(prelim_file)"
    end
    prelim = load_preliminaries(prelim_file)

    return (
        name = "l63",
        nx = nx,
        ny = ny,
        dt = dt,
        T = T,
        config = config,
        obs = obs,
        phi_truth = phi_truth,
        truth_θ = truth_θ,
        prior = prior,
        build = θ -> EnsembleMemberConfig(θ),
        x0 = prelim.x0,
        y = prelim.y,
        R = prelim.R,
        ic_cov_sqrt = prelim.ic_cov_sqrt,
        param_names = ["ρ", "β"],
        output_names = ["mean x", "mean y", "mean z", "var x", "var y", "var z", "cov xy", "cov xz", "cov yz"],
    )
end
