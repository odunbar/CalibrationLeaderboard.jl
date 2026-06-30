# lm — L96 preliminaries
# Computes and saves the shared L96 problem quantities (truth trajectory, obs
# covariance, IC spread) for one forcing case.  Run once per case before
# submitting the corresponding run_array job.
#
# Local:  EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
#         EXPERIMENT=l96_vec   julia --project=. l96_preliminaries.jl
#         EXPERIMENT=l96_flux  julia --project=. l96_preliminaries.jl
# HPC:    sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_const preliminaries.sbatch
#         sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec   preliminaries.sbatch
#         sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_flux  preliminaries.sbatch

using BSON
using Distributions
using Flux
using JLD2
using LinearAlgebra
using Random

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

function main()
    experiment = l96_experiment()
    cfg        = experiment_config(experiment)
    case       = cfg.force_case

    rng_i = MersenneTwister(11)
    t     = 0.01

    if case == "const-force"
        nx = 40; T = 14.0; T_start = 4.0; inff = 2.0
        phi = ConstantEMC(8.0)

    elseif case == "vec-force"
        nx = 40; T = 54.0; T_start = 4.0; inff = 2.0
        sinusoid = 8 .+ 6 * sin.((4 * π * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)

    elseif case == "flux-force"
        nx = 100; T = 54.0; T_start = 4.0; inff = 2.5
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * π * x) / 10)
        x_train  = collect(-5.0:0.01:5.0)
        y_train  = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, _ = train_network(phi_structure, x_train, y_train)
        sample_range  = Float32.(collect(-5.0:0.1:4.9))
        phi           = FluxEMC(true_model, sample_range)
    else
        throw(ArgumentError("Unknown L96 case: $case"))
    end

    T_long     = 1000.0
    lorenz_cfg = LorenzConfig(t, T)
    obs_cfg    = ObservationConfig(T_start, T)

    output_dir  = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prelim_file = joinpath(output_dir, "l96_computed_preliminaries_$(case).jld2")

    x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
    pdc = compute_perfect_data(
        phi, nx,
        LorenzConfig(t, T_long), x_initial,
        lorenz_cfg, obs_cfg;
        R_inflation = inff,
    )
    save_preliminaries(pdc, prelim_file)
    @info "Saved L96 ($case) preliminaries to $prelim_file"
end

main()
