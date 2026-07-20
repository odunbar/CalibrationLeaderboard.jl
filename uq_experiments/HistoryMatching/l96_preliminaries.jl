# HistoryMatching — L96 preliminaries (truth data + observations)
#
# Computes the perfect-model truth trajectory and observations once, serially,
# for one force_case (const / vec / flux), and saves them to
# output/l96_computed_preliminaries_<force_case>.jld2. Run this before the
# calibrate stage — extracting it out of calibrate_l96.jl avoids every SLURM
# array task racing to compute and write the same file. Identical in spirit
# to uq_experiments/calibrate_emulate_sample/l96_preliminaries.jl (truth
# generation is method-agnostic).
#
# Local: EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
# SLURM: invoked via preliminaries.sbatch with SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_const

using Distributions
using LinearAlgebra
using Random
using JLD2
using Flux

include(joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz96.jl"))
include("experiment_config.jl")

# Mirrors calibrate_l96.jl's force-case forcing construction — only nx/phi/T/inff
# are needed here to build the truth trajectory; prior/phi_structure/sample_range
# are per-task setup and stay in calibrate_l96.jl.
function force_case_setup(force_case::AbstractString)
    if force_case == "const-force"
        nx = 40
        phi = ConstantEMC(8.0)
        T = 14.0
        inff = 2
    elseif force_case == "vec-force"
        nx = 40
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
        phi = VectorEMC(sinusoid)
        T = 54.0
        inff = 2
    elseif force_case == "flux-force"
        nx = 100
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * pi * x) / 10)
        x_train = collect(-5.0:0.01:5.0)
        Random.seed!(20260529)
        y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, _ = train_network(deepcopy(phi_structure), x_train, y_train)
        sample_range = Float32.(collect(-5.0:0.1:4.9))
        phi = FluxEMC(true_model, sample_range)
        T = 54.0
        inff = 2.5
    else
        throw(ArgumentError("Unknown force_case: $force_case"))
    end
    return (nx = nx, phi = phi, T = T, inff = inff)
end

function main()
    experiment = l96_experiment()
    @assert experiment in (:l96_const, :l96_vec, :l96_flux) "l96_preliminaries.jl requires EXPERIMENT to be :l96_const, :l96_vec, or :l96_flux (got $experiment)"
    cfg = experiment_config(experiment)
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prelim_file = joinpath(output_dir, prelim_filename(cfg))

    setup = force_case_setup(cfg.force_case)
    nx, phi, T, inff = setup.nx, setup.phi, setup.T, setup.inff
    t = 0.01

    rng_i = MersenneTwister(11)
    pdc = compute_perfect_data(
        phi, nx,
        LorenzConfig(t, 1000.0), rand(rng_i, Normal(0.0, 1.0), nx),
        LorenzConfig(t, T), ObservationConfig(4.0, T);
        R_inflation = inff,
    )
    save_preliminaries(pdc, prelim_file)
    @info "Saved preliminaries to $(prelim_file)"
end

main()
