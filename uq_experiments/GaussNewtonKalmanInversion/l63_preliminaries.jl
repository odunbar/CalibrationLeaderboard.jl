# GaussNewtonKalmanInversion — L63 preliminaries (truth data + observations)
#
# Computes the perfect-model truth trajectory and observations once, serially,
# and saves them to output/l63_computed_preliminaries.jld2. Run this before the
# calibrate stage — extracting it out of calibrate_l63.jl avoids every SLURM
# array task racing to compute and write the same file.
#
# Local: julia --project=. l63_preliminaries.jl
# SLURM: invoked via preliminaries.sbatch with SCRIPT=l63_preliminaries.jl

using Distributions
using Random
using JLD2

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

function main()
    cfg = experiment_config(:l63)
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prelim_file = joinpath(output_dir, prelim_filename(cfg))

    nx = 3  # state dimension
    ny = 9  # observation dimension
    truth_params = EnsembleMemberConfig([28.0, 8.0 / 3.0])
    t = 0.01
    T = 40.0

    rng_i = MersenneTwister(11)
    pdc = compute_perfect_data(
        truth_params, nx, ny,
        LorenzConfig(t, 1000.0), rand(rng_i, Normal(0.0, 1.0), nx),
        LorenzConfig(t, T), ObservationConfig(30.0, T),
    )
    save_preliminaries(pdc, prelim_file)
    @info "Saved preliminaries to $(prelim_file)"
end

main()
