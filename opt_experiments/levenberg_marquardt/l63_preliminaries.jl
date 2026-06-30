# lm — L63 preliminaries
# Computes and saves the shared L63 problem quantities (truth trajectory, obs
# covariance, IC spread) that all array tasks read.  Run this once before
# submitting the run_array job.
#
# Local:  julia --project=. l63_preliminaries.jl
# HPC:    sbatch --export=ALL,SCRIPT=l63_preliminaries.jl,EXPERIMENT=l63 preliminaries.sbatch

using Distributions
using JLD2
using LinearAlgebra
using Random

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))

function main()
    rng_i     = MersenneTwister(11)
    t = 0.01; T = 40.0; nx = 3; ny = 9
    u_truth   = EnsembleMemberConfig([28.0, 8.0 / 3.0])
    x_initial = rand(rng_i, Normal(0.0, 1.0), nx)

    output_dir  = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")

    pdc = compute_perfect_data(
        u_truth, nx, ny,
        LorenzConfig(t, 1000.0), x_initial,
        LorenzConfig(t, T), ObservationConfig(30.0, T);
        R_n_samples = 36,
    )
    save_preliminaries(pdc, prelim_file)
    @info "Saved L63 preliminaries to $prelim_file"
end

main()
