# ReinforcementLearning — L63 opt experiment: DDPG actor-critic (see README.md).
# The forcing parameters θ = (log ρ, log β) are the (state-independent) policy; a learned
# critic Q_ϑ(x, F) supplies ∇_F Q, which replaces the Jacobian used by adam / LM.
# No ForwardDiff, no GFDT: 1 RL update per simulated window (:long) or many fractional-window updates (:short).
#
# Algorithm (:ddpg / :td3) and update frequency (:short / :long) are set by the ALGORITHM / UPDATE_FREQUENCY
# toggles in experiment_config.jl, or overridden by the env vars of the same name (or ARGS[4] / ARGS[3]):
#   UPDATE_FREQUENCY=short ALGORITHM=ddpg julia --project=. run_l63_ddpg.jl [task_index]
#   UPDATE_FREQUENCY=long  ALGORITHM=td3  julia --project=. run_l63_ddpg.jl [task_index]

using Distributions
using Flux
using JLD2
using LinearAlgebra
using Optimisers
using Random
using Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")
include("ddpg_core.jl")
include("rl_env.jl")
include("rl_run.jl")

########################################################################
###############  Problem setup  #######################################
########################################################################

function build_l63_problem(cfg, output_dir)
    prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
    isfile(prelim_file) || error("Prelim file not found: $prelim_file\nRun l63_preliminaries.jl first.")
    ld = load_preliminaries(prelim_file)
    @info "Loaded L63 preliminaries from $prelim_file"
    # L63: parameters are (log ρ, log β), matching the EKI convention exp.(θ) = [ρ, β]
    return (; x0 = ld.x0, y = ld.y, R = ld.R, R_inv_var = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config,
              nx = 3, ny = length(ld.y), nu = 2,
              prior_mean = [3.3, 1.2], prior_cov = diagm([0.15^2, 0.5^2]),
              make_forcing = F -> EnsembleMemberConfig(exp.(F)))
end

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Method core lives in ddpg_core.jl (critic regression + actor ascent),   ║
# ║  rl_env.jl (long = full-window / short = fractional-window environment) and rl_run.jl (training loop).     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

rl_main(:l63, build_l63_problem)
