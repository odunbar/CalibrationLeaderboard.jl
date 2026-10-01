# ReinforcementLearning — L96 opt experiment: DDPG actor-critic (see README.md).
# Three forcing cases: const-force (nu=1), vec-force (nu=40), flux-force (nu=61).
#
# Case selected by EXPERIMENT env var / ARGS[2]; update_frequency (:short / :long) and algorithm (:ddpg / :td3) by the
# UPDATE_FREQUENCY / ALGORITHM toggles in experiment_config.jl, overridden by the env vars of the same name
# (or ARGS[3] / ARGS[4]):
#   ALGORITHM=ddpg UPDATE_FREQUENCY=short EXPERIMENT=l96_const julia --project=. run_l96_ddpg.jl [task_index]
#   ALGORITHM=td3  UPDATE_FREQUENCY=long  EXPERIMENT=l96_vec   julia --project=. run_l96_ddpg.jl [task_index]

using BSON
using Distributions
using Flux
using JLD2
using LinearAlgebra
using Optimisers
using Random
using Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")
include("ddpg_core.jl")
include("rl_env.jl")
include("rl_run.jl")
include("l96_problem.jl")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Method core lives in ddpg_core.jl (critic regression + actor ascent),   ║
# ║  rl_env.jl (long = full-window / short = fractional-window environment) and rl_run.jl (training loop).     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

rl_main(l96_experiment(), (cfg, output_dir) -> build_l96_problem(cfg.force_case, output_dir))
