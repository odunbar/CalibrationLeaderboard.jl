#!/bin/bash
# SLURM array job for ReinforcementLearning (DDPG / TD3 actor-critic) runs (L63 or L96).
# One task per (rmse_target, rng_idx) cell.
#
# N_TASKS (all cases):  length(rmse_targets) * n_repeats = 3 * 30 = 90
# (same for every case, algorithm and update frequency)
# If rmse_targets or n_repeats change in experiment_config.jl, update --array
# in the submit scripts (which override this default with --array=1-N).
#
# The arm is selected by the ALGORITHM (ddpg|td3) and UPDATE_FREQUENCY (short|long) env vars,
# read by experiment_config.jl:
#
# L63:
#   sbatch --array=1-90 --export=ALL,SCRIPT=run_l63_ddpg.jl,EXPERIMENT=l63,ALGORITHM=ddpg,UPDATE_FREQUENCY=short run_array.sbatch
# L96:
#   sbatch --array=1-90 --export=ALL,SCRIPT=run_l96_ddpg.jl,EXPERIMENT=l96_const,ALGORITHM=td3,UPDATE_FREQUENCY=long run_array.sbatch
#   sbatch --array=1-90 --export=ALL,SCRIPT=run_l96_ddpg.jl,EXPERIMENT=l96_vec,ALGORITHM=ddpg,UPDATE_FREQUENCY=short run_array.sbatch
#   sbatch --array=1-90 --export=ALL,SCRIPT=run_l96_ddpg.jl,EXPERIMENT=l96_flux,ALGORITHM=td3,UPDATE_FREQUENCY=short run_array.sbatch
#
# Smoke test (verify paths and output writing before full run):
#   sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_ddpg.jl,EXPERIMENT=l63,ALGORITHM=ddpg,UPDATE_FREQUENCY=long run_array.sbatch
#
# Wall time: a cell runs until RMSE < target or the max_evals budget is spent. The short arm
# makes ~10x more RL updates per unit of cost than the long arm, so non-converging cells
# (e.g. L96 vec / flux) are the slow ones. Override per submission with --time=... if needed.

#SBATCH --job-name=run_rl
#SBATCH --output=../output/slurm/run_%A_%a.out
#SBATCH --error=../output/slurm/run_%A_%a.err
#SBATCH --array=1-90
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with precompile.sbatch

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-run_l63_ddpg.jl}

# hpc-variant/ holds ONLY sbatch + submit scripts (no .jl, no experiment_config.jl):
# --project=.. uses ReinforcementLearning/Project.toml, and ../${SCRIPT} runs from
# ReinforcementLearning/ so @__DIR__ resolves both common/ and experiment_config.jl.
# The run date is pinned via RUN_DATE, the arm via ALGORITHM / UPDATE_FREQUENCY.
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
