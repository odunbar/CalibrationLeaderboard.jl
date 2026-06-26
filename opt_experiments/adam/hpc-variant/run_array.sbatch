#!/bin/bash
# SLURM array job for Adam runs (L63 or L96).
# One task per (N_ens, rng_idx) cell.
#
# N_TASKS = length(rmse_targets) * n_repeats = 3 * 100 = 300
# If rmse_targets or n_repeats change in experiment_config.jl, update --array upper bound.
#
# L63:
#   sbatch --export=ALL,SCRIPT=run_l63_adam.jl,EXPERIMENT=l63 run_array.sbatch
# L96:
#   sbatch --export=ALL,SCRIPT=run_l96_adam.jl,EXPERIMENT=l96_const run_array.sbatch
#   sbatch --export=ALL,SCRIPT=run_l96_adam.jl,EXPERIMENT=l96_vec   run_array.sbatch
#   sbatch --export=ALL,SCRIPT=run_l96_adam.jl,EXPERIMENT=l96_flux  run_array.sbatch
#
# Smoke test (verify paths and output writing before full run):
#   sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_adam.jl,EXPERIMENT=l63 run_array.sbatch

#SBATCH --job-name=run_adam
#SBATCH --output=../output/slurm/run_%A_%a.out
#SBATCH --error=../output/slurm/run_%A_%a.err
#SBATCH --array=1-300%100
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with precompile.sbatch

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-run_l63_adam.jl}

# Stay in hpc-variant/ so include("experiment_config.jl") picks up the
# HPC-pinned version here; --project=.. uses adam/Project.toml;
# ../${SCRIPT} runs from adam/ so @__DIR__ resolves common/ paths correctly.
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
