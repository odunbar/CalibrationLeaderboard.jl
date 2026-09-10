#!/bin/bash
# SLURM array job for score-based Adam (GFDT) runs (L63 or L96).
# One task per (N_ens, rng_idx) cell.
#
# N_TASKS (all cases):   length(N_ens_sizes) * length(rmse_targets) * n_repeats = 10 * 3 * 30 = 900
# If N_ens_sizes, rmse_targets or n_repeats change in experiment_config.jl, update --array
# in the submit scripts (which override this default with --array=1-N).
#
# L63:
#   sbatch --array=1-900 --export=ALL,SCRIPT=run_l63_sbadam.jl,EXPERIMENT=l63 run_array.sbatch
# L96:
#   sbatch --array=1-900 --export=ALL,SCRIPT=run_l96_sbadam.jl,EXPERIMENT=l96_const run_array.sbatch
#   sbatch --array=1-900 --export=ALL,SCRIPT=run_l96_sbadam.jl,EXPERIMENT=l96_vec   run_array.sbatch
#   sbatch --array=1-900 --export=ALL,SCRIPT=run_l96_sbadam.jl,EXPERIMENT=l96_flux  run_array.sbatch
#
# Smoke test (verify paths and output writing before full run):
#   sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_sbadam.jl,EXPERIMENT=l63 run_array.sbatch

#SBATCH --job-name=run_sbadam
#SBATCH --output=../output/slurm/run_%A_%a.out
#SBATCH --error=../output/slurm/run_%A_%a.err
#SBATCH --array=1-900
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with precompile.sbatch

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-run_l63_sbadam.jl}

# hpc-variant/ holds ONLY sbatch + submit scripts (no .jl, no experiment_config.jl):
# --project=.. uses score_based_adam/Project.toml, and ../${SCRIPT} runs from
# score_based_adam/ so @__DIR__ resolves both common/ and experiment_config.jl.
# The run date is pinned via RUN_DATE, not via a second copy of the config.
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
