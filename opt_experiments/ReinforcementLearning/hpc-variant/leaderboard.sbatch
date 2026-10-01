#!/bin/bash
# Single SLURM job: converts per-cell JLD2 results to a leaderboard NetCDF for ONE
# (ALGORITHM, UPDATE_FREQUENCY) arm of one case.
# Depends on run_array of the same arm (afterany, so partial results are still processed).
#
# Manual use:
#   L63: sbatch --dependency=afterany:<RUN_JID> --export=ALL,EXPERIMENT=l63,ALGORITHM=ddpg,UPDATE_FREQUENCY=short leaderboard.sbatch
#   L96: sbatch --dependency=afterany:<RUN_JID> --export=ALL,EXPERIMENT=l96_const,ALGORITHM=td3,UPDATE_FREQUENCY=long leaderboard.sbatch

#SBATCH --job-name=leaderboard_rl
#SBATCH --output=../output/slurm/leaderboard_%j.out
#SBATCH --error=../output/slurm/leaderboard_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. ../run_to_leaderboard.jl
