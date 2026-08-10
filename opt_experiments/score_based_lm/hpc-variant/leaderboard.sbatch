#!/bin/bash
# Single SLURM job: converts per-cell JLD2 results to a leaderboard NetCDF.
# Runs serially over all (N_ens, rng_idx) cells in one job (no array needed).
# Depends on run_array completing successfully.
#
# Manual use:
#   L63: sbatch --dependency=afterok:<RUN_JID> --export=ALL,EXPERIMENT=l63 leaderboard.sbatch
#   L96: sbatch --dependency=afterok:<RUN_JID> --export=ALL,EXPERIMENT=l96_const leaderboard.sbatch

#SBATCH --job-name=leaderboard_lm
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
