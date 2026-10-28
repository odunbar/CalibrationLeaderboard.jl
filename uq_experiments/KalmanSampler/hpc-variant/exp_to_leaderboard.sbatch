#!/bin/bash
# Single job: converts stored calibrate + pushforward results to a leaderboard
# NetCDF. Runs all (N_ens, rng_idx) cells serially in one job (no array needed).
#
# Invoked automatically by submit_l*.sh after pushforward_from_posterior completes.
# Manual use:
#   L63: sbatch --dependency=afterany:<pushfwd_jobid> --export=ALL,CALIBRATE_DATE=<yyyy-mm-dd> \
#               exp_to_leaderboard.sbatch
#   L96: sbatch --dependency=afterany:<pushfwd_jobid> \
#               --export=ALL,EXPERIMENT=l96_flux,CALIBRATE_DATE=<yyyy-mm-dd> \
#               exp_to_leaderboard.sbatch

#SBATCH --job-name=leaderboard
#SBATCH --output=../output/slurm/leaderboard_%j.out
#SBATCH --error=../output/slurm/leaderboard_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Only sbatch/submit scripts live in hpc-variant/ — the .jl scripts and
# Project.toml stay in the parent method directory (one copy, no duplication).
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. ../exp_to_leaderboard.jl
