#!/bin/bash
# Array job: one task per (N_ens, rng_idx) pair.
# Each task loads the stored ensemble history for one calibrate cell, pushes
# the raw ensemble at every stored GNKI iteration through the Lorenz forward
# map, and saves output samples back into the same results JLD2 file.
# Downstream consumers (exp_to_leaderboard) load these stored arrays instead
# of rerunning the expensive forward map.
#
# L63 (180 tasks for default config):
#   sbatch --dependency=afterok:<calib_jobid> --export=ALL,CALIBRATE_DATE=<yyyy-mm-dd> \
#          pushforward_from_posterior.sbatch
#
# L96 (180 tasks per case):
#   sbatch --dependency=afterok:<calib_jobid> \
#          --export=ALL,EXPERIMENT=l96_flux,CALIBRATE_DATE=<yyyy-mm-dd> \
#          pushforward_from_posterior.sbatch
#
# N_TASKS = length(N_ens_sizes) * n_repeats = 9 * 20 = 180 (all four cases).
# If either value changes in experiment_config.jl, update the --array upper bound.

#SBATCH --job-name=pushfwd
#SBATCH --output=../output/slurm/pushfwd_%A_%a.out
#SBATCH --error=../output/slurm/pushfwd_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=00:30:00
#SBATCH --mem=16G
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

if [[ -z "${EXPERIMENT:-}" || "${EXPERIMENT}" == "l63" ]]; then
    julia --project=.. ../pushforward_from_posterior_l63.jl "${SLURM_ARRAY_TASK_ID}"
else
    julia --project=.. ../pushforward_from_posterior_l96.jl "${SLURM_ARRAY_TASK_ID}"
fi
