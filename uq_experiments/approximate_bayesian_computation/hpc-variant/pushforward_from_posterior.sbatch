#!/bin/bash
# Array job: one task per (N_ens, rng_idx) pair (indexed via flat_tasks(cfg),
# unlike calibrate_array.sbatch which indexes by rng_idx alone).
# Each task loads the stored accepted-sample-pool history for one calibrate
# cell, fits a Gaussian to the pool at every stored round k, pushes
# n_pushforward_samples=1000 resampled points through the Lorenz forward map,
# and saves output samples back into the same results JLD2 file.
#
# L63 / L96 const (180 tasks = 9 N_ens x 20 repeats):
#   sbatch --dependency=afterok:<calib_jobid> --export=ALL,CALIBRATE_DATE=<yyyy-mm-dd> \
#          pushforward_from_posterior.sbatch
#
# L96 vec/flux (also 180 tasks):
#   sbatch --dependency=afterok:<calib_jobid> \
#          --export=ALL,EXPERIMENT=l96_flux,CALIBRATE_DATE=<yyyy-mm-dd> \
#          pushforward_from_posterior.sbatch
#
# N_TASKS = length(N_ens_sizes) * n_repeats = 9 * 20 = 180 (all four cases).
# If either value changes in experiment_config.jl, update the --array upper bound.
#
# Per-cell cost now scales with the number of stored rounds (up to N_iter =
# 500 for l63/l96_const, 111 for l96_vec/l96_flux), each round pushing forward
# 1000 samples — far more than GNKI's ~10-15 rounds. Run the smoke test below
# and check actual wall-clock before trusting the --time below for the full array.
#
# The per-round sample loop is Threads.@threads-parallel (embarrassingly
# parallel, independent per-sample RNG) — set --cpus-per-task to use more
# cores. OPENBLAS_NUM_THREADS=1 avoids oversubscription now that Julia
# threads do the parallel work.

#SBATCH --job-name=pushfwd
#SBATCH --output=../output/slurm/pushfwd_%A_%a.out
#SBATCH --error=../output/slurm/pushfwd_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

# Only sbatch/submit scripts live in hpc-variant/ — the .jl scripts and
# Project.toml stay in the parent method directory (one copy, no duplication).
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

if [[ -z "${EXPERIMENT:-}" || "${EXPERIMENT}" == "l63" ]]; then
    julia --project=.. ../pushforward_from_posterior_l63.jl "${SLURM_ARRAY_TASK_ID}"
else
    julia --project=.. ../pushforward_from_posterior_l96.jl "${SLURM_ARRAY_TASK_ID}"
fi
