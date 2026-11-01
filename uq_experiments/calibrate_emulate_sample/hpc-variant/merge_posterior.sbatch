#!/bin/bash
# Array job: one task per (N_ens, rng_idx) cell. Assembles the per-k emulate_sample files
# (output/<calib_dir>/emu_k/) into the combined posterior file that pushforward_from_posterior,
# the diagnostic plots and exp_to_leaderboard read. Cells with missing k are skipped with a
# warning (see merge_posterior_k.jl); set KEEP_K_FILES=1 to keep the per-k files.
#
# L63 (180 tasks for default config):
#   sbatch --dependency=afterany:<emu_jobids> --export=ALL,EXPERIMENT=l63 merge_posterior.sbatch
#
# L96 (180 tasks per case):
#   sbatch --dependency=afterany:<emu_jobids> --export=ALL,EXPERIMENT=l96_flux merge_posterior.sbatch
#
# EXPERIMENT must be set explicitly (including l63).

#SBATCH --job-name=mergepost
#SBATCH --output=../output/slurm/mergepost_%A_%a.out
#SBATCH --error=../output/slurm/mergepost_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. ../merge_posterior_k.jl "${SLURM_ARRAY_TASK_ID}"
