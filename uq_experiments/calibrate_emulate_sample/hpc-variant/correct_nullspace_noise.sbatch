#!/bin/bash
# Array job: one task per (N_ens, rng_idx) cell. Removes injected nullspace noise from existing
# :PCA combined posterior files (see ../correct_nullspace_noise.jl for the method and safety
# guarantees), drops their stale pushforward arrays, and moves the originals to
# output/<calib_dir>/injected_orig/ (KEEP_ORIG=0 deletes them instead). Rerun
# pushforward_from_posterior afterwards; submit_correct_nullspace.sh chains the whole sequence.
#
# Must be run with the same CALIBRATE_METHOD / DIM_REDUCTION (PCA) / CALIBRATE_DATE as the
# run being corrected; EXPERIMENT must be set explicitly (including l63):
#   sbatch --export=ALL,EXPERIMENT=l96_flux,CALIBRATE_METHOD=eks,CALIBRATE_DATE=2026-11-01 \
#          correct_nullspace_noise.sbatch

#SBATCH --job-name=fixnoise
#SBATCH --output=../output/slurm/fixnoise_%A_%a.out
#SBATCH --error=../output/slurm/fixnoise_%A_%a.err
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

julia --project=.. ../correct_nullspace_noise.jl "${SLURM_ARRAY_TASK_ID}"
