#!/bin/bash
# SLURM array job for the calibrate stage.
# array=1-20 means run array 1-20, one task per rng_idx (NOT per (N_ens, rng_idx):
# calibrate_l63.jl/calibrate_l96.jl draw one flat i.i.d. stream per rng_idx and
# compute every N_ens_sizes entry from prefixes of it, see calibrate_l*.jl headers).
#
# L63 (20 tasks = n_repeats):
#   sbatch --export=ALL,SCRIPT=calibrate_l63.jl,CALIBRATE_DATE=<yyyy-mm-dd> calibrate_array.sbatch
#
# L96 (same count; adjust SCRIPT and set EXPERIMENT):
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const,CALIBRATE_DATE=<yyyy-mm-dd> calibrate_array.sbatch
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_vec,CALIBRATE_DATE=<yyyy-mm-dd>   calibrate_array.sbatch
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_flux,CALIBRATE_DATE=<yyyy-mm-dd>  calibrate_array.sbatch
#
# N_TASKS = n_repeats = 20 (all four cases). If n_repeats changes in
# experiment_config.jl, update the --array upper bound.
#
# Each task forward-evaluates N_ens_max*N_iter candidates (~10,000) instead of
# GNKI's single-N_ens-per-task workload, so per-task cost is much higher here
# even though there are far fewer tasks — run the smoke test below and check
# the actual wall-clock before trusting the --time below for a full submission.
#
# The candidate loop is Threads.@threads-parallel (embarrassingly parallel,
# independent per-draw RNG) — set --cpus-per-task to use more cores.
# OPENBLAS_NUM_THREADS=1 avoids oversubscription (many Julia threads x many
# BLAS threads each) now that Julia threads do the parallel work.
#
# Requires the prelim file from preliminaries.sbatch to already exist —
# submit_l*.sh chains this automatically with --dependency=afterok.
#
# One-time precompile before submitting:
#   mkdir -p ../output/slurm
#   julia --project=.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

#SBATCH --job-name=calib
#SBATCH --output=../output/slurm/calib_%A_%a.out
#SBATCH --error=../output/slurm/calib_%A_%a.err
#SBATCH --array=1-20
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # pin version as needed, e.g.: module load julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-calibrate_l63.jl}

# Only sbatch/submit scripts live in hpc-variant/ — the .jl scripts and
# Project.toml stay in the parent method directory (one copy, no duplication).
# "../${SCRIPT}" keeps the script's own @__DIR__ resolved to the method dir;
# --project=.. points at the method dir's Project.toml.
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0 # already precompiled so should not get all jobs to re-precompile

julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
