#!/bin/bash
# SLURM array job for the calibrate stage.
# array=1-180%100 means run array 1-180, with max of 100 at a time.
#
# L63 (180 tasks = 9 N_ens x 20 repeats):
#   sbatch --export=ALL,SCRIPT=calibrate_l63.jl,CALIBRATE_DATE=<yyyy-mm-dd> calibrate_array.sbatch
#
# L96 (same count; adjust SCRIPT and set EXPERIMENT):
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const,CALIBRATE_DATE=<yyyy-mm-dd> calibrate_array.sbatch
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_vec,CALIBRATE_DATE=<yyyy-mm-dd>   calibrate_array.sbatch
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_flux,CALIBRATE_DATE=<yyyy-mm-dd>  calibrate_array.sbatch
#
# N_TASKS = length(N_ens_sizes) * n_repeats = 9 * 20 = 180 (all four cases).
# If either value changes in experiment_config.jl, update the --array upper bound.
#
# Requires the prelim file from preliminaries.sbatch to already exist —
# submit_l*.sh chains this automatically with --dependency=afterok.
#
# One-time precompile before submitting (avoids 100 simultaneous precompiles):
#   mkdir -p ../output/slurm
#   julia --project=.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

#SBATCH --job-name=calib
#SBATCH --output=../output/slurm/calib_%A_%a.out
#SBATCH --error=../output/slurm/calib_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # pin version as needed, e.g.: module load julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-calibrate_l63.jl}

# Only sbatch/submit scripts live in hpc-variant/ — the .jl scripts and
# Project.toml stay in the parent method directory (one copy, no duplication).
# "../${SCRIPT}" keeps the script's own @__DIR__ resolved to the method dir;
# --project=.. points at the method dir's Project.toml.
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0 # already precompiled so should not get all jobs to re-precompile

julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
