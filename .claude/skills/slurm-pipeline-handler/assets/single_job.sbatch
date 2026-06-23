#!/bin/bash
# Single SLURM job for <STAGE_NAME>.
# Runs all (N_ens, rng_idx) cells serially in one job (no array needed).
#
# Manual use:
#   L63: sbatch --dependency=afterany:<PREV_JID> <STAGE_NAME>.sbatch
#   L96: sbatch --dependency=afterany:<PREV_JID> --export=ALL,EXPERIMENT=l96_const <STAGE_NAME>.sbatch

#SBATCH --job-name=<STAGE_SHORT>
#SBATCH --output=output/slurm/<STAGE_SHORT>_%j.out
#SBATCH --error=output/slurm/<STAGE_SHORT>_%j.err
#SBATCH --time=<HH:MM:SS>
#SBATCH --mem=<MEM>G
#SBATCH --cpus-per-task=<NCPUS>
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=. <SCRIPT_NAME>.jl
