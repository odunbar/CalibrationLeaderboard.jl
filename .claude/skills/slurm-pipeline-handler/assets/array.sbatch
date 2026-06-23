#!/bin/bash
# SLURM array job for <STAGE_NAME> stage.
# One task per (N_ens, rng_idx) cell.  Array upper bound = length(N_ens_sizes) * n_repeats.
#
# L63:
#   sbatch --export=ALL,SCRIPT=<SCRIPT_L63> <STAGE_NAME>_array.sbatch
# L96:
#   sbatch --export=ALL,SCRIPT=<SCRIPT_L96>,EXPERIMENT=l96_const <STAGE_NAME>_array.sbatch
#   sbatch --export=ALL,SCRIPT=<SCRIPT_L96>,EXPERIMENT=l96_vec   <STAGE_NAME>_array.sbatch
#   sbatch --export=ALL,SCRIPT=<SCRIPT_L96>,EXPERIMENT=l96_flux  <STAGE_NAME>_array.sbatch
#
# If N_ens_sizes or n_repeats change, update --array upper bound to:
#   length(N_ens_sizes) * n_repeats (currently <N_TASKS>)

#SBATCH --job-name=<STAGE_SHORT>
#SBATCH --output=output/slurm/<STAGE_SHORT>_%A_%a.out
#SBATCH --error=output/slurm/<STAGE_SHORT>_%A_%a.err
#SBATCH --array=1-<N_TASKS>%100
#SBATCH --time=<HH:MM:SS>
#SBATCH --mem=<MEM>G
#SBATCH --cpus-per-task=<NCPUS>
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with precompile.sbatch

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default script if not set via --export
SCRIPT=${SCRIPT:-<SCRIPT_L63>}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=. "${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
