#!/bin/bash
# Single job: computes and saves the shared truth-data preliminaries once,
# before the calibrate array starts. Extracting this out of calibrate_l*.jl
# avoids every array task racing to compute and write the same prelim file.
#
# L63:
#   sbatch --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch
#
# L96 (set EXPERIMENT to select the force case):
#   sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_const preliminaries.sbatch
#   sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec   preliminaries.sbatch
#   sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_flux  preliminaries.sbatch

#SBATCH --job-name=prelim
#SBATCH --output=../output/slurm/prelim_%j.out
#SBATCH --error=../output/slurm/prelim_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-l63_preliminaries.jl}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0 # already precompiled so should not get this job to re-precompile

julia --project=.. "../${SCRIPT}"
