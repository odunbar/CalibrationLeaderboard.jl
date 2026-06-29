#!/bin/bash
# SLURM single job: compute and save problem preliminaries.
# Runs serially (no array); must complete before run_array starts so all
# array tasks find the prelim file ready.
#
# L63: sbatch --export=ALL,SCRIPT=l63_preliminaries.jl,EXPERIMENT=l63 preliminaries.sbatch
# L96: sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_const preliminaries.sbatch
#      sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec   preliminaries.sbatch
#      sbatch --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_flux  preliminaries.sbatch

#SBATCH --job-name=prelim_lm
#SBATCH --output=../output/slurm/prelim_%j.out
#SBATCH --error=../output/slurm/prelim_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with precompile.sbatch

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT=${SCRIPT:-l63_preliminaries.jl}

# Stay in hpc-variant/ so include("experiment_config.jl") picks up the
# HPC-pinned version here; --project=.. uses levenberg_marquardt/Project.toml;
# ../${SCRIPT} runs from levenberg_marquardt/ so @__DIR__ resolves common/ paths correctly.
cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=.. "../${SCRIPT}"
