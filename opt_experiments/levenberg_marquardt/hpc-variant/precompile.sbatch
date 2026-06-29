#!/bin/bash
# Single SLURM job that instantiates and precompiles the levenberg_marquardt project.
# Run via submit_precompile.sh before launching array jobs so that all
# array tasks start with a warm package cache and never race to precompile.

#SBATCH --job-name=precompile_lm
#SBATCH --output=../output/slurm/precompile_%j.out
#SBATCH --error=../output/slurm/precompile_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with run_array.sbatch

cd "${SLURM_SUBMIT_DIR}"

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Compile on the same architecture the array tasks will run on:
export JULIA_CPU_TARGET="cascadelake"
export JULIA_PKG_PRECOMPILE_AUTO=1

echo "JULIA_CPU_TARGET=$JULIA_CPU_TARGET"
julia -e '
println("CPU = ", Sys.CPU_NAME)
println("JULIA_CPU_TARGET = ", get(ENV,"JULIA_CPU_TARGET","<unset>"))
'

julia --project=.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
