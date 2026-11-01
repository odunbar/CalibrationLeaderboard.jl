#!/bin/bash
# SLURM array job for the emulate-sample stage.
#
# Normally submitted per band of k by submit_emulate_bands.sh (sourced from submit_l*.sh),
# which overrides --array and --time on the command line and sets K_LO/K_HI: each task is
# then ONE (N_ens, rng_idx, k) (k-major index within the band; <=1000 tasks per array) and
# writes output/<calib_dir>/emu_k/*_k<k>.jld2. Run merge_posterior.sbatch afterwards to
# assemble the per-k files into the combined posterior file.
#
# Without K_LO/K_HI (the defaults below) each task is a whole (N_ens, rng_idx) cell looping
# over all k serially and writing the combined posterior file directly (180 tasks = 9 N_ens
# x 20 repeats, 12h): the original behaviour, no merge needed.
#
# Per-k, one band (L96 example):
#   sbatch --dependency=afterok:<calib_jobid> --array=1-900%100 --time=01:00:00 \
#          --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const,K_LO=1,K_HI=5 \
#          emulate_sample_array.sbatch
#
# Per-cell (L63; L96 additionally needs EXPERIMENT=l96_const etc.):
#   sbatch --dependency=afterok:<calib_jobid> \
#          --export=ALL,SCRIPT=emulate_sample_l63.jl emulate_sample_array.sbatch

#SBATCH --job-name=emusample
#SBATCH --output=../output/slurm/emusample_%A_%a.out
#SBATCH --error=../output/slurm/emusample_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT=${SCRIPT:-emulate_sample_l63.jl}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0 # already precompiled so should not get all jobs to re-precompile

julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
