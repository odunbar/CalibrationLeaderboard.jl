#!/bin/bash
# Submit the score-based LM L96 flux-force pipeline.
# Dependency chain: preliminaries →(afterok)→ run_array →(afterok)→ leaderboard
#
# N_TASKS = length(N_ens_sizes) * length(rmse_targets) * n_repeats = 3 * 3 * 100 = 900
# If N_ens_sizes, rmse_targets or n_repeats change in experiment_config.jl, update --array below.
#
# Usage: bash submit_l96_flux.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names,
#                      e.g. "run2" → jobs appear as "run_l96f_run2".
#
# Note: flux-force trains a Flux.jl NN per cell and runs ForwardDiff through
# it (nu ≈ 61 params, nx=100). If tasks time out, resubmit with --time=08:00:00.

set -euo pipefail

EXP_ID=${1:-}
LABEL="l96f_${SCORE_KIND:-dsm}_${BUDGET_MODE:-serial}${EXP_ID:+_${EXP_ID}}"
RUN_DATE=$(date +%Y-%m-%d)
# Arm selection. Each (SCORE_KIND, BUDGET_MODE) pair is a SEPARATE submission
# writing its own netcdf under its own algorithm_type.
#   SCORE_KIND  : dsm (learned score, DSM) | kgmm (learned score, KGMM) | gaussian (quasi-Gaussian FDT baseline)
#   BUDGET_MODE : serial (one long window) | parallel (N_ens independent branches off one spin-up)
SCORE_KIND=${SCORE_KIND:-dsm}
BUDGET_MODE=${BUDGET_MODE:-serial}
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p ../output/slurm

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."
echo "  run_date pinned to ${RUN_DATE} for all jobs in this pipeline."
echo "  arm: SCORE_KIND=${SCORE_KIND}  BUDGET_MODE=${BUDGET_MODE}"

echo "=== Submitting preliminaries (L96 flux-force) ==="
PRELIM_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="prelim_${LABEL}" \
                    --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_flux,RUN_DATE=${RUN_DATE},SCORE_KIND=${SCORE_KIND},BUDGET_MODE=${BUDGET_MODE} \
                    preliminaries.sbatch)
echo "  preliminaries job ID: ${PRELIM_JID}"

echo "=== Submitting run_array (L96 flux-force, lm, after ${PRELIM_JID}) ==="
RUN_JID=$(sbatch --parsable \
                 -A esm \
                 --job-name="run_${LABEL}" \
                 --array=1-900 \
                 --dependency=afterok:${PRELIM_JID} \
                 --kill-on-invalid-dep=yes \
                 --export=ALL,SCRIPT=run_l96_sblm.jl,EXPERIMENT=l96_flux,RUN_DATE=${RUN_DATE},SCORE_KIND=${SCORE_KIND},BUDGET_MODE=${BUDGET_MODE} \
                 run_array.sbatch)
echo "  run_array job ID: ${RUN_JID}"

echo "=== Submitting leaderboard (L96 flux-force, after ${RUN_JID}) ==="
LB_JID=$(sbatch --parsable \
                -A esm \
                --job-name="leaderboard_${LABEL}" \
                --dependency=afterany:${RUN_JID} \
                --kill-on-invalid-dep=yes \
                --export=ALL,EXPERIMENT=l96_flux,RUN_DATE=${RUN_DATE},SCORE_KIND=${SCORE_KIND},BUDGET_MODE=${BUDGET_MODE} \
                leaderboard.sbatch)
echo "  leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="

# ---------------------------------------------------------------------------
# Manual resubmission reference (not executed).
# Copy-paste to rerun ONE stage without reconstructing its --export flags --
# e.g. after finding a truncated output file once the pipeline has finished.
# Substitute the real RUN_DATE for <yyyy-mm-dd>, and the arm you want.
#
# preliminaries:
#   sbatch -A esm --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_flux,RUN_DATE=<yyyy-mm-dd>,SCORE_KIND=dsm,BUDGET_MODE=serial preliminaries.sbatch
#
# one array cell (smoke test):
#   sbatch -A esm --array=1-1 --export=ALL,SCRIPT=run_l96_sblm.jl,EXPERIMENT=l96_flux,RUN_DATE=<yyyy-mm-dd>,SCORE_KIND=dsm,BUDGET_MODE=serial run_array.sbatch
#
# full array:
#   sbatch -A esm --array=1-900 --export=ALL,SCRIPT=run_l96_sblm.jl,EXPERIMENT=l96_flux,RUN_DATE=<yyyy-mm-dd>,SCORE_KIND=dsm,BUDGET_MODE=serial run_array.sbatch
#
# leaderboard only:
#   sbatch -A esm --export=ALL,EXPERIMENT=l96_flux,RUN_DATE=<yyyy-mm-dd>,SCORE_KIND=dsm,BUDGET_MODE=serial leaderboard.sbatch
#
# the other arms:
#   SCORE_KIND=dsm       BUDGET_MODE=parallel  bash submit_l96_flux.sh
#   SCORE_KIND=gaussian BUDGET_MODE=serial      bash submit_l96_flux.sh
# ---------------------------------------------------------------------------
