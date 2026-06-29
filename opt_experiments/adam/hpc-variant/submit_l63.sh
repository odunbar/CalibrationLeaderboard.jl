#!/bin/bash
# Submit the adam L63 pipeline.
# Dependency chain: preliminaries →(afterok)→ run_array →(afterok)→ leaderboard
#
# Usage: bash submit_l63.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when multiple cases run simultaneously,
#                      e.g. "run2" → jobs appear as "run_l63_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="l63${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p ../output/slurm

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."
echo "NOTE: Pin run_date in experiment_config.jl before submitting to ensure all"
echo "      array tasks write to the same output directory."

echo "=== Submitting preliminaries (L63) ==="
PRELIM_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="prelim_${LABEL}" \
                    --export=ALL,SCRIPT=l63_preliminaries.jl,EXPERIMENT=l63 \
                    preliminaries.sbatch)
echo "  preliminaries job ID: ${PRELIM_JID}"

echo "=== Submitting run_array (L63, adam, after ${PRELIM_JID}) ==="
RUN_JID=$(sbatch --parsable \
                 -A esm \
                 --job-name="run_${LABEL}" \
                 --dependency=afterok:${PRELIM_JID} \
                 --kill-on-invalid-dep=yes \
                 --export=ALL,SCRIPT=run_l63_adam.jl,EXPERIMENT=l63 \
                 run_array.sbatch)
echo "  run_array job ID: ${RUN_JID}"

echo "=== Submitting leaderboard (L63, after ${RUN_JID}) ==="
LB_JID=$(sbatch --parsable \
                -A esm \
                --job-name="leaderboard_${LABEL}" \
                --dependency=afterany:${RUN_JID} \
                --kill-on-invalid-dep=yes \
                --export=ALL,EXPERIMENT=l63 \
                leaderboard.sbatch)
echo "  leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="
