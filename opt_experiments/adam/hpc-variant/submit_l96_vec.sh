#!/bin/bash
# Submit the adam L96 vec-force pipeline.
# Dependency chain: preliminaries →(afterok)→ run_array →(afterok)→ leaderboard
#
# Usage: bash submit_l96_vec.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names,
#                      e.g. "run2" → jobs appear as "run_l96v_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="l96v${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p ../output/slurm

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."
echo "NOTE: Pin run_date in experiment_config.jl before submitting to ensure all"
echo "      array tasks write to the same output directory."

echo "=== Submitting preliminaries (L96 vec-force) ==="
PRELIM_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="prelim_${LABEL}" \
                    --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec \
                    preliminaries.sbatch)
echo "  preliminaries job ID: ${PRELIM_JID}"

echo "=== Submitting run_array (L96 vec-force, adam, after ${PRELIM_JID}) ==="
RUN_JID=$(sbatch --parsable \
                 -A esm \
                 --job-name="run_${LABEL}" \
                 --dependency=afterok:${PRELIM_JID} \
                 --kill-on-invalid-dep=yes \
                 --export=ALL,SCRIPT=run_l96_adam.jl,EXPERIMENT=l96_vec \
                 run_array.sbatch)
echo "  run_array job ID: ${RUN_JID}"

echo "=== Submitting leaderboard (L96 vec-force, after ${RUN_JID}) ==="
LB_JID=$(sbatch --parsable \
                -A esm \
                --job-name="leaderboard_${LABEL}" \
                --dependency=afterok:${RUN_JID} \
                --kill-on-invalid-dep=yes \
                --export=ALL,EXPERIMENT=l96_vec \
                leaderboard.sbatch)
echo "  leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="
