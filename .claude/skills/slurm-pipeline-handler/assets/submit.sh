#!/bin/bash
# Submit <EXPERIMENT_CASE> pipeline for <METHOD_NAME>.
#
# Usage: bash submit_<CASE>.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when multiple cases run simultaneously,
#                      e.g. "run2" → jobs appear as "<STAGE_SHORT>_<CASE>_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="<CASE_SHORT>${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p output/slurm

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."

# ── Stage 1: array job ────────────────────────────────────────────────────────
echo "=== Submitting <STAGE1_NAME> (<CASE_LABEL>) ==="
STAGE1_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="<STAGE1_SHORT>_${LABEL}" \
                    --export=ALL,SCRIPT=<SCRIPT_L63_OR_L96>,EXPERIMENT=<EXPERIMENT_VAL> \
                    <STAGE1_SBATCH>)
echo "  <STAGE1_NAME> job ID: ${STAGE1_JID}"

# ── Stage 2: depends on stage 1 ──────────────────────────────────────────────
echo "=== Submitting <STAGE2_NAME> (<CASE_LABEL>, after ${STAGE1_JID}) ==="
STAGE2_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="<STAGE2_SHORT>_${LABEL}" \
                    --dependency=afterok:${STAGE1_JID} \
                    --kill-on-invalid-dep=yes \
                    --export=ALL,EXPERIMENT=<EXPERIMENT_VAL> \
                    <STAGE2_SBATCH>)
echo "  <STAGE2_NAME> job ID: ${STAGE2_JID}"

# Add further stages following the same pattern.
# Use afterok when the next stage REQUIRES success; afterany when it should
# always run (e.g. diagnostics that process partial results).

echo "=== Done. Monitor with: squeue -u \$USER ==="
