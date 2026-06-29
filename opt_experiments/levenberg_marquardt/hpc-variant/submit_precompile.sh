#!/bin/bash
# Submit the one-off precompile job for levenberg_marquardt.
# Run this once before launching array jobs (or after any package update).
#
# Usage: bash submit_precompile.sh [EXP_ID]
#   EXP_ID (optional): label appended to the SLURM job name.

set -euo pipefail

EXP_ID=${1:-}
LABEL="precompile_lm${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p ../output/slurm

echo "=== Submitting precompile (levenberg_marquardt) ==="
PRECOMPILE_JID=$(sbatch --parsable \
                        -A esm \
                        --job-name="${LABEL}" \
                        precompile.sbatch)
echo "  precompile job ID: ${PRECOMPILE_JID}"
echo "=== Done. Monitor with: squeue -u \$USER ==="
