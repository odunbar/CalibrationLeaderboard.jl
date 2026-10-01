#!/bin/bash
# Submit the ReinforcementLearning (DDPG / TD3 actor-critic) L96 vec-force pipeline.
# Dependency chain (per arm):
#   preliminaries →(afterok)→ run_array →(afterany)→ leaderboard
# The preliminaries job is shared by all arms submitted by one invocation.
#
# N_TASKS = length(rmse_targets) * n_repeats = 3 * 30 = 90   (per arm)
# If rmse_targets or n_repeats change in experiment_config.jl, update N_TASKS below.
#
# ── Toggles (env vars; values can be a single option or "all") ──────────────────
#   ALGORITHM        : ddpg | td3 | all          (default: ddpg)
#   UPDATE_FREQUENCY : short | long | all        (default: short)
# Each (ALGORITHM, UPDATE_FREQUENCY) pair is a SEPARATE arm: its own array job and its
# own leaderboard netcdf with algorithm_type = rl-<algorithm>-<update_frequency>.
#   "short" = RL step is a fraction of the statistics window (many RL updates per window)
#   "long"  = RL step is the full statistics window (one RL update per forward evaluation)
#
# Usage: [ALGORITHM=..] [UPDATE_FREQUENCY=..] bash submit_l96_vec.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when multiple cases run simultaneously,
#                      e.g. "run2" → jobs appear as "run_l96v_ddpg_short_run2".
#
# Examples:
#   bash submit_l96_vec.sh                                          # ddpg, short
#   ALGORITHM=td3 UPDATE_FREQUENCY=long bash submit_l96_vec.sh      # td3, long
#   ALGORITHM=all UPDATE_FREQUENCY=all  bash submit_l96_vec.sh      # all four arms

set -euo pipefail

EXP_ID=${1:-}
RUN_DATE=$(date +%Y-%m-%d)
N_TASKS=90
ALGORITHM=${ALGORITHM:-ddpg}
UPDATE_FREQUENCY=${UPDATE_FREQUENCY:-short}
DIR="$(cd "$(dirname "$0")" && pwd)"

case "${ALGORITHM}" in
    all)       ALGORITHMS=(ddpg td3) ;;
    ddpg|td3)  ALGORITHMS=("${ALGORITHM}") ;;
    *) echo "ERROR: ALGORITHM must be ddpg, td3 or all (got '${ALGORITHM}')" >&2; exit 1 ;;
esac
case "${UPDATE_FREQUENCY}" in
    all)         FREQUENCIES=(short long) ;;
    short|long)  FREQUENCIES=("${UPDATE_FREQUENCY}") ;;
    *) echo "ERROR: UPDATE_FREQUENCY must be short, long or all (got '${UPDATE_FREQUENCY}')" >&2; exit 1 ;;
esac

cd "$DIR"
mkdir -p ../output/slurm

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."
echo "  run_date pinned to ${RUN_DATE} for all jobs in this pipeline."
echo "  arms: ALGORITHM=${ALGORITHMS[*]}  UPDATE_FREQUENCY=${FREQUENCIES[*]}"

echo "=== Submitting preliminaries (L96 vec-force) ==="
PRELIM_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="prelim_l96v${EXP_ID:+_${EXP_ID}}" \
                    --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec,RUN_DATE=${RUN_DATE} \
                    preliminaries.sbatch)
echo "  preliminaries job ID: ${PRELIM_JID}"

for ALG in "${ALGORITHMS[@]}"; do
for FREQ in "${FREQUENCIES[@]}"; do
    LABEL="l96v_${ALG}_${FREQ}${EXP_ID:+_${EXP_ID}}"
    ARM="ALGORITHM=${ALG},UPDATE_FREQUENCY=${FREQ}"

    echo "=== Submitting run_array (L96 vec-force, ${ALG}-${FREQ}, after ${PRELIM_JID}) ==="
    RUN_JID=$(sbatch --parsable \
                     -A esm \
                     --job-name="run_${LABEL}" \
                     --array=1-${N_TASKS} \
                     --dependency=afterok:${PRELIM_JID} \
                     --kill-on-invalid-dep=yes \
                     --export=ALL,SCRIPT=run_l96_ddpg.jl,EXPERIMENT=l96_vec,RUN_DATE=${RUN_DATE},${ARM} \
                     run_array.sbatch)
    echo "  run_array job ID: ${RUN_JID}"

    echo "=== Submitting leaderboard (L96 vec-force, ${ALG}-${FREQ}, after ${RUN_JID}) ==="
    LB_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="leaderboard_${LABEL}" \
                    --dependency=afterany:${RUN_JID} \
                    --kill-on-invalid-dep=yes \
                    --export=ALL,EXPERIMENT=l96_vec,RUN_DATE=${RUN_DATE},${ARM} \
                    leaderboard.sbatch)
    echo "  leaderboard job ID: ${LB_JID}"
done
done

echo "=== Done. Monitor with: squeue -u \$USER ==="

# ---------------------------------------------------------------------------
# Manual resubmission reference (not executed).
# Copy-paste to rerun ONE stage without reconstructing its --export flags --
# e.g. after finding a truncated output file once the pipeline has finished.
# Substitute the real RUN_DATE for <yyyy-mm-dd>, and the arm you want
# (ALGORITHM=ddpg|td3, UPDATE_FREQUENCY=short|long).
#
# preliminaries:
#   sbatch -A esm --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec,RUN_DATE=<yyyy-mm-dd> preliminaries.sbatch
#
# one array cell (smoke test):
#   sbatch -A esm --array=1-1 --export=ALL,SCRIPT=run_l96_ddpg.jl,EXPERIMENT=l96_vec,RUN_DATE=<yyyy-mm-dd>,ALGORITHM=ddpg,UPDATE_FREQUENCY=short run_array.sbatch
#
# full array:
#   sbatch -A esm --array=1-90 --export=ALL,SCRIPT=run_l96_ddpg.jl,EXPERIMENT=l96_vec,RUN_DATE=<yyyy-mm-dd>,ALGORITHM=ddpg,UPDATE_FREQUENCY=short run_array.sbatch
#
# leaderboard only:
#   sbatch -A esm --export=ALL,EXPERIMENT=l96_vec,RUN_DATE=<yyyy-mm-dd>,ALGORITHM=ddpg,UPDATE_FREQUENCY=short leaderboard.sbatch
#
# the other arms:
#   ALGORITHM=td3 UPDATE_FREQUENCY=long  bash submit_l96_vec.sh
#   ALGORITHM=all UPDATE_FREQUENCY=all   bash submit_l96_vec.sh
# ---------------------------------------------------------------------------
