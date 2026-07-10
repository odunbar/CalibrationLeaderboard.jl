#!/bin/bash
# Submit preliminaries + calibrate + emulate_sample + diagnostic plots for the L96 vec-force case.
#
# Usage: bash submit_l96_vec.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when all four cases run simultaneously,
#                      e.g. "run2" → jobs appear as "calib_l96v_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="l96v${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p ../output/slurm

# Pin the calibrate date once so every stage in this chain agrees, even if
# the pipeline runs across a midnight boundary.
CALIBRATE_DATE=$(date +%Y-%m-%d)
echo "CALIBRATE_DATE=${CALIBRATE_DATE}"

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."

echo "=== Submitting preliminaries (L96 vec-force) ==="
PRELIM_JID=$(sbatch --parsable \
		   -A esm \
		   --job-name="prelim_${LABEL}" \
		   --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_vec \
		   preliminaries.sbatch)
echo "  preliminaries job ID: ${PRELIM_JID}"

echo "=== Submitting calibrate (L96 vec-force, after ${PRELIM_JID}) ==="
CALIB_JID=$(sbatch --parsable \
		   -A esm \
		   --job-name="calib_${LABEL}" \
		   --dependency=afterok:${PRELIM_JID} \
		   --kill-on-invalid-dep=yes \
		   --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_vec,CALIBRATE_DATE=${CALIBRATE_DATE} \
		   calibrate_array.sbatch)
echo "  calibrate job ID: ${CALIB_JID}"

echo "=== Submitting calibration_diagnostic_plots (L96 vec-force, after ${CALIB_JID}) ==="
CALIB_DIAG_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="calib_diag_${LABEL}" \
		 --dependency=afterok:${CALIB_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE=${CALIBRATE_DATE} \
		 calibration_diagnostic_plots_l96.sbatch)
echo "  calibration_diagnostic_plots job ID: ${CALIB_DIAG_JID}"

echo "=== Submitting emulate_sample (L96 vec-force, after ${CALIB_JID}) ==="
EMU_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="emu_${LABEL}" \
		 --dependency=afterok:${CALIB_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_vec,CALIBRATE_DATE=${CALIBRATE_DATE} \
		 emulate_sample_array.sbatch)
echo "  emulate_sample job ID: ${EMU_JID}"

echo "=== Submitting pushforward_from_posterior (L96 vec-force, after ${EMU_JID}) ==="
PUSHFWD_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="pushfwd_${LABEL}" \
		 --dependency=afterany:${EMU_JID} \
		 --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE=${CALIBRATE_DATE} \
		 pushforward_from_posterior.sbatch)
echo "  pushforward_from_posterior job ID: ${PUSHFWD_JID}"

echo "=== Submitting posterior_diagnostic_plots (L96 vec-force, after ${PUSHFWD_JID}) ==="
POST_DIAG_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="post_diag_${LABEL}" \
		 --dependency=afterany:${PUSHFWD_JID} \
		 --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE=${CALIBRATE_DATE} \
		 posterior_diagnostic_plots_l96.sbatch)
echo "  posterior_diagnostic_plots job ID: ${POST_DIAG_JID}"

echo "=== Submitting exp_to_leaderboard (L96 vec-force, after ${PUSHFWD_JID}) ==="
LB_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="leaderboard_${LABEL}" \
		 --dependency=afterany:${PUSHFWD_JID} \
		 --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE=${CALIBRATE_DATE} \
		 exp_to_leaderboard.sbatch)
echo "  exp_to_leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="


# sbatch --parsable -A esm --job-name="pushfwd_l96_vec" --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE="2026-07-09" pushforward_from_posterior.sbatch

# sbatch -A esm --job-name="post_diag_l96_vec" --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE="2026-07-09" posterior_diagnostic_plots_l96.sbatch

# sbatch -A esm --job-name="leaderboard_l96_vec" --export=ALL,EXPERIMENT=l96_vec,CALIBRATE_DATE="2026-07-09" exp_to_leaderboard.sbatch
