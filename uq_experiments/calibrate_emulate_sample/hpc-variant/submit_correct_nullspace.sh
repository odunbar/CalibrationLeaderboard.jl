#!/bin/bash
# Correct an existing :PCA run whose posteriors were decoded with nullspace-noise injection,
# then redo the stages that depend on the posterior samples:
#   correct_nullspace_noise -> pushforward_from_posterior -> posterior_diagnostic_plots
#                                                         -> exp_to_leaderboard
# No emulate_sample / MCMC is rerun (the correction is exact for the saved chain; see
# ../correct_nullspace_noise.jl). Originals are kept in output/<calib_dir>/injected_orig/.
#
# Usage: CALIBRATE_DATE=<date of the run> bash submit_correct_nullspace.sh <EXPERIMENT> [EXP_ID]
#   EXPERIMENT: l96_flux | l96_vec   (the cases where the injector fired; l63 / l96_const were never
#               affected, but the script is a no-op for them)
#   EXP_ID (optional): label appended to the SLURM job names
#
# CALIBRATE_DATE is required and must be the date directory of the run to correct. Set the same
# CALIBRATE_METHOD as that run (default eki; the EKS runs need CALIBRATE_METHOD=eks), e.g.
#   CALIBRATE_METHOD=eks CALIBRATE_DATE=2026-11-01 bash submit_correct_nullspace.sh l96_flux

set -euo pipefail

EXPERIMENT=${1:?usage: CALIBRATE_DATE=<date> bash submit_correct_nullspace.sh <l63|l96_const|l96_vec|l96_flux> [EXP_ID]}
EXP_ID=${2:-}
: "${CALIBRATE_DATE:?set CALIBRATE_DATE to the date directory of the run to correct}"
LABEL="${EXPERIMENT}${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p ../output/slurm

if [[ "$EXPERIMENT" == "l63" ]]; then
    DIAG_SBATCH=posterior_diagnostic_plots_l63.sbatch
else
    DIAG_SBATCH=posterior_diagnostic_plots_l96.sbatch
fi

echo "=== Submitting correct_nullspace_noise (${EXPERIMENT}, CALIBRATE_DATE=${CALIBRATE_DATE}) ==="
FIX_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="fixnoise_${LABEL}" \
		 --export=ALL,EXPERIMENT=${EXPERIMENT},CALIBRATE_DATE=${CALIBRATE_DATE} \
		 correct_nullspace_noise.sbatch)
echo "  correct_nullspace_noise job ID: ${FIX_JID}"

echo "=== Submitting pushforward_from_posterior (after ${FIX_JID}) ==="
PUSHFWD_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="pushfwd_${LABEL}" \
		 --dependency=afterok:${FIX_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,EXPERIMENT=${EXPERIMENT},CALIBRATE_DATE=${CALIBRATE_DATE} \
		 pushforward_from_posterior.sbatch)
echo "  pushforward_from_posterior job ID: ${PUSHFWD_JID}"

echo "=== Submitting posterior_diagnostic_plots (after ${PUSHFWD_JID}) ==="
POST_DIAG_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="post_diag_${LABEL}" \
		 --dependency=afterany:${PUSHFWD_JID} \
		 --export=ALL,EXPERIMENT=${EXPERIMENT},CALIBRATE_DATE=${CALIBRATE_DATE} \
		 ${DIAG_SBATCH})
echo "  posterior_diagnostic_plots job ID: ${POST_DIAG_JID}"

echo "=== Submitting exp_to_leaderboard (after ${PUSHFWD_JID}) ==="
LB_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="leaderboard_${LABEL}" \
		 --dependency=afterany:${PUSHFWD_JID} \
		 --export=ALL,EXPERIMENT=${EXPERIMENT},CALIBRATE_DATE=${CALIBRATE_DATE} \
		 exp_to_leaderboard.sbatch)
echo "  exp_to_leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="
