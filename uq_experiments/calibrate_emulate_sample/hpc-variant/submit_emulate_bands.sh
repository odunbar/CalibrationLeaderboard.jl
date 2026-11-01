#!/bin/bash
# Sourced by submit_l*.sh (not run directly).
#
# Submits the emulate_sample stage as one SLURM array PER BAND OF k, instead of one task per
# (N_ens, rng_idx) cell looping over all k. Every (N_ens, rng_idx, k) is then its own task
# (emulate_sample_l63.jl / emulate_sample_l96.jl in per-k mode, selected by K_LO/K_HI),
# writing output/<calib_dir>/emu_k/*_k<k>.jld2; merge_posterior.sbatch assembles them afterwards.
#
# Why bands: the cluster caps arrays at 1000 tasks, and the per-k cost grows with k (training on
# N_ens*k points), so low-k bands get a short wall-clock request and high-k bands a longer one.
#
#   submit_emulate_bands LABEL SCRIPT EXPERIMENT DEP_JID CALIBRATE_DATE MAX_ITER
#
# Sets EMU_JIDS to the colon-separated list of submitted job IDs (for --dependency=afterany:...).
#
# Knobs (override via env, e.g. K_PER_BAND=3 BAND_TIMES="00:30:00 01:30:00" bash submit_l96_vec.sh):
#   N_PER_K        tasks per k = length(N_ens_sizes) * n_repeats. MUST match experiment_config.jl
#                  (9 * 20 = 180); a too-small array silently skips cells.
#   K_PER_BAND     k values per array; K_PER_BAND * N_PER_K must be <= MAX_ARRAY_SIZE.
#   BAND_TIMES     wall-clock per band, space separated, band 1 (lowest k) first; the last entry
#                  is reused if there are more bands than entries. Starting guesses: tune them
#                  from `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS` on a first run.
#   THROTTLE       max simultaneously running tasks per array.
#   MAX_ARRAY_SIZE cluster limit (1000 on Resnick).

N_PER_K=${N_PER_K:-180}
K_PER_BAND=${K_PER_BAND:-5}
MAX_ARRAY_SIZE=${MAX_ARRAY_SIZE:-1000}
THROTTLE=${THROTTLE:-100}
BAND_TIMES=${BAND_TIMES:-"01:00:00 02:00:00 03:00:00"}

submit_emulate_bands() {
    local label=$1 script=$2 experiment=$3 dep=$4 cal_date=$5 max_iter=$6
    local -a times
    read -r -a times <<< "${BAND_TIMES}"

    if (( K_PER_BAND * N_PER_K > MAX_ARRAY_SIZE )); then
        echo "ERROR: K_PER_BAND*N_PER_K = $((K_PER_BAND * N_PER_K)) exceeds MAX_ARRAY_SIZE=${MAX_ARRAY_SIZE}" >&2
        return 1
    fi

    EMU_JIDS=""
    local lo=1 band=0 hi n t jid ti
    while (( lo <= max_iter )); do
        hi=$(( lo + K_PER_BAND - 1 ))
        (( hi > max_iter )) && hi=${max_iter}
        n=$(( (hi - lo + 1) * N_PER_K ))
        ti=$(( band < ${#times[@]} ? band : ${#times[@]} - 1 ))
        t=${times[$ti]}
        jid=$(sbatch --parsable \
                     -A esm \
                     --job-name="emu_${label}_k${lo}-${hi}" \
                     --dependency=afterok:${dep} \
                     --kill-on-invalid-dep=yes \
                     --array=1-${n}%${THROTTLE} \
                     --time=${t} \
                     --export=ALL,SCRIPT=${script},EXPERIMENT=${experiment},CALIBRATE_DATE=${cal_date},K_LO=${lo},K_HI=${hi} \
                     emulate_sample_array.sbatch)
        echo "  emulate_sample k=${lo}-${hi}: job ID ${jid} (array 1-${n}, ${t})"
        EMU_JIDS="${EMU_JIDS:+${EMU_JIDS}:}${jid}"
        lo=$(( hi + 1 ))
        band=$(( band + 1 ))
    done
}
