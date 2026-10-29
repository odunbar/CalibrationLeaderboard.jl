#!/bin/bash
# Combine the per-cell GBOED diagnostics of every variant and case into one file per (variant, case):
#   diagnostics/output/diagnostics_<boed-variant key>_<case>_<date>.jld2
# (all seeds x N_ens in one small file; diagnostics/test3_eig_placement.jl and test4_eig_helpfulness.jl read these).
#
# Usage: bash submit_combine_diagnostics.sh [DATE]
#   DATE (optional): the calibrate date of the cell directories, yyyy-mm-dd, applied to every variant. Default: the LATEST date found for each
#                    variant, i.e. the newest output/boed-<variant>_<yyyy-mm-dd>/ directory, so variants run on different dates (e.g. tmcmc on
#                    2026-10-16, iekf on 2026-10-19) are each combined at their own latest date.
#   Env overrides (optional):
#     VARIANTS="tmcmc_eig tmcmc iekf_eig iekf"     which variants to combine (default: these four; add eks_eig eks for the EKS variants)
#     CASES="l63 l96_const l96_vec l96_flux"       which cases to combine (default: all four)
#     DRY_RUN=1                                    print what would be submitted, submit nothing
#   Examples:
#     bash submit_combine_diagnostics.sh                                  # each variant at its latest date
#     bash submit_combine_diagnostics.sh 2026-10-19                       # all variants with cells dated 2026-10-19
#     VARIANTS="tmcmc tmcmc_eig" bash submit_combine_diagnostics.sh 2026-10-16
#     CASES=l96_vec bash submit_combine_diagnostics.sh                    # latest dates, one case
#
# One SLURM job per variant (the variants run in parallel; each job loops over the cases). A variant is skipped, with a message, if
# no output/boed-<variant>_<date>/ directory exists (for the given date, or at all). Each case reads the cell JLD2s there and the leaderboard netcdf
# output/<key>_<case>_<date>.nc (for the reached/partial/never status; it is only a warning if it is missing).
# Run it after the calibrate stage (and the leaderboard stage, for the status labels). Needs the packages of the method Project (no precompile here:
# run submit_precompile.sh first after a fresh checkout or package update).
#
# Copy the results back with, from your machine:
#   rsync -av hpc:<repo>/uq_experiments/BayesianOptimalExperimentalDesign/diagnostics/output/diagnostics_*_<date>.jld2 diagnostics/output/

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
METHOD_DIR="$(cd "$DIR/.." && pwd)"
cd "$DIR"
mkdir -p ../output/slurm

ARG_DATE=${1:-}                           # empty: use each variant's latest date
if [ -n "$ARG_DATE" ] && ! [[ "$ARG_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: DATE must be yyyy-mm-dd, got '$ARG_DATE'" >&2
    exit 1
fi
VARIANTS=${VARIANTS:-"tmcmc_eig tmcmc iekf_eig iekf"}
CASES=${CASES:-"l63 l96_const l96_vec l96_flux"}
DRY_RUN=${DRY_RUN:-0}

[ -f "$METHOD_DIR/diagnostics/combine_diagnostics.jl" ] || { echo "ERROR: $METHOD_DIR/diagnostics/combine_diagnostics.jl not found (git pull?)" >&2; exit 1; }

echo "Combining diagnostics; date: ${ARG_DATE:-<latest found per variant>}; variants: ${VARIANTS}; cases: ${CASES}"

n_submitted=0
for V in $VARIANTS; do
    case "$V" in
        tmcmc_eig|tmcmc|iekf_eig|iekf|eks_eig|eks) ;;
        *) echo "ERROR: unknown variant '$V' (expected tmcmc_eig, tmcmc, iekf_eig, iekf, eks_eig, eks)" >&2; exit 1 ;;
    esac
    KEY="boed-${V//_/-}"                      # leaderboard key, e.g. boed-tmcmc-eig (experiment_config.jl's method_key)
    if [ -n "$ARG_DATE" ]; then
        RUN_DATE=$ARG_DATE
    else   # newest output/<key>_yyyy-mm-dd (the underscore keeps boed-tmcmc from matching boed-tmcmc-eig)
        RUN_DATE=$(ls -d ../output/${KEY}_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] 2>/dev/null | sed "s|.*/${KEY}_||" | sort | tail -n 1 || true)
    fi
    CELLS="../output/${KEY}_${RUN_DATE}"
    if [ -z "$RUN_DATE" ] || [ ! -d "$CELLS" ]; then
        echo "  [skip] ${V}: no cell directory output/${KEY}_${RUN_DATE:-<date>}"
        continue
    fi
    echo "  ${V}: date ${RUN_DATE}, $(ls "$CELLS" | wc -l) files in output/${KEY}_${RUN_DATE}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry run] would submit combine_${V} (cases: ${CASES})"
        continue
    fi
    # passed through the environment (sbatch --export=ALL is the default): CASES holds spaces, which --export cannot carry
    export BOED_VARIANT=${V} CALIBRATE_DATE=${RUN_DATE} CASES METHOD_DIR
    JID=$(sbatch --parsable \
        -A esm \
        --job-name="combine_${V}" \
        --output="../output/slurm/combine_${V}_%j.out" \
        --error="../output/slurm/combine_${V}_%j.err" \
        --time=01:00:00 --mem=8G --cpus-per-task=4 --ntasks=1 \
        --constraint=cascadelake <<'EOF'
#!/bin/bash
module load julia/1.12.2
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PKG_PRECOMPILE_AUTO=0
cd "${METHOD_DIR}"
echo "combine: variant=${BOED_VARIANT} date=${CALIBRATE_DATE} cases=${CASES}"
n_fail=0
for c in ${CASES}; do
    echo "=== ${BOED_VARIANT} ${c} ==="
    EXPERIMENT=${c} julia --project=. diagnostics/combine_diagnostics.jl || { echo "FAILED: ${BOED_VARIANT} ${c}" >&2; n_fail=$((n_fail + 1)); }
done
echo "combine done: ${n_fail} failed case(s)"
ls -la diagnostics/output/ | grep -E "diagnostics_boed-.*_${CALIBRATE_DATE}\.jld2" || true
exit $((n_fail > 0))
EOF
    )
    echo "  submitted combine_${V}: job ${JID}"
    n_submitted=$((n_submitted + 1))
done

echo "=== ${n_submitted} job(s) submitted. Monitor with: squeue -u \$USER ; logs in output/slurm/combine_*.out ==="
echo "Outputs: diagnostics/output/diagnostics_boed-<variant>_<case>_<date>.jld2 (dates as listed above)"
