---
name: slurm-pipeline-handler
description: >-
  Creates and manages consistent local + HPC (SLURM) pipeline variants for
  uq or opt experiments in CalibrationLeaderboard.jl. Use this skill whenever
  the user wants to: add SLURM sbatch files or submit_*.sh scripts to an
  experiment that currently only runs locally; wire up a dependency-chained
  pipeline (calibrate → emulate_sample → pushforward → diagnostics/leaderboard
  for uq; or run → leaderboard for opt); create a precompile job; adjust the
  --array upper bound when N_ens_sizes or n_repeats change; add a new experiment
  case (l63/l96_const/l96_vec/l96_flux) to an existing submit script; understand
  the SLURM pipeline structure; or debug job dependency failures.
  Trigger even when the user says "add HPC support", "make it run on the cluster",
  "create sbatch files", "set up slurm for opt", "chain the jobs", "my array
  size is wrong", "how do I submit on Resnick", "separate out preliminaries",
  "race condition on the prelim file", or "pre-stage before the array job".
---

# SLURM Pipeline Handler

The canonical reference for how pipelines should look is
`uq_experiments/calibrate_emulate_sample/`. The UQ pipeline has
excellent SLURM organization; the OPT experiments currently have **none**.
This skill creates or extends SLURM pipelines following those conventions.

Read `references/pipeline-uq.md` for the full UQ pipeline spec.
Read `references/pipeline-opt.md` for the OPT pipeline spec (to be built).

## Core design principle: one set of .jl scripts, two run modes

Local and HPC use the **same Julia scripts**. The dispatch mechanism lives in
`experiment_config.jl` (or equivalent) and must be present in every
array-capable entry point:

```julia
# In every array-capable entry-point script, before main():
include("experiment_config.jl")   # or the @__DIR__-relative path to common/config/

function main()
    tidx = task_index_from_args()     # SLURM_ARRAY_TASK_ID > ARGS[1] > nothing (run all)
    experiment = l96_experiment()     # EXPERIMENT env var > ARGS[2] > toggle in config
    tasks = flat_tasks(experiment_config(experiment))

    if tidx === nothing
        for (N_ens, rng_idx) in tasks
            run_one(N_ens, rng_idx)
        end
    else
        (N_ens, rng_idx) = tasks[tidx]
        run_one(N_ens, rng_idx)
    end
end

main()
```

`task_index_from_args()` and `l96_experiment()` are defined in `experiment_config.jl`.
**Never hard-code a case index** (`case = cases[2]`) — always use the env-var / ARGS path.

## The --array upper bound is a footgun

Every `--array=1-N` in every sbatch file must equal `length(N_ens_sizes) * n_repeats`
from `experiment_config.jl`. When these get out of sync, tasks either don't run
or index out of bounds. Always check and document this:

```bash
# In the sbatch comment header (model: calibrate_array.sbatch):
# If N_ens_sizes or n_repeats change, update --array upper bound to:
# length(N_ens_sizes) * n_repeats
```

The UQ example uses 180 = 9 N_ens_sizes × 20 repeats. OPT example uses 6 = 3 × 2.
Different methods will have different values — compute and set explicitly.

## Cluster-specific settings (Caltech Resnick / cascadelake)

These are parameterized in every sbatch template:

| Setting | Value | Where |
|---|---|---|
| Account | `-A esm` | `sbatch` call in `submit_*.sh` |
| Julia module | `module load julia/1.12.2` | All sbatch files |
| Constraint | `--constraint=cascadelake` | All sbatch files |
| CPU target | `JULIA_CPU_TARGET="cascadelake"` | `precompile.sbatch` only |
| Thread count | `${SLURM_CPUS_PER_TASK}` | `JULIA_NUM_THREADS` + `OPENBLAS_NUM_THREADS` |
| No auto-precompile | `JULIA_PKG_PRECOMPILE_AUTO=0` | All non-precompile sbatch files |
| Log dir | `output/slurm/` | All `--output` / `--error` |

When adapting for a different cluster, only these settings change — the Julia
script invocation and dependency chain stay the same.

## Pin the run date before submitting

In `experiment_config.jl`, comment out the `today()` line and pin the date:
```julia
# UQ experiments use calibrate_date:
calibrate_date = Date("2026-06-04", "yyyy-mm-dd")

# OPT experiments use run_date:
run_date = Date("2026-06-25", "yyyy-mm-dd")
```
This ensures all array tasks write into the same output directory even when
jobs run past midnight or across days. The submit scripts should remind users:

```bash
echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first."
echo "NOTE: Pin run_date in experiment_config.jl before submitting."
```

For OPT, the HPC-pinned config lives in `hpc-variant/experiment_config.jl` (separate from
the local `experiment_config.jl` which keeps `today()`). Pin the date there before submitting.

## Precompile job — always separate

`precompile.sbatch` + `submit_precompile.sh` must exist as standalone files.
The `submit_l*.sh` scripts **never** submit a precompile — they only remind the
user to run it first. This avoids dozens of array tasks racing to precompile.

## Serial pre-stage job (preliminaries pattern)

When run scripts share expensive setup that all array tasks need — computing truth data,
obs covariances, initial conditions — that setup creates a race condition if left inside
the run script with a "compute-if-missing" guard: multiple array tasks may start
simultaneously and all try to write the same file.

The fix: extract the setup into a dedicated `l*_preliminaries.jl` script and run it
as a single serial SLURM job before the array starts.

### When to apply

Look for a `build_*_problem()` function (or equivalent) that contains:
```julia
if isfile(prelim_file)
    ld = load_preliminaries(prelim_file)
else
    pdc = compute_perfect_data(...)
    save_preliminaries(pdc, prelim_file)
end
```
That guard is a local-run convenience that becomes a race condition on HPC.
Extract the `else` branch into its own script.

### The three-part change

**1. Create `l63_preliminaries.jl` / `l96_preliminaries.jl` (in `<method>/`)**

The script computes and saves unconditionally — no existence check:

```julia
function main()
    rng_i = MersenneTwister(11)    # fixed seed for reproducibility
    # ... same setup as the old else-branch in build_*_problem ...
    pdc = compute_perfect_data(...)
    save_preliminaries(pdc, prelim_file)
    @info "Saved preliminaries to $prelim_file"
end
main()
```

For L96, use `l96_experiment()` and `experiment_config(experiment).force_case` to
select the case — the same env-var dispatch as the run scripts, so
`EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl` works locally too.

**2. Simplify `build_*_problem()` in the run scripts to load-or-error**

```julia
function build_l63_problem(output_dir)
    prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
    isfile(prelim_file) || error("Prelim file not found: $prelim_file\nRun l63_preliminaries.jl first.")
    ld = load_preliminaries(prelim_file)
    @info "Loaded L63 preliminaries from $prelim_file"
    return (; x0 = ld.x0, y = ld.y, R = ld.R, R_inv_var = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config, nx = 3)
end
```

Any per-task setup (prior distributions, forcing parameters, NN structure) stays in
`build_*_problem()` — only the shared expensive computation moves to the prelim script.

**3. Create `hpc-variant/preliminaries.sbatch`**

Mirror `leaderboard.sbatch` — single job, no array, same `cd ${SLURM_SUBMIT_DIR}` path trick:

```bash
#SBATCH --job-name=prelim_<method>
#SBATCH --output=../output/slurm/prelim_%j.out
#SBATCH --error=../output/slurm/prelim_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

SCRIPT=${SCRIPT:-l63_preliminaries.jl}
cd "${SLURM_SUBMIT_DIR}"
export JULIA_PKG_PRECOMPILE_AUTO=0
julia --project=.. "../${SCRIPT}"
```

Unlike `run_array.sbatch`, there is **no** `"${SLURM_ARRAY_TASK_ID}"` argument —
the prelim script runs once over the whole problem, not once per cell.

### Updated submit chain with preliminaries

```
preliminaries  →(afterok)→  run_array  →(afterok)→  leaderboard
```

In each `submit_l*.sh`, submit `preliminaries.sbatch` first and chain `run_array` off it:

```bash
PRELIM_JID=$(sbatch --parsable \
                    -A esm \
                    --job-name="prelim_${LABEL}" \
                    --export=ALL,SCRIPT=l63_preliminaries.jl,EXPERIMENT=l63 \
                    preliminaries.sbatch)
echo "  preliminaries job ID: ${PRELIM_JID}"

RUN_JID=$(sbatch --parsable \
                 -A esm \
                 --job-name="run_${LABEL}" \
                 --dependency=afterok:${PRELIM_JID} \
                 --kill-on-invalid-dep=yes \
                 --export=ALL,SCRIPT=run_l63_<method>.jl,EXPERIMENT=l63 \
                 run_array.sbatch)
```

### Updated hpc-variant layout (when preliminaries are present)

```
<method>/
├── experiment_config.jl         ← local (run_date = today())
├── l63_preliminaries.jl         ← runs once before l63 array
├── l96_preliminaries.jl         ← case selected via EXPERIMENT env var
├── run_l63_<method>.jl
├── run_l96_<method>.jl
├── run_to_leaderboard.jl
└── hpc-variant/
    ├── experiment_config.jl     ← HPC version (pin run_date here)
    ├── preliminaries.sbatch     ← single serial pre-stage job
    ├── run_array.sbatch
    ├── leaderboard.sbatch
    ├── precompile.sbatch
    ├── submit_precompile.sh
    ├── submit_l63.sh
    └── submit_l96_*.sh
```

## Creating a pipeline for a new experiment

### UQ pipeline (follow calibrate_emulate_sample as template)

1. Identify the stages: `calibrate`, `emulate_sample`, `pushforward_from_posterior`,
   `diagnostic_plots`, `exp_to_leaderboard`.
2. For each stage that runs per `(N_ens, rng_idx)` cell: use `array.sbatch` template.
3. For stages that run once over all cells serially: use `single_job.sbatch` template.
4. Create `submit_l63.sh`, `submit_l96_const.sh`, `submit_l96_vec.sh`, `submit_l96_flux.sh`
   from the `submit.sh` template — each chains the stages with `--dependency=afterok/afterany`.
5. Create `precompile.sbatch` + `submit_precompile.sh` from templates.
6. Write or update `README.md` using `assets/README-skeleton.md`.

See `references/pipeline-uq.md` for the full dependency graph and sbatch table.

### OPT pipeline

`opt_experiments/adam/hpc-variant/` is the canonical reference for OPT pipelines —
it was built first and follows the patterns below. When adding SLURM to a new OPT method:

1. Check whether the run scripts already have `task_index_from_args()`, `l96_experiment()`,
   and `flat_tasks()` in their `main()`. If not, add them (see the core design principle
   above). Adam already has these; older methods (EKP, CBO) may not.
2. Create `experiment_config.jl` in the method directory with `run_date = today()`.
3. Create `hpc-variant/` — put only sbatch + submit files + `experiment_config.jl` there
   (see "OPT hpc-variant layout" section below — do NOT copy the run scripts).
4. Array sbatch = one task per `(N_ens, rng_idx)` cell.
5. After the array completes, a single `leaderboard.sbatch` job runs
   `run_to_leaderboard.jl` (saves netcdf via `write_results_nc`).
6. If the run scripts share expensive setup (truth data, obs covariance, ICs), apply
   the **serial pre-stage pattern**: create `l*_preliminaries.jl` scripts and
   `preliminaries.sbatch`. See the "Serial pre-stage job" section above.
7. `submit_l63.sh` (with prelims): `preliminaries` →(afterok)→ `run_array` →(afterok)→ `leaderboard`.
   Without prelims: `run_array` →(afterok)→ `leaderboard`.
8. `submit_l96_<case>.sh`: same pattern with `EXPERIMENT=l96_const|l96_vec|l96_flux`.

See `references/pipeline-opt.md` for the dependency graph and stage table.

## OPT hpc-variant layout: avoid copying scripts

OPT run scripts use `@__DIR__` for `common/` includes but bare `include("experiment_config.jl")`
for config. This asymmetry lets you avoid copying scripts into `hpc-variant/`:

```
adam/
├── experiment_config.jl      ← local (run_date = today())
├── run_l63_adam.jl           ← NOT copied; stays here
├── run_l96_adam.jl
├── run_to_leaderboard.jl
└── hpc-variant/
    ├── experiment_config.jl  ← HPC version (pin run_date here)
    ├── run_array.sbatch
    ├── leaderboard.sbatch
    ├── precompile.sbatch
    ├── submit_precompile.sh
    ├── submit_l63.sh
    └── submit_l96_*.sh
```

**The trick — submit from `hpc-variant/`, run scripts from `adam/`:**

In every OPT sbatch file:
```bash
cd "${SLURM_SUBMIT_DIR}"   # = hpc-variant/ (set by SLURM to submission directory)
julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
```

Why each part matters:
- `cd "${SLURM_SUBMIT_DIR}"` → pwd = `hpc-variant/`, so `include("experiment_config.jl")`
  inside the script picks up the HPC-pinned version in `hpc-variant/` ✓
- `--project=..` → uses `adam/Project.toml` (one level up) ✓
- `"../${SCRIPT}"` → Julia sees the script at its real location (`adam/`), so `@__DIR__`
  inside the script resolves to `adam/`, keeping `../../common` paths correct ✓

**Log paths** for OPT sbatch files use `../output/slurm/` (not `output/slurm/`):
```bash
#SBATCH --output=../output/slurm/run_%A_%a.out
```

**Submit scripts** use `mkdir -p ../output/slurm` and can `cd "$DIR"` where `$DIR` is
the `hpc-variant/` directory (sbatch is submitted from there):
```bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p ../output/slurm
```

**For UQ experiments**, this trick is not needed — the UQ `hpc-variant/` has its own
copies of all scripts with adjusted `../../../common` paths (3 levels vs OPT's 2). When
working on UQ, follow `calibrate_emulate_sample/hpc-variant/` as the template instead.

## Adjusting array size

When `N_ens_sizes` or `n_repeats` change in `experiment_config.jl`:
- Update `--array=1-N` in **every** array sbatch file for that experiment.
- Update the comment at the top of each sbatch file that documents the formula.
- The `%100` concurrency cap can be raised or removed for faster turnaround;
  keep it as a cluster-courtesy default.

## Smoke test before full submission

Always offer a smoke test one-liner before suggesting a full array run:

```bash
# UQ example — run only task 1 to verify file resolution and output writing:
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch

# OPT equivalent:
sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_<method>.jl run_array.sbatch
```

## Template files (in assets/)

Copy and fill in `<PLACEHOLDER>` values:

| Template | Purpose |
|---|---|
| `assets/array.sbatch` | Array job for a per-cell stage |
| `assets/single_job.sbatch` | Single job for a serial stage |
| `assets/precompile.sbatch` | One-time precompile job |
| `assets/submit.sh` | Per-case submission script (chains stages) |
| `assets/submit_precompile.sh` | Submits just the precompile job |
| `assets/README-skeleton.md` | README template (pipeline graph + tables) |

## Final step — improve this skill

After finishing, offer to improve the **slurm-pipeline-handler** skill via
skill-creator: "Would you like to improve the **slurm-pipeline-handler** skill
using skill-creator? You can share suggestions, or I can analyse what came up
this session — e.g. cluster settings that differed, a dependency pattern that
was tricky, or steps that were unclear — to refine the skill for next time."
