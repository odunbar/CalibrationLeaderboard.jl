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
  size is wrong", or "how do I submit on Resnick".
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

## Pin calibrate_date before submitting

In `experiment_config.jl`, comment out `calibrate_date = today()` and pin it:
```julia
calibrate_date = Date("2026-06-04", "yyyy-mm-dd")
```
This ensures all array tasks write into the same output directory even when
jobs run past midnight or across days. The submit scripts should remind users:

```bash
echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first."
```

## Precompile job — always separate

`precompile.sbatch` + `submit_precompile.sh` must exist as standalone files.
The `submit_l*.sh` scripts **never** submit a precompile — they only remind the
user to run it first. This avoids dozens of array tasks racing to precompile.

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

### OPT pipeline (new — does not exist yet)

The OPT experiments currently have no SLURM support. When adding it:

1. Each `run_l63_<method>.jl` and `run_l96_<method>.jl` must be updated to
   use `task_index_from_args()` + `l96_experiment()` (they currently use hard-coded
   `case = cases[N]`).
2. Create an `experiment_config.jl` for the method (or wire to `common/config/`
   once that exists — see [[common-handler]]).
3. Array sbatch = one task per `(N_ens, rng_idx)` cell.
4. After the array completes, a single `leaderboard.sbatch` job runs
   `run_to_leaderboard.jl` (saves netcdf via `write_results_nc`).
5. `submit_l63.sh`: `run_array` →(afterok)→ `leaderboard`.
6. `submit_l96_<case>.sh`: same pattern with `EXPERIMENT=l96_const|l96_vec|l96_flux`.

See `references/pipeline-opt.md` for the dependency graph and stage table.

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
