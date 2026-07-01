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
`experiment_config.jl` and must be present in every array-capable entry point:

```julia
function main()
    tidx       = task_index_from_args()   # SLURM_ARRAY_TASK_ID > ARGS[1] > nothing
    experiment = l96_experiment()         # EXPERIMENT env var > ARGS[2] > config toggle
    tasks      = flat_tasks(experiment_config(experiment))

    if tidx === nothing
        for (N_ens, rng_idx) in tasks; run_one(N_ens, rng_idx); end
    else
        (N_ens, rng_idx) = tasks[tidx]; run_one(N_ens, rng_idx)
    end
end
main()
```

**Never hard-code a case index** (`case = cases[2]`) — always use the env-var / ARGS path.

## The --array upper bound is a footgun

Every `--array=1-N` must equal `length(flat_tasks(cfg))` from `experiment_config.jl`.
When these get out of sync, tasks either don't run or index out of bounds.
Document the formula in every sbatch header:

```bash
# N_TASKS = length(rmse_targets) * n_repeats = 3 * 100 = 300  (OPT)
# N_TASKS = length(N_ens_sizes) * n_repeats  = 9 * 20  = 180  (UQ)
# If either value changes in experiment_config.jl, update --array upper bound.
```

### When array size varies by case

If different cases produce different task counts (e.g. L63 uses 4 `N_ens_sizes` while
L96 uses 3), don't rely on a single `#SBATCH --array=1-N` in `run_array.sbatch`.
Instead, pass `--array=1-N` in each `submit_l*.sh` — command-line args override
`#SBATCH` directives:

```bash
sbatch --array=1-80 --export=ALL,SCRIPT=run_l63_<method>.jl ... run_array.sbatch
sbatch --array=1-60 --export=ALL,SCRIPT=run_l96_<method>.jl ... run_array.sbatch
```

Document the per-case formula in the submit script comment so it stays in sync.

## Cluster-specific settings (Caltech Resnick / cascadelake)

| Setting | Value | Where |
|---|---|---|
| Account | `-A esm` | `sbatch` call in `submit_*.sh` |
| Julia module | `module load julia/1.12.2` | All sbatch files |
| Constraint | `--constraint=cascadelake` | All sbatch files |
| CPU target | `JULIA_CPU_TARGET="cascadelake"` | `precompile.sbatch` only |
| Thread count | `${SLURM_CPUS_PER_TASK}` | `JULIA_NUM_THREADS` + `OPENBLAS_NUM_THREADS` |
| No auto-precompile | `JULIA_PKG_PRECOMPILE_AUTO=0` | All non-precompile sbatch files |
| Log dir | `../output/slurm/` | All `--output` / `--error` (OPT); `output/slurm/` (UQ) |

## Pin the run date via RUN_DATE, not by hand-editing experiment_config.jl

All array tasks in a pipeline must write to the same output directory, so the run
date has to be fixed once at submission time. Two older approaches both fail this:
`today()` can drift across a midnight boundary mid-pipeline, and manually pinning
`run_date = Date("2026-06-25", "yyyy-mm-dd")` in `experiment_config.jl` is easy to
forget to undo (the next local run silently reuses a stale date).

The fix: the **submit script** decides and owns the date — `experiment_config.jl`
just reads it, falling back to `today()` only when nothing was passed in (i.e. local
runs).

**In `submit_*.sh`**, compute it once, right after `LABEL` is set and before any
`sbatch` calls:
```bash
RUN_DATE=$(date +%Y-%m-%d)
```
Then thread it through **every** `sbatch --export=ALL,...` call in the chain —
preliminaries, run_array, leaderboard, every stage, no exceptions:
```bash
--export=ALL,SCRIPT=l63_preliminaries.jl,EXPERIMENT=l63,RUN_DATE=${RUN_DATE}
--export=ALL,SCRIPT=run_l63_<method>.jl,EXPERIMENT=l63,RUN_DATE=${RUN_DATE}
--export=ALL,EXPERIMENT=l63,RUN_DATE=${RUN_DATE}
```
Missing `RUN_DATE` on even one stage reintroduces the original bug: that stage's
tasks fall back to `today()` and can land in a different output directory than the
rest of the pipeline.

**In `experiment_config.jl`** (env var name matches the variable it controls —
`RUN_DATE` for `run_date` in OPT, `CALIBRATE_DATE` for `calibrate_date` in UQ):
```julia
run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()
```
Local runs have no `RUN_DATE` in `ENV`, so they fall through to `today()` automatically
— no manual pin/unpin step, and nothing left over to forget after a run finishes.

`opt_experiments/levenberg_marquardt/` is the worked, canonical example of this
end-to-end: see `experiment_config.jl` and `hpc-variant/submit_l63.sh` /
`submit_l96_const.sh` / `submit_l96_vec.sh` / `submit_l96_flux.sh`.

**One config only.** No separate `hpc-variant/experiment_config.jl`. Local and HPC share.

## Precompile job — always separate

`precompile.sbatch` + `submit_precompile.sh` must exist as standalone files.
The `submit_l*.sh` scripts never submit a precompile — they only remind the user.
This avoids dozens of array tasks racing to precompile.

## Serial pre-stage job (preliminaries pattern)

When run scripts share expensive setup (truth data, obs covariances, ICs), a
`"compute-if-missing"` guard inside the run script becomes a race condition on HPC:
multiple array tasks start simultaneously and all try to write the same file.

**The fix:** extract the setup into `l*_preliminaries.jl` and run it as a single
serial SLURM job before the array starts.

### When to apply

Look for a `build_*_problem()` function containing:
```julia
if isfile(prelim_file)
    ld = load_preliminaries(prelim_file)
else
    pdc = compute_perfect_data(...)
    save_preliminaries(pdc, prelim_file)
end
```
That guard is a local-run convenience. Extract the `else` branch into its own script.

### The three-part change

**1. Create `l63_preliminaries.jl` / `l96_preliminaries.jl` (in `<method>/`)**

Computes and saves unconditionally — no existence check:

```julia
function main()
    rng_i = MersenneTwister(11)    # fixed seed for reproducibility
    pdc = compute_perfect_data(...)
    save_preliminaries(pdc, prelim_file)
    @info "Saved preliminaries to $prelim_file"
end
main()
```

For L96, use `l96_experiment()` / `experiment_config(experiment).force_case` so
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

Per-task setup (prior distributions, forcing parameters, NN structure) stays in
`build_*_problem()` — only the shared expensive computation moves to the prelim script.

**3. Create `hpc-variant/preliminaries.sbatch`**

Mirror `leaderboard.sbatch`: single job, no `--array`, same `cd ${SLURM_SUBMIT_DIR}`
path trick. No `${SLURM_ARRAY_TASK_ID}` argument — the prelim script runs once, not
once per cell. Use `SCRIPT=${SCRIPT:-l63_preliminaries.jl}` for the env-var dispatch.

### Submit chain with preliminaries

```
preliminaries  →(afterok)→  run_array  →(afterany)→  leaderboard
```

Use `afterany` (not `afterok`) for the leaderboard: `run_to_leaderboard` reads whatever
JLD2 files exist and warns on missing ones, so it should fire even if some array tasks failed.

```bash
RUN_DATE=$(date +%Y-%m-%d)   # computed once in submit_*.sh; threaded through every stage below

PRELIM_JID=$(sbatch --parsable -A esm \
                    --job-name="prelim_${LABEL}" \
                    --export=ALL,SCRIPT=l63_preliminaries.jl,EXPERIMENT=l63,RUN_DATE=${RUN_DATE} \
                    preliminaries.sbatch)

RUN_JID=$(sbatch --parsable -A esm \
                 --job-name="run_${LABEL}" \
                 --dependency=afterok:${PRELIM_JID} --kill-on-invalid-dep=yes \
                 --export=ALL,SCRIPT=run_l63_<method>.jl,EXPERIMENT=l63,RUN_DATE=${RUN_DATE} \
                 run_array.sbatch)

LB_JID=$(sbatch --parsable -A esm \
                --job-name="leaderboard_${LABEL}" \
                --dependency=afterany:${RUN_JID} --kill-on-invalid-dep=yes \
                --export=ALL,EXPERIMENT=l63,RUN_DATE=${RUN_DATE} \
                leaderboard.sbatch)
```

### hpc-variant layout (with preliminaries)

```
<method>/
├── experiment_config.jl         ← single config (reads RUN_DATE env var, falls back to today())
├── l63_preliminaries.jl
├── l96_preliminaries.jl
├── run_l63_<method>.jl
├── run_l96_<method>.jl
├── run_to_leaderboard.jl
└── hpc-variant/
    ├── preliminaries.sbatch
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
2. Per-cell stages: use `array.sbatch` template. Serial stages: use `single_job.sbatch`.
3. Create `submit_l63.sh`, `submit_l96_const.sh`, `submit_l96_vec.sh`, `submit_l96_flux.sh`
   — each chains stages with `afterok` between array jobs and `afterany` before the leaderboard.
4. Create `precompile.sbatch` + `submit_precompile.sh` from templates.
5. Write or update `README.md` using `assets/README-skeleton.md`.

See `references/pipeline-uq.md` for the full dependency graph and sbatch table.

### OPT pipeline

`opt_experiments/adam/hpc-variant/` is the canonical reference. When adding SLURM to a new OPT method:

1. Check whether the run scripts already have `task_index_from_args()`, `l96_experiment()`,
   and `flat_tasks()` in their `main()`. If not, add them. Adam already has these; older
   methods (EKP, CBO) may not.
2. Create `experiment_config.jl` in the method directory with
   `run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()` (see
   "Pin the run date via RUN_DATE" above; `opt_experiments/levenberg_marquardt/`
   is the worked example).
3. Create `hpc-variant/` — put only sbatch + submit files there, no Julia scripts.
4. Array sbatch = one task per `(N_ens, rng_idx)` cell.
5. After the array completes, a single `leaderboard.sbatch` runs `run_to_leaderboard.jl`.
6. If run scripts share expensive setup, apply the **serial pre-stage pattern** above.
7. Dependency chain: `preliminaries` →(afterok)→ `run_array` →(afterany)→ `leaderboard`.
   Without prelims: `run_array` →(afterany)→ `leaderboard`.
8. `submit_l96_<case>.sh`: same pattern with `EXPERIMENT=l96_const|l96_vec|l96_flux`.

See `references/pipeline-opt.md` for the dependency graph and stage table.

## OPT hpc-variant layout: avoid copying scripts

OPT run scripts stay in the method directory. `hpc-variant/` contains only sbatch and
submit files — no `experiment_config.jl`, no Julia scripts:

```
adam/
├── experiment_config.jl
├── run_l63_adam.jl
├── run_l96_adam.jl
├── run_to_leaderboard.jl
└── hpc-variant/
    ├── run_array.sbatch
    ├── leaderboard.sbatch
    ├── precompile.sbatch
    ├── submit_precompile.sh
    ├── submit_l63.sh
    └── submit_l96_*.sh
```

In every OPT sbatch file:
```bash
cd "${SLURM_SUBMIT_DIR}"   # = hpc-variant/
julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
```

- `--project=..` uses `adam/Project.toml`
- `"../${SCRIPT}"` makes `@__DIR__` resolve to `adam/`, keeping `../../common` paths correct
- `include("experiment_config.jl")` in the script resolves relative to the script's directory
  (`adam/`), not the CWD — so there is always exactly one config

Log paths use `../output/slurm/`; submit scripts use `mkdir -p ../output/slurm`.

## Adjusting array size

When `rmse_targets`, `N_ens_sizes`, or `n_repeats` change in `experiment_config.jl`:
- Recompute `length(flat_tasks(cfg))` = `length(rmse_targets) * n_repeats` (OPT)
  or `length(N_ens_sizes) * n_repeats` (UQ).
- Update `--array=1-N` in every array sbatch file and every submit script for that experiment.
- Update the formula comment at the top of each sbatch file.
- The `%100` concurrency cap can be raised or removed; keep it as a cluster-courtesy default.

## Smoke test before full submission

Always offer a smoke test one-liner before suggesting a full array run:

```bash
# OPT:
sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_<method>.jl,EXPERIMENT=l63 run_array.sbatch

# UQ:
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
```

## Template files (in assets/)

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
this session — e.g. a dependency pattern that was tricky, or steps that were
unclear — to refine the skill for next time."
