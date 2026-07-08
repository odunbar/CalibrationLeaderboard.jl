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

`uq_experiments/GaussNewtonKalmanInversion/`, `uq_experiments/calibrate_emulate_sample/`,
and `opt_experiments/levenberg_marquardt/` are the canonical examples: one copy
of every `.jl` script and of `experiment_config.jl`, living in the method
directory; `hpc-variant/` holds only sbatch + submit scripts (see "hpc-variant
layout" below for why and how). `calibrate_emulate_sample` is the fuller UQ
example — it has the extra `emulate_sample` / `diagnostic_plots` stages that
`GaussNewtonKalmanInversion` skips (see `references/pipeline-uq.md`); read
either for the current shape, they follow the same conventions.

Read `references/pipeline-uq.md` for the full UQ pipeline spec.
Read `references/pipeline-opt.md` for the full OPT pipeline spec.

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
| Log dir | `../output/slurm/` | All `--output`/`--error` |

## Pin the run date via RUN_DATE / CALIBRATE_DATE

All array tasks in a pipeline must write to the same output directory, so the
date has to be fixed once at submission time. `today()` drifts across a
midnight boundary mid-pipeline; hand-pinning `run_date = Date("2026-06-25", ...)`
in `experiment_config.jl` is easy to forget to undo. The fix: the **submit
script** decides and owns the date — `experiment_config.jl` just reads it,
falling back to `today()` for local runs.

**In `submit_*.sh`**, compute it once and thread it through **every**
`sbatch --export=ALL,...` call in the chain — preliminaries, array, leaderboard,
no exceptions:
```bash
RUN_DATE=$(date +%Y-%m-%d)
# ...
--export=ALL,SCRIPT=run_l63_<method>.jl,EXPERIMENT=l63,RUN_DATE=${RUN_DATE}
```
Missing it on even one stage reintroduces the bug: that stage falls back to
`today()` and can land in a different output directory than the rest.

**In `experiment_config.jl`** (env var name matches the variable — `RUN_DATE`
for OPT's `run_date`, `CALIBRATE_DATE` for UQ's `calibrate_date`):
```julia
run_date = haskey(ENV, "RUN_DATE") ? Date(ENV["RUN_DATE"]) : today()
```
Treat this as standard practice for any new pipeline, UQ or OPT — not an
aspirational nice-to-have. Worked examples: `opt_experiments/levenberg_marquardt/`,
`uq_experiments/GaussNewtonKalmanInversion/`, `uq_experiments/calibrate_emulate_sample/`.

## Precompile job — always separate

`precompile.sbatch` + `submit_precompile.sh` must exist as standalone files.
The `submit_l*.sh` scripts never submit a precompile — they only remind the user.
This avoids dozens of array tasks racing to precompile.

## Serial pre-stage job (preliminaries pattern)

When run scripts share expensive setup (truth data, obs covariances, ICs), a
`"compute-if-missing"` guard becomes a race condition on HPC — multiple array
tasks start simultaneously and all try to compute/write the same file. Extract
the setup into `l*_preliminaries.jl` and run it as one serial SLURM job before
the array starts.

### When to apply

Look for:
```julia
if isfile(prelim_file)
    ld = load_preliminaries(prelim_file)
else
    pdc = compute_perfect_data(...)
    save_preliminaries(pdc, prelim_file)
end
```
Extract the `else` branch into its own script.

**Watch for the partial fix**, where only the *write* is gated and the
computation above it still runs unconditionally every task:
```julia
setup = build_setup(cfg)   # computed every task, regardless
if !isfile(prelim_file)
    try; save_preliminaries(setup.pdc, prelim_file); catch; end
end
```
This looks safe (`isfile` + `try/catch`) but isn't: it wastes the computation
N times (e.g. retraining a small neural net 180 times instead of once), and
the write can still race — the `try/catch` only stops the crash, not the torn
write. This exact shape was found and fixed in `calibrate_l63.jl` during
`calibrate_emulate_sample`'s migration to the preliminaries pattern — grep any
new run script for an unconditional compute before an `isfile`-gated save,
that's the tell. The real fix moves the *whole* computation, not just the save
call, into the prelim script.

### The three-part change

1. **`l63_preliminaries.jl` / `l96_preliminaries.jl`** (in `<method>/`): compute
   and save unconditionally, no existence check. For L96, dispatch via
   `l96_experiment()` / `experiment_config(experiment).force_case` so
   `EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl` works locally too.
2. **Run scripts become load-or-error**: `isfile(prelim_file) || error("... run l63_preliminaries.jl first.")`,
   then `load_preliminaries(prelim_file)`. Per-task setup (priors, forcing
   params, NN structure) stays in the run script — only the shared expensive
   part moves to the prelim script.
3. **`hpc-variant/preliminaries.sbatch`**: single job, no `--array`, same
   `cd "${SLURM_SUBMIT_DIR}"` path trick as the other stages, `SCRIPT=${SCRIPT:-l63_preliminaries.jl}`
   for env-var dispatch. Mirrors `leaderboard.sbatch`.

Chain: `preliminaries →(afterok)→ run_array →(afterany)→ leaderboard` —
`afterok` + `--kill-on-invalid-dep=yes` so a failed prelim stops wasted work;
`afterany` on the leaderboard so it processes whatever ran, even if some array
tasks failed. Worked examples: `uq_experiments/GaussNewtonKalmanInversion/`
(the original) and `uq_experiments/calibrate_emulate_sample/` (retrofitted
from the partial-fix anti-pattern above). See `references/pipeline-uq.md` /
`pipeline-opt.md` for full worked `sbatch --export=ALL,...` chains.

## Creating a pipeline for a new experiment

### UQ pipeline

1. Identify the stages: `calibrate`, optionally `emulate_sample`,
   `pushforward_from_posterior`, optionally `diagnostic_plots`,
   `exp_to_leaderboard`. Not every method needs all of them —
   `GaussNewtonKalmanInversion` skips `emulate_sample`/`diagnostic_plots`
   entirely because the raw ensemble at each iteration already is the
   posterior sample set.
2. Per-cell stages: use `array.sbatch` template. Serial stages: use `single_job.sbatch`.
3. Follow the "hpc-variant layout" convention below. Use
   `uq_experiments/GaussNewtonKalmanInversion/` (3-stage) or
   `uq_experiments/calibrate_emulate_sample/` (5-stage, with emulate_sample and
   diagnostic_plots) as the template, depending on which stage set the new
   method needs.
4. If run scripts share expensive setup, apply the **serial pre-stage
   pattern** above — check for the write-only-gated partial fix too.
5. Create `submit_l63.sh`, `submit_l96_const.sh`, `submit_l96_vec.sh`, `submit_l96_flux.sh`
   — each chains stages with `afterok` between array jobs and `afterany` before the leaderboard.
6. Create `precompile.sbatch` + `submit_precompile.sh` from templates.
7. Write or update `README.md` using `assets/README-skeleton.md`.

See `references/pipeline-uq.md` for the full dependency graph and sbatch table.

### OPT pipeline

`opt_experiments/adam/hpc-variant/` is the layout reference;
`opt_experiments/levenberg_marquardt/` is the `RUN_DATE` reference. When adding
SLURM to a new OPT method:

1. Check whether the run scripts already have `task_index_from_args()`,
   `l96_experiment()`, and `flat_tasks()` in their `main()`. Add them if not —
   older methods (EKP, CBO) may lack these.
2. Create `experiment_config.jl` with the `RUN_DATE` pattern above.
3. Follow the "hpc-variant layout" convention below.
4. Array sbatch = one task per `(N_ens, rng_idx)` cell; a single
   `leaderboard.sbatch` runs `run_to_leaderboard.jl` after.
5. If run scripts share expensive setup, apply the **serial pre-stage
   pattern** above.
6. `submit_l96_<case>.sh`: same pattern with `EXPERIMENT=l96_const|l96_vec|l96_flux`.

See `references/pipeline-opt.md` for the dependency graph and stage table.

## hpc-variant layout: one copy, no duplication

`hpc-variant/` holds only sbatch + submit scripts — no `experiment_config.jl`,
no `.jl` files, no `Project.toml`. Everything else lives exactly once, in
`<method>/`:

```
<method>/
├── experiment_config.jl
├── l63_preliminaries.jl / l96_preliminaries.jl   ← see preliminaries pattern
├── calibrate_l63.jl / run_l63_<method>.jl
├── calibrate_l96.jl / run_l96_<method>.jl
├── exp_to_leaderboard.jl / run_to_leaderboard.jl
└── hpc-variant/
    ├── preliminaries.sbatch
    ├── calibrate_array.sbatch / run_array.sbatch
    ├── exp_to_leaderboard.sbatch / leaderboard.sbatch
    ├── precompile.sbatch
    ├── submit_precompile.sh
    ├── submit_l63.sh
    └── submit_l96_*.sh
```

Every sbatch file:
```bash
cd "${SLURM_SUBMIT_DIR}"   # = hpc-variant/
julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
```
- `--project=..` → `<method>/Project.toml`
- `"../${SCRIPT}"` → the script's own `@__DIR__` resolves to `<method>/`, so
  `common/` includes and `include("experiment_config.jl")` behave exactly as
  in a local run. Verify a new path with
  `julia -e 'println(isdir(normpath(joinpath(@__DIR__, "..", "..", "common"))))'`
  from inside `hpc-variant/` before trusting it.

Log paths use `../output/slurm/`; submit scripts do `mkdir -p ../output/slurm`.
Worked examples: `levenberg_marquardt/`, `adam/`, `GaussNewtonKalmanInversion/`,
`calibrate_emulate_sample/`.

**Why not copy scripts into `hpc-variant/` instead?** It isn't broken — Julia's
`include()` resolves relative to the including file's own directory, so a
copied script genuinely does load a config copied alongside it. The problem is
drift: bump `N_ens_sizes` in one copy and forget the other, and array tasks
silently run stale settings. Referencing the original via `../${SCRIPT}`
removes the second copy, so there's nothing left to drift. (`calibrate_emulate_sample`
used to copy scripts into `hpc-variant/` this way — it has since been migrated
to the one-copy layout.)

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
