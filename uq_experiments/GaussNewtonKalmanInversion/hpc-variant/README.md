# GaussNewtonKalmanInversion HPC variant

This directory holds only SLURM sbatch and submit scripts — no `.jl` files, no
`experiment_config.jl`, no `Project.toml`. Every `.jl` script and the single
`experiment_config.jl` live in the parent method directory and are shared
verbatim between local and HPC runs; there is exactly one copy of each.

Every sbatch file does:
```bash
cd "${SLURM_SUBMIT_DIR}"        # = hpc-variant/
julia --project=.. "../${SCRIPT}" ...
```
so `@__DIR__` inside the invoked script still resolves to the method
directory — it picks up the same `experiment_config.jl`, the same `common/`
paths, and writes into the same `output/` tree as a local run. `--project=..`
points at the method directory's `Project.toml`. (See "avoid copying scripts"
in the `slurm-pipeline-handler` skill — this mirrors `opt_experiments/adam/`
and `opt_experiments/levenberg_marquardt/`, not `calibrate_emulate_sample`.)

There is no separate emulate_sample stage — the raw GNKI ensemble at each
iteration is itself the UQ sample set (see the top-level README).

## One-time setup

Precompilation is handled by a dedicated `submit_precompile.sh` script that
queues `precompile.sbatch` as a compute job. Run it once before your experiments
and again whenever the environment changes (fresh checkout, package updates).
The `submit_l*.sh` scripts do not precompile — they will remind you at
submission time.

`submit_l*.sh` computes the calibrate date once (`RUN_DATE=$(date +%Y-%m-%d)`)
and threads it through every downstream stage via `CALIBRATE_DATE`. This keeps
all array tasks and the leaderboard writing into the same output directory
even if the pipeline runs past midnight — no manual pin/unpin step needed.

## Pipeline

### L63
```
l63_preliminaries  ─(afterok)→  calibrate_array  ─(afterany)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```

### L96 (const / vec / flux)
```
l96_preliminaries  ─(afterok)→  calibrate_array  ─(afterany)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```

`preliminaries` computes and saves the shared truth-data/observations once,
serially, before any calibrate task starts — this avoids every array task
racing to compute and write the same file. `calibrate_array` then runs GNKI
to (approximate) convergence for every `(N_ens, rng_idx)` cell, storing the
ensemble at each iteration. `pushforward_from_posterior` pushes the stored
ensembles through the Lorenz forward map and saves output samples back into
each cell's results JLD2. `exp_to_leaderboard` loads all cells serially and
writes the leaderboard NetCDF.

## Standalone (serial, from the parent directory)

Run with no arguments to sweep all `(N_ens, rng_idx)` cells sequentially.
These are the same commands as in the top-level README — HPC and local share
one set of scripts.

```bash
# L63
julia --project=. l63_preliminaries.jl
julia --project=. calibrate_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. exp_to_leaderboard.jl

# L96 — set EXPERIMENT env var
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl
```

You can also run a single cell by passing its 1-based task index for the
array-capable scripts:

```bash
julia --project=. calibrate_l63.jl 1                         # first (N_ens, rng_idx) cell only
julia --project=. pushforward_from_posterior_l63.jl 1         # first cell only
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl 5
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl 5
```

## HPC (Caltech Resnick cluster, SLURM)

### Submission scripts (recommended)

Precompile once (or whenever the environment changes), then submit the cases:

```bash
bash submit_precompile.sh [EXP_ID]

bash submit_l63.sh        [EXP_ID]
bash submit_l96_const.sh  [EXP_ID]
bash submit_l96_vec.sh    [EXP_ID]
bash submit_l96_flux.sh   [EXP_ID]
```

Each `submit_l*.sh` script chains the full pipeline for its case automatically.
All four cases can be launched simultaneously — output files are case-specific
so there are no write conflicts. The optional `EXP_ID` argument suffixes SLURM
job names to keep the queue readable.

If you are launching all four cases together you only need one precompile run:

```bash
bash submit_precompile.sh run1
for s in submit_l63.sh submit_l96_const.sh submit_l96_vec.sh submit_l96_flux.sh; do
    bash "$s" run1 &
done
wait
```

### Manual submission

Precompile via `submit_precompile.sh` (or directly), then submit each stage:

```bash
RUN_DATE=$(date +%Y-%m-%d)

# L63
PRELIM_JID=$(sbatch --parsable -A esm \
             --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch)
CALIB_JID=$(sbatch --parsable -A esm \
            --dependency=afterok:${PRELIM_JID} --kill-on-invalid-dep=yes \
            --export=ALL,SCRIPT=calibrate_l63.jl,CALIBRATE_DATE=${RUN_DATE} \
            calibrate_array.sbatch)
PUSHFWD_JID=$(sbatch --parsable -A esm \
              --dependency=afterany:${CALIB_JID} \
              --export=ALL,CALIBRATE_DATE=${RUN_DATE} \
              pushforward_from_posterior.sbatch)
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l63,CALIBRATE_DATE=${RUN_DATE} \
       exp_to_leaderboard.sbatch

# L96
PRELIM_JID=$(sbatch --parsable -A esm \
             --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_const \
             preliminaries.sbatch)
CALIB_JID=$(sbatch --parsable -A esm \
            --dependency=afterok:${PRELIM_JID} --kill-on-invalid-dep=yes \
            --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const,CALIBRATE_DATE=${RUN_DATE} \
            calibrate_array.sbatch)
PUSHFWD_JID=$(sbatch --parsable -A esm \
              --dependency=afterany:${CALIB_JID} \
              --export=ALL,EXPERIMENT=l96_const,CALIBRATE_DATE=${RUN_DATE} \
              pushforward_from_posterior.sbatch)
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l96_const,CALIBRATE_DATE=${RUN_DATE} \
       exp_to_leaderboard.sbatch
```

### Sbatch files reference

| File | Type | Description |
|------|------|-------------|
| `preliminaries.sbatch` | single job | Computes + saves shared truth data/observations once (SCRIPT env var picks l63/l96) |
| `calibrate_array.sbatch` | array (1–180) | One task per `(N_ens, rng_idx)` cell; model-agnostic (SCRIPT env var) |
| `pushforward_from_posterior.sbatch` | array (1–180) | Posterior pushforward, one task per cell; saves results into the results JLD2 |
| `exp_to_leaderboard.sbatch` | single job | NetCDF leaderboard file, all cells serially |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

All SLURM logs and output data are written under `../output/` (the same
directory local runs use) — `../output/slurm/` for logs, `../output/gnki_<date>/`
for calibrate results, matching the top-level README's paths exactly.

### Adjusting array size

The sbatch files default to `--array=1-180` (9 ensemble sizes x 20 repeats, the
same for all four cases). If you change `N_ens_sizes` or `n_repeats` in
`experiment_config.jl` (the single copy in the parent directory), update the
upper bound to `length(N_ens_sizes) * n_repeats` in both `calibrate_array.sbatch`
and `pushforward_from_posterior.sbatch`. The `%100` suffix caps concurrent
tasks as a cluster-courtesy limit; raise or remove it if you want faster
turnaround.

### Smoke test

Before a full submission, run a single-task array to verify the job finds its
input files and writes output correctly (after the preliminaries file exists):

```bash
sbatch --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
```
