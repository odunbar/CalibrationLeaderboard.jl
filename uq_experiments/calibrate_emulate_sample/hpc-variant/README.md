# EKIRace HPC variant

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
points at the method directory's `Project.toml`.

Each script can run serially (identical behaviour to a local run) or as a
SLURM job array where every `(N_ens, rng_idx)` cell is an independent task.

## One-time setup

Each `submit_l*.sh` script pins `CALIBRATE_DATE=$(date +%Y-%m-%d)` once at
submission time and passes it via `--export=ALL,...,CALIBRATE_DATE=${CALIBRATE_DATE}`
to every stage in its chain. `experiment_config.jl` reads it
(`ENV["CALIBRATE_DATE"]`, falling back to `today()` for local runs), so all
stages agree on the output directory even if the pipeline runs past midnight
or across days — no manual editing required. For manual `sbatch` submission
(below), remember to pass `CALIBRATE_DATE` on every call yourself.

Precompilation is handled by a dedicated `submit_precompile.sh` script that
queues `precompile.sbatch` as a compute job. Run it once before your experiments
and again whenever the environment changes (fresh checkout, package updates).
The `submit_l*.sh` scripts do not precompile — they will remind you at
submission time.

## Pipeline

### L63

```
                                                                                                                  ┌──afterany──►  posterior_diagnostic_plots_l63
l63_preliminaries  ──afterok──►  calibrate_array  ──afterok──►  emulate_sample_array (per-k bands)  ──afterany──►  merge_posterior  ──afterany──►  pushforward_from_posterior ─┤
                                                                                                                  └──afterany──►  exp_to_leaderboard
```

### L96 (const / vec / flux)

```
                                                    ┌──afterok──►  calibration_diagnostic_plots_l96
l96_preliminaries  ──afterok──►  calibrate_array  ──afterok──►  emulate_sample_array (per-k bands)  ──afterany──►  merge_posterior  ──afterany──►  pushforward_from_posterior  ──afterany──►  posterior_diagnostic_plots_l96
                                                                                                                                  ──afterany──►  exp_to_leaderboard
```

`preliminaries` computes and saves the shared truth-data/observations once,
serially, before any calibrate task starts — this avoids every array task
racing to compute and write the same file (see `l63_preliminaries.jl` /
`l96_preliminaries.jl` in the parent directory). For L96, `calibration_diagnostic_plots`
and `emulate_sample` both start once calibrate succeeds (they run in parallel).
For all cases, `pushforward_from_posterior` starts once `merge_posterior`
finishes and runs the Lorenz forward map for every posterior cell in parallel,
saving forcing and output samples back into each posterior JLD2 file.
`posterior_diagnostic_plots` and `exp_to_leaderboard` both start once
`pushforward_from_posterior` finishes — they load the precomputed samples
rather than re-running the forward map.

### Per-k emulate_sample arrays

`emulate_sample` is parallelised over `(N_ens, rng_idx, k)`, not just
`(N_ens, rng_idx)`: every `k` (number of EKP iterations used as training data)
is independent, and the cost grows with `k`. The cluster caps arrays at 1000
tasks, so `submit_emulate_bands.sh` (sourced by every `submit_l*.sh`) submits
**one array per band of `k`**, each with its own wall-clock request:

| case | `max_iter` | bands (`K_PER_BAND=5`) | array size per band |
|------|-----------|------------------------|---------------------|
| L63 | 10 | k=1–5, 6–10 | 900 |
| L96 | 15 | k=1–5, 6–10, 11–15 | 900 |

Tasks are indexed k-major within a band, i.e. `idx → (N_ens, rng_idx, k)` via
`flat_tasks_k(cfg; ks = K_LO:K_HI)`. The band is passed to the Julia script by
the `K_LO` / `K_HI` env vars (their presence switches on per-k mode); each task
writes `output/<calib_dir>/emu_k/<model>_posterior[_pca]_<case>_k<k>.jld2`.
The number of valid `k` per cell, `K = min(EKP iterations - 2, max_iter)`, is
only known after calibrate, so tasks with `k > K` start, find nothing to do and
exit successfully (wasted but cheap). Tasks whose per-k file already exists are
skipped too, so a band can be resubmitted to fill gaps.

`merge_posterior.sbatch` (one task per cell, runs `../merge_posterior_k.jl`)
then assembles the per-k files into the usual combined posterior file, so
`pushforward_from_posterior`, the diagnostic plots and `exp_to_leaderboard` are
unchanged. A cell is merged only if its per-k files cover `k = 1:K`
contiguously; otherwise it is skipped with a warning and its per-k files are
left in place (rerun the missing `k`, e.g. `K_LO=7 K_HI=7`, then rerun the merge).
Merging is idempotent (an existing combined file is never overwritten, which
protects a pushforward already appended to it); per-k files are deleted after a
successful merge unless `KEEP_K_FILES=1`.

Knobs (env vars, e.g. `K_PER_BAND=3 BAND_TIMES="00:30:00 02:00:00" bash submit_l96_vec.sh`):

| var | default | meaning |
|-----|---------|---------|
| `K_PER_BAND` | 5 | `k` values per array; `K_PER_BAND * N_PER_K` must be ≤ `MAX_ARRAY_SIZE` |
| `BAND_TIMES` | `"01:00:00 02:00:00 03:00:00"` | wall-clock per band, lowest `k` first (last entry reused) |
| `THROTTLE` | 100 | max concurrent tasks per array |
| `N_PER_K` | 180 | `length(N_ens_sizes) * n_repeats`, must match `experiment_config.jl` |
| `MAX_ARRAY_SIZE` | 1000 | cluster array limit |

The default wall-clock values are starting guesses, not measurements: check
`sacct -j <jobid> --format=JobID,Elapsed,MaxRSS` after a first run and adjust
`BAND_TIMES`. Running `emulate_sample_array.sbatch` directly without `K_LO`/`K_HI`
keeps the original per-cell behaviour (all `k` serially, combined file, no merge).

## Standalone (serial, from the parent directory)

Run with no arguments to sweep all `(N_ens, rng_idx)` cells sequentially.
These are the same commands as in the top-level README — HPC and local share
one set of scripts:

```bash
# L63 (run from the calibrate_emulate_sample/ directory)
julia --project=. l63_preliminaries.jl
julia --project=. calibrate_l63.jl
julia --project=. emulate_sample_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. posterior_diagnostic_plots_l63.jl
julia --project=. exp_to_leaderboard.jl

# L96 — set EXPERIMENT env var or edit the toggle in experiment_config.jl
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. calibration_diagnostic_plots_l96.jl
EXPERIMENT=l96_const julia --project=. emulate_sample_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. posterior_diagnostic_plots_l96.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl
```

You can also run a single cell by passing its 1-based task index for the
array-capable scripts (again, from the parent directory):

```bash
julia --project=. calibrate_l63.jl 1                         # first (N_ens, rng_idx) cell only
julia --project=. emulate_sample_l63.jl 5                    # fifth cell only
julia --project=. pushforward_from_posterior_l63.jl 5        # fifth cell only
julia --project=. posterior_diagnostic_plots_l63.jl 5        # fifth cell only
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl 3
EXPERIMENT=l96_const julia --project=. posterior_diagnostic_plots_l96.jl 3
```

`exp_to_leaderboard.jl` runs all cells serially in a single call (requires
pushforward to have been run first). By default it only computes and saves
the `*_minimal.nc` dataset; set `SAVE_FULL_NC=true` to also compute and save
the full `*.nc` dataset:

```bash
julia --project=. exp_to_leaderboard.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl
SAVE_FULL_NC=true julia --project=. exp_to_leaderboard.jl
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
so there are no write conflicts.  The optional `EXP_ID` argument suffixes SLURM
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

Precompile via `submit_precompile.sh` (or directly), then submit each stage.
Pin `CALIBRATE_DATE` once and pass it to every call so all stages agree:

```bash
CALIBRATE_DATE=$(date +%Y-%m-%d)

# L63
PRELIM_JID=$(sbatch --parsable -A esm \
             --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch)
CALIB_JID=$(sbatch --parsable -A esm \
            --dependency=afterok:${PRELIM_JID} --kill-on-invalid-dep=yes \
            --export=ALL,SCRIPT=calibrate_l63.jl,CALIBRATE_DATE=${CALIBRATE_DATE} \
            calibrate_array.sbatch)
EMU_JID=$(sbatch --parsable -A esm \
          --dependency=afterok:${CALIB_JID} --kill-on-invalid-dep=yes \
          --export=ALL,SCRIPT=emulate_sample_l63.jl,CALIBRATE_DATE=${CALIBRATE_DATE} \
          emulate_sample_array.sbatch)
PUSHFWD_JID=$(sbatch --parsable -A esm \
              --dependency=afterany:${EMU_JID} \
              --export=ALL,CALIBRATE_DATE=${CALIBRATE_DATE} \
              pushforward_from_posterior.sbatch)
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,CALIBRATE_DATE=${CALIBRATE_DATE} \
       posterior_diagnostic_plots_l63.sbatch
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l63,CALIBRATE_DATE=${CALIBRATE_DATE} \
       exp_to_leaderboard.sbatch

# L96
PRELIM_JID=$(sbatch --parsable -A esm \
             --export=ALL,SCRIPT=l96_preliminaries.jl,EXPERIMENT=l96_const \
             preliminaries.sbatch)
CALIB_JID=$(sbatch --parsable -A esm \
            --dependency=afterok:${PRELIM_JID} --kill-on-invalid-dep=yes \
            --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const,CALIBRATE_DATE=${CALIBRATE_DATE} \
            calibrate_array.sbatch)
sbatch -A esm \
       --dependency=afterok:${CALIB_JID} --kill-on-invalid-dep=yes \
       --export=ALL,EXPERIMENT=l96_const,CALIBRATE_DATE=${CALIBRATE_DATE} \
       calibration_diagnostic_plots_l96.sbatch
EMU_JID=$(sbatch --parsable -A esm \
          --dependency=afterok:${CALIB_JID} --kill-on-invalid-dep=yes \
          --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const,CALIBRATE_DATE=${CALIBRATE_DATE} \
          emulate_sample_array.sbatch)
PUSHFWD_JID=$(sbatch --parsable -A esm \
              --dependency=afterany:${EMU_JID} \
              --export=ALL,EXPERIMENT=l96_const,CALIBRATE_DATE=${CALIBRATE_DATE} \
              pushforward_from_posterior.sbatch)
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l96_const,CALIBRATE_DATE=${CALIBRATE_DATE} \
       posterior_diagnostic_plots_l96.sbatch
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l96_const,CALIBRATE_DATE=${CALIBRATE_DATE} \
       exp_to_leaderboard.sbatch
```

### Sbatch files reference

| File | Type | Description |
|------|------|-------------|
| `preliminaries.sbatch` | single job | Computes + saves shared truth data/observations once (SCRIPT env var picks l63/l96) |
| `calibrate_array.sbatch` | array (1–180) | One task per `(N_ens, rng_idx)` cell |
| `emulate_sample_array.sbatch` | array (per band of k, ≤1000) | One task per `(N_ens, rng_idx, k)`; submitted per band by `submit_emulate_bands.sh` (see above). Without `K_LO`/`K_HI`: array (1–180), one task per cell, all `k` serially |
| `merge_posterior.sbatch` | array (1–180) | Assembles each cell's per-k files into the combined posterior JLD2 |
| `pushforward_from_posterior.sbatch` | array (1–180) | Posterior pushforward (forcing + output), one task per cell; saves results into the posterior JLD2 |
| `calibration_diagnostic_plots_l96.sbatch` | single job | Calibration figures, all cells serially (L96 only) |
| `posterior_diagnostic_plots_l63.sbatch` | array (1–180) | Posterior ribbon/scatter figures, one task per cell (L63); loads pushforward from JLD2 |
| `posterior_diagnostic_plots_l96.sbatch` | array (1–180) | Posterior ribbon/scatter figures, one task per cell (L96); loads pushforward from JLD2 |
| `exp_to_leaderboard.sbatch` | single job | NetCDF leaderboard file, all cells serially; loads pushforward from JLD2 |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

### Adjusting array size

The sbatch files default to `--array=1-180` (9 ensemble sizes × 20 repeats).
If you change `N_ens_sizes` or `n_repeats` in `experiment_config.jl`, update
the upper bound to `length(N_ens_sizes) * n_repeats` in every array sbatch file,
including `pushforward_from_posterior.sbatch` and `merge_posterior.sbatch`. For the
per-k emulate arrays, also update `N_PER_K` and the per-case `MAX_ITER` in
`submit_emulate_bands.sh` / `submit_l*.sh` (`MAX_ITER` must equal `max_iter` in
`experiment_config.jl`; the band arrays are sized `K_PER_BAND * N_PER_K`).  The `%100` suffix caps concurrent
tasks as a cluster-courtesy limit; raise or remove it if you want faster turnaround.

### Smoke test

Before a full submission, run the preliminaries job followed by a single-task
array to verify the job finds its input files and writes output correctly:

```bash
sbatch --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
```
