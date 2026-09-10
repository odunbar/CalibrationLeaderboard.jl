# Approximate Bayesian Computation HPC variant

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

There is no separate emulate_sample stage — the accepted-sample pool at each
ABC round is itself the UQ sample set (see the top-level README).

## Task indexing differs by stage

- **`calibrate_array.sbatch`** indexes by `rng_idx` alone
  (`1:cfg.n_repeats` = 20). `calibrate_l63.jl`/`calibrate_l96.jl` draw one
  flat i.i.d. candidate stream per `rng_idx` and reconstruct every
  `N_ens_sizes` entry from prefixes of it, so one task computes all 9
  `N_ens` values for its seed — see the header comments in either calibrate
  script.
- **`pushforward_from_posterior.sbatch`** and **`exp_to_leaderboard.sbatch`**
  still index by `flat_tasks(cfg)`, i.e. `(N_ens, rng_idx)` pairs (180 tasks),
  unchanged from other UQ methods in this repo.

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
serially, before any calibrate task starts. `calibrate_array` runs ABC
rejection sampling for every `rng_idx`, writing one results file per
`(N_ens, rng_idx)` cell. `pushforward_from_posterior` fits a Gaussian to each
cell's accepted-sample pool at every stored round and pushes
`n_pushforward_samples = 1000` resampled points through the Lorenz forward
map. `exp_to_leaderboard` loads all cells serially and writes the leaderboard
NetCDF.

## Standalone (serial, from the parent directory)

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
array-capable scripts — remember calibrate's index means `rng_idx`, while
pushforward's means a `flat_tasks(cfg)` position:

```bash
julia --project=. calibrate_l63.jl 1                         # rng_idx=1, all N_ens
julia --project=. pushforward_from_posterior_l63.jl 1         # first (N_ens, rng_idx) cell only
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
| `calibrate_array.sbatch` | array (1–20) | One task per `rng_idx`; computes every `N_ens_sizes` entry from one flat draw stream (SCRIPT env var) |
| `pushforward_from_posterior.sbatch` | array (1–180) | Posterior pushforward, one task per `(N_ens, rng_idx)` cell; saves results into the results JLD2 |
| `exp_to_leaderboard.sbatch` | single job | NetCDF leaderboard file, all cells serially |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

All SLURM logs and output data are written under `../output/` (the same
directory local runs use) — `../output/slurm/` for logs, `../output/abc_<date>/`
for calibrate results, matching the top-level README's paths exactly.

### Per-task cost is much higher here than in other UQ methods

Because `N_iter`/`max_iter` were raised so `N_ens_max * N_iter ~ 10,000`
(see the top-level README), each `calibrate_array` task now forward-evaluates
~10,000 candidates (vs. GNKI's per-`N_ens` workload), and each
`pushforward_from_posterior` task pushes 1000 samples through the forward map
at up to `N_iter` rounds (500 for l63/l96_const, 111 for l96_vec/l96_flux) —
far more than GNKI's ~10-15 rounds. The `--time` values in the sbatch files
are placeholders; run the smoke test below on one cell first and adjust
`--time`/`--mem` to the observed per-task wall-clock before submitting the
full array.

Both hot loops (`calibrate`'s candidate loop, `pushforward`'s per-round sample
loop) are `Threads.@threads`-parallel with an independent RNG per draw, so
results don't depend on thread count. `calibrate_array.sbatch`/
`pushforward_from_posterior.sbatch` request `--cpus-per-task=16` and set
`OPENBLAS_NUM_THREADS=1` (avoids oversubscription now that Julia threads do
the parallel work) — raise/lower `--cpus-per-task` to match node core counts.

### Adjusting array size

- `calibrate_array.sbatch`: `--array` upper bound = `n_repeats` (20). Update
  if `n_repeats` changes in `experiment_config.jl`.
- `pushforward_from_posterior.sbatch`: `--array` upper bound =
  `length(N_ens_sizes) * n_repeats` (9 x 20 = 180). Update if either changes.

The `%100` suffix on `pushforward_from_posterior.sbatch` caps concurrent tasks
as a cluster-courtesy limit; raise or remove it if you want faster turnaround.

### Smoke test

Before a full submission, run a single-task array to verify the job finds its
input files and writes output correctly (after the preliminaries file exists),
and to measure actual per-task wall-clock:

```bash
sbatch --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
```
