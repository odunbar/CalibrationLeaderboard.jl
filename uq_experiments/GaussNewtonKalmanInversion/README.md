# GaussNewtonKalmanInversion — uq experiment

Uses the `GaussNewtonInversion` (GNKI) process from `EnsembleKalmanProcesses.jl`
— an iterated ensemble Kalman filter with statistical linearization that
converges (in pseudotime) towards the exact Bayesian posterior. Unlike
`uq_experiments/calibrate_emulate_sample`, there is **no emulate_sample
stage**: the raw ensemble at each GNKI iteration is itself treated as the
posterior estimate, keeping the pipeline to three stages (calibrate →
pushforward → leaderboard). `post_mean`/`post_cov` come directly from this
raw `N_ens`-member ensemble. For the output-space coverage metric, though,
`N_ens` alone (as low as 4 for L63) makes quantile estimates noisy, so
`pushforward_from_posterior_l*.jl` instead fits a Gaussian to the ensemble
(in unconstrained space) and resamples a fixed, larger number of points from
it (`n_pushforward_samples = 1000`, matching `calibrate_emulate_sample`)
before pushing forward — no fitted emulator or MCMC chain involved, just a
moment-matched Gaussian resample of the ensemble itself.

## One-time setup

Before calibrating, compute the shared truth-data preliminaries once (this is
extracted out of `calibrate_l63.jl`/`calibrate_l96.jl` to avoid every SLURM
array task racing to compute and write the same file):
```bash
julia --project=. l63_preliminaries.jl
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
```

`calibrate_date` in `experiment_config.jl` is set via the `CALIBRATE_DATE` env
var (falls back to `today()` for local runs); `hpc-variant/submit_*.sh` fixes
it once per submission so every stage agrees on the same output directory.

## Pipeline

### L63
```
l63_preliminaries  ─(afterok)→  calibrate_array  ─(afterany)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```

### L96 (const / vec / flux)
```
l96_preliminaries  ─(afterok)→  calibrate_array  ─(afterany)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```

## Standalone (serial / local)

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

# Single cell:
julia --project=. calibrate_l63.jl 1
EXPERIMENT=l96_vec julia --project=. calibrate_l96.jl 5
```

## HPC (Caltech Resnick cluster, SLURM)

`hpc-variant/` holds only sbatch and submit scripts — there is a single copy
of every `.jl` script and of `experiment_config.jl` (this one, in the
directory you're reading now), shared verbatim between local and HPC runs.
Each sbatch job `cd`s into `hpc-variant/` and invokes `julia --project=..
"../${SCRIPT}"`, so the script's own `@__DIR__` still resolves here — same
config, same `common/` paths, same `output/` tree as a local run. See
`hpc-variant/README.md` for the full dependency graph, sbatch/submit script
reference, and manual submission examples. Quick start:

```bash
cd hpc-variant
bash submit_precompile.sh                    # once, or after any package update
bash submit_l63.sh
bash submit_l96_const.sh
bash submit_l96_vec.sh
bash submit_l96_flux.sh
```

Array upper bound = `length(N_ens_sizes) * n_repeats` = 9 x 20 = 180 (all four
cases). If either changes in `experiment_config.jl`, update `--array` in
`hpc-variant/calibrate_array.sbatch` and
`hpc-variant/pushforward_from_posterior.sbatch` — there's only the one config
file to edit.

## Leaderboard metric

For each `(N_ens, rng_idx, k)` cell:
- `post_mean`, `post_cov` — mean and covariance of the raw `N_ens` ensemble in
  parameter (constrained) space.
- `output_coverage` — R-whitened PCA coverage in output space (see
  `exp_to_leaderboard.jl` header), computed from `n_pushforward_samples = 1000`
  points resampled from the Gaussian implied by the ensemble's own mean/cov
  (fit in unconstrained space), not from the raw `N_ens` ensemble directly.
  This reduces quantile-estimation noise relative to using `N_ens` samples,
  but does not reduce the underlying estimation error of the ensemble
  mean/cov itself, and assumes the ensemble is well approximated by a
  Gaussian.
- `output_budget_to_target` / `output_iters_to_target` — smallest
  `N_ens × k` (forward-model evaluations) to reach calibrated coverage, per
  quantile and per tolerance scaling `c`.
