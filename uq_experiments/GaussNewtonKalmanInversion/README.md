# GaussNewtonKalmanInversion — uq experiment

Uses the `GaussNewtonInversion` (GNKI) process from `EnsembleKalmanProcesses.jl`
— an iterated ensemble Kalman filter with statistical linearization that
converges (in pseudotime) towards the exact Bayesian posterior. Unlike
`uq_experiments/calibrate_emulate_sample`, there is **no emulate_sample
stage**: the raw ensemble at each GNKI iteration is itself treated as the set
of UQ samples and is pushed forward directly. This keeps the pipeline to three
stages (calibrate → pushforward → leaderboard) and makes the sample count at
iteration `k` equal to `N_ens` (no artificial upsampling via a fitted
emulator/MCMC chain).

## One-time setup

In `experiment_config.jl`, pin the calibrate date before starting a run:
```julia
calibrate_date = Date("<YYYY-MM-DD>", "yyyy-mm-dd")
```

## Pipeline

### L63
```
calibrate_array  ─(afterok)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```

### L96 (const / vec / flux)
```
calibrate_array  ─(afterok)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```

## Standalone (serial / local)

```bash
# L63
julia --project=. calibrate_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. exp_to_leaderboard.jl

# L96 — set EXPERIMENT env var
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl

# Single cell:
julia --project=. calibrate_l63.jl 1
EXPERIMENT=l96_vec julia --project=. calibrate_l96.jl 5
```

## HPC (Caltech Resnick cluster, SLURM)

Use the `slurm-pipeline-handler` skill to add sbatch and submit scripts.

Array upper bound = `length(N_ens_sizes) * n_repeats`.
Update in every array sbatch file when either changes in `experiment_config.jl`.

## Leaderboard metric

For each `(N_ens, rng_idx, k)` cell:
- `post_mean`, `post_cov` — mean and covariance of the raw ensemble in
  parameter (constrained) space, no emulator/MCMC involved.
- `output_coverage` — marginal coverage fraction in output space, computed
  directly from the `N_ens` ensemble-pushforward samples at iteration `k`
  (no upsampling).
- `output_budget_to_target` / `output_iters_to_target` — smallest
  `N_ens × k` (forward-model evaluations) to reach calibrated coverage, per
  quantile and per tolerance scaling `c`.

Because the sample count equals `N_ens` (as low as 4 for L63), coverage
estimates are noisier at small ensemble sizes than the emulator/MCMC-based
methods in `calibrate_emulate_sample` — this is an inherent property of
using the raw ensemble as the posterior surrogate, not a bug.
