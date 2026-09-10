# Approximate Bayesian Computation — uq experiment

Simulation-based inference via ABC rejection sampling: draw i.i.d. candidates
from the prior, forward-evaluate each through the Lorenz model, and accept a
candidate iff its implausibility against the target summary statistic falls
below a Chi-squared threshold. Accepted candidates accumulate into a growing
posterior sample pool.

Like `uq_experiments/GaussNewtonKalmanInversion`, there is no emulate_sample
stage — the accepted-sample pool at each round IS the posterior estimate. No
surrogate/emulator is fit anywhere: every accept/reject decision and every
pushforward sample is a direct Lorenz forward-model evaluation.

## Method details

- **Rejection criterion**: accept `θ` iff `(G(θ) - y)ᵀ(2R)⁻¹(G(θ) - y) ≤
  χ²_{ny}(1 - α)`, with `α = alpha_reject = 0.05`. `y`/`R` come from
  `l63_preliminaries.jl` / `l96_preliminaries.jl`.
- **Prior**: shared with `GaussNewtonKalmanInversion` / `calibrate_emulate_sample`
  for a fair leaderboard comparison.
- **Ensemble size ≡ batch size**: `N_ens` is the ABC batch size, reused as the
  leaderboard's `ensemble_size` axis so `N_ens · k_iter` (forward-model
  evaluations) is comparable to EKI-based methods. Since ABC accepts/rejects
  each draw independently, `calibrate_l63.jl`/`calibrate_l96.jl` draw one flat
  stream of `N_ens_max · N_iter` candidates per `rng_idx`, and for each
  `N_ens` take the accepted pool among its first `N_ens · k` draws as round `k`.
- **k_iter**: `phi_stored[k]` holds every accepted sample from rounds `1..k`.
- **Pushforward**: `pushforward_from_posterior_l*.jl` fits a Gaussian to the
  pool (unconstrained space) and resamples `n_pushforward_samples = 1000`
  points from it, skipping rounds before the pool reaches `nu + 2` members.

## One-time setup

Compute the shared truth-data preliminaries once:
```bash
julia --project=. l63_preliminaries.jl
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_vec julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_flux julia --project=. l96_preliminaries.jl
```

`calibrate_date` in `experiment_config.jl` is set via `CALIBRATE_DATE` (falls
back to `today()`).

## Pipeline

### L63
```
l63_preliminaries → calibrate → pushforward_from_posterior → exp_to_leaderboard
```

### L96 (const / vec / flux)
```
l96_preliminaries → calibrate → pushforward_from_posterior → exp_to_leaderboard
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

# Single cell: calibrate indexes by rng_idx; pushforward/leaderboard index by
# flat_tasks(cfg) position, i.e. (N_ens, rng_idx)
julia --project=. calibrate_l63.jl 1
EXPERIMENT=l96_vec julia --project=. calibrate_l96.jl 5
```

No HPC/SLURM variant has been scaffolded yet; add one with the
`slurm-pipeline-handler` skill when needed.

## Leaderboard metric

For each `(N_ens, rng_idx, k)` cell:
- `post_mean`, `post_cov` — mean/covariance of the accepted pool in parameter
  space at round `k`.
- `pool_size` — number of accepted samples in the pool at round `k`.
- `output_coverage` — R-whitened PCA coverage in output space, from
  `n_pushforward_samples = 1000` Gaussian-resampled points.
- `output_budget_to_target` / `output_iters_to_target` — smallest `N_ens × k`
  / `k` to reach calibrated coverage, per quantile and tolerance scaling `c`.
