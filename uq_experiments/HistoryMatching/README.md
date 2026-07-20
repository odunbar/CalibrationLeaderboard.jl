# HistoryMatching — uq experiment

Bayesian History Matching (HM): an iterative "wave" scheme. At each wave, an
ensemble of parameters is forward-simulated, one independent Gaussian Process
(via [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl),
ARD squared-exponential kernel) is fit per output statistic, and a
chi-squared *implausibility* test against every wave's GPs fitted so far
defines the Not-Ruled-Out-Yet (NROY) region that the next wave's ensemble is
rejection-sampled from.

Both the GP's output and input spaces are truncated-whitened before fitting
(`history_matching_core.jl`'s `HMProblem`), reusing
`common/uq_metrics/coverage_metrics.jl`'s generic whitening machinery — the
same math already used there for the leaderboard coverage metric:
- **Output space**: always whitened against the observation covariance `R`,
  fixed for the whole cell and known before any wave runs. This also
  simplifies implausibility to a plain weighted sum (no matrix solve), since
  after whitening `R`'s contribution is exactly the identity.
- **Input space**: always whitened against the *prior* covariance, likewise
  fixed for the whole cell and known before any data exists.
- **`l96_flux` only**: an *additional* wave-local PCA on top of the
  prior-whitened coordinates, fit fresh from that wave's own ensemble — the
  one thing that can't be known before seeing forward evaluations (see
  "l96_flux caveat" below).

Translated from
`Dropbox/Caltech/RobRebecca/Calibration-Race-Bayesian-methods/calib_race_hm_l63.py`
(Python/JAX/gpjax), which ran this algorithm as an **optimization race**
(measuring `waves × N_ens` forward evaluations to hit an RMSE target). This
port drops that framing entirely: each wave becomes a `k_iter` checkpoint on
the **UQ leaderboard**, and the wave's NROY sample set is the "posterior"
that gets pushed forward and scored on output-space coverage — the same
convention used by `calibrate_emulate_sample` and `GaussNewtonKalmanInversion`.

## Algorithm summary

1. **Wave 1**: draw `N_ens` parameters via Latin-hypercube sampling from the
   prior (generalizes the notebook's independent-lognormal-margins LHS to
   arbitrary Gaussian priors — needed for `l96_vec`'s correlated 40-D prior
   and `l96_flux`'s 61-D prior).
2. Forward-evaluate the ensemble; fit one GP per output statistic on
   `(theta_ens, results)`.
3. **Later waves**: rejection-sample from the prior, keeping only points
   whose implausibility is below a chi-squared threshold under *every*
   wave's GPs fitted so far, until either `~1000` samples are found or
   `max_rejection_samples` candidates have been tried (see
   `experiment_config.jl`). This one batch serves double duty: it's stored
   as wave `k`'s "posterior" (pushed forward and coverage-scored exactly like
   every other UQ method's posterior), and its first `N_ens` columns become
   the next wave's training ensemble — fit a fresh set of GPs on that
   ensemble; repeat.
4. Continues for `max_waves` waves, or fewer if a wave's NROY batch comes up
   short of `N_ens` samples (recorded as `n_waves_completed < max_waves`).

There is no separate "emulate" or "sample" stage: emulation (the GP fit)
happens inside every wave of `calibrate_<MODEL>.jl`, not in a distinct step,
and the "posterior at wave k" / "training ensemble for wave k+1" are the same
NROY draw rather than two separate samples — so `calibrate_<MODEL>.jl` goes
straight to `pushforward_from_posterior_<MODEL>.jl` (the same reasoning
`GaussNewtonKalmanInversion` uses to justify merging its own stages).

## History Matching is fundamentally curse-of-dimensionality-limited here

`rejection_sample_nroy`'s candidate-and-filter approach (matching
`calib_race_hm_l63.py`'s own `make_nroy_sampler`) is efficient for `l63`'s 2-D
and `l96_const`'s 1-D theta, but its acceptance rate collapses fast as
dimension grows: a smoke test of `l96_vec` (40-D theta) measured roughly a
1-in-10,000 acceptance rate against just the wave-1 GPs, and stayed at
essentially zero even after adding prior-based input whitening with a
deliberately low `retain_var_input=0.9` (26 effective dimensions instead of
40). This is expected, not a bug: naive rejection sampling's acceptance rate
falls off exponentially in the *effective* dimension, so even a real,
correctly-implemented ~35% dimension cut doesn't meaningfully move the
needle — this is exactly the curse-of-dimensionality limitation History
Matching is well known for, unlike ensemble/gradient-based approaches. Later
waves for `l96_vec` / `l96_flux` routinely exhaust `max_rejection_samples`
and the wave loop stops early (recorded as `n_waves_completed < max_waves`,
NaN-padded by `exp_to_leaderboard.jl`). This is left as-is rather than
"fixed" — the point of this port is a sensible, correctly-implemented
translation of the algorithm, not working around a limitation that's
intrinsic to the method itself. A production-grade implementation
targeting genuinely high-dimensional problems would replace naive rejection
with an NROY-region-aware sampler (e.g. MCMC confined to the NROY region, or
an ellipsoidal/simplex approximation of it).

## l96_flux caveat

`l96_flux`'s parameter vector is the flattened weight vector of a small NN
(`Flux.destructure`), which is non-identifiable under hidden-unit permutation
and `tanh` sign-flip symmetries — a property of the *forward map*, not of the
prior, so prior-based whitening alone can't reveal it (and separately,
`N_ens` starts below 61, so a covariance estimated from the raw ensemble
would be rank-deficient anyway). For this case only, `fit_wave_gps` is called
with `ensemble_retain_var = cfg.retain_var_input`, which fits an *additional* PCA
on top of the prior-whitened coordinates from that wave's own ensemble, and
un-projects consistently at prediction time (`history_matching_core.jl`).
This ensemble-level PCA is always computed fresh from a single wave's own
data — never accumulated across waves — so it never uses information a later
wave wouldn't have had at fit time either.

## One-time setup

In `experiment_config.jl`, pin the calibrate date before starting a run:
```julia
calibrate_date = Date("<YYYY-MM-DD>", "yyyy-mm-dd")
```

## Pipeline

### L63
```
l63_preliminaries → calibrate_l63 (wave loop, incl. NROY posterior draw) → pushforward_from_posterior_l63 → exp_to_leaderboard
```

### L96 (const / vec / flux)
```
l96_preliminaries → calibrate_l96 (wave loop, incl. NROY posterior draw) → pushforward_from_posterior_l96 → exp_to_leaderboard
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

Not yet set up for this method — see the `slurm-pipeline-handler` skill to
add `hpc-variant/` sbatch/submit scripts, following the pattern in
`uq_experiments/calibrate_emulate_sample/hpc-variant/` and
`uq_experiments/GaussNewtonKalmanInversion/hpc-variant/`.

## Shared code

`common/uq_metrics/coverage_metrics.jl` holds the generic truncated-whitening
math (`WhitenedPCABasis`/`whitened_pca_basis`/`whiten_vector`/
`whiten_samples`), marginal coverage, and per-quantile budget-to-target —
extracted here since `exp_to_leaderboard.jl`'s R-whitened PCA coverage
computation was the third independent copy of the same computation (after
`calibrate_emulate_sample` and `GaussNewtonKalmanInversion`, which still have
their own inline copies). Each method's netcdf schema stays local, since
what "k_iter" means differs per method. `history_matching_core.jl` also
reuses the same whitening functions for GP input/output preprocessing (see
the top of this README) — it's generic linear algebra, not leaderboard-metric
-specific.

## Leaderboard metric

Primary metric: output-space coverage (R-whitened PCA) at marginal
quantiles, computed at each History Matching wave `k`.
Budget metric: smallest `N_ens × wave` (forward-model evaluations) to achieve
calibrated coverage.
