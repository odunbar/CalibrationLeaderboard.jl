# BayesianOptimalExperimentalDesign — uq experiment

Implements **GBOED** (Goal-Oriented Bayesian Optimal Experimental Design),
Algorithm 2 of Holthuijzen, Chakraborty, Krath, Catanach (2026),
*"Surrogate-based Bayesian calibration methods for chaotic systems: a
comparison of traditional and non-traditional approaches"*
([arXiv:2508.13071](https://arxiv.org/abs/2508.13071)).

An iterative scheme: at each iteration a Gaussian Process surrogate (via
[GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl), a
hand-written Matérn-7/2 ARD kernel — see "Matérn-7/2 kernel" below) is fit on
the *cumulative* set of forward-model input/output pairs seen so far;
[TransitionalMCMC.jl](https://github.com/AnderGray/TransitionalMCMC.jl) (an
ST-MCMC/SMC sampler implementing Ching & Chen's 2007 Transitional MCMC — the
paper's own cited ST-MCMC reference) draws approximate posterior samples from
that surrogate fit; a new batch of candidate simulation inputs is then
*jointly* optimized (via `Optim.jl`'s `Fminbox(LBFGS())` with `ForwardDiff.jl`
gradients — the paper used JAX autodiff instead) to maximize a closed-form
Expected Information Gain (EIG) targeted at those posterior samples; and the
optimized batch is forward-evaluated and folded into the training set for the
next iteration's GP fit.

This is the "goal-oriented" variant (GBOED), not "standard BOED": standard
BOED optimizes EIG against a fixed, generic test set of GP-input locations,
while GBOED draws MCMC posterior samples each iteration and targets the EIG
acquisition at those — exactly the loop implemented here.

## How this differs from HistoryMatching

Structurally the closest existing method: a GP is refit every iteration to
decide where to sample the forward model next, and (like `HistoryMatching`
and `GaussNewtonKalmanInversion`) there is no separate "emulate"/"sample"
stage — the GP fit, MCMC posterior sampling, and acquisition-batch
optimization all live inside every iteration of `calibrate_<MODEL>.jl`.

The key difference is *direction*: History Matching *rules out* implausible
regions (rejection sampling against a chi-squared implausibility threshold,
refitting an independent GP per wave on that wave's own ensemble only).
GBOED *optimizes toward* good candidates (gradient-based EIG maximization,
refitting one GP on the full, ever-growing cumulative dataset every
iteration).

Both methods reuse `common/uq_metrics/coverage_metrics.jl`'s generic
truncated-whitening machinery (`WhitenedPCABasis`) to whiten the GP's output
space against the observation covariance `R` and its input space against the
prior covariance, both fixed for the whole cell. GBOED additionally needs to
go the other direction — decoding LHS/ST-MCMC/EIG-optimized candidates in the
truncated whitened space back to raw parameter space before calling the
forward model — so it also uses this file's new `unwhiten_vector`/
`unwhiten_samples` functions (History Matching only ever whitens forward).

## Algorithm summary (`boed_core.jl`)

1. **Iteration 1**: draw `N_ens` candidates via Latin-hypercube sampling in
   the truncated, whitened prior space (PCA-whitening makes the retained
   coordinates' marginal prior exactly standard normal, so LHS is done
   directly in that space, then decoded back to raw parameter space).
   Forward-evaluate; fit one independent GP per whitened output statistic.
2. Draw `n_posterior_samples` approximate posterior samples via ST-MCMC
   (`TransitionalMCMC.tmcmc`), using a Gaussian log-likelihood that combines
   the GP's predictive mean/variance with the (whitened, ≈ identity)
   observation covariance.
3. Jointly optimize a new candidate batch of size `N_ens` to maximize
   ```
   EIG(X_cand) = 0.5·log(det(K_{X',X'}) / det(Σ'))
   Σ' = K_{X',X'} − K_{X',X_cand}·K_{X_cand,X_cand}⁻¹·K_{X_cand,X'}
   ```
   where `X'` is the ST-MCMC posterior draw, `K` is the (fixed,
   currently-fitted) Matérn-7/2 kernel, via `Optim.jl`'s `Fminbox(LBFGS())` +
   `ForwardDiff` autodiff.
4. Forward-evaluate the optimized batch; augment the cumulative dataset;
   refit the GP; repeat from step 2.
5. Continues for `max_iters` iterations (`experiment_config.jl`).

### Iteration-indexing convention

Iteration `k_iter=1` is the initial LHS design (no acquisition yet — mirrors
the paper's setup steps before its own "for k=1..K" loop); iterations
`k_iter=2..max_iters` each add one EIG-optimized acquisition batch. `N_ens`
doubles as both the initial LHS design size (`n0` in the paper) and the
acquisition batch size (`B` in the paper, fixed across iterations, matching
the paper's own experiments where `n0 = B`). This is a deliberate
reinterpretation of Algorithm 2, chosen so that `budget = N_ens × k_iter`
stays identical to every other UQ method's leaderboard convention with zero
changes to `common/uq_metrics/coverage_metrics.jl`'s `budget_to_target` — not
a literal transcription of the paper's own single-final-budget framing.

## Matérn-7/2 kernel

The paper fixes the GP kernel's smoothness at ν=3.5 (=7/2).
`GaussianProcesses.jl`'s own `Matern(ν, ll, lσ)` constructor only supports
ν ∈ {1/2, 3/2, 5/2}, so `boed_core.jl` hand-writes a `Mat72Ard <:
GaussianProcesses.MaternARD` kernel type (closed form:
`k(r) = σ²(1 + √7r + 14r²/5 + 7√7r³/15)·exp(-√7r)`), slotting into the
library's existing ARD chain-rule machinery the same way its own built-in
`Mat52Ard` does. A second, standalone Matérn-7/2 kernel-matrix builder
(`matern72_cov_matrix`) is used only inside the EIG objective, deliberately
decoupled from `GaussianProcesses.jl`'s internals so its `ForwardDiff`
compatibility doesn't depend on library internals not designed for autodiff.

## One-time setup

In `experiment_config.jl`, pin the calibrate date before starting a run:
```julia
calibrate_date = Date("<YYYY-MM-DD>", "yyyy-mm-dd")
```

## Pipeline

### L63
```
l63_preliminaries → calibrate_l63 (GBOED loop, incl. ST-MCMC posterior draw) → pushforward_from_posterior_l63 → exp_to_leaderboard
```

### L96 (const / vec / flux)
```
l96_preliminaries → calibrate_l96 (GBOED loop, incl. ST-MCMC posterior draw) → pushforward_from_posterior_l96 → exp_to_leaderboard
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

Wired up in `hpc-variant/`, following the same 3-stage pipeline
(preliminaries → calibrate → pushforward → leaderboard) as
`uq_experiments/HistoryMatching/hpc-variant/` and
`uq_experiments/GaussNewtonKalmanInversion/hpc-variant/`. See
`hpc-variant/README.md` for submission scripts and the full sbatch reference.

## Packages this method introduces to the repo

- [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) — `Fminbox(LBFGS())`
  for the joint-batch EIG optimization.
- [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) — autodiff
  gradients for the EIG objective.
- [`TransitionalMCMC.jl`](https://github.com/AnderGray/TransitionalMCMC.jl) —
  ST-MCMC posterior sampling from the GP surrogate.

`GaussianProcesses.jl` and `PDMats.jl` are reused from `HistoryMatching`
(including its `GaussianProcesses.jl`/`PDMats.jl` `ldiv!` ambiguity fix,
needed for the same reason here).

## Shared code

`common/uq_metrics/coverage_metrics.jl` holds the generic truncated-whitening
math (`WhitenedPCABasis`/`whitened_pca_basis`/`whiten_vector`/
`whiten_samples`, plus the new `unwhiten_vector`/`unwhiten_samples` this
method added), marginal coverage, and per-quantile budget-to-target. Each
method's netcdf schema stays local, since what "k_iter" means differs per
method.

## Leaderboard metric

Primary metric: output-space coverage (R-whitened PCA) at marginal
quantiles, computed at each GBOED iteration `k_iter`. Budget metric: smallest
`N_ens × k_iter` (forward-model evaluations) to achieve calibrated coverage.
