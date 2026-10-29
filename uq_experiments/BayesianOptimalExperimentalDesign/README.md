# BayesianOptimalExperimentalDesign — uq experiment

Implements **GBOED** (Goal-Oriented Bayesian Optimal Experimental Design),
Algorithm 2 of Holthuijzen, Chakraborty, Krath, Catanach (2026),
*"Surrogate-based Bayesian calibration methods for chaotic systems: a
comparison of traditional and non-traditional approaches"*
([arXiv:2508.13071](https://arxiv.org/abs/2508.13071)).

An iterative scheme: at each iteration a Gaussian Process surrogate (via
[GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl), the
library's built-in `SEArd` squared-exponential ARD kernel — matching
HistoryMatching's own kernel choice, rather than the paper's fixed-smoothness
Matérn-7/2) is fit on the *cumulative* set of forward-model input/output pairs
seen so far;
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
and `KalmanSampler`) there is no separate "emulate"/"sample"
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
2. Draw `n_posterior_samples` (1000) approximate posterior samples via ST-MCMC
   (`TransitionalMCMC.tmcmc`), using a Gaussian log-likelihood that combines
   the GP's predictive mean/variance with the (whitened, ≈ identity)
   observation covariance.
3. Jointly optimize a new candidate batch of size `N_ens` to maximize
   ```
   EIG(X_cand) = 0.5·log(det(K_{X',X'}) / det(Σ'))
   Σ' = K_{X',X'} − K_{X',X_cand}·K_{X_cand,X_cand}⁻¹·K_{X_cand,X'}
   ```
   where `X'` is a subsample of the posterior draw (ST-MCMC or IEKF, per
   variant) and `K` is the (fixed, currently-fitted) SE-ARD kernel of each
   output mode (the objective is averaged over modes). The batch is optimized
   with `Optim.jl`'s LBFGS and `Zygote` reverse-mode gradients, inside a
   χ² ball (`eig_region = :ball`, via a radial squash) or a `Fminbox` box
   (`:box`).
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

## Variants / ablations (`BOED_VARIANT`)

Six variants (three posterior samplers × with/without EIG) form an ablation grid. All share preliminaries, the GP fit,
pushforward and leaderboard; they differ in the posterior sampler and the
acquisition step inside `calibrate_<MODEL>.jl`. Select with the `BOED_VARIANT`
env var (or `BOED_VARIANT` in `experiment_config.jl`); set it for **every**
stage, since it picks the output directory/netcdf (`method_key`).

| `BOED_VARIANT` | Leaderboard key | Posterior sampler | Acquisition (batch of `N_ens`) |
|---|---|---|---|
| `tmcmc_eig` (default) | `boed-tmcmc-eig` | ST-MCMC | Joint-batch EIG maximization (above) |
| `tmcmc` | `boed-tmcmc` | ST-MCMC | **No EIG:** `N_ens` draws from the ST-MCMC posterior |
| `iekf_eig` | `boed-iekf-eig` | **IEKF on the GP surrogate** (no ST-MCMC) | Joint-batch EIG maximization against (a subsample of) the IEKF ensemble |
| `iekf` | `boed-iekf` | **IEKF on the GP surrogate** (no ST-MCMC) | **No EIG:** `N_ens` draws from the IEKF posterior |
| `eks_eig` | `boed-eks-eig` | **Ensemble Kalman sampler (EKS) on the GP surrogate**, 200 particles | Joint-batch EIG maximization against (a subsample of) the EKS ensemble |
| `eks` | `boed-eks` | **Ensemble Kalman sampler (EKS) on the GP surrogate**, 200 particles | **No EIG:** `N_ens` draws from the EKS posterior |

Hyphens are accepted in the env var (`BOED_VARIANT=tmcmc-eig`). The former
`eig` variant / `boed` key is now `tmcmc_eig` / `boed-tmcmc-eig`.

`tmcmc` isolates how much the EIG optimization buys over simply sampling the
posterior. `iekf` additionally replaces ST-MCMC with a cheap sampler:
`GaussNewtonInversion` (fixed-step `DefaultScheduler(0.1)`, 30 steps, so T = 3; it was 10 steps, T = 1)
run on the GP-*mean* forward map for an ensemble of `n_posterior_samples`
LHS-prior draws, in the truncated prior-whitened input space (prior `N(0, I)`)
against the truncated output-whitened observation (noise `I`). No forward-model
evaluations are used by the sampler. Unlike ST-MCMC's likelihood it ignores the
GP predictive variance. The final IEKF ensemble is the stored per-iteration
"posterior" and the pool the next batch is drawn from. Dials: `iekf_step`,
`iekf_iters` in `experiment_config.jl`. `iekf_eig` feeds that same IEKF
ensemble to the EIG optimizer exactly as `tmcmc_eig` does with the ST-MCMC
samples, isolating the sampler's effect with EIG held fixed.

`eks` / `eks_eig` replace the finite-time IEKF with the ensemble Kalman *sampler*
(`EKP.Sampler(prior; sampler_type = "eks")`, Garbuno-Inigo et al. 2019) on the same GP-mean forward map,
prior, noise and LHS initial ensemble, with `EKSStableScheduler()` (data-dependent step) run until the
scheduler numerator `eks_alpha = 0.25` (offline scans against `tmcmc_long` found smaller steps closer to the reference; `alpha` > 1 is unstable)
for a fixed `eks_max_iters = 500` steps (`eks_T = Inf`; a finite `eks_T` stops earlier at that algorithm time, and `run_eks` warns if
the step cap bites first), recording the algorithm time reached, using `eks_n_particles = 200` members. Unlike the IEKF it targets the posterior as
T → ∞, at ~10× the number of GP-mean evaluations. The EKS posterior has 200 draws (not `n_posterior_samples`), so
pushforward pads its sample set with NaN rows, which the leaderboard stage drops; the EIG subsample is 50 of the 200
and the next batch is drawn from the 200. Diagnostics: `eks_steps`, `eks_T_reached`, `eks_secs`. The offline test2 evidence
for it (`diagnostics/output/test2_sampler_convergence_combined.png`) is on posterior accuracy only.

```bash
BOED_VARIANT=tmcmc julia --project=. calibrate_l63.jl 1
BOED_VARIANT=iekf  julia --project=. calibrate_l63.jl 1
BOED_VARIANT=iekf_eig julia --project=. calibrate_l63.jl 1
BOED_VARIANT=iekf  bash hpc-variant/submit_l63.sh      # HPC: env var is forwarded to all stages
```

## SE-ARD kernel

The paper fixes the GP kernel's smoothness at ν=3.5 (Matérn-7/2); this
implementation instead uses `GaussianProcesses.jl`'s built-in `SEArd`
(squared-exponential ARD) kernel, matching HistoryMatching's own kernel
choice (`history_matching_core.jl`'s `fit_wave_gps`) rather than
transcribing the paper's fixed smoothness — a deliberate departure, traded
for consistency with the rest of this repo's GP-based methods and the
library's own well-tested implementation. A second, standalone SE-ARD
kernel-matrix builder (`se_cov_matrix`) is used only inside the EIG
objective, deliberately decoupled from `GaussianProcesses.jl`'s internals so
its `ForwardDiff` compatibility doesn't depend on library internals not
designed for autodiff.

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
`uq_experiments/KalmanSampler/hpc-variant/`. See
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
