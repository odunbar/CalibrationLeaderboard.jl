# score_based_adam — Adam with a GFDT / score-function Jacobian

Adam (`opt_experiments/adam`) on `(y - G(θ))' R⁻¹ (y - G(θ))`, but the Jacobian
comes from the **generalized fluctuation-dissipation theorem** with a
**learned score** instead of from ForwardDiff — the same estimator used by
`opt_experiments/score_based_lm`, reused verbatim here, only the update rule
that consumes the Jacobian changes (Adam instead of Levenberg-Marquardt).

```
∂⟨A⟩/∂θ_j = ∫₀^∞ ⟨ δA(x(τ)) · δB_j(x(0)) ⟩ dτ
B_j(x)    = −[ ∇·(∂f/∂θ_j)(x) + (∂f/∂θ_j)(x) · s(x) ],    s(x) = ∇ log p_s(x)
```

One unperturbed trajectory gives the whole `ny × nu` Jacobian, so the
per-iteration cost is **independent of `nu`** — exactly as in `score_based_lm`.

## Why not ForwardDiff, and why Adam instead of Levenberg-Marquardt

The Jacobian rationale (naive AD blows up on a chaotic trajectory like
`e^{λ₁t}`, GFDT estimates the response of the invariant measure instead, and
there is an AD fallback where the attractor collapses to a fixed point) is
identical to `score_based_lm` — see that experiment's `README.md` and
`VALIDATION.md` for the measured numbers and the current gate status.

**This directory only changes the parameter-update rule**, to compare the
score-based Jacobian against a *second* optimizer, not just against LM.
`opt_experiments/adam` already showed how a derivative-based method degrades
gracefully to IC-averaged noisy gradients; this experiment asks the same
question when the "derivative" itself is a statistical estimate rather than an
exact Jacobian.

One consequence is worth flagging: **Adam has no gain-ratio / trust-region
signal**. In `score_based_lm`, a low LM gain ratio ρ means either "step too
long" or "the score was poor this iteration," and shrinking the trust region
is the right response to both. A fixed-`α` Adam step has no such backstop — a
bad-iteration score (e.g. a Stein defect that is large but still under
`stein_recalibrate`'s `max_defect` gate) feeds directly into the momentum
accumulators with no damping. This is a known risk, not yet quantified here;
treat leaderboard numbers from this directory with the same caution
`score_based_lm/README.md` and `VALIDATION.md` already document for the raw
Jacobian estimator (in particular: **the L63 Gate 3 check does not currently
pass** for the DSM/KGMM scores).

## Layout

| File | Role |
|---|---|
| `experiment_config.jl` | config, filename builders, array dispatch, arm selection (score kind, budget mode) — GFDT hyperparameters unchanged from `score_based_lm`, plus `adam_alpha/beta1/beta2/eps` |
| `l63_preliminaries.jl`, `l96_preliminaries.jl` | shared truth data — **byte-identical to `levenberg_marquardt`'s / `score_based_lm`'s**, so leaderboards are directly comparable |
| `score_gfdt.jl` | `ScoreModel` interface, DSM training, KGMM (k-means + Gaussian mixture) fitting, Stein recalibration, correlation integral — copied verbatim from `score_based_lm` (model-agnostic, optimizer-agnostic) |
| `score_nets.jl` | MLP (L63) and circular 1-D U-Net (L96) for the DSM arm, plus the `make_score_model`/`refresh_score!` lifecycle — copied verbatim |
| `gfdt_l63.jl`, `gfdt_l96.jl`, `l96_problem.jl` | raw moments `A(x)`, `G = φ(m)`, `∂φ/∂m`, conjugate variables `B_j` — copied verbatim |
| `run_l63_sbadam.jl`, `run_l96_sbadam.jl` | the Adam loop, consuming the GFDT/score Jacobian in place of `ForwardDiff.jacobian` |
| `run_to_leaderboard.jl` | per-cell JLD2 → leaderboard netcdf |

`gfdt_l63.jl` and `gfdt_l96.jl` must be separate: `common/forward_maps/Lorenz63.jl`
and `Lorenz96.jl` define clashing `LorenzConfig` / `EnsembleMemberConfig` /
`ObservationConfig`, so they can never be loaded together.

## Arms

Two independent switches, each combination a separate submission and netcdf:

| `SCORE_KIND` | `BUDGET_MODE` | `algorithm_type` |
|---|---|---|
| `dsm` | `serial` | `Score-based Adam (GFDT, DSM) (extended window)` |
| `dsm` | `parallel` | `Score-based Adam (GFDT, DSM) (parallel branches)` |
| `kgmm` | `serial` | `Score-based Adam (GFDT, KGMM) (extended window)` |
| `kgmm` | `parallel` | `Score-based Adam (GFDT, KGMM) (parallel branches)` |
| `gaussian` | `serial` | `Quasi-Gaussian FDT Adam (extended window)` |
| `gaussian` | `parallel` | `Quasi-Gaussian FDT Adam (parallel branches)` |

`dsm` is the single-sigma denoising-score-matching neural net (a training loop,
warm-started across outer iterations). `kgmm` is a k-means Gaussian-mixture
score estimator (arXiv:2503.18054) — no training loop, just a k-means fit plus
`K` per-cluster covariance estimates each iteration. `gaussian` is the
quasi-Gaussian FDT baseline, `s(x) = -C⁻¹(x-μ)`.

**Budget modes.** Both spend `N_ens` forward-model evaluations per outer
iteration and integrate the same total model time, `T_start` once + `N_ens·W`;
they differ only in how that `N_ens·W` is arranged:

- `:serial` — one continuous window `N_ens` times longer than the base window.
  Inherently sequential, but the whole span is one attractor sample, giving the
  longest usable lags in the correlation integral for a given cost.
- `:parallel` — spin up ONCE per outer iteration, then fork `N_ens` independent
  base-length branches from small (`ic_cov_sqrt`) perturbations of that
  on-attractor state. Mutually independent, so they run concurrently via
  `Threads.@threads` (needs `JULIA_NUM_THREADS` > 1 to actually parallelize).

At `N_ens = 1` the two modes are identical.

## Cost metric

```
conv_score = outer_iter * N_ens          # vs opt_experiments/adam's * N_ens * (nu + 1)
```

No trial-step evaluation is counted (there is none — Adam takes the step
unconditionally, unlike LM's gain-ratio accept/reject). `n_fwd_actual` is
stored alongside as an audit counter measuring integration time in base-run
equivalents; score-network training time is real compute this metric does not
capture.

## Running

```bash
julia --project=. l63_preliminaries.jl
for c in l96_const l96_vec l96_flux; do EXPERIMENT=$c julia --project=. l96_preliminaries.jl; done

# one cell / all cells
julia --project=. run_l63_sbadam.jl 1
EXPERIMENT=l96_const julia --project=. run_l96_sbadam.jl
julia --project=. run_to_leaderboard.jl

# arms
SCORE_KIND=dsm  BUDGET_MODE=parallel julia --project=. run_l63_sbadam.jl
SCORE_KIND=kgmm BUDGET_MODE=serial   julia --project=. run_l63_sbadam.jl
```

## HPC

`preliminaries →(afterok)→ run_array →(afterany)→ leaderboard`; precompile is
separate and never submitted by `submit_l*.sh`. Cloned from
`score_based_lm/hpc-variant/` (which has the dependency chain and the
`--cpus-per-task`/`JULIA_NUM_THREADS` wiring the `:parallel` budget mode needs)
rather than `adam/hpc-variant/` (which lacks the preliminaries stage, since
plain `adam` never parallelizes internally).

```bash
cd hpc-variant
bash submit_precompile.sh                       # once, after a checkout or package update
bash submit_l63.sh                              # defaults to SCORE_KIND=dsm BUDGET_MODE=parallel
SCORE_KIND=dsm  BUDGET_MODE=serial   bash submit_l63.sh
SCORE_KIND=kgmm BUDGET_MODE=parallel bash submit_l63.sh
SCORE_KIND=gaussian bash submit_l96_const.sh
```

`N_TASKS = length(N_ens_sizes) * length(rmse_targets) * n_repeats = 10*3*30 = 900` (all cases).
If any of those change in `experiment_config.jl`, update `--array` in
`run_array.sbatch` **and** in every `submit_l*.sh`.

`run_array.sbatch` uses `--time=8:00:00 --mem=16G --cpus-per-task=4`, matching
`score_based_lm` (score-network training/fitting dominates wall-clock, same as
there; `:parallel` runs its `N_ens` branches as `Threads.@threads`-parallel,
with `--cpus-per-task=4` passed through to `JULIA_NUM_THREADS`). Resubmit with
a longer `--time` if `l96_flux` DSM cells run long, per `score_based_lm`'s
notes on that arm.
