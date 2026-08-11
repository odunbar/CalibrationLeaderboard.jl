# score_based_lm — Levenberg-Marquardt with a GFDT / score-function Jacobian

Levenberg-Marquardt on `(y - G(θ))' R⁻¹ (y - G(θ))`, but the Jacobian comes from
the **generalized fluctuation-dissipation theorem** with a **learned score**
instead of from ForwardDiff.

```
∂⟨A⟩/∂θ_j = ∫₀^∞ ⟨ δA(x(τ)) · δB_j(x(0)) ⟩ dτ
B_j(x)    = −[ ∇·(∂f/∂θ_j)(x) + (∂f/∂θ_j)(x) · s(x) ],    s(x) = ∇ log p_s(x)
```

One unperturbed trajectory gives the whole `ny × nu` Jacobian, so the
per-iteration cost is **independent of `nu`**.

## Why not ForwardDiff

The motivation is not only cost. Naive AD through a chaotic trajectory does not
converge to the derivative of a long-time average — the tangent-linear solution
grows like `e^{λ₁t}`. Measured for this experiment's exact L63 configuration
(`dt=0.01`, `T=40`, statistics window `[30,40]`):

```
t= 5.0  |dx/dρ| = 1.9e2        t=30.0  |dx/dρ| = 6.9e11
t=10.0  |dx/dρ| = 1.2e5        t=35.0  |dx/dρ| = 2.6e14
t=20.0  |dx/dρ| = 5.0e7        t=40.0  |dx/dρ| = 4.5e15
                               (Lyapunov prediction e^{0.906·40} = 5.5e15)
```

`‖J_AD‖` over 10 IC draws at θ_true: min 7.4e14, median 4.2e16, max 4.5e17 —
while the true statistical response has `‖J‖ ≈ 2e2`. GFDT estimates the response
of the invariant measure instead, without differentiating through the flow.

## Hybrid: FDT where the measure is smooth, AD where the tangent linear is stable

Roughly **30% of draws from this experiment's L63 prior are not chaotic** — large
β collapses the system onto a fixed point. There the invariant measure is a point
mass, `∇ log p_s` does not exist, and GFDT returns a large *meaningless* Jacobian
(measured `‖J‖ = 9.7e4` against a true response of `≈36`). ForwardDiff, by
contrast, is perfectly well-behaved on exactly those parameters, because with no
chaos there is no tangent-linear blow-up (`‖J_AD‖ ≈ 30`).

So the motivation inverts by region, and the run scripts dispatch on it:

```julia
if any(is_collapsed_window, Xs)      # min(var) < 1e-2; the regimes are 4 orders apart
    J = ForwardDiff.jacobian(...)    # charged N_ens * nu extra, matching LM's (nu+1)
else
    J = gfdt_jacobian(...)           # charged nothing extra
end
```

`conv_score` is therefore **accumulated per iteration** rather than
`outer_iter * N_ens`, and each cell records `ad_iters` / `gfdt_iters` so the
fallback fraction is auditable.

## ⚠️ Status: the L63 Jacobian gate does NOT pass

See `VALIDATION.md` for measured numbers. Gate 1 (estimator vs. closed form)
passes at 3.5%. Gate 3 does not: the learned score on a 1001-sample window has a
Stein defect `‖Ξ‖` far above zero, and the resulting Gauss-Newton step direction
is not reliably aligned with the long-run finite-difference reference.

The binding constraint is the sample count, not the optimiser: on *Gaussian* data
at n=1001 a tuned DSM score still has ~15% error versus 0.17% for simply fitting a
Gaussian, and training 10× longer does not help. **KGMM (`SCORE_KIND=kgmm`)** is
now implemented as a second learned-score estimator alongside DSM
(`SCORE_KIND=dsm`), precisely to test whether a k-means Gaussian-mixture score
fares better in this sample-starved regime — see `VALIDATION.md` for the
comparison.

**Do not trust a leaderboard number from this directory until Gate 3 passes.**
The scaffolding, both budget modes, all four arms and the HPC pipeline are
complete and smoke-tested (including `l96_vec`/`l96_flux`, at reduced
iteration count — see `VALIDATION.md`).

**Also do not submit the `l96_flux` (and likely `l96_vec`) DSM arm at scale
yet.** A 5-iteration smoke test on `l96_flux` took 39 minutes, extrapolating to
~6.5h for a real 50-iteration cell — past the `--time=12:00:00` already set in
`run_array.sbatch` and unaffordable at 900 cells/arm. `kgmm` has no training
loop (one k-means fit + K covariance estimates per iteration instead), so it is
expected to sidestep this wall-clock problem — untested at scale yet, but see
`VALIDATION.md` for the score-sensibility comparison. Cutting the DSM
wall-clock, or preferring `kgmm` outright on L96, is the nearest-term
priority, ahead of the L63 Gate 3 fix.

## Layout

| File | Role |
|---|---|
| `experiment_config.jl` | config, filename builders, array dispatch, arm selection |
| `l63_preliminaries.jl`, `l96_preliminaries.jl` | shared truth data — **byte-identical to `levenberg_marquardt`'s**, so leaderboards are directly comparable |
| `score_gfdt.jl` | `ScoreModel` interface, DSM training, KGMM (k-means + Gaussian mixture) fitting, Stein recalibration, correlation integral |
| `score_nets.jl` | MLP (L63) and circular 1-D U-Net (L96), for the DSM arm; also the `make_score_model`/`refresh_score!` lifecycle dispatch shared by all three score kinds |
| `gfdt_l63.jl`, `gfdt_l96.jl` | raw moments `A(x)`, `G = φ(m)`, `∂φ/∂m`, conjugate variables `B_j` |
| `run_l63_sblm.jl`, `run_l96_sblm.jl` | the LM loop |
| `run_to_leaderboard.jl` | per-cell JLD2 → leaderboard netcdf |
| `validate_gfdt_ou.jl` | **Gate 1** — estimator vs. a closed form |
| `validate_jacobian_l63.jl` | **Gates 2/3** — Jacobian vs. long-run finite differences |

`gfdt_l63.jl` and `gfdt_l96.jl` must be separate: `common/forward_maps/Lorenz63.jl`
and `Lorenz96.jl` define clashing `LorenzConfig` / `EnsembleMemberConfig` /
`ObservationConfig`, so they can never be loaded together.

## Arms

Two independent switches, each combination a separate submission and netcdf:

| `SCORE_KIND` | `BUDGET_MODE` | `algorithm_type` |
|---|---|---|
| `dsm` | `serial` | `Score-based LM (GFDT, DSM) (extended window)` |
| `dsm` | `parallel` | `Score-based LM (GFDT, DSM) (parallel branches)` |
| `kgmm` | `serial` | `Score-based LM (GFDT, KGMM) (extended window)` |
| `kgmm` | `parallel` | `Score-based LM (GFDT, KGMM) (parallel branches)` |
| `gaussian` | `serial` | `Quasi-Gaussian FDT LM (extended window)` |
| `gaussian` | `parallel` | `Quasi-Gaussian FDT LM (parallel branches)` |

`dsm` is the single-sigma denoising-score-matching neural net (a training loop,
warm-started across LM iterations). `kgmm` is a k-means Gaussian-mixture score
estimator (arXiv:2503.18054) — no training loop, just a k-means fit plus `K`
per-cluster covariance estimates each iteration, added specifically because DSM
is known to struggle in this sample-starved regime (see `VALIDATION.md`).

**Budget modes.** Both spend `N_ens` forward-model evaluations per LM iteration
and integrate the same total model time, `T_start` once + `N_ens·W`; they differ
only in how that `N_ens·W` is arranged:

- `:serial` — one continuous window `N_ens` times longer than the base window.
  Inherently sequential (each step depends on the last), but the whole span is
  one attractor sample, giving the longest usable lags in the correlation
  integral for a given cost — the binding constraint on the 10-model-time-unit
  `l63` / `l96_const` windows.
- `:parallel` — spin up ONCE per LM iteration, then fork `N_ens` independent
  base-length branches from small (`ic_cov_sqrt`) perturbations of that
  on-attractor state. The branches carry no burn-in of their own and are
  mutually independent, so they can be integrated concurrently — same total
  integration cost as `:serial`, but parallelisable instead of one long serial
  run. (Replaces the old `:fair` mode, which paid `T_start` redundantly for
  every member; at `N_ens=1` `:fair` was already identical to `:serial`, so
  nothing is lost.)

At `N_ens = 1` the two modes are identical — that row is the strict
apples-to-apples comparison against `levenberg_marquardt`.

Charging `N_ens` forward evaluations either way means the metric doesn't
distinguish wall-clock parallelism from serial cost; the exact per-cell
integration time is recorded in `n_fwd_actual` regardless of mode.

## Cost metric

```
conv_score = outer_iter * N_ens          # vs levenberg_marquardt's * N_ens * (nu + 1)
```

The trial evaluation is not counted, matching `run_l63_lm.jl:85`, so the
comparison stays like-for-like. `n_fwd_actual` is stored alongside as an audit
counter measuring integration time in base-run equivalents.

**Score-network training is real compute that this metric does not capture.**
The leaderboard metric is forward-model evaluations, so the number is honest by
that definition — but it is not a wall-clock claim.

## Running

```bash
# gates first — do not skip
julia --project=. validate_gfdt_ou.jl                          # Gate 1
julia --project=. l63_preliminaries.jl
SCORE_KIND=gaussian julia --project=. validate_jacobian_l63.jl # Gate 2 (baseline)
SCORE_KIND=dsm      julia --project=. validate_jacobian_l63.jl # Gate 3 (DSM)
SCORE_KIND=kgmm     julia --project=. validate_jacobian_l63.jl # Gate 3 (KGMM)

# preliminaries per L96 case
for c in l96_const l96_vec l96_flux; do EXPERIMENT=$c julia --project=. l96_preliminaries.jl; done

# one cell / all cells
julia --project=. run_l63_sblm.jl 1
EXPERIMENT=l96_const julia --project=. run_l96_sblm.jl
julia --project=. run_to_leaderboard.jl
```

## HPC

`preliminaries →(afterok)→ run_array →(afterany)→ leaderboard`; precompile is
separate and never submitted by `submit_l*.sh`.

```bash
cd hpc-variant
bash submit_precompile.sh                       # once, after a checkout or package update
bash submit_l63.sh                              # defaults to SCORE_KIND=dsm BUDGET_MODE=serial
SCORE_KIND=dsm  BUDGET_MODE=parallel bash submit_l63.sh
SCORE_KIND=kgmm BUDGET_MODE=serial   bash submit_l63.sh
SCORE_KIND=gaussian bash submit_l96_const.sh
```

`N_TASKS = length(N_ens_sizes) * length(rmse_targets) * n_repeats = 3*3*100 = 900`.
If any of those change in `experiment_config.jl`, update `--array` in
`run_array.sbatch` **and** in every `submit_l*.sh`.

`run_array.sbatch` uses `--time=12:00:00 --mem=24G` (vs LM's `04:00:00`/`16G`)
because score training dominates wall-clock and `:serial` at `N_ens=10`
integrates 10× the window. `:parallel` integrates the same total but as
`Threads.@threads`-parallel branches (`--cpus-per-task=4` in `run_array.sbatch`
is passed through to `JULIA_NUM_THREADS`), so it stands to gain the most
wall-clock from that allocation.
