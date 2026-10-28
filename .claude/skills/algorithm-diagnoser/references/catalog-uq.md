# Diagnostic catalog: uq methods

The uq leaderboard scores **output-space coverage** of the pushforward samples
against the truth, at each iteration k, and `budget_to_target` (the first k with
|S_k(q) − q| ≤ c·√(q(1−q)/k_R) per quantile). A failing uq run is almost always
one of:
- (a) overconfident: spread collapses faster than the misfit, so the truth falls outside
- (b) biased: centred on the wrong place, typically because the truth was excluded
- (c) under-converged: too few iterations or waves
- (d) a broken emulator or sampler

Pick diagnostics that separate these.

Prefer post-hoc computation from saved objects (SKILL.md § 2). Notation: u = unconstrained
parameters, Σ_p = prior covariance (unconstrained), R = observation noise
covariance, G_k = forward outputs of the ensemble at iteration k.

## Contents
- Coverage-side diagnostics (every uq method, from the nc alone)
- GNKI / EKI
- CES (calibrate → emulate → sample)
- History matching
- BOED
- ABC
- MCMC samplers

## Coverage-side diagnostics (every uq method, from the nc alone)

`output_coverage[seed, ens, k, q]` already gives S_k(q). Free per k:

| Key | Formula | Separates |
|---|---|---|
| `cov_err` | mean_q \|S_k(q) − q\| | added automatically by `combine_diagnostics_uq.jl` (aligned on iteration value; verify the k ↔ record alignment) |
| `cov_slope` | slope of S_k(q) against q | < 1 = under-dispersed (overconfident), > 1 = over-dispersed |
| `cov_bias` | S_k(0.5) − 0.5 | biased centre: truth systematically above or below |

These show *how* coverage fails. The method-specific keys below show *why*.

## GNKI / EKI (KalmanSampler; also the calibrate step of CES)

GNKI saves only `phi_stored`. **Save the `ekpobj`** in `calibrate_l{63,96}.jl`
(see SKILL.md § 2; ≈ 4 MB at flux N_ens=90). Then, post-hoc per k,
with `u_k = get_u(ekp)[k]` and `g_k = get_g(ekp)[k]`:

| Key | Formula | Separates |
|---|---|---|
| `avg_rmse`, `loss`, `bayes_loss`, `crps` | `get_error_metrics(ekp)` (EKP ≥ 2.7 computes them at every update) | free misfit trajectory |
| `out_z` | ‖R^{-1/2}(y − mean(g_k))‖ / √tr(R^{-1/2} Cov(g_k) R^{-1/2}) | misfit relative to output spread: ≫ 1 = overconfident ensemble → output under-coverage (a) |
| `spread_w` | √(tr(Σ_p⁻¹ Cov(u_k))/nu) | collapse rate; compare its slope with the misfit's |
| `eff_rank` | participation ratio (Σλ)²/Σλ² of the whitened Cov(u_k) | subspace collapse; matters when N_ens < nu (vec: N_ens 4–20 < 40) |
| `truth_dist_over_spread` | ‖Σ_p^{-1/2}(u_true − mean u_k)‖ / (√nu · spread_w) | truth many spreads away = (a) or (b) |
| `truth_outside_span` | fraction of the whitened (u_true − mean u_k) outside the anomaly span | truth unreachable by the ensemble subspace |
| `n_failed` | members with non-finite columns in g_k | blown-up members |
| `dt` | `get_Δt(ekp)` | constant under `DefaultScheduler`: record but don't plot |

Without `ekpobj` (old cells), use `phi_stored` for `spread_w`, `eff_rank` and
`truth_dist_over_spread` only. u_true comes from `truth_phi` via
`transform_constrained_to_unconstrained(prior, ·)` where defined. Flux:
forcing-space error.

Alignment: `get_g` has N_iter entries matching `u[1:N_iter]`; the final u has
no G. Set those keys NaN at the last record.

## CES (calibrate → emulate → sample)

- Calibrate: `ekpobj` already saved in `l96_ekp_*.jld2`, so the GNKI keys are post-hoc.
- Emulator: weighted MSE at the truth (currently only printed,
  `emulate_sample_l96.jl:119`). Recompute post-hoc from the saved emulator
  inputs/outputs where possible, or record it (it is computed anyway).
- MCMC: acceptance rate, `optimize_stepsize` result (printed only, `:124`), ESS
  from the saved chains post-hoc.

## History matching (HistoryMatching)

Saved: `waves` (GPs per wave), `emulator_diagnostics` (LOO per wave),
`posteriors_by_k`, `n_waves_completed`. Most diagnostics are post-hoc via **GP
predictions on the saved waves** (not forward runs, so the rules allow them).

| Key | Formula | Separates |
|---|---|---|
| `I2_truth_over_thr` | implausibility of θ_true at wave w ÷ the χ² threshold (`calibrate_l96.jl:133`) | **> 1 means the truth is ruled out at that wave**, a direct cause of (b). Uses `is_nroy`-style evaluation at one point |
| `truth_nroy` | cumulative product of "truth survives wave ≤ w" | the wave where the truth dies |
| `n_post`, `post_frac` | `size(posteriors_by_k[w], 2)`, ÷ n_wanted | NROY exhaustion (< n_wanted ⇒ sampler gave up) |
| `accept_rate` | n_accepted / n_tried (instrument `rejection_sample_nroy`, see snippets) | NROY volume per wave; collapse to ~0 = curse of dimensionality |
| `post_spread_w`, `truth_dist_over_spread` | from `posteriors_by_k` | as GNKI |
| `loo_msse_med`, `loo_cov95_med` | median over leading modes of the saved `emulator_diagnostics` | msse ≫ 1 or cov95 ≪ 0.95 = overconfident emulator, which is the usual reason the truth gets cut |
| `ls_at_bound_frac`, `log_mll` | GP lengthscales / mll from the saved GPs | lengthscales at bounds = emulator can't resolve the response |

Expected causal chain for HM failure: overconfident emulator (msse ≫ 1) → truth
implausibility exceeds the threshold → NROY excludes the truth → coverage never
reached. The vec case (nu = 40) also tends to show `accept_rate` → 0.

Don't save more GP objects: flux cells are already 2.5 GB. For flux, skip
θ_true-based keys (no identifiable truth). `HistoryMatching/emulator_diagnostics.jl`
prints the LOO table for one file; the combine script supersedes it across seeds.

## BOED (BayesianOptimalExperimentalDesign)

Save **summary fields**, not objects:

| Key | Source | Separates |
|---|---|---|
| GP hyperparameters per iteration | `extract_hyperparams` (`boed_core.jl:270`) | emulator drift or degeneracy |
| `log_ev` | TMCMC log-evidence, currently discarded (`boed_core.jl:240`) | sampler stability across iterations |
| `eig_converged`, `eig_calls` | `run_fminbox_with_call_limit` (`boed_core.jl:317-374`, warns only) | design optimiser hitting its limits |
| `design_mindist` | min distance of the new design to existing points | design collapse (re-sampling the same point) |
| plus the HM-style truth implausibility and posterior spread | | |

Prior hand-made versions live in `BayesianOptimalExperimentalDesign/diag_*.jl`
(text only, one cell). Reuse their logic, not their single-cell structure.

## ABC (approximate_bayesian_computation)

Fully post-hoc: `pool_sizes[k]`, `threshold`, and `phi_stored[k]` (accepted pool) are saved.

| Key | Formula | Separates |
|---|---|---|
| `accept_rate` | pool_sizes[k] / (N_ens · k) | threshold too tight (→ 0) or too loose |
| `pool_size` | pool_sizes[k] | (already in the nc) |
| `spread_w`, `truth_dist_over_spread` | from `phi_stored[k]` | posterior too wide (loose threshold) vs biased |

## MCMC samplers (CES sample step, BOED TMCMC)

| Key | Source | Separates |
|---|---|---|
| `accept_rate` | chain acceptance (compute post-hoc from repeated states) | step size badly tuned |
| `step_size` | `optimize_stepsize` output | |
| `ess` | effective sample size from the saved chain | too few independent samples → noisy coverage |
| `rhat` (if several chains) | | non-convergence |
