# Diagnostic catalog: opt methods

Each entry lists cheap diagnostics, computed only from what the loop already has
(SKILL.md § Rules). For each: the formula, why it separates success from failure, and its
blind spot. Notation:
- J̃ = whitened Jacobian, r̃ = whitened residual, g = −J̃ᵀr̃, RMSE = ‖r̃‖/√ny
- Lp = cholesky(prior cov).L; "whitened" means multiplied by Lp⁻¹ (parameters)
  or Lpᵀ (gradients)

Record with `record!` at scheduled iterations; accumulate per-step quantities
with `accumulate!` (flushed as `_wmean`/`_wmin`/`_wmax`). See
`assets/instrument-snippets.jl` for placement.

## Contents
- Shared core (every opt method)
- Gradient / Adam (adam, score_based_adam)
- Levenberg–Marquardt / Gauss–Newton (levenberg_marquardt, score_based_lm)
- GFDT / score-based Jacobians (score_based_*)
- Consensus-based optimisation (CBO)
- Ensemble Kalman as optimiser (ensemble_kalman_processes)
- Reinforcement learning (DDPG/TD3)

## Shared core (every opt method)

| Key | Formula | Separates | Blind spot |
|---|---|---|---|
| `rmse` | ‖r̃‖/√ny at record | outcome trajectory | single-IC noise: one draw can dip below target by luck |
| `rmse_wmin` | window min of RMSE | real progress vs a lucky stop | none really; pairs with `rmse` |
| `rmse_up_wmean` | fraction of steps with RMSE increase | ≈ 0.5 = random walk; ≪ 0.5 = descent | noisy objective inflates it even with a good direction |
| `dist_truth` | ‖Lp⁻¹(θ − θ_true)‖ | approaching the truth vs drifting or stuck | multiple basins / non-identifiable directions; flux has no θ_true (see repo-layout) |
| `nonfinite_J` | count of non-finite J̃ entries | overflow | 0 doesn't mean J̃ is sane (see `log10_J`) |

## Canonical keys

The same quantity must carry the same name and unit in every method, so a PANELS
list or a comparison table works across methods. `DiagFiles.KEY_ALIASES`
(`common/diagnostics/diag_files.jl`) maps each canonical key to what a method
stores; `resolve_series(c, "cos_truth")` returns the first one present, converted
to the canonical unit (an exact stored key always wins). Record new quantities
under these names when the quantity is the same, and add an alias entry when a
method must keep a different stored name.

| Canonical key | Stored as | Unit / meaning |
|---|---|---|
| `rmse` | `rmse` | whitened RMSE at the record |
| `log10_J`, `jac_plaus`, `log10_g`, `dist_truth` | same name | as in the tables below |
| `cos_truth` | `cos_truth` (point: LM, EKP) or `cos_truth_wmean` (window mean: Adam) | cos(−Lpᵀg, Lp⁻¹(θ_true − θ)) |
| `cos_prev` | `cos_prev` or `cos_prev_wmean` | cos(g_t, g_{t−1}) |
| `log10_step` | `log10_step_wmean` (LM), `step_w_wmean` / `step_w` (Adam, EKP; converted with log10, zeros floored at 1e-16) | log10 ‖Lp⁻¹Δθ‖ |

A `_wmean` / `_wmin` / `_wmax` suffix is only for values the recorder
aggregated over the window since the last record (`accumulate!`); a point value
recorded with `record!` has no suffix. Method-specific keys (`vhat_ratio_wmean`,
`log10_lambda`, `stein_norm`, ...) stay free; describe each one in `describe!`.

**Noise floor.** Every method evaluates RMSE on one IC sample per
iteration (N_ens = 1 for adam/LM). If RMSE at θ_true is close to the target,
"success" is partly chance. `score_based_lm/VALIDATION.md` reports
RMSE(θ_true) ≈ 0.68 for one IC on L63. Compare `rmse_wmin` of failures with this
floor before blaming the optimiser.

## Gradient / Adam (adam, score_based_adam)

The loop overwrites J̃, g, m, v every iteration, so record scalars (there is no
object to save).

| Key | Formula | Separates | Blind spot |
|---|---|---|---|
| `log10_J` | log10‖J̃‖_F | **magnitude**: healthy O(1–10²); AD through a chaotic window gives 1e9–1e22 | a correct-but-large Jacobian near a bifurcation also reads high |
| `jac_plaus` | log10(‖J̃ Lp‖ / ‖r̃‖) | predicted residual change per prior std relative to the current residual; ≫ 3 is physically impossible because attractor statistics are bounded. A reference-free magnitude test | assumes the prior scale is meaningful |
| `log10_g` | log10‖g‖ | magnitude through the loss | Adam normalises it away, so pair with `step_w` |
| `step_w_wmean` | ‖Lp⁻¹(θ_new − θ_pre)‖ | ≈ α early (sign-descent on noise); collapsing to 1e-10 or exactly 0 = frozen | plot on log axis with a floor for exact zeros |
| `vhat_ratio_wmean` | log10(√mean v̂ / rms g) | **frozen Adam**: one old spike in v̂ outlives the momentum (β₂ = 0.999 vs β₁ = 0.9), so after ~100 iterations the step ≈ α·10^−ratio even when the current g is sane | none for the freeze itself; doesn't say whether g is right (pair with `cos_truth`) |
| `snr_wmean` | mean_i m̂ᵢ²/v̂ᵢ | i.i.d. zero-mean g gives ≈ (1−β₁)/(1+β₁) ≈ 0.053 (draw as a reference line); consistent g → near 1 | heavy-tailed g (occasional 1e15 spikes) pushes v̂ up and SNR down even with a mean signal |
| `cos_prev_wmean` | cos(g_t, g_{t−1}) | **direction**: each iteration uses a fresh IC, so ≈ 0 means g is dominated by IC noise | a genuinely curved path also lowers it; nu = 1 → ±1 only, read as sign-agreement fraction |
| `cos_truth_wmean` | cos(−Lpᵀg, Lp⁻¹(θ_true − θ)) | **direction** against the known optimum: > 0 consistently = descent toward truth | weak far from θ_true in a non-convex landscape; use window means, not single values |

Seen in adam (see `worked-examples.md` §1): in chaotic windows `log10_J` ≫ 3,
`cos_truth_wmean` ≈ 0; the run then freezes with `vhat_ratio_wmean` ≈ 10 and
`step_w_wmean` ≈ 1e-11, often ending in a collapsed window where AD is sane
(‖J̃‖ ≈ 135, cos ≈ 0.96) but unusable. Successes converge within ~100
iterations, before the freeze.

Regime flag: a collapsed (fixed-point) window has near-zero variance
statistics in the forward output, available free as `y - r_sum/N_ens`. L63:
`log10_statvar` = log10 min of `G[4:6]` (check the `stats` layout in
`common/forward_maps/Lorenz63.jl`); L96: log10 mean of `G[nx+1:end].^2`.
< −2 = collapsed.

## Levenberg–Marquardt / Gauss–Newton (levenberg_marquardt, score_based_lm)

LM evaluates a trial step with **the same IC sample** (`x0p_all` reused). That
gives a free, clean directional-derivative check.

| Key | Formula | Separates | Blind spot |
|---|---|---|---|
| `log10_J`, `jac_plaus` | as Adam | magnitude | |
| `cond_J` | log10 cond(J̃), recorded iterations only | ill-conditioning | Inf/NaN J̃ → skip |
| `sigma_J` | ‖J̃ᵀr̃‖/(‖J̃‖‖r̃‖) | residual outside range(J̃): ≈ 0 means the model can't explain the misfit locally | |
| `lin_cos` | cos(r̃_trial − r̃, −J̃Δθ) | **direction at the step scale**: ≈ 1 = linearisation valid | mixes derivative error with nonlinearity over the step |
| `lin_ratio` | log10(‖r̃_trial − r̃‖ / ‖J̃Δθ‖) | **magnitude at the step scale**: ≈ 0 valid; ≪ 0 means J̃ overstates the response | same |
| `rho_clip_wmean` | mean clamp(ρ, −2, 2) | gain ratio; reference lines 0, 0.25, 0.75 | raw ρ is unbounded, so the clip keeps means readable |
| `accepted_wmean` | fraction ρ > 0 | stuck = 0 | |
| `log10_lambda` | log10 λ at record | damping pinned at 1e8 = given up | |
| `log10_step` | log10‖Lp⁻¹Δθ‖ | MINPACK scaling with ‖J̃‖ ~ 1e15 gives Δθ ~ 1e-15: θ is frozen | |
| `step_quant` | ‖(θ+Δθ) − θ‖/‖Δθ‖ | ≠ 1 means the step is below Float64 resolution. Decisive and free | |
| `cos_step_truth` | cos(Lp⁻¹Δθ, Lp⁻¹(θ_true − θ)) | direction vs optimum | as Adam |
| `rmse_trial_wmin` | min trial RMSE | how close rejected steps got | |

Expected signature for derivative failure in LM: `log10_J` ≫ 3, `log10_step`
≈ −12…−15, `step_quant` ≠ 1 or `lin_cos` ≈ 1 with tiny steps ("valid but
useless": the linear radius is about machine epsilon), `rho` erratic,
`log10_lambda` climbing to the cap, `dist_truth` flat across all 50 iterations.

Side note: LM's `conv_score` doesn't count the trial-step evaluation.

## GFDT / score-based Jacobians (score_based_*)

`gfdt_jacobian` already returns quality signals. Record them:

| Key | Source | Separates |
|---|---|---|
| `stein_norm` | `score_gfdt.jl` return value ‖Ξ‖ | Stein-identity defect: large = poor score model |
| `b_mean_ratio` | mean(B)/std(B) | response signal vs noise |
| `collapsed` / `ad_fallback` | `is_collapsed_window` branch | GFDT undefined there, so the run used ForwardDiff |
| plus the LM or Adam keys above, depending on the optimiser | | |

Also `ad_iters` / `gfdt_iters` counters (already saved at the end) as
per-iteration flags. Offline ground truth for these estimators is
`validate_jacobian_l63.jl` / `validate_jacobian_l96.jl` / `validate_gfdt_ou.jl`
(never inside the run).

## Consensus-based optimisation (CBO)

No derivatives. The particle ensemble and Gibbs weights exist each iteration.

| Key | Formula | Separates |
|---|---|---|
| `spread_w` | √(tr(Lp⁻¹ Cov(X) Lp⁻ᵀ)/nu) | collapse rate vs misfit decrease (premature collapse = failure) |
| `weight_ess` | (Σw)²/Σw² over N_ens | weight degeneracy: ESS ≈ 1 means one particle dominates |
| `cons_dist_truth` | ‖Lp⁻¹(x̄_cons − θ_true)‖ | consensus heading to the truth |
| `frac_near_cons` | fraction of particles within 1 prior std of the consensus | |
| `best_rmse_wmin` | min member RMSE in the window | |

CBO's Project has no Plots: ask before adding it.

## Ensemble Kalman as optimiser (ensemble_kalman_processes)

Save the `ekpobj` if it fits the storage cap (SKILL.md § 2). Post-hoc:
ensemble spread vs misfit, `get_error_metrics`, Δt per iteration, failed
members. Same formulas as GNKI in `catalog-uq.md`.

## Reinforcement learning (DDPG/TD3, `rl_run.jl`, `ddpg_core.jl`)

| Key | Source | Separates |
|---|---|---|
| `rmse` | `sqrt(Φ/ny)` (`rl_run.jl:46`) | outcome |
| `critic_loss_wmean` | return value of `ddpg_update!` | critic not learning (flat or exploding) |
| `policy_step_w` | ‖Lp⁻¹ΔF‖ between actions | policy frozen or thrashing |
| `action_sat_wmean` | fraction of action components at bounds | saturation |
| `actor_grad_norm` | ‖∇θ actor loss‖ (`ddpg_core.jl:178`) | vanishing or exploding actor updates |

Exploration noise draws from the RNG. Recording the action *after* noise is
fine; drawing extra noise is not (SKILL.md § Rules).
