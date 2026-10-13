# Worked examples

Four diagnoses, each written as assessed cause → diagnostic that identifies it
→ panels. Numbers come from the 2026-06/10 runs and
`score_based_lm/VALIDATION.md`; re-measure, don't quote.

## Contents
1. adam: derivative magnitude, direction, and the frozen Adam step (instrument path)
2. levenberg_marquardt: the free linear-prediction check (instrument path)
3. GNKI: save the ekpobj, diagnose collapse post-hoc
4. HistoryMatching: is the truth ruled out? (post-hoc + small instrument)

---

## 1. adam: derivative magnitude, direction, and the frozen Adam step

**Failure map** (from-hpc 2026-06-26, fraction NaN): l63 0.88, l96_const 0.56,
l96_vec 1.00. `adam_alpha` changed on 2026-09-10, after these runs, so the map
only guides seed choice. l63 smoke seeds: 68 (converges at iteration 35) and 1
(fails); task = rng_idx for target 1.0.

**Assessed cause (before running).** ForwardDiff through the chaotic T=40
window gives ‖J̃‖ ~ 1e9–1e22 against a true ~221, so the gradient is huge and
points nowhere. About 30% of the l63 prior gives a window that collapses to a
fixed point, where AD is sane (VALIDATION.md).

**Diagnostics** (`opt_experiments/adam/diagnostics/adam_diagnostics.jl`):
`rmse`, `log10_J` (+ `_wmax`), `cos_truth_wmean`, `step_w_wmean`,
`vhat_ratio_wmean` = log10(√mean v̂ / rms g), `log10_statvar` (collapse flag
from the forward output's variance statistics), `lin_ratio_wmean` (actual vs
J̃-predicted residual change between iterations), `dist_truth`.

**What the runs showed (assessment revised).** Magnitude and direction are
garbage in chaotic windows (median log10‖J̃‖ ≈ 9.5, cos to truth ≈ 0.07,
residual change over-predicted ~10⁷×). But the run dies for a different
reason: one early spike fills v̂, and with β₂ = 0.999 it outlives the momentum
(β₁ = 0.9). After ~110–140 iterations the step is ≈ α·10^−vhat_ratio, about
1e-10, sometimes exactly 0 in Float64. l63 failures end in a collapsed window
where the gradient is *correct* (‖J̃‖ ≈ 135, cos ≈ 0.96) but the step is frozen.
Successes reach the target within the first ~100 iterations, by luck, before
the freeze.

**PANELS** (4): `rmse` (ref :slot), `log10_J` (ref 3), `cos_truth_wmean`
(ref 0), `step_w_wmean` (log, floor 1e-16 for exact zeros).

---

## 2. levenberg_marquardt: the free linear-prediction check

**Failure map** (from-hpc, flat, 2026-06-29): l63 0.89, l96_const 0.77, l96_vec 1.00.

**Assessed cause.** Same ‖J̃‖ ~ 1e15 as adam, but LM's MINPACK column scaling
`d = ‖J̃[:,j]‖` turns it into Δθ ~ ‖r̃‖/‖J̃‖ ~ 1e-15, so θ is frozen from the
start.

**Diagnostics** (snippet 2 in `assets/instrument-snippets.jl`): `rmse`,
`log10_J`, `log10_step_wmean`, `step_quant_wmean` (≠ 1: θ + Δθ == θ in
Float64), `lin_cos_wmean` / `lin_ratio_wmean` (LM's trial step reuses the same
IC, so this linear check is clean), `log10_lambda`, `dist_truth`. N_iter = 50,
so every iteration is recorded.

**Confirming values:** `log10_step` ≈ −12…−15, `step_quant` ≠ 1, `dist_truth`
flat for all 50 iterations. If instead `lin_cos` ≈ 1 with tiny steps, the
linear model is right only within machine epsilon. `log10_lambda` at 8 with no
accepted steps means the damping ran away.

**PANELS:** `rmse` (ref :slot), `log10_J`, `log10_step_wmean`,
`step_quant_wmean` (ref 1).

Side note: `conv_score` omits the trial evaluation (`outer_iter*N_ens*(nu+1)`).

---

## 3. GNKI: save the ekpobj, diagnose collapse post-hoc

**Failure map:** GNKI budget_to_target NaN fraction at c = 1: l63 ~0.01, flux
~0.14. Status is graded (`frac_q_reached`). l63 "failures" are mostly *large
budgets*.

**Assessed cause.** The ensemble becomes overconfident: spread collapses faster
than the misfit falls, so coverage is never reached.

**Instrument** (snippet 3): add `"ekpobj", ekpobj` to the `JLD2.save` in
`calibrate_l63.jl` and `calibrate_l96.jl`. Check `Base.summarysize(ekpobj)`
against the storage cap (≈ 4 MB at flux N_ens = 90). Smoke on l63 plus one small flux cell.

**`cell_diagnostics` for `combine_diagnostics.jl`** (sketch: check the EKP
accessor names against the installed version):
```julia
using EnsembleKalmanProcesses, EnsembleKalmanProcesses.ParameterDistributions
function cell_diagnostics(f, cfg)
    haskey(f, "ekpobj") || return phi_only_diagnostics(f, cfg)     # old cells
    ekp, prior, R, y = f["ekpobj"], f["prior"], f["R"], f["y"]
    us = get_u(ekp); gs = get_g(ekp)                                 # N_iter+1 and N_iter entries
    Σp = cov(prior); Wp = inv(cholesky(Symmetric(Σp)).L)
    Rw = inv(cholesky(Symmetric(R)).L)
    nu = size(us[1], 1); K = length(us)
    out = Dict{String, Vector{Float64}}("iter" => collect(0:(K - 1)))
    out["spread_w"] = [sqrt(tr(Wp * cov(u; dims = 2) * Wp') / nu) for u in us]
    out["eff_rank"] = [(λ = eigvals(Symmetric(Wp * cov(u; dims = 2) * Wp')); sum(λ)^2 / sum(abs2, λ)) for u in us]
    out["out_z"] = [k <= length(gs) ? norm(Rw * (y - vec(mean(gs[k]; dims = 2)))) /
                                      sqrt(tr(Rw * cov(gs[k]; dims = 2) * Rw')) : NaN for k in 1:K]
    out["n_failed"] = [k <= length(gs) ? count(j -> !all(isfinite, gs[k][:, j]), axes(gs[k], 2)) : NaN for k in 1:K]
    # optional: EKP error metrics (one value per update → align to k = 1:N_iter, NaN at k = 0)
    # truth: u_true = transform_constrained_to_unconstrained(prior, truth vector); skip for flux
    return out
end
```

**Confirming values:** `out_z` grows ≫ 1 while `spread_w` falls geometrically.
If instead `eff_rank` ≪ nu, the truth is outside the ensemble subspace; if
misfit is flat with `out_z` ≈ 1, it plateaued before coverage.

**PANELS:** `out_z` (ref 1), `spread_w` (log), `eff_rank`,
`truth_dist_over_spread` (ref 1, 2). x axis = k.

---

## 4. HistoryMatching: is the truth ruled out?

**Failure map:** HM budget NaN fraction: l63 0.13, flux 0.75, vec 1.00.
from-hpc 2026-09-15 holds one full cell per case: use l63/const/vec, **not**
the 2.5 GB flux file.

**Assessed cause.** The implausibility cut excludes the truth at some wave,
likely because the emulator is overconfident there; coverage can never recover.

**Post-hoc** (no rerun): rebuild the `HMProblem` exactly as the calibrate
script does (`make_hm_problem(y, R, PRIOR_MEAN_LOG, prior_cov, log; ...)`,
with `y`/`R` from the cell and prior constants and `cfg.retain_var*` from the
calibrate script/config), and the threshold `quantile(Chisq(prob.output_basis.k_R), cfg.confidence)`.
Then per wave w:
```julia
θt = reshape(θ_true_constrained, :, 1)                    # l63: [28, 8/3]; vec/const: truth_phi; skip flux
mu, var = predict_wave(prob, waves[w], θt)
I2 = implausibility_sq_whitened(mu[1, :], prob.y_whitened, var[1, :])
out["I2_truth_over_thr"][w] = I2 / threshold              # > 1: truth ruled out at wave w
out["truth_nroy"][w] = is_nroy(prob, θt, waves[1:w], threshold)[1]
out["n_post"][w] = size(posteriors_by_k[w], 2)
```
plus `loo_msse_med` / `loo_cov95_med` from the saved `emulator_diagnostics`
(median over the leading 5 modes, as `emulator_diagnostics.jl` prints).
Include `history_matching_core.jl` in the combine script for `predict_wave`,
`is_nroy`, `implausibility_sq_whitened`.

**Instrument** (snippet 4): `diag` keyword on `rejection_sample_nroy` →
`"hm_wave_diagnostics"` with `n_tried` / `n_accepted` → `accept_rate`. Both
calibrate scripts pass it; the existing `emulator_diagnostics` key is untouched.

**Confirming values:** `I2_truth_over_thr` crosses 1 at wave w* and
`truth_nroy` drops to 0 there; `loo_msse_med` ≫ 1 or `loo_cov95_med` ≪ 0.95 at
or before w*. If instead `n_post` < n_wanted with `accept_rate` → 0, the NROY
sampler is exhausted (expected for vec, nu = 40).

**PANELS:** `I2_truth_over_thr` (log, ref 1), `truth_nroy`, `loo_msse_med`
(log, ref 1), `accept_rate` (log). x axis = wave.
