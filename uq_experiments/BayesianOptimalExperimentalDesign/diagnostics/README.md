# GBOED diagnostics

Diagnostics for the four GBOED variants (`BOED_VARIANT` = `tmcmc_eig`, `tmcmc`, `iekf_eig`, `iekf`), built around four questions:

| # | Question | Where it is answered |
|---|---|---|
| 1 | Does the emulator improve over the iterations, where it matters? | `test1_emulator_quality.jl` (offline, one cell, uses the true forward model) + the `ll_*`, `misfit_*`, `gp_*`, `loo_*` keys recorded in every run |
| 2 | Have ST-MCMC and the IEKF converged? | `test2_sampler_convergence.jl` (offline, frozen GP, no forward model) + `tmcmc_*` / `iekf_*` keys |
| 3 | Where does EIG place points relative to the posterior? | `test3_eig_placement.jl` (all seeds, from the combined files) + `cand_*`, `eig_*` keys |
| 4 | Does the EIG stage help? | `test4_eig_helpfulness.jl` (paired EIG vs no-EIG, from the combined files) |

## What the run records

`calibrate_l63.jl` / `calibrate_l96.jl` add a `"diagnostics"` key to each cell's `*_calibrate_results_*.jld2` (rewritten after every wave, like the rest of the file).
`DIAG=0` turns it off. The additions are read-only: no RNG draw, no forward evaluation, no change to `conv_score`-style outputs. For `iekf` a run with and without diagnostics
gives bitwise identical posteriors (checked). The code is `boed_diagnostics.jl` (included by `boed_core.jl`; it includes the shared recorder
`common/diagnostics/diag_recorder.jl`); `boed_core.jl` gained optional `diag = nothing` keywords on `run_tmcmc`, `run_iekf`, `sample_posterior`, `run_fminbox_with_call_limit`,
`optimize_batch` and `acquire_batch` that fill a `Dict` when given one.

Per iteration k (every iteration is recorded; keys are described in `boed_diagnostics.jl`'s `describe!` and stored under `diagnostics["descriptions"]`):

| Group | Keys |
|---|---|
| Emulator at the truth and in the posterior | `ll_truth`, `misfit_truth`, `gp_std_truth`, `ll_post_med`, `misfit_post`, `gp_std_post`, `n_train` |
| Emulator health | `loo_msse_med`, `loo_cov95_med`, `gp_ls_med`, `gp_ls_min`, `gp_ls_bound_frac`, `gp_noise_ratio_med`, `gp_sig_med`, `gp_mll_sum` |
| Posterior | `post_spread_w`, `post_mean_rms`, `post_eff_rank`, `post_uniq_frac`, `truth_dist_over_spread` |
| Sampler | `tmcmc_log_ev`, `tmcmc_loglik_calls`, `tmcmc_secs`; `iekf_misfit_first/last/flat`, `iekf_spread_ratio`, `iekf_secs` |
| Acquisition (the batch added at k, judged against the k−1 posterior) | `eig_batch`, `eig_init`, `eig_postdraw`, `eig_priordesign`, `eig_opt_final/calls/secs`, `eig_converged`, `eig_hit_limit`, `cand_r_mean`, `post_r_mean`, `cand_r_over_ball`, `cand_at_ball_frac`, `cand_mahal_med`, `post_mahal_med`, `cand_move`, `cand_mindist_med`, `post_mindist_med`, `batch_mindist_med`, `cand_gp_std`, `post_gp_std` |

The truth keys (`ll_truth`, `misfit_truth`, `gp_std_truth`, `truth_dist_over_spread`) are NaN for flux (no identifiable truth). The truth is projected into the truncated
whitened input space, so the discarded prior modes are not charged to the emulator.

Objects saved next to the scalars (read by the offline tests, not combined): `Z` (cumulative design, whitened inputs), `Yw` (whitened outputs), `gp_hyper_by_k`
(`[iℓ2…, σ2, noise variance]` per output mode and iteration), `X_post_sub_by_k` (100 evenly spaced posterior draws per iteration), `X_cand0_by_k` (the EIG optimiser's starting batch),
`prior_V/λ/mean`, `output_V/λ`, `y_whitened`, `z_true`. Together they rebuild GP_k exactly (`rebuild_gps`; checked against the run's recorded log target to 1e-5) without refitting.
Storage: roughly 0.5 MB per vec/flux cell (a cell's posteriors alone are 2.5–4 MB), and ≈ 60 KB for l63.

## Files

| File | Role |
|---|---|
| `boed_diagnostics.jl` | per-iteration diagnostics included by the run |
| `combine_diagnostics.jl` | all seeds × N_ens of one (variant, case) → `output/diagnostics_<variant key>_<case>_<date>.jld2` (layout: `common/diagnostics/combine_core.jl`) |
| `experiment_diagnosis_visualization.jl` | one nine-panel figure per (variant, case): `output/diagnosis_<…>.{png,pdf}` |
| `offline_common.jl` | cell loader, GP rebuild, true-forward evaluation for tests 1–2 |
| `offline_tmcmc.jl` | threaded and lock-step batched ST-MCMC for test2 (`DIAG_TMCMC=package\|threaded\|batched`, default `batched`); offline only, the run's `run_tmcmc` is unchanged |
| `validate_offline_tmcmc.jl` | checks `threaded` / `batched` against `TransitionalMCMC.tmcmc` on a frozen GP (log-evidence, energy distance vs the MC floor, timings) |
| `compare_common.jl` | combined-file loader for tests 3–4 |
| `test1_…jl` … `test4_…jl` | the four tests |
| `test5_eig_optimiser_gain.jl` | within-stage EIG gain of the optimiser (`eig_batch − eig_init`, same fixed subsample), split by hit-call-limit; from the combined files of the EIG variants |

## Commands (from `BayesianOptimalExperimentalDesign/`)

```bash
# 1. run cells with diagnostics (default on). One variant at a time; set the same CALIBRATE_DATE and BOED_VARIANT for every stage.
BOED_VARIANT=iekf_eig EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-20 julia --project=. -t 8 calibrate_l96.jl <task>

# 2. combine + figure, per variant and case (DIAG_NC=<leaderboard nc> supplies the success/failure labels; from-hpc nc files work)
BOED_VARIANT=iekf_eig EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-20 julia --project=. diagnostics/combine_diagnostics.jl
BOED_VARIANT=iekf_eig EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-20 julia --project=. diagnostics/experiment_diagnosis_visualization.jl

# 3. tests 1-2 (one cell; take it from the same output directory)
BOED_VARIANT=tmcmc_eig EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-20 DIAG_N_ENS=50 julia --project=. -t 8 diagnostics/test1_emulator_quality.jl
BOED_VARIANT=tmcmc_eig EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-20 DIAG_N_ENS=50 julia --project=. -t 8 diagnostics/test2_sampler_convergence.jl

# 3b. test2 uses the lock-step batched ST-MCMC by default (same algorithm as TransitionalMCMC.tmcmc, far faster on l96_vec/flux; see offline_tmcmc.jl).
#     DIAG_TMCMC=package reproduces the old serial behaviour; validate_offline_tmcmc.jl compares the three implementations on one cell.
BOED_VARIANT=iekf EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-20 DIAG_N_ENS=50 julia --project=. -t 16 diagnostics/validate_offline_tmcmc.jl

# 4. tests 3-4 (need the combined files of several variants; DIAG_VARIANTS restricts; CALIBRATE_DATE pins the date)
EXPERIMENT=l96_vec julia --project=. diagnostics/test3_eig_placement.jl
EXPERIMENT=l96_vec julia --project=. diagnostics/test4_eig_helpfulness.jl

# 5. does the EIG optimiser improve the batch within each stage? (EIG variants only; same env options as test3)
EXPERIMENT=l96_vec julia --project=. diagnostics/test5_eig_optimiser_gain.jl
```

For HPC, add the combine job after the leaderboard job (slurm-pipeline-handler) and copy `diagnostics/output/diagnostics_*.jld2` back with the nc; the offline tests need the cell files themselves (`output/<variant key>_<date>/`), which for flux are large.

## Known limits

* `TransitionalMCMC.tmcmc` returns only `(samples, log_evidence)` and draws its Metropolis moves from Julia's **global** RNG, so (a) the tempering stages and β schedule are not recoverable
  (`tmcmc_loglik_calls` is the proxy), and (b) the ST-MCMC variants are not reproducible per seed, with or without diagnostics. Test 4 therefore pairs the tmcmc variants only by (seed, N_ens).
* The in-run emulator keys use GP predictions only. They say whether the emulator reproduces the *observation* at the truth and in the posterior; whether it reproduces the *true model* there
  needs forward evaluations, which is what `test1` adds.
* Cells run before the instrumentation (all HPC cells up to 2026-10-12) have no `"diagnostics"` key, and `combine_diagnostics.jl` will skip them: rerun to get diagnostics.
