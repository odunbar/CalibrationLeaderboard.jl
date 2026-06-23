# UQ Pipeline Reference

Source: `uq_experiments/calibrate_emulate_sample/`

## Dependency graph

### L63
```
calibrate_array  ─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior ─┬─(afterany)→  posterior_diagnostic_plots_l63
                                                                                                └─(afterany)→  exp_to_leaderboard
```

### L96 (const / vec / flux)
```
                               ┌─(afterok)→  calibration_diagnostic_plots_l96
calibrate_array  ─(afterok)→──┤
                               └─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior ─┬─(afterany)→  posterior_diagnostic_plots_l96
                                                                                                              └─(afterany)→  exp_to_leaderboard
```

Key: `afterok` = all array tasks succeeded; `afterany` = all finished (success or failure).
Use `afterok` + `--kill-on-invalid-dep=yes` on calibrate→emulate to stop wasted work if
calibrate fails. Use `afterany` on pushforward→diagnostics/leaderboard so they always
attempt to process whatever posterior files exist.

## Sbatch file reference

| File | Type | Description |
|---|---|---|
| `calibrate_array.sbatch` | array `1-N%100` | One task per `(N_ens, rng_idx)` cell; model-agnostic (SCRIPT env var) |
| `emulate_sample_array.sbatch` | array `1-N%100` | One task per cell; model-agnostic |
| `pushforward_from_posterior.sbatch` | array `1-N%100` | Posterior pushforward; saves results into JLD2 |
| `calibration_diagnostic_plots_l96.sbatch` | single job | Calibration figures, all cells serially (L96 only) |
| `posterior_diagnostic_plots_l63.sbatch` | array `1-N%100` | Posterior plots, one cell per task (L63) |
| `posterior_diagnostic_plots_l96.sbatch` | array `1-N%100` | Posterior plots, one cell per task (L96) |
| `exp_to_leaderboard.sbatch` | single job | NetCDF leaderboard, all cells serially |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

Array upper bound: `length(N_ens_sizes) * n_repeats`
(UQ example: 9 × 20 = 180; `%100` caps concurrency)

## Submit script reference

| Script | Chains |
|---|---|
| `submit_precompile.sh [EXP_ID]` | precompile.sbatch only |
| `submit_l63.sh [EXP_ID]` | calibrate → emu → pushfwd → post_diag + leaderboard |
| `submit_l96_const.sh [EXP_ID]` | calibrate → calib_diag + emu → pushfwd → post_diag + leaderboard |
| `submit_l96_vec.sh [EXP_ID]` | same as const with EXPERIMENT=l96_vec |
| `submit_l96_flux.sh [EXP_ID]` | same as const with EXPERIMENT=l96_flux |

EXP_ID is optional; it suffix-labels SLURM job names (e.g. `calib_l63_run2`).
All four cases can submit simultaneously — output paths are case-specific.

## Serial (local) invocation

```bash
# L63
julia --project=. calibrate_l63.jl
julia --project=. emulate_sample_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. posterior_diagnostic_plots_l63.jl
julia --project=. exp_to_leaderboard.jl

# L96 — set EXPERIMENT env var
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. calibration_diagnostic_plots_l96.jl
EXPERIMENT=l96_const julia --project=. emulate_sample_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. posterior_diagnostic_plots_l96.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl

# Single-cell (pass 1-based task index):
julia --project=. calibrate_l63.jl 3
EXPERIMENT=l96_vec julia --project=. calibrate_l96.jl 7
```

## Script roles

- **calibrate_l63/96.jl**: `build_setup` → `write_priors` → `calibrate_one(N_ens, rng_idx)`
  (EKP loop with DataMisfitController). Saves `ekp_*.jld2` + `*_calibrate_results_*.jld2`.
- **emulate_sample_l63/96.jl**: For each iteration k, builds `ScalarRandomFeatureInterface`
  emulator, runs RWMH MCMC (`MCMCWrapper`). Saves `*_posterior_*.jld2`.
- **pushforward_from_posterior_l63/96.jl**: Draws `n_pushforward_samples=1000` from each
  posterior, pushes through Lorenz forward map, saves `pushforward_output_samples` (+ forcing)
  back into `posterior_*.jld2`.
- **exp_to_leaderboard.jl**: Loads all posterior JLD2 files, computes coverage metrics,
  writes leaderboard `.nc`.
- **posterior_diagnostic_plots_l63/96.jl**: Loads pushforward samples (no re-running),
  generates ribbon / scatter figures.
- **calibration_diagnostic_plots_l96.jl**: Calibration-stage figures (all cells serially).

## One-time setup checklist

1. Pin `calibrate_date` in `experiment_config.jl` before submitting.
2. Run `submit_precompile.sh [EXP_ID]` once (or after any package update).
3. Submit cases simultaneously: `for s in submit_l63.sh ...; do bash "$s" run1 & done; wait`.
4. Smoke test first: `sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch`.
