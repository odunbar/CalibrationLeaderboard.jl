# UQ Pipeline Reference

Two worked examples, covering different corners of the UQ pipeline shape.
Both follow the current conventions (one copy of every script, `CALIBRATE_DATE`
pinning, a preliminaries pre-stage) — the difference is stage count, not
maturity:

- `uq_experiments/calibrate_emulate_sample/` — the fuller 5-stage pipeline
  (preliminaries → calibrate → emulate_sample → pushforward →
  diagnostics/leaderboard). Good reference for stage names, array-sbatch
  conventions, and a case where per-task setup (priors, forcing params, NN
  structure) is non-trivial enough to be worth reading in `build_setup`.
- `uq_experiments/GaussNewtonKalmanInversion/` — a leaner 3-stage pipeline (no
  `emulate_sample`, no `diagnostic_plots`, because the raw ensemble at each
  iteration already is the posterior sample set). Good reference for the
  minimal shape a new UQ method can get away with.

## Dependency graph

### calibrate_emulate_sample — L63
```
l63_preliminaries  ─(afterok)→  calibrate_array  ─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior ─┬─(afterany)→  posterior_diagnostic_plots_l63
                                                                                                                                └─(afterany)→  exp_to_leaderboard
```

### calibrate_emulate_sample — L96 (const / vec / flux)
```
                                                    ┌─(afterok)→  calibration_diagnostic_plots_l96
l96_preliminaries  ─(afterok)→  calibrate_array  ──┤
                                                    └─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior ─┬─(afterany)→  posterior_diagnostic_plots_l96
                                                                                                                                    └─(afterany)→  exp_to_leaderboard
```

### GaussNewtonKalmanInversion — L63 and L96 (const / vec / flux)
```
l63_preliminaries / l96_preliminaries  ─(afterok)→  calibrate_array  ─(afterany)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```
No emulate_sample, no diagnostic_plots — the GNKI ensemble itself is the UQ
sample set at each iteration.

In both pipelines, `preliminaries` computes the shared truth data/observations
once, serially, before calibrate_array starts — see SKILL.md's "Serial
pre-stage job (preliminaries pattern)".

Key: `afterok` = all array tasks succeeded; `afterany` = all finished (success
or failure). Use `afterok` + `--kill-on-invalid-dep=yes` between stages that
would waste work downstream if an upstream stage fails outright (preliminaries
→ calibrate, calibrate → emulate). Use `afterany` on pushforward →
diagnostics/leaderboard so they always attempt to process whatever files exist.

## Sbatch file reference

| File | Type | Description |
|---|---|---|
| `preliminaries.sbatch` | single job | Computes + saves shared truth data/observations once (SCRIPT env var picks l63/l96) |
| `calibrate_array.sbatch` | array `1-N%100` | One task per `(N_ens, rng_idx)` cell; model-agnostic (SCRIPT env var) |
| `emulate_sample_array.sbatch` | array `1-N%100` | One task per cell; model-agnostic (calibrate_emulate_sample only) |
| `pushforward_from_posterior.sbatch` | array `1-N%100` | Posterior pushforward; saves results into JLD2 |
| `calibration_diagnostic_plots_l96.sbatch` | single job | Calibration figures, all cells serially (calibrate_emulate_sample, L96 only) |
| `posterior_diagnostic_plots_l63.sbatch` | array `1-N%100` | Posterior plots, one cell per task (calibrate_emulate_sample, L63) |
| `posterior_diagnostic_plots_l96.sbatch` | array `1-N%100` | Posterior plots, one cell per task (calibrate_emulate_sample, L96) |
| `exp_to_leaderboard.sbatch` | single job | NetCDF leaderboard, all cells serially |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

Array upper bound: `length(N_ens_sizes) * n_repeats`
(both examples: 9 × 20 = 180; `%100` caps concurrency)

## Submit script reference

| Script | Chains |
|---|---|
| `submit_precompile.sh [EXP_ID]` | precompile.sbatch only |
| `submit_l63.sh [EXP_ID]` | preliminaries → calibrate → pushfwd → leaderboard (GNKI); preliminaries → calibrate → emu → pushfwd → post_diag + leaderboard (CES) |
| `submit_l96_const.sh [EXP_ID]` | preliminaries → calibrate → pushfwd → leaderboard (GNKI); preliminaries → calibrate → calib_diag + emu → pushfwd → post_diag + leaderboard (CES) |
| `submit_l96_vec.sh [EXP_ID]` | same as const with EXPERIMENT=l96_vec |
| `submit_l96_flux.sh [EXP_ID]` | same as const with EXPERIMENT=l96_flux |

EXP_ID is optional; it suffix-labels SLURM job names (e.g. `calib_l63_run2`).
All four cases can submit simultaneously — output paths are case-specific.

## Serial (local) invocation

```bash
# GaussNewtonKalmanInversion — L63
julia --project=. l63_preliminaries.jl
julia --project=. calibrate_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. exp_to_leaderboard.jl

# GaussNewtonKalmanInversion — L96 (set EXPERIMENT env var)
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl

# calibrate_emulate_sample — L63
julia --project=. l63_preliminaries.jl
julia --project=. calibrate_l63.jl
julia --project=. emulate_sample_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. posterior_diagnostic_plots_l63.jl
julia --project=. exp_to_leaderboard.jl

# Single-cell (pass 1-based task index):
julia --project=. calibrate_l63.jl 3
EXPERIMENT=l96_vec julia --project=. calibrate_l96.jl 7
```

## Script roles

- **l63/l96_preliminaries.jl**: computes truth data/observations once,
  serially, and saves to `output/*_computed_preliminaries*.jld2`. Run before
  calibrate; calibrate errors if the file is missing rather than computing it
  itself — see SKILL.md's preliminaries pattern. In `calibrate_emulate_sample`'s
  L96 case, the per-force-case forcing/NN construction (`force_case_setup`) is
  duplicated between this script and `calibrate_l96.jl` rather than shared —
  that's intentional, see `common-handler`'s SKILL.md for why.
- **calibrate_l63/96.jl**: runs the calibration loop (EKI/GNKI/etc. with
  DataMisfitController) for one `(N_ens, rng_idx)` cell. Loads the
  preliminaries file (errors if missing) rather than computing it. In
  `calibrate_emulate_sample` this also does `build_setup` → `write_priors` for
  all 4 EKP method variants; in GNKI it's a single method's per-cell loop.
- **emulate_sample_l63/96.jl** (calibrate_emulate_sample only): for each
  iteration k, builds a `ScalarRandomFeatureInterface` emulator, runs RWMH MCMC
  (`MCMCWrapper`). Saves `*_posterior_*.jld2`. GNKI has no equivalent — its raw
  ensemble at each iteration already is the posterior sample set.
- **pushforward_from_posterior_l63/96.jl**: pushes stored samples (posterior
  MCMC draws, or the raw ensemble for GNKI) through the Lorenz forward map,
  saves `pushforward_output_samples` (+ forcing) back into the results/posterior
  JLD2.
- **exp_to_leaderboard.jl**: loads all cells, computes coverage metrics, writes
  leaderboard `.nc`.
- **posterior_diagnostic_plots_l63/96.jl**, **calibration_diagnostic_plots_l96.jl**
  (calibrate_emulate_sample only): load precomputed samples, generate figures.

## One-time setup checklist

1. `calibrate_date` follows the `CALIBRATE_DATE` env-var convention (see "Pin
   the run date via RUN_DATE" in SKILL.md): `submit_*.sh` computes the date
   once and passes it via `--export=ALL,...,CALIBRATE_DATE=${CALIBRATE_DATE}`
   on every stage, and `experiment_config.jl` reads it with a `today()`
   fallback for local runs. Both worked examples follow this.
2. Run `submit_precompile.sh [EXP_ID]` once (or after any package update).
3. Submit cases simultaneously: `for s in submit_l63.sh ...; do bash "$s" run1 & done; wait`.
4. Smoke test first: `sbatch --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch`
   then `sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch`.
