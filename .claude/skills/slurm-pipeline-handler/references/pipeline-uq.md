# UQ Pipeline Reference

Two worked examples, covering different corners of the UQ pipeline shape:

- `uq_experiments/calibrate_emulate_sample/` — the original 5-stage pipeline
  (calibrate → emulate_sample → pushforward → diagnostics/leaderboard). Good
  reference for stage names and array-sbatch conventions, but its
  `hpc-variant/` copies every `.jl` script and a second `experiment_config.jl`
  into the hpc-variant directory (an older convention — see "hpc-variant
  layout: avoid copying scripts" in SKILL.md) and its prelim-file handling has
  the write-only-gated partial fix described in SKILL.md's "Watch for the
  partial fix" — not yet retrofitted. Read it for shape, not for these two
  details.
- `uq_experiments/GaussNewtonKalmanInversion/` — a 3-stage pipeline (no
  `emulate_sample`, no `diagnostic_plots`, because the raw ensemble at each
  iteration already is the posterior sample set) with a proper preliminaries
  pre-stage and `CALIBRATE_DATE` env-var date pinning. This is the current
  template for new UQ pipelines: single `experiment_config.jl`, no copied
  scripts, `hpc-variant/` holds only sbatch + submit files.

## Dependency graph

### calibrate_emulate_sample — L63
```
calibrate_array  ─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior ─┬─(afterany)→  posterior_diagnostic_plots_l63
                                                                                                └─(afterany)→  exp_to_leaderboard
```

### calibrate_emulate_sample — L96 (const / vec / flux)
```
                               ┌─(afterok)→  calibration_diagnostic_plots_l96
calibrate_array  ─(afterok)→──┤
                               └─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior ─┬─(afterany)→  posterior_diagnostic_plots_l96
                                                                                                              └─(afterany)→  exp_to_leaderboard
```

### GaussNewtonKalmanInversion — L63 and L96 (const / vec / flux)
```
l63_preliminaries / l96_preliminaries  ─(afterok)→  calibrate_array  ─(afterany)→  pushforward_from_posterior  ─(afterany)→  exp_to_leaderboard
```
No emulate_sample, no diagnostic_plots — the GNKI ensemble itself is the UQ
sample set at each iteration. The preliminaries stage computes the shared
truth data/observations once, serially, before calibrate_array starts —
see SKILL.md's "Serial pre-stage job (preliminaries pattern)".

Key: `afterok` = all array tasks succeeded; `afterany` = all finished (success
or failure). Use `afterok` + `--kill-on-invalid-dep=yes` between stages that
would waste work downstream if an upstream stage fails outright (preliminaries
→ calibrate, calibrate → emulate). Use `afterany` on pushforward →
diagnostics/leaderboard so they always attempt to process whatever files exist.

## Sbatch file reference

| File | Type | Description |
|---|---|---|
| `preliminaries.sbatch` | single job | Computes + saves shared truth data/observations once (GNKI only; SCRIPT env var picks l63/l96) |
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
| `submit_l63.sh [EXP_ID]` | (GNKI) preliminaries → calibrate → pushfwd → leaderboard; (CES) calibrate → emu → pushfwd → post_diag + leaderboard |
| `submit_l96_const.sh [EXP_ID]` | (GNKI) preliminaries → calibrate → pushfwd → leaderboard; (CES) calibrate → calib_diag + emu → pushfwd → post_diag + leaderboard |
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

- **l63/l96_preliminaries.jl** (GNKI): computes truth data/observations once,
  serially, and saves to `output/*_computed_preliminaries*.jld2`. Run before
  calibrate; calibrate errors if the file is missing rather than computing it
  itself — see SKILL.md's preliminaries pattern.
- **calibrate_l63/96.jl**: runs the calibration loop (EKI/GNKI/etc. with
  DataMisfitController) for one `(N_ens, rng_idx)` cell. Saves per-cell results
  JLD2. In `calibrate_emulate_sample` this also does `build_setup` →
  `write_priors`; in GNKI it loads the preliminaries file instead of computing
  it.
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
   the run date via RUN_DATE" in SKILL.md and
   `uq_experiments/GaussNewtonKalmanInversion/` for the worked UQ example):
   `submit_*.sh` computes the date once and passes it via
   `--export=ALL,...,CALIBRATE_DATE=${CALIBRATE_DATE}` on every stage, and
   `experiment_config.jl` reads it with a `today()` fallback for local runs.
   Treat this as standard practice for any new UQ pipeline —
   `calibrate_emulate_sample` still hand-pins `calibrate_date`; that's a known
   gap in the older pipeline, not something to replicate.
2. Run `submit_precompile.sh [EXP_ID]` once (or after any package update).
3. Submit cases simultaneously: `for s in submit_l63.sh ...; do bash "$s" run1 & done; wait`.
4. Smoke test first: `sbatch --export=ALL,SCRIPT=l63_preliminaries.jl preliminaries.sbatch`
   then `sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch`.
