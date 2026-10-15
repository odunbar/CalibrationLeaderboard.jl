# Repo layout for diagnosis

Where things live in `opt_experiments/<method>/` and `uq_experiments/<method>/`,
how success is defined, and the quirks that trip up a combine script. Line
numbers are as of 2026-10 and drift. Search for the quoted code if they don't
match.

## Contents
- OPT anatomy
- UQ anatomy
- Task indexing (nc seed → CLI task)
- Status: success vs failure
- Dates
- from-hpc layouts
- Truth sources per case
- Packages per method
- Cross-method comparison directories
- Traps

## OPT anatomy

Files per method (adam, levenberg_marquardt, score_based_lm, score_based_adam,
ReinforcementLearning):

| Role | File |
|---|---|
| config, filename builders, `flat_tasks`, `task_index_from_args`, `l96_experiment` | `experiment_config.jl` |
| truth data → `output/l63_computed_preliminaries.jld2`, `output/l96_computed_preliminaries_<fc>.jld2` | `l63_preliminaries.jl`, `l96_preliminaries.jl` |
| algorithm (`run_one` holds the loop) | `run_l63_<tag>.jl`, `run_l96_<tag>.jl` (RL: thin drivers + `rl_run.jl`) |
| leaderboard (per-seed JLD2 → nc) | `run_to_leaderboard.jl` |
| SLURM | `hpc-variant/` |

- `run_one(cfg, rmse_target, rng_idx, problem)` keeps every per-iteration
  quantity local and returns `(; conv_score, final_params, final_output)`.
  `main()` saves those with `JLD2.save(fn, "key", val, ...)`.
- The convergence check comes **before** the step. `conv_score` is the cost
  when the whitened RMSE first drops below the target, NaN if never:
  - adam / LM: `outer_iter * N_ens * (nu + 1)`, N_ens hard-coded to 1. LM's
    trial-step evaluation is not counted (a known side note; don't fix it).
  - score_based_*: `cost_accum` (N_ens per iteration, + N_ens·nu on AD-fallback iterations).
  - RL: window-fraction-weighted forward-evaluation equivalents.
  - CBO: `count * N_ens`.
- Loop landmarks:
  - adam: loop `run_l63_adam.jl:77`, Jacobian `:91`, J̃/r̃/RMSE `:93-95`,
    check `:97-102`, g + update `:111-116`. In `run_l96_adam.jl`: loop `:154`,
    J `:165`, g `:185`, update `:190`.
  - LM: loop `run_l63_lm.jl:64`, J `:78`, RMSE `:82`, QR step `:103-106`,
    trial residual (same IC) `:108-114`, ρ `:117`, accept/λ `:119-126`. In
    `run_l96_lm.jl`: `:143`, `:154`, `:182`, `:192`.
  - score_based_lm: loop `run_l63_sblm.jl:118`, collapsed-window AD fallback
    `:161-172`, GFDT `:174-184` (returns `stein_norm`, `b_mean_ratio`), LM step
    `:196-219`. The L96 scripts use ASCII names (`theta`, `Jt`, `rt`, `dtheta`).
  - score_based_lm prototypes: `trace_l63_body.jl` (per-iteration NamedTuple
    trace) and `trace_l96_body.jl` (prints `sigma_J`).
- Per-seed result: `output/<model>_<tag>_result_[<fc>_][<N_ens>_]<tgt with . → p>_<rng_idx>_<run_date>.jld2`,
  built by `result_filename(cfg, ...)`. Local adam/LM `output/` files from
  June 2026 use an older scheme (`l63_adam_result_1_10_2026-06-24.jld2`) that
  `result_filename` won't find. Report found vs expected rather than guessing.
- Leaderboard nc: `leaderboard_<tag>_<model>[_<fc>]_<date>.nc`, variable
  `metric[random_seed, ensemble_size, rmse_target, algorithm_type]`
  (`common/opt_metrics/write_results_nc.jl`).

## UQ anatomy

Stages: `calibrate_l{63,96}.jl` → (`emulate_sample_l{63,96}.jl` for CES) →
`pushforward_from_posterior_l{63,96}.jl` → `exp_to_leaderboard.jl`.

- Per cell: `output/<method_key>_<calibrate_date>/<model>_calibrate_results_[<fc>_]<N_ens>_<rng_idx>.jld2`
  (`calib_directory(cfg)`, `results_filename(cfg, N_ens, rng_idx)`). The
  pushforward stage appends `pushforward_output_samples` / `pushforward_k_values`
  with `jldopen(fn, "r+")`. Readers open single keys, so extra keys are safe.
- What each method saves (and discards):

| Method (`method_key`) | Saved per cell | Discarded / missing |
|---|---|---|
| GNKI (`gnki`) | `phi_stored` (ensemble per iteration, [1] = prior), `prior`, `y`, `R`, `truth_phi`, ... | `G_ens` (`calibrate_l96.jl:118`), `ekpobj`. Uses `DefaultScheduler(0.1)`, so Δt is constant |
| HistoryMatching (`history-matching`) | `waves` (GPs), `emulator_diagnostics` (LOO per wave), `posteriors_by_k`, `n_waves_completed` | NROY `n_tried` (`history_matching_core.jl` `rejection_sample_nroy`). Flux cells are ~2.5 GB |
| BOED (`boed`) | `posteriors_by_k`, `k_values`, `n_iters_completed` | GP hyperparameters (`extract_hyperparams` exists, `boed_core.jl:270`), TMCMC `_log_ev` (`:240`), EIG optimiser convergence, design `Z` |
| ABC (`abc`) | `phi_stored` (accepted pool per k), `pool_sizes`, `threshold` | per-draw implausibility (acceptance is derivable) |
| CES | full `ekpobj` in `l96_ekp_*.jld2`, `posteriors_by_k`, `input_output_pairs_by_k` | MCMC acceptance rate, step size (printed only) |

- Leaderboard nc variables (dims `random_seed, ensemble_size, ...`):
  `output_coverage[seed, ens, k_iter, coverage_quantile]`,
  `output_budget_to_target[seed, ens, target_scaling, coverage_quantile]`
  (NaN = never reached), `output_iters_to_target`, plus `post_mean`/`post_cov`.
  ABC adds `pool_size`. The metric is in `common/uq_metrics/coverage_metrics.jl`
  (`marginal_coverage`, `budget_to_target`).
- Some nc names have no `leaderboard_` prefix (HM, BOED, CES:
  `<key>_<model>_ensemble_results_<date>`). The templates' `diag_filename` handles both.

## Task indexing (nc seed → CLI task)

`flat_tasks(cfg)` lists tasks with `rng_idx` innermost, so:
- adam / LM / RL: task `(rmse_target, rng_idx)` → `t = (target_idx - 1) * n_repeats + rng_idx`
- score_based_*: `(N_ens, rmse_target, rng_idx)` → `t = ((ens_idx - 1) * n_targets + target_idx - 1) * n_repeats + rng_idx`
- uq: `(N_ens, rng_idx)` → `t = (ens_idx - 1) * n_repeats + rng_idx`.
  ABC runs every N_ens in one task per seed.
- CES maps `rng_idx` through `randperm(MersenneTwister(20260529), …)`
  (`calibrate_l96.jl:35`), so check the script before assuming the seed.

Run one cell: `julia --project=<method> <method>/run_l63_<tag>.jl <t>`. For
L96: `EXPERIMENT=l96_const julia --project=<method> <method>/run_l96_<tag>.jl <t>`.

## Status: success vs failure

- **opt:** `converged` if `conv_score` is finite, else `failed`. `no_diag` =
  file without a `"diagnostics"` key; `missing` = no file. Within one
  `rng_idx`, the three targets share the same RNG stream and trajectory up to
  the stopping point (adam/LM). Verify this before relying on it; score_based_*
  thread their sums.
- **uq (graded):** from `output_budget_to_target[seed, ens, 1, :]`
  (target_scaling index 1 = c 1.0). `frac_q_reached` = fraction of quantiles
  with a finite budget. Labels: `reached` (all), `partial`, `never` (none),
  `missing` (no nc entry). `metric` = median finite budget. Failure is often a
  *large* budget rather than NaN (GNKI l63 is ~1% NaN), so plots use the
  continuous score.

## Dates

- opt: `RUN_DATE` (else `today()`), stored as `cfg.run_date`, in every result filename.
- uq: `CALIBRATE_DATE`, stored as `cfg.calibrate_date`, in the cell directory name.
- Smoke runs: use a date no real run uses (e.g. `2099-01-01`) so they never
  overwrite real outputs. Check `output/` first; earlier tests left some
  (e.g. GNKI `output/gnki_2099-01-02/`).

## from-hpc layouts

- Usually `from-hpc/<YYYY-MM-DD>/` (copy date; files inside may carry another
  run date). LM is flat (`from-hpc/*.nc`). Some have subfolders
  (`score_based_adam/from-hpc/2026-09-10/scaled-alphas/`) or notes
  (`notes.md`, `note.md`). Read the notes.
- Mostly nc only. Exception: `HistoryMatching/from-hpc/2026-09-15/history-matching_2026-09-15/`
  holds one calibrate cell per case (l63 1 MB, const 20 MB, vec 130 MB, flux 2.5 GB).
- Pick "latest" by the date in the filename, then check `git log` for code
  changes since that date.
- From Python/h5py the nc dimension order is reversed (alg, target, ens, seed).

## Truth sources per case

The opt preliminaries files do **not** store the truth (`save_preliminaries`
drops `truth_params`). Copy θ_true from the method's `*_preliminaries.jl`:

| Case | θ_true (parameter space used by the run) | Note |
|---|---|---|
| l63 | `log.([28, 8/3])` | |
| l96_const | `log(8.0)` | nu = 1: cosines degenerate to sign agreement |
| l96_vec | the sinusoidal forcing vector in `l96_preliminaries.jl` | nu = 40 |
| l96_flux | none usable | NN weights are non-identifiable, and `run_l96_adam.jl:57-61` retrains the truth NN without a seed. Use forcing-space error on `sample_range` instead (an NN evaluation, not an ODE solve) |

UQ cells save `truth_phi` (and some save `truth_params_constrained`).

## Packages per method

Every experiment `Project.toml` has JLD2 and Plots/StatsPlots. uq ones also have NCDatasets.
- consensus_based_optimization: no Plots. Ask before adding it for the visualization script.
- ensemble_kalman_processes: no NCDatasets.
- Opt methods have Plots and StatsPlots listed but never used.
- Run scripts must not gain packages (SKILL.md § Rules). The combine and visualization
  scripts (and `common/diagnostics/{combine,diagnosis_plot}_core.jl`, which need JLD2,
  Statistics and, for the figure, Plots) may use anything already in the method's Project.

## Cross-method comparison directories

- Per-method: `<family>/<method>/diagnostics/{output/,*.jl}`: the method's
  `*_diagnostics.jl`, a thin `combine_diagnostics.jl` and
  `experiment_diagnosis_visualization.jl`.
- Shared code: `common/diagnostics/`: `diag_recorder.jl`, `lm_diagnostics.jl`,
  `regime_diagnostics.jl`, `checkpoint.jl` (mid-run `<result>_partial.jld2` for HPC time-outs; `partial_path`, `save_checkpoint`) (included by `run_*.jl`), and the offline modules
  `diag_files.jl` (`DiagFiles`: filename grammar, canonical-key aliases,
  `find_combined`, `find_combined_by_tag`, `resolve_series`), `combine_core.jl`
  (`DiagCombine.combine_opt`, `write_combined`) and `diagnosis_plot_core.jl`
  (`DiagPlot.run_diagnosis`, `visualize_file`).
- Cross-method: `<family>/diagnostics/<topic>/{*.jl,output/}`, where family is
  `opt_experiments` or `uq_experiments`. Existing: `opt_experiments/diagnostics/jacobian_exact_and_gfdt_comparison/`
  (adam/LM/score_based_* exact vs GFDT) and `opt_experiments/diagnostics/score-regime-dependence/`.
  `uq_experiments/diagnostics/` doesn't exist yet.
- Scripts there run with a method's project (`--project=levenberg_marquardt`)
  and take paths from `@__DIR__`, so moving a topic directory means rechecking
  its `dirname(...)` depth.
- ensemble_kalman_processes has one output file for all variants × N_ens × seeds
  (`output/l63_output_<date>.jld2`, `output/l96_output_<case>-force_<date>.jld2`,
  `cfg["case"]` = `"const-force"` etc.), so its `combine_diagnostics.jl` reshapes
  that file: one combined file per N_ens, slots = the variants. It has no
  Manifest; run it with `--project=../adam`.

## Traps

- `experiment_config.jl` sets `EXPERIMENT = experiments[3]` (l96_vec) as the
  default, and `l96_experiment()` reads `ARGS[2]`. Combine and plot scripts take
  `EXPERIMENT` from the environment only.
- The L63 run scripts hard-code `experiment = :l63`.
- Old local output naming (see OPT anatomy).
- `common-handler/references/common-layout.md` is stale (lists files that
  don't exist). Mention it if relevant; don't fix it here.
