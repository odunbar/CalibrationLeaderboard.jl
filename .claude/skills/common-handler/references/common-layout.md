# common/ canonical layout

```
common/
├── forward_maps/
│   ├── Lorenz63.jl          ← L63 model (lorenz_forward, lorenz_solve, stats, f, RK4)
│   └── Lorenz96.jl          ← L96 model (const/vec/flux forcing subtypes + train_network)
├── opt_metrics/
│   ├── opt_score.jl         ← RMSE-to-target convergence score (from EKP/CBO run scripts)
│   └── write_results_nc.jl  ← write_results_nc() from CBO's jld2_to_netcdf_for_leaderboard.jl
└── uq_metrics/
    ├── coverage_metrics.jl  ← coverage-at-quantiles + budget-to-target computation
    └── write_uq_nc.jl       ← NetCDF writer extracted from exp_to_leaderboard.jl
```

Additional subdirs to create when needed (discuss with user first):
```
├── plotting/
│   └── simulation_plots.jl  ← shared pushforward ribbon / scatter plots
└── config/
    └── experiment_config.jl ← unified experiment_config(case) for all experiments
```

---

## What each subdir holds

### forward_maps/

The Lorenz forward maps are the single most duplicated code in the repo.
They are currently byte-identical in:
- `opt_experiments/ensemble_kalman_processes/Lorenz63.jl` (opt)
- `uq_experiments/calibrate_emulate_sample/Lorenz63.jl` (uq)
- (Lorenz96.jl likewise)

**Lorenz63.jl interface** (must be preserved when migrating):
- Structs: `LorenzConfig{FT1,FT2}(dt, T)`, `EnsembleMemberConfig{VV}(u)`,
  `ObservationConfig{FT1,FT2}(T_start, T_end)`
- Functions: `lorenz_forward`, `lorenz_solve`, `stats` (→ 9-element: 3 means, 3 vars, 3 cross-covs),
  `f(params, x)`, `RK4(params, xold, config)`

**Lorenz96.jl interface**:
- Abstract `EnsembleMemberConfig` with subtypes: `ConstantEMC`, `VectorEMC`, `FluxEMC`
- `build_forcing(::T, val, args...)`, `forcing(params, x, i)`, `forcing(params, x)`
- `train_network(model, x_train, y_train)` (Flux, 5000 epochs, Adam, MSE)
- `lorenz_forward`, `lorenz_solve`, `stats` (→ `2*N_state`: means then stds), `f`, `RK4`

### opt_metrics/

**opt_score.jl** — the optimization leaderboard metric:
```julia
# Forward-model evaluation count to convergence:
conv_score = count * Ne   # count = iterations taken, Ne = ensemble size
# NaN means "did not converge within N_iter"
```
Dimensions of the results array: `[n_methods, n_ens_sizes, n_rng_seeds]`

**write_results_nc.jl** — extracted from
`opt_experiments/consensus_based_optimization/jld2_to_netcdf_for_leaderboard.jl`:
```julia
function write_results_nc(filename;
    random_seed, ensemble_size, rmse_target, algorithm_type, metric)
```
Writes dims `random_seed, ensemble_size, rmse_target, algorithm_type` and a
`metric` var described as "Number of forward model evaluations (i.e., algorithm
cost)", `fillvalue=NaN`. The EKP opt experiment currently saves only JLD2 —
wiring it to `write_results_nc` is part of the opt migration.

### uq_metrics/

**coverage_metrics.jl** — extracted from `uq_experiments/calibrate_emulate_sample/exp_to_leaderboard.jl`:
```julia
# Key parameters
marginal_coverage_quantiles = collect(0.05:0.05:0.95)
budget_target_scalings = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
n_lowrank_modes = 5   # for PLR Mahalanobis (computed but de-emphasised)

# Primary metric: output-space coverage fraction at each quantile
output_coverage[q] = mean(y_samples .<= [quantile(os[:,d], q) for d in 1:n_output])

# Budget metric: smallest k where |S(q) - q| ≤ c·√(q(1-q)/N_y) for all q/c
# → budget_to_target = N_ens * k,  iters_to_target = k
```

**write_uq_nc.jl** — extracted from `exp_to_leaderboard.jl` (lines 291–426).
Opens `NCDataset(nc_save_filename, "c")`, defines dims and coord vars, writes
`output_coverage`, `output_budget_to_target`, `output_iters_to_target`, plus
param-space and (for L96) forcing-space arrays. The Mahalanobis/PLR variants
are computed-but-de-emphasised (display_metrics defaults skip them).

---

## Include-path table

All experiment scripts are one level below the repo root:
`{uq,opt}_experiments/<method_name>/script.jl`

| Target file | Include statement |
|---|---|
| `common/forward_maps/Lorenz63.jl` | `include(joinpath(@__DIR__, "..", "common", "forward_maps", "Lorenz63.jl"))` |
| `common/forward_maps/Lorenz96.jl` | `include(joinpath(@__DIR__, "..", "common", "forward_maps", "Lorenz96.jl"))` |
| `common/opt_metrics/write_results_nc.jl` | `include(joinpath(@__DIR__, "..", "common", "opt_metrics", "write_results_nc.jl"))` |
| `common/uq_metrics/coverage_metrics.jl` | `include(joinpath(@__DIR__, "..", "common", "uq_metrics", "coverage_metrics.jl"))` |

Always use `@__DIR__` (the directory of the current *file*), never `pwd()` or
a bare relative path, so scripts run correctly regardless of the working directory
when Julia was invoked.

---

## Current state of common/ (as of 2026-06-22)

All three subdirs exist and are **empty** — created as scaffolding.
Run `find common -type f | sort` to see the current state before any task.

## Files NOT to migrate (stay local)

- `uq_experiments/calibrate_emulate_sample/experiment_config.jl` — currently UQ-only;
  migrate to `common/config/` only when opt experiments adopt the same config structure.
- `uq_experiments/calibrate_emulate_sample/l63_exp_to_leaderboard_utilities.jl` /
  `l96_*` — legacy superseded by `exp_to_leaderboard.jl`; archive rather than migrate.
- Any method-specific calibrate/emulate/pushforward scripts — these are
  algorithm-specific and belong in the method's own experiment directory.
