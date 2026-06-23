# <METHOD_NAME> — opt experiment

<One-line description of the method and its optimization approach.>

## One-time setup

In `experiment_config.jl`, pin the run date before starting a batch:
```julia
run_date = Date("<YYYY-MM-DD>", "yyyy-mm-dd")
```

## Pipeline

```
run_array  ─(afterok)→  leaderboard
```

Each array task runs one `(N_ens, rng_idx)` cell. After all cells complete,
the leaderboard job collects results and writes the netcdf.

## Standalone (serial / local)

```bash
# L63 — all cells
julia --project=. run_l63_<METHOD_NAME>.jl

# L63 — single cell
julia --project=. run_l63_<METHOD_NAME>.jl 1

# L96 — all cells
EXPERIMENT=l96_const julia --project=. run_l96_<METHOD_NAME>.jl
EXPERIMENT=l96_vec   julia --project=. run_l96_<METHOD_NAME>.jl
EXPERIMENT=l96_flux  julia --project=. run_l96_<METHOD_NAME>.jl

# Leaderboard (after all cells):
julia --project=. run_to_leaderboard.jl
EXPERIMENT=l96_const julia --project=. run_to_leaderboard.jl
```

## HPC (Caltech Resnick cluster, SLURM)

See the `slurm-pipeline-handler` skill to add sbatch files.

Array upper bound = `length(N_ens_sizes) * n_repeats` (update in sbatch files when
these values change in `experiment_config.jl`).

## Leaderboard metric

Metric: number of forward-model evaluations to reach `target_rmse`.
Stored as `count * Ne` per cell; `NaN` if the method did not converge within `N_iter`.
