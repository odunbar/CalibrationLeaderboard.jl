# <METHOD_NAME> — uq experiment

<One-line description of the method and its UQ approach.>

## One-time setup

In `experiment_config.jl`, pin the calibrate date before starting a run:
```julia
calibrate_date = Date("<YYYY-MM-DD>", "yyyy-mm-dd")
```

Precompile once (or after any package update):
```bash
bash submit_precompile.sh [EXP_ID]
```

## Pipeline

### L63
```
calibrate_array  ─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior  ─┬─(afterany)→  posterior_diagnostic_plots_l63
                                                                                                 └─(afterany)→  exp_to_leaderboard
```

### L96 (const / vec / flux)
```
                               ┌─(afterok)→  calibration_diagnostic_plots_l96
calibrate_array  ─(afterok)→──┤
                               └─(afterok)→  emulate_sample_array  ─(afterany)→  pushforward_from_posterior  ─┬─(afterany)→  posterior_diagnostic_plots_l96
                                                                                                               └─(afterany)→  exp_to_leaderboard
```

## Standalone (serial / local)

```bash
# L63
julia --project=. calibrate_l63.jl
julia --project=. emulate_sample_l63.jl
julia --project=. pushforward_from_posterior_l63.jl
julia --project=. exp_to_leaderboard.jl

# L96 — set EXPERIMENT env var
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. emulate_sample_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. exp_to_leaderboard.jl

# Single cell:
julia --project=. calibrate_l63.jl 1
EXPERIMENT=l96_vec julia --project=. calibrate_l96.jl 5
```

## HPC (Caltech Resnick cluster, SLURM)

Use the `slurm-pipeline-handler` skill to add sbatch and submit scripts.

```bash
bash submit_precompile.sh [EXP_ID]
bash submit_l63.sh        [EXP_ID]
bash submit_l96_const.sh  [EXP_ID]
bash submit_l96_vec.sh    [EXP_ID]
bash submit_l96_flux.sh   [EXP_ID]
```

Array upper bound = `length(N_ens_sizes) * n_repeats`.
Update in every array sbatch file when either changes in `experiment_config.jl`.

## Leaderboard metric

Primary metric: output-space coverage at marginal quantiles.
Budget metric: smallest `N_ens × k` (forward-model evaluations) to achieve calibrated coverage.
