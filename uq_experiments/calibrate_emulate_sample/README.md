# EKIRace

Calibrate, emulate-sample, and posterior-pushforward scripts for the Lorenz 63
and Lorenz 96 benchmark experiments. These are the same scripts used for both
local (serial) runs and SLURM job-array runs — there is exactly one copy of
each; see [`hpc-variant/README.md`](hpc-variant/README.md) for the HPC side.

## Experiments

| Symbol | Model | Forcing type |
|--------|-------|-------------|
| `:l63` | Lorenz 63 | — |
| `:l96_const` | Lorenz 96 | scalar constant |
| `:l96_vec` | Lorenz 96 | spatial vector |
| `:l96_flux` | Lorenz 96 | neural-network flux |

## One-time setup

In `experiment_config.jl`, set `EXPERIMENT` to the case you want to run.

```julia
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT  = experiments[3]   # :l96_vec
```

`calibrate_date` defaults to `today()` but is overridden by the `CALIBRATE_DATE`
env var when set — the HPC `submit_l*.sh` scripts pin it automatically so every
stage in a chain agrees, even across a midnight boundary. For a manual local
run spanning multiple days, set it yourself before starting:

```bash
CALIBRATE_DATE=2026-06-04 julia --project=. -e 'include("calibrate_l63.jl")'
```

All subsequent stages include `experiment_config.jl` and read this value, so
keeping it fixed ensures every stage finds the right output directory.

## Running scripts

Run all scripts from the `examples/EKIRace/` directory using:

```bash
julia --project=. -e 'include("scriptname.jl")'
```

Each script loops over all `(N_ens, rng_idx)` cells defined in `experiment_config.jl`
and writes output to `output/<method>_<date>/`.

## Stages

### 1. Preliminaries

```bash
# L63 (EXPERIMENT setting is ignored)
julia --project=. -e 'include("l63_preliminaries.jl")'

# L96 — reads EXPERIMENT from experiment_config.jl
julia --project=. -e 'include("l96_preliminaries.jl")'
```

Computes the shared truth-data trajectory and observations once, serially,
and writes it to `output/`:

- `l63_preliminaries.jl` → `output/l63_computed_preliminaries.jld2`
- `l96_preliminaries.jl` → `output/l96_computed_preliminaries_<force_case>.jld2`

Run this before calibrate — on SLURM, extracting it out of `calibrate_l*.jl`
avoids every array task racing to compute and write the same file (see
`hpc-variant/README.md`). Calibrate errors out if the prelim file is missing.

### 2. Calibrate

```bash
# L63 (EXPERIMENT setting is ignored)
julia --project=. -e 'include("calibrate_l63.jl")'

# L96 — reads EXPERIMENT from experiment_config.jl
julia --project=. -e 'include("calibrate_l96.jl")'
```

### 3. Calibration diagnostic plots (L96 only)

```bash
julia --project=. -e 'include("calibration_diagnostic_plots_l96.jl")'
```

Reads calibration output and produces data, solution (mean ensemble), and
full-ensemble spread figures for each `(N_ens, rng_idx)` cell.  Can run
in parallel with stage 4.

### 4. Emulate and sample

```bash
# L63
julia --project=. -e 'include("emulate_sample_l63.jl")'

# L96
julia --project=. -e 'include("emulate_sample_l96.jl")'
```

Reads the calibration output for each `(N_ens, rng_idx)` cell, trains an
emulator, runs MCMC, and writes per-cell posterior `.jld2` files to the
calibration output directory.

### 5. Pushforward from posterior

```bash
# L63
julia --project=. -e 'include("pushforward_from_posterior_l63.jl")'

# L96
julia --project=. -e 'include("pushforward_from_posterior_l96.jl")'
```

For each posterior `.jld2`, draws samples from the fitted posterior and runs
them through the Lorenz forward map, saving the resulting forcing and output
sample arrays back into the same `.jld2` file.  This step is required before
running stage 6 (posterior diagnostics) or stage 7 (leaderboard).

### 6. Posterior diagnostic plots (L96 only)

```bash
julia --project=. -e 'include("posterior_diagnostic_plots_l96.jl")'
```

Reads each posterior `.jld2` (including the pushforward samples written in
stage 5) and writes pushforward and posterior-ribbon diagnostic plots into
the calibration output directory.

### 7. Leaderboard

Convert the per-cell posterior files into a leaderboard NetCDF file:

```bash
# L63
julia --project=. -e 'include("l63_exp_to_leaderboard_utilities.jl")'

# L96
julia --project=. -e 'include("l96_exp_to_leaderboard_utilities.jl")'
```

These scripts require the pushforward samples from stage 5 to be present in
each posterior `.jld2`.

To compute summary metrics from the resulting NetCDF file, pair the `filename`
and `prelim_jld2_file` variables at the top of `compute_leaderboard_metrics.jl`
to point to the target file and its prelim JLD2, then run:

```bash
julia --project=. -e 'include("compute_leaderboard_metrics.jl")'
```

This prints per-ensemble-size calibration scores (Mahalanobis, log-posterior
ratio, marginal coverage) against chi-squared reference quantiles, and saves
budget-for-coverage figures to `indir/`.  The prelim JLD2 is needed for
R-whitened PCA coverage; without it, only raw coverage is reported.

## Full run example (L96 vector forcing)

```julia
# In experiment_config.jl:
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT  = experiments[3]   # :l96_vec
```

```bash
export CALIBRATE_DATE=2026-06-04   # pins the output directory for every stage below
julia --project=. -e 'include("l96_preliminaries.jl")'
julia --project=. -e 'include("calibrate_l96.jl")'
julia --project=. -e 'include("calibration_diagnostic_plots_l96.jl")'   # optional
julia --project=. -e 'include("emulate_sample_l96.jl")'
julia --project=. -e 'include("pushforward_from_posterior_l96.jl")'
julia --project=. -e 'include("posterior_diagnostic_plots_l96.jl")'     # optional
julia --project=. -e 'include("l96_exp_to_leaderboard_utilities.jl")'
```

## HPC variant

For parallelised SLURM job-array runs on the Caltech Resnick cluster, see
[`hpc-variant/README.md`](hpc-variant/README.md).
