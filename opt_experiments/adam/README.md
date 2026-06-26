# adam — opt experiment

Adam optimization of the quadratic likelihood

```
L(θ) = (y − G(θ))ᵀ R⁻¹ (y − G(θ))
```

Jacobians `∂G/∂θ` are computed by **ForwardDiff.jl** (forward-mode AD through
the Lorenz ODE solver). The gradient `∇L = -J̃ᵀ r̃` (where `J̃ = R^{-1/2} J`
and `r̃ = R^{-1/2} (y - G(θ))`) is fed to the Adam update rule with
bias-corrected first and second moment estimates.

`N_ens` IC perturbations are averaged at each step to reduce gradient noise.

## Cost metric

```
conv_score = outer_iter × N_ens × (nu + 1)
```

where `nu` is the parameter dimension and the `+1` accounts for the residual
evaluation. The Jacobian costs `nu` forward-model evaluations (one per
parameter, via ForwardDiff forward-mode).

| Experiment   | nu  | cost factor (nu+1) |
|---|---|---|
| L63          | 2   | 3 |
| L96 const    | 1   | 2 |
| L96 vec      | 40  | 41 |
| L96 flux     | 61  | 62 |

## Adam hyperparameters

| Parameter | Value | Role |
|---|---|---|
| α   | 0.01  | Step size |
| β₁  | 0.9   | First-moment decay |
| β₂  | 0.999 | Second-moment decay |
| ε   | 1e-8  | Numerical stability |

Tune `α` in `run_l63_adam.jl` / `run_l96_adam.jl` if convergence is slow or
unstable. Larger problems (L96 vec/flux) may benefit from a smaller `α`.

## One-time setup

Pin the run date in `experiment_config.jl` before starting a batch:
```julia
run_date = Date("YYYY-MM-DD", "yyyy-mm-dd")
```

## Standalone (serial / local)

```bash
# L63 — all cells
julia --project=. run_l63_adam.jl

# L63 — single cell
julia --project=. run_l63_adam.jl 1

# L96 — all cells (one case at a time)
EXPERIMENT=l96_const julia --project=. run_l96_adam.jl
EXPERIMENT=l96_vec   julia --project=. run_l96_adam.jl
EXPERIMENT=l96_flux  julia --project=. run_l96_adam.jl

# Write leaderboard netcdf (after all cells for a given experiment):
julia --project=. run_to_leaderboard.jl
EXPERIMENT=l96_const julia --project=. run_to_leaderboard.jl
EXPERIMENT=l96_vec   julia --project=. run_to_leaderboard.jl
EXPERIMENT=l96_flux  julia --project=. run_to_leaderboard.jl
```

## HPC (SLURM)

All HPC files live in `hpc-variant/`. Run every command below from that
directory.

### Before the first submission (or after any package update)

```bash
cd hpc-variant/
bash submit_precompile.sh
```

Wait for the precompile job to finish before launching array jobs (`squeue -u $USER`).

### Before every batch submission

**1. Pin the run date** in `hpc-variant/experiment_config.jl` so all array
tasks write to the same output directory even when jobs span midnight:

```julia
# hpc-variant/experiment_config.jl
run_date = Date("2026-06-25", "yyyy-mm-dd")   # replace with today's date
```

**2. Verify the array size** in `hpc-variant/run_array.sbatch` matches
`length(N_ens_sizes) × n_repeats` from `experiment_config.jl`:

```
#SBATCH --array=1-20%100   # must equal length(N_ens_sizes) * n_repeats
```

With the current defaults (`N_ens_sizes = [1]`, `n_repeats = 20`) the bound
is `1 × 20 = 20`. If you change either value in `experiment_config.jl`, update
`--array` to match — mismatched bounds will either skip cells or submit
out-of-range task IDs that fail immediately.

> **Note:** there are two `experiment_config.jl` files — `adam/experiment_config.jl`
> (used for local runs) and `hpc-variant/experiment_config.jl` (used by SLURM
> jobs). Edit `hpc-variant/experiment_config.jl` for HPC submissions. Keep both
> in sync when changing `n_repeats` or `N_ens_sizes`.

### Submitting

```bash
cd hpc-variant/

# L63
bash submit_l63.sh

# L96 cases (submit independently; all can run concurrently)
bash submit_l96_const.sh
bash submit_l96_vec.sh
bash submit_l96_flux.sh
```

An optional `EXP_ID` label keeps the queue readable when running multiple
batches simultaneously:

```bash
bash submit_l63.sh run2   # jobs appear as run_l63_run2, leaderboard_l63_run2
```

Each `submit_x.sh` chains two SLURM jobs automatically:

```
run_array (--array=1-N) →(afterok)→ leaderboard (single task)
```

The leaderboard job is cancelled automatically (`--kill-on-invalid-dep=yes`)
if any array task fails.

### Smoke test (single task before full submission)

```bash
cd hpc-variant/
sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_adam.jl,EXPERIMENT=l63 run_array.sbatch
```

### Monitoring

```bash
squeue -u $USER                       # live queue
cat ../output/slurm/run_<JOBID>_<TASKID>.out   # stdout for a specific task
cat ../output/slurm/run_<JOBID>_<TASKID>.err   # stderr / errors
```

## Pipeline

```
run_l63/l96  (array) ─(afterok)→  run_to_leaderboard (single)
```
