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

See `slurm-pipeline-handler` to generate sbatch files.
Array upper bound = `length(N_ens_sizes) × n_repeats` (update when those
change in `experiment_config.jl`).

## Pipeline

```
run_l63/l96  (array) ─(afterok)→  run_to_leaderboard (single)
```
