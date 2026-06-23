# gradient_descent — opt experiment

Levenberg-Marquardt optimization of the quadratic likelihood

```
L(θ) = (y − G(θ))ᵀ R⁻¹ (y − G(θ))
```

Jacobians `∂G/∂θ` are computed by **ForwardDiff.jl** (forward-mode AD through
the Lorenz ODE solver). Each outer iteration runs `N_ens` independent LM
restarts in parallel (one step per restart), stopping when any restart reaches
`target_rmse`.

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

> **Note for L96 vec/flux**: the Jacobian is expensive (nu=40 or 61 ODE passes
> per restart per outer iteration). LM is most efficient for small-nu problems.
> For L96 flux, ForwardDiff compatibility depends on Flux.Chain supporting dual-
> number arithmetic; this is expected to work but is less tested.

## One-time setup

Pin the run date in `experiment_config.jl` before starting a batch:
```julia
run_date = Date("YYYY-MM-DD", "yyyy-mm-dd")
```

## Standalone (serial / local)

```bash
# L63 — all cells
julia --project=. run_l63_gradient_descent.jl

# L63 — single cell
julia --project=. run_l63_gradient_descent.jl 1

# L96 — all cells (one case at a time)
EXPERIMENT=l96_const julia --project=. run_l96_gradient_descent.jl
EXPERIMENT=l96_vec   julia --project=. run_l96_gradient_descent.jl
EXPERIMENT=l96_flux  julia --project=. run_l96_gradient_descent.jl

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
