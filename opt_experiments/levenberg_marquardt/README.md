# gradient_descent — opt experiment

Levenberg-Marquardt optimization of the quadratic likelihood

```
L(θ) = (y − G(θ))ᵀ R⁻¹ (y − G(θ))
```

Jacobians `∂G/∂θ` are computed by **ForwardDiff.jl** (forward-mode AD through
the Lorenz ODE solver). Each outer iteration runs one LM step, stopping when
the whitened RMSE drops below `target_rmse`.

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
> per outer iteration). LM is most efficient for small-nu problems.
> For L96 flux, ForwardDiff compatibility depends on Flux.Chain supporting dual-
> number arithmetic; this is expected to work but is less tested.

## LM hyperparameters

| Parameter | Value | Role |
|---|---|---|
| `λ` (initial) | 1.0 | Levenberg-Marquardt damping |
| λ increase | `× 4` when ρ < 0.25 or non-finite | Reject step, increase regularisation |
| λ decrease | `÷ 3` when ρ > 0.75 | Accept step, reduce regularisation |
| `λ_max` | 1e8 | Damping ceiling |
| `λ_min` | 1e-10 | Damping floor |
| `N_iter` | 50 | Max outer iterations (LM converges fast) |
| `target_rmse` | 1.2 | Convergence criterion (whitened RMSE) |
| `N_ens` | 1 | IC perturbations averaged per step (pure LM) |

The LM step uses augmented-system QR (`vcat(J̃, √λ·diag(d))`) with MINPACK
column scaling, so the condition number scales as κ(J̃) rather than κ(J̃)².

## One-time setup

Pin the run date in `experiment_config.jl` before starting a batch:
```julia
run_date = Date("YYYY-MM-DD", "yyyy-mm-dd")
```

## Standalone (serial / local)

Run prelims once before the first run (or if the output directory is empty):

```bash
# L63 prelims
julia --project=. l63_preliminaries.jl

# L96 prelims (one case at a time)
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_vec   julia --project=. l96_preliminaries.jl
EXPERIMENT=l96_flux  julia --project=. l96_preliminaries.jl
```

Then run the optimisation:

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

All HPC files live in `hpc-variant/`. Run every command below from that directory.

### Before the first submission (or after any package update)

```bash
cd hpc-variant/
bash submit_precompile.sh
```

Wait for the precompile job to finish before launching array jobs (`squeue -u $USER`).

### Before every batch submission

**1. Pin the run date** in `experiment_config.jl` so all array tasks write to the
same output directory even when jobs span midnight:

```julia
# experiment_config.jl
run_date = Date("2026-06-29", "yyyy-mm-dd")   # replace with today's date
```

**2. Verify the array size** in `hpc-variant/run_array.sbatch` matches
`length(N_ens_sizes) × n_repeats` from `experiment_config.jl`:

```
#SBATCH --array=1-20   # must equal length(N_ens_sizes) * n_repeats
```

With the current defaults (`N_ens_sizes = [1]`, `n_repeats = 20`) the bound is
`1 × 20 = 20`. If you change either value in `experiment_config.jl`, update
`--array` in both `run_array.sbatch` and the relevant `submit_l*.sh` to match.

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

Each `submit_x.sh` chains three SLURM jobs automatically:

```
preliminaries (single) →(afterok)→ run_array (--array=1-20) →(afterany)→ leaderboard (single)
```

The `afterany` dependency on the leaderboard means it fires even if some array
tasks fail — `run_to_leaderboard` reads whatever JLD2 files are present and
warns on missing ones.

### Smoke test (single task before full submission)

```bash
cd hpc-variant/
sbatch --array=1-1 --export=ALL,SCRIPT=run_l63_gradient_descent.jl,EXPERIMENT=l63 run_array.sbatch
```

### Monitoring

```bash
squeue -u $USER                                          # live queue
cat ../output/slurm/run_<JOBID>_<TASKID>.out            # stdout for a specific task
cat ../output/slurm/run_<JOBID>_<TASKID>.err            # stderr / errors
cat ../output/slurm/prelim_<JOBID>.out                  # preliminaries job output
```

## Pipeline

```
l63_preliminaries (single)
        ↓ afterok
run_l63_gradient_descent (array, 20 tasks)
        ↓ afterany
run_to_leaderboard (single)

l96_preliminaries (single, per case)
        ↓ afterok
run_l96_gradient_descent (array, 20 tasks per case)
        ↓ afterany
run_to_leaderboard (single, per case)
```
