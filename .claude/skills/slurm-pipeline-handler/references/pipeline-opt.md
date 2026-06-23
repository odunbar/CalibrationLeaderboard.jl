# OPT Pipeline Reference

The OPT experiments (`opt_experiments/`) currently have **no SLURM support**.
This document specifies the target pipeline to build, modeled on the UQ example.

## Current state (as of 2026-06-22)

| Experiment | Local scripts | SLURM | experiment_config.jl |
|---|---|---|---|
| `ensemble_kalman_processes/` | run_l63_example.jl, run_l96_example.jl | ❌ | ❌ |
| `consensus_based_optimization/` | run_l63_example_cbo.jl, run_l96_example_cbo.jl | ❌ | ❌ |
| `batch_stochastic_gradient_descent/` | — (empty) | ❌ | ❌ |

## Target dependency graph

### L63 opt pipeline
```
run_array  ─(afterok)→  leaderboard
```
(run_array = per-(N_ens, rng_idx) cell; leaderboard = write_results_nc, all cells)

### L96 opt pipeline
```
run_array  ─(afterok)→  leaderboard
```
(same, with EXPERIMENT=l96_const|l96_vec|l96_flux on the array job)

There is no multi-stage dependency here (unlike UQ's calibrate→emulate→pushforward
chain) because OPT methods run to completion in a single pass.

## Sbatch file spec for OPT

| File | Type | Description |
|---|---|---|
| `run_array.sbatch` | array `1-N%100` | One task per `(N_ens, rng_idx)` cell; model-agnostic via SCRIPT env var |
| `leaderboard.sbatch` | single job | Runs write_results_nc over all cells; depends on run_array |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

Array upper bound: `length(N_ens_sizes) * n_repeats`
(EKP example uses 3 × 2 = 6 cells; future methods will vary)

## Submit script spec

| Script | Chains |
|---|---|
| `submit_precompile.sh [EXP_ID]` | precompile.sbatch only |
| `submit_l63.sh [EXP_ID]` | run_array(SCRIPT=run_l63_<method>.jl) →(afterok)→ leaderboard |
| `submit_l96_const.sh [EXP_ID]` | run_array(SCRIPT=run_l96_<method>.jl, EXPERIMENT=l96_const) →(afterok)→ leaderboard |
| `submit_l96_vec.sh [EXP_ID]` | same with EXPERIMENT=l96_vec |
| `submit_l96_flux.sh [EXP_ID]` | same with EXPERIMENT=l96_flux |

## Prerequisites before adding SLURM to an OPT experiment

The existing run scripts need these two changes first:

### 1. Add experiment_config.jl (or wire to common/config/)

Create `experiment_config.jl` in the method's directory (or in `common/config/`)
providing:
- `experiment_config(case::Symbol)` → `(N_ens_sizes, n_repeats, N_iter, target_rmse, ...)`
- `flat_tasks(cfg)` → `[(N_ens, rng_idx) for ...]`
- `task_index_from_args()` — reads `SLURM_ARRAY_TASK_ID` → `ARGS[1]` → nothing
- `l96_experiment()` — reads `EXPERIMENT` env var → `ARGS[2]` → toggle

### 2. Refactor the run scripts

Replace the current monolithic loop structure with `main()`:

```julia
function run_one(cfg, N_ens, rng_idx)
    # single (N_ens, rng_idx) cell
    # ...your method's iteration loop...
    # conv_score = count * Ne   (or NaN if not converged)
    return conv_score, final_params, final_output
end

function main()
    experiment = l96_experiment()
    cfg = experiment_config(experiment)
    tidx = task_index_from_args()
    tasks = flat_tasks(cfg)
    
    if tidx === nothing
        for (N_ens, rng_idx) in tasks
            score, params, output = run_one(cfg, N_ens, rng_idx)
            save_result(score, params, output, N_ens, rng_idx)
        end
    else
        (N_ens, rng_idx) = tasks[tidx]
        score, params, output = run_one(cfg, N_ens, rng_idx)
        save_result(score, params, output, N_ens, rng_idx)
    end
end

main()
```

Replace hard-coded `case = cases[N]` toggles with `experiment = l96_experiment()`.

### 3. Method-outermost loop (intended refactor)

The current EKP example nests methods as the INNERMOST loop — one JLD2 file
holds all four methods. The intended structure is method-outermost: one run
script per method (already the case for CBO), producing one score set / one
leaderboard file per method. The `new-method-builder` skill scaffolds new
methods this way from the start.

## Serial (local) invocation for OPT

```bash
# All cells:
julia --project=. run_l63_<method>.jl
EXPERIMENT=l96_const julia --project=. run_l96_<method>.jl

# Single cell:
julia --project=. run_l63_<method>.jl 3
EXPERIMENT=l96_vec julia --project=. run_l96_<method>.jl 5

# Leaderboard (after all cells have run):
julia --project=. run_to_leaderboard.jl
```
