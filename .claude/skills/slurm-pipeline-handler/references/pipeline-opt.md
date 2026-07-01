# OPT Pipeline Reference

Most OPT experiments (`opt_experiments/`) still have **no SLURM support**.
This document specifies the target pipeline to build, modeled on the UQ example.

## Current state (as of 2026-06-30)

| Experiment | Local scripts | SLURM | experiment_config.jl |
|---|---|---|---|
| `adam/` | run_l63_adam.jl, run_l96_adam.jl, run_to_leaderboard.jl | ✅ hpc-variant/ | ✅ |
| `levenberg_marquardt/` | run_l63_lm.jl, run_l96_lm.jl, run_to_leaderboard.jl | ✅ hpc-variant/ | ✅ (RUN_DATE convention) |
| `ensemble_kalman_processes/` | run_l63_example.jl, run_l96_example.jl | ❌ | ❌ |
| `consensus_based_optimization/` | run_l63_example_cbo.jl, run_l96_example_cbo.jl | ❌ | ❌ |
| `batch_stochastic_gradient_descent/` | — (empty) | ❌ | ❌ |

`opt_experiments/adam/hpc-variant/` is the canonical reference for overall layout.
`opt_experiments/levenberg_marquardt/` is the canonical reference for the `RUN_DATE`
convention (see "Pin the run date via RUN_DATE" in SKILL.md) — prefer it as the
template for new methods, and consider porting `adam/` to match.

## Target dependency graph

### L63 opt pipeline (with shared preliminaries)
```
preliminaries  ─(afterok)→  run_array  ─(afterok)→  leaderboard
```
(preliminaries = single serial job; run_array = per-(N_ens, rng_idx) cell; leaderboard = write_results_nc, all cells)

### L96 opt pipeline (with shared preliminaries)
```
preliminaries  ─(afterok)→  run_array  ─(afterok)→  leaderboard
```
(same, with EXPERIMENT=l96_const|l96_vec|l96_flux on both preliminaries and run_array)

The `preliminaries` pre-stage is needed when run scripts share expensive setup (truth
data, obs covariance, ICs) that would create a race condition if each array task computed
it independently. Without shared setup, omit `preliminaries` and use:
```
run_array  ─(afterok)→  leaderboard
```

There is no multi-stage dependency here (unlike UQ's calibrate→emulate→pushforward
chain) because OPT methods run to completion in a single pass.

## Sbatch file spec for OPT

| File | Type | Description |
|---|---|---|
| `preliminaries.sbatch` | single job | Compute and save shared setup (truth data, obs cov, ICs); omit if no shared setup |
| `run_array.sbatch` | array `1-N%100` | One task per `(N_ens, rng_idx)` cell; model-agnostic via SCRIPT env var |
| `leaderboard.sbatch` | single job | Runs write_results_nc over all cells; depends on run_array |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

Array upper bound: `length(N_ens_sizes) * n_repeats`
(EKP example uses 3 × 2 = 6 cells; future methods will vary)

## Submit script spec

| Script | Chains |
|---|---|
| `submit_precompile.sh [EXP_ID]` | precompile.sbatch only |
| `submit_l63.sh [EXP_ID]` | preliminaries(l63) →(afterok)→ run_array(l63) →(afterok)→ leaderboard |
| `submit_l96_const.sh [EXP_ID]` | preliminaries(l96_const) →(afterok)→ run_array(l96_const) →(afterok)→ leaderboard |
| `submit_l96_vec.sh [EXP_ID]` | same with EXPERIMENT=l96_vec |
| `submit_l96_flux.sh [EXP_ID]` | same with EXPERIMENT=l96_flux |

(Omit the `preliminaries` step for methods that have no shared expensive setup.)

## Prerequisites before adding SLURM to an OPT experiment

Use `opt_experiments/adam/` as the canonical example — it has everything in place.
The existing run scripts for EKP and CBO need these changes first:

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
# If the method uses shared preliminaries, run this first:
julia --project=. l63_preliminaries.jl
EXPERIMENT=l96_const julia --project=. l96_preliminaries.jl

# All cells:
julia --project=. run_l63_<method>.jl
EXPERIMENT=l96_const julia --project=. run_l96_<method>.jl

# Single cell:
julia --project=. run_l63_<method>.jl 3
EXPERIMENT=l96_vec julia --project=. run_l96_<method>.jl 5

# Leaderboard (after all cells have run):
julia --project=. run_to_leaderboard.jl
```

## hpc-variant/ layout for OPT (adam is the canonical example)

OPT run scripts use `@__DIR__` for `common/` includes but a bare `include("experiment_config.jl")`
for config. This means you do NOT need to copy scripts into `hpc-variant/`. Instead:

**Directory structure (with shared preliminaries — levenberg_marquardt is the
canonical example of the RUN_DATE convention; adam is the canonical example of
the overall layout):**
```
<method>/
├── experiment_config.jl         ← single config: reads RUN_DATE env var, falls back to today()
├── l63_preliminaries.jl         ← compute/save shared L63 setup once before array
├── l96_preliminaries.jl         ← same for L96; case via EXPERIMENT env var
├── run_l63_<method>.jl          ← stays here; NOT copied to hpc-variant/
├── run_l96_<method>.jl
├── run_to_leaderboard.jl
└── hpc-variant/
    ├── preliminaries.sbatch     ← single serial pre-stage job
    ├── run_array.sbatch
    ├── leaderboard.sbatch
    ├── precompile.sbatch
    ├── submit_precompile.sh
    ├── submit_l63.sh
    └── submit_l96_*.sh
```

**Without shared preliminaries** (method has no expensive shared setup), omit the
`l*_preliminaries.jl` scripts and `preliminaries.sbatch` entirely — the layout and
chains revert to the simpler `run_array →(afterok)→ leaderboard` form.

**sbatch invocation pattern** (submit from `hpc-variant/`, run from `<method>/`):
```bash
cd "${SLURM_SUBMIT_DIR}"                              # = hpc-variant/
julia --project=.. "../${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
```

- `include("experiment_config.jl")` resolves relative to the script's own directory
  (`<method>/`, not `pwd`) → always picks up the single shared config ✓
- `@__DIR__ = <method>/` (script's real location) → `../../common` paths resolve correctly ✓
- `--project=..` → uses `<method>/Project.toml` ✓
- `RUN_DATE` set by `submit_*.sh` and passed via `--export` is what makes that shared
  config resolve to the right date for this run — see "Pin the run date via RUN_DATE"
  in SKILL.md.

**Log paths** in OPT sbatch must use `../output/slurm/` (not `output/slurm/`):
```
#SBATCH --output=../output/slurm/run_%A_%a.out
```

**Submit scripts** do `mkdir -p ../output/slurm` and submit sbatch from within `hpc-variant/`.
