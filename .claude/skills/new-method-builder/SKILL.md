---
name: new-method-builder
description: >-
  Scaffolds a new opt or uq experiment directory in CalibrationLeaderboard.jl
  following the established structure and the common/ framework. Use
  this skill whenever the user wants to: add a new optimization or UQ method to
  the leaderboard; create the skeleton of a new opt_experiments/<method>/ or
  uq_experiments/<method>/ directory; wire a new method to the shared forward
  maps and metric code in common/; understand how a new method should be
  structured; or start implementing a method and need the boilerplate done
  correctly so they can focus on the algorithm.
  Trigger even when the user says "add a new method", "create a new experiment",
  "scaffold <MethodName>", "I want to try <algorithm> on the leaderboard",
  "build a skeleton for my optimizer", or "how do I add a uq method".
---

# New Method Builder

This skill creates the skeleton for a new `opt` or `uq` experiment, wired to
`common/` and following the conventions of the existing experiments.

**Key assumption**: `common/forward_maps/` is already populated with
`Lorenz63.jl` and `Lorenz96.jl`. If it isn't, run the `common-handler` skill
first to migrate those files.

## Before starting: gather the required information

Ask the user (or infer from context):
1. **Method name** — e.g. `my_optimizer`, `consensus_based_v2`. This becomes
   the directory name and is used throughout the skeleton.
2. **Leaderboard type** — `opt` (optimization, scored on forward-eval count to
   convergence) or `uq` (uncertainty quantification, scored on coverage at quantiles).
3. **Which experiments to cover** — default is all four: `l63`, `l96_const`,
   `l96_vec`, `l96_flux`. Confirm if the user wants fewer.

## Directory to create

```
<opt|uq>_experiments/<method_name>/
```

## Step 1 — Copy the right skeleton from assets/

- OPT: copy files from `assets/opt-skeleton/` into the new directory.
- UQ: copy files from `assets/uq-skeleton/` into the new directory.

Then do a find-and-replace of `<METHOD_NAME>` with the actual method name
throughout all files.

## Step 2 — Project.toml

Copy the template from assets and fill in:
- **OPT**: deps matching `opt_experiments/ensemble_kalman_processes/Project.toml`
  (EnsembleKalmanProcesses, Distributions, JLD2, Plots, StatsPlots, StatsBase,
  Flux, BSON, FFTW, DocStringExtensions) **plus** NCDatasets (for write_results_nc).
  Add the new method's package if it's an unregistered local dep — document in
  `path_to_<method>.txt`.
- **UQ**: deps matching `uq_experiments/calibrate_emulate_sample/Project.toml`
  (CalibrateEmulateSample, EnsembleKalmanProcesses, NCDatasets, JLD2, Plots,
  StatsPlots, Distributions, DataFrames, StatsBase, FFTW, BSON, Flux,
  DocStringExtensions).
- **Gradient-based OPT methods**: also add `ForwardDiff` to Project.toml.
  ForwardDiff is not in the standard skeleton list.

## Step 3 — Wire common/ includes

At the top of every run/stage script, replace the include stubs with real paths:

```julia
const _COMMON = joinpath(@__DIR__, "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
# OPT:
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
# UQ:
include(joinpath(_COMMON, "uq_metrics", "coverage_metrics.jl"))
include(joinpath(_COMMON, "uq_metrics", "write_uq_nc.jl"))
```

Always use `@__DIR__`-relative paths — never bare relative includes — so the
scripts work regardless of what directory Julia was invoked from.

**Bootstrap check — `common/opt_metrics/write_results_nc.jl`**: this file is
referenced by every OPT script but ships as an empty directory. Before wiring
the includes, verify it exists:

```bash
ls common/opt_metrics/write_results_nc.jl
```

If missing, copy the implementation from the CBO experiment:

```bash
cp opt_experiments/consensus_based_optimization/jld2_to_netcdf_for_leaderboard.jl \
   common/opt_metrics/write_results_nc.jl
```

Then strip the CBO-specific top-level code and keep only the `write_results_nc`
function definition. Without this step the skeleton's `include` will fail at
runtime with a file-not-found error.

## Step 4 — Experiment config

Copy `experiment_config.jl` from the UQ example (or a future `common/config/`
version when it exists) into the new method's directory. The config provides:
- `experiment_config(case::Symbol)` → `(N_ens_sizes, n_repeats, N_iter, ...)`
  for each of `:l63`, `:l96_const`, `:l96_vec`, `:l96_flux`
- `flat_tasks(cfg)`, `task_index_from_args()`, `l96_experiment()` (dispatch)
- Filename builders: `calib_directory`, `nc_filename`, etc.

**Never use hard-coded `case = cases[N]` toggles.** Always read the experiment
from the env var / ARGS dispatch in `l96_experiment()`.

## Step 5 — Mark the customization point clearly

In the run/stage scripts, the method's core algorithm step is marked with:

```julia
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REPLACE: your method's ensemble/parameter update goes here             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Example placeholder (remove and replace with your method):
updated_params = params_i  # no-op placeholder
```

Everything else (problem setup, convergence check, score computation, netcdf
saving) is already wired up in the skeleton.

## OPT-specific: method-outermost loop structure

The existing `ensemble_kalman_processes` example nests methods as the *innermost*
loop — one JLD2 holds all four methods. The **intended** structure for new methods
is **one script per method** (already how CBO is organized), producing one result
file and one leaderboard netcdf per method.

The OPT skeleton enforces this: `run_l63_<METHOD>.jl` runs exactly one method.
If the user's method has variants (e.g. `method_v1` and `method_v2`), create
separate scripts for each.

## OPT-specific: convergence score

The opt leaderboard metric is the number of forward-model evaluations to reach
`target_rmse`. The skeleton computes this as:

```julia
if RMSE_e < target_rmse
    conv_score[ee, rr] = count * Ne   # forward eval count
    ...
    break
end
```

`NaN` if the method did not converge within `N_iter` iterations.
Results are `[n_ens_sizes, n_rng_seeds]`-dimensional (no method axis — one
script per method).

**Gradient-based methods**: replace `count * Ne` with `count * N_ens * (nu + 1)`
where `nu` is the parameter dimension and `+1` is the residual evaluation. The
`nu` factor accounts for the `nu` forward-mode AD passes that ForwardDiff uses
to compute the Jacobian. Typical values:

| Experiment | nu  | cost factor (nu+1) |
|---|---|---|
| L63        | 2   | 3  |
| L96 const  | 1   | 2  |
| L96 vec    | 40  | 41 |
| L96 flux   | 61  | 62 |

## OPT-specific: combined convergence-check + step loop

Do not separate the convergence check and the algorithm step into two sequential
loops over ensemble members. That forces a redundant `G` evaluation (once for
the RMSE check, again at the top of the update step). Instead, compute `G_j`
once and use it for both:

```julia
for j in 1:N_ens
    G_j    = G_func(θ_j)          # one evaluation
    r_j    = y - G_j
    RMSE_j = norm(R_inv_var * r_j) / sqrt(ny)

    if RMSE_j < cfg.target_rmse   # convergence check
        conv_score = ...; break
    end

    # algorithm step — reuses G_j / r_j computed above
    J_j = ForwardDiff.jacobian(G_func, θ_j)
    Δθ  = lm_solve(J_j, r_j, ...)
    θ_j = θ_j + Δθ
end
```

## OPT-specific: gradient-based methods

If the new method uses exact gradients (via ForwardDiff or another AD backend)
rather than an ensemble, several additional conventions apply.

### N_ens is IC-averaging count, not restart count

For ensemble methods, `N_ens` = number of particles, each contributing one
forward evaluation per iteration. For a gradient-based method with exact
Jacobians, multi-start optimization does not use the ensemble in any principled
way — the gradient computation is independent of N_ens.

The correct analogue is **IC-averaging**: draw `N_ens` IC perturbations per
step, average the resulting Jacobians and residuals, then take one gradient
step. This reduces noise from stochastic IC perturbations and gives N_ens the
same "budget per iteration" interpretation as in EKI.

```julia
J_sum = zeros(ny, nu); r_sum = zeros(ny)
for k in 1:N_ens
    x0p_k  = x0 .+ ic_cov_sqrt * randn(rng, nx)   # draw OUTSIDE closure (see below)
    G_func = θ -> lorenz_forward(..., x0p_k, ...)
    G_k    = G_func(θ)
    r_sum += y - G_k
    J_sum += ForwardDiff.jacobian(G_func, θ)
end
J_avg = J_sum / N_ens
r_avg = r_sum / N_ens
```

Set `N_ens_sizes = [1, 5, 10]` (not `[20, 25, 30]`). `N_ens = 1` is pure
gradient descent with a single noisy Jacobian; larger values average out IC
noise. The sweep shows the variance-reduction benefit, not a particle-count
benefit.

### IC perturbation must be drawn outside the ForwardDiff closure

`ForwardDiff.jacobian` traces dual numbers through every operation in the
closure. If `randn` is called inside the closure, ForwardDiff will attempt to
differentiate through the random draw and either error or silently return zeros.
Always fix the perturbation before creating the closure:

```julia
# CORRECT — x0p is plain Float64, only θ is dual
x0p    = x0 .+ ic_cov_sqrt * randn(rng, nx)
G_func = θ -> lorenz_forward(..., x0p, ...)
J      = ForwardDiff.jacobian(G_func, θ)

# WRONG — randn inside the closure
G_func = θ -> lorenz_forward(..., x0 .+ ic_cov_sqrt * randn(rng, nx), ...)
J      = ForwardDiff.jacobian(G_func, θ)   # differentiates through randn!
```

### Reuse IC perturbations for the trial step

When doing a gain-ratio check (LM accept/reject), evaluate `G(θ + Δθ)` under
the **same** IC perturbations used for the Jacobian, not fresh draws. This
makes the RMSE comparison purely attributable to Δθ:

```julia
x0p_all = x0 .+ ic_cov_sqrt * randn(rng, nx, N_ens)  # fix once per outer iter

# ... compute J_avg, r_avg using x0p_all[:, k] ...

θ_trial = θ + Δθ
r_trial = mean(y - lorenz_forward(..., x0p_all[:, k], ...) for k in 1:N_ens)
```

### Lorenz ODE ForwardDiff compatibility

Both `Lorenz63.jl` and `Lorenz96.jl` are ForwardDiff-compatible: they use
`promote_type(eltype(params.u), eltype(x0))` and `zeros(promoted_type, ...)`
throughout the RK4 integrator, so dual numbers propagate cleanly from `params.u`
through the entire ODE. No modifications to the forward maps are needed.

Exception: the **flux-force** L96 case (`FluxEMC`). Flux chains store Float32
weights internally. ForwardDiff should propagate dual numbers through the chain
(via `Flux.destructure` / `reconstructor`), but this path is less tested. If
the flux-force Jacobian errors, the likely fix is casting the params vector to
Float32 duals or switching to a reverse-mode backend (Zygote).

## UQ-specific: pipeline stages

The UQ skeleton has four stage scripts, all with `main()` + dispatch:

| Script | Role | Array? |
|---|---|---|
| `calibrate_<MODEL>.jl` | EKP calibration loop | ✓ per cell |
| `emulate_sample_<MODEL>.jl` | Build emulator, run MCMC | ✓ per cell |
| `pushforward_from_posterior_<MODEL>.jl` | Push posterior through forward map | ✓ per cell |
| `exp_to_leaderboard.jl` | Coverage metrics → netcdf | single (all cells) |

**Required fifth stage — preliminaries.** `calibrate_<MODEL>.jl`'s problem
setup includes an expensive shared computation (a Lorenz spin-up +
synthetic-observation generation via `compute_perfect_data`). Never compute it
inline in the calibrate script — every array task would recompute it, and a
naive `isfile`-guarded save races when many tasks start at once. Always split
it into its own `l63_preliminaries.jl` / `l96_preliminaries.jl` that computes
and saves unconditionally (single serial job, no race), and have
`calibrate_<MODEL>.jl` load-or-error instead of compute-or-load. Scaffold this
for every new UQ method — two worked examples to follow:
`uq_experiments/GaussNewtonKalmanInversion/` and
`uq_experiments/calibrate_emulate_sample/`.

After scaffolding, run `slurm-pipeline-handler` to add the HPC variant (it
covers the preliminaries stage's SLURM wiring too).

## Step 6 — README.md

Copy `assets/opt-skeleton/README-skeleton.md` or `assets/uq-skeleton/README-skeleton.md`,
fill in method name, experiment type, pipeline graph, and serial/HPC commands.
For HPC commands, refer to `slurm-pipeline-handler`.

## Step 7 — Ensure common/ consistency via common-handler

After scaffolding, invoke the **`common-handler`** skill on the newly created and
modified files. This catches three classes of problem that the scaffolding steps
alone cannot prevent:

1. **Missing common/ files** — a new method may reference a `common/` file that
   does not yet exist (e.g., `write_results_nc.jl` was bootstrapped from CBO but
   was never reviewed for completeness, or a metrics file was declared but left
   empty). `common-handler` verifies every `include(joinpath(_COMMON, ...))` call
   resolves to a real, non-empty file.

2. **Duplicated logic** — the scaffolded scripts may contain helper code (e.g. a
   bespoke RMSE function, a custom netcdf writer) that already exists in another
   experiment and belongs in `common/` instead. `common-handler` identifies these
   candidates and extracts them.

3. **Path drift** — if `common/` was restructured since the skeleton assets were
   last updated, the include paths in the new method will be wrong. `common-handler`
   can detect and correct the drift.

To invoke it, pass the list of files created or modified in Steps 1–6. In practice
this means telling the skill: **"Use common-handler to check consistency for these
files: `<opt|uq>_experiments/<method_name>/` and any `common/` files touched above."**

Do not skip this step even if Steps 1–6 appeared to go smoothly — the scaffolding
is template-driven and cannot know whether `common/` has drifted since the templates
were written.

## Verification checklist

After creating the skeleton:
- [ ] `ls <opt|uq>_experiments/<method_name>/` shows expected files
- [ ] UQ only: `l63_preliminaries.jl` / `l96_preliminaries.jl` exist and
      `calibrate_<MODEL>.jl` loads from them (`isfile(prelim_file) || error(...)`)
      rather than computing truth data inline
- [ ] `grep -r 'include.*Lorenz' <opt|uq>_experiments/<method_name>/`
      shows only `@__DIR__`-relative common/ paths (no bare includes)
- [ ] `ls common/opt_metrics/write_results_nc.jl` exists (OPT only —
      create it if missing; see Step 3)
- [ ] `julia --project=<opt|uq>_experiments/<method_name> -e 'include("<script>.jl")'`
      resolves includes without error (pkg loads may fail if Project.toml deps
      aren't instantiated, but the include paths themselves should resolve)
- [ ] `<METHOD_NAME>` placeholder is replaced everywhere
- [ ] `# REPLACE` markers are present at the algorithm customization points
- [ ] A README.md exists in the new directory
- [ ] Gradient-based methods: `ForwardDiff` is in `Project.toml` and IC
      perturbations are drawn **outside** all ForwardDiff closures

## Cross-skill references

- Run `common-handler` first if `common/forward_maps/` is still empty (pre-condition).
- Run `common-handler` again after scaffolding (Step 7) to validate and fix common/ consistency.
- Run `slurm-pipeline-handler` after scaffolding to add SLURM support.

## Final step — improve this skill

After finishing, offer to improve the **new-method-builder** skill via
skill-creator: "Would you like to improve the **new-method-builder** skill using
skill-creator? You can share suggestions, or I can analyse what came up this
session — e.g. a package that wasn't in the template, a stage that needed extra
wiring, or a pattern worth encoding — to refine the skill for next time."
