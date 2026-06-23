---
name: common-handler
description: >-
  Manages shared code in common/ in CalibrationLeaderboard.jl — both
  migrating existing duplicated files into common/ and adding new shared
  abstractions (structs, methods, helpers) directly to common/ files.
  Use this skill whenever the user wants to: consolidate duplicated forward maps
  (Lorenz63.jl / Lorenz96.jl); populate common/forward_maps/,
  common/opt_metrics/, or common/uq_metrics/ with code
  extracted from individual experiments; fix the broken include("../Lorenz63.jl")
  paths in the CBO scripts; rewire any experiment's include paths to point to
  common/ rather than local copies; add a new shared struct or method to a
  common/ file; extract an abstraction that both experiments need; write new
  code that belongs in common/; or decide what belongs in common/ vs. stays
  experiment-specific.
  Trigger even when the user says "consolidate", "move to shared", "deduplicate
  Lorenz files", "fix the CBO includes", "extract metrics to common",
  "add a shared struct", "extract an abstraction to common/",
  "write new code that both experiments need", "add methods to a common file",
  or "what goes in common/".
---

# Common Handler

The `common/` directory is the single home for code shared across
all experiments and both leaderboard types:

```
common/
├── forward_maps/    ← Lorenz63.jl, Lorenz96.jl (and future models)
├── opt_metrics/     ← RMSE-to-target scorer + write_results_nc()
└── uq_metrics/      ← coverage-at-quantiles scorer + uq NetCDF writer
```

See `references/common-layout.md` for the full canonical layout.

There are two distinct workflows depending on what the user is doing. Read
the right one — most of the steps don't overlap.

---

## Workflow A — Migrating existing duplicated code into common/

Use this when the same file already exists in two or more experiment
directories and needs a single authoritative home.

### Step A1 — Diff before moving

```bash
diff uq_experiments/calibrate_emulate_sample/Lorenz63.jl \
     opt_experiments/ensemble_kalman_processes/Lorenz63.jl
```

- **Byte-identical**: safe to move one copy to `common/` and delete the others.
- **Diverge**: show the diff to the user and ask which version is authoritative,
  or whether the divergence is intentional. Never silently clobber a meaningful
  experiment-specific change.

### Step A2 — Copy to common/, delete originals

```bash
cp uq_experiments/calibrate_emulate_sample/Lorenz63.jl \
   common/forward_maps/Lorenz63.jl
# delete duplicates after rewiring (Step A3)
```

### Step A3 — Rewrite include paths

Replace every in-experiment `include("Lorenz63.jl")` or `include("../Lorenz63.jl")`
with the canonical `@__DIR__`-relative path. The key reason for `@__DIR__` (not
a cwd-relative path) is that Julia scripts are invoked from many working
directories — SLURM array tasks, local runs, and interactive sessions all `cd`
to different places.

**Path depth table:**

| Script location | Correct include path |
|---|---|
| `uq_experiments/<method>/foo.jl` | `joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz63.jl")` |
| `opt_experiments/<method>/foo.jl` | `joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz63.jl")` |
| `uq_experiments/<method>/hpc-variant/foo.jl` | `joinpath(@__DIR__, "..", "..", "..", "common", "forward_maps", "Lorenz63.jl")` |

See `assets/common-include-header.jl` for a ready-to-paste snippet.

**When more than ~5 files need rewiring, use a script rather than individual
edits.** A Python loop that opens each file, replaces the old include string,
and prints `OK / SKIPPED / MISSING` per file is faster, less error-prone, and
self-documenting.

**Known broken includes to fix immediately:**

`opt_experiments/consensus_based_optimization/run_l63_example_cbo.jl`
and `run_l96_example_cbo.jl` previously had broken `include("../LorenzNN.jl")`
paths — these have been fixed as of 2026-06-22, but check for regressions.

### Step A4 — Smoke-test each touched experiment

```bash
julia --project=uq_experiments/calibrate_emulate_sample \
      -e 'include("uq_experiments/calibrate_emulate_sample/calibrate_l63.jl"); println("OK")'
```

### Step A5 — Flag naming inconsistencies, don't silently fix them

The EKP opt experiment labels `Inversion(prior)` as `"TEKI"`, while the UQ
experiment labels `Inversion()` as `"EKI"`. These are distinct: `Inversion()`
is standard EKI; `Inversion(prior)` is Tikhonov-regularised (TEKI). Surface
any inconsistency to the user before touching any name.

**Distinguish naming context from a change request.** If the user responds with
a clarification ("TEKI uses `Inversion(prior)`, not `Inversion`") rather than
an explicit instruction to rename, treat it as informational context only and
continue with the originally requested migration task. Do not attempt a rename
until the user explicitly asks for one.

### After migration — verification checklist

- [ ] `find common -type f` shows the migrated files
- [ ] No stale copies remain on disk:
      `find . -name "Lorenz*.jl" ! -path "*/common/*"`
      (should print nothing)
- [ ] No include lines still point to a local copy:
      `grep -rn 'include.*Lorenz' . --include="*.jl" | grep -v 'common'`
      (should print nothing)
- [ ] Each smoke-test passes (Step A4) for every touched experiment
- [ ] Git diff confirms only the expected files moved/changed
- [ ] CBO scripts have a working include path and (ideally) a Project.toml

---

## Workflow B — Adding new shared code to common/ files

Use this when the user wants to write a new struct, method, or helper that
belongs in common/ rather than migrating an existing file.

### Step B1 — Confirm it belongs in common/

Apply the same decision guide as for migration:

**Goes to common/** when:
- The code is meaningful to every (experiment × leaderboard_type) pair.
- Future methods will need it without modification.
- It expresses a concept at the level of the forward model or metric, not at
  the level of one method's algorithmic pipeline.

**Stays experiment-local** when:
- It is tightly coupled to one method's structure (e.g. the CES emulate/sample
  pipeline, a method-specific plotting script).
- It would need to be modified for each new method that adopts it.

### Step B2 — Follow Julia shared-library conventions

Code in common/ is included by every experiment, so it must be written to the
standard of a shared library, not a script. The most important rule:

**Use parametric structs with abstract type bounds — never hardcode concrete
element types in struct field definitions.**

Concrete types like `Vector{Float64}` or `Matrix{Float64}` prevent the struct
from working with other numeric types and defeat Julia's type inference. Use
abstract bounds instead:

```julia
# Wrong — locks out Float32, sparse matrices, GPU arrays, etc.
struct MyConfig
    x0  :: Vector{Float64}
    R   :: Matrix{Float64}
    cfg :: LorenzConfig    # fine — already a concrete parametric type
end

# Right — parametric on the actual storage types
struct MyConfig{VV <: AbstractVector, MM <: AbstractMatrix, LC <: LorenzConfig}
    x0  :: VV
    R   :: MM
    cfg :: LC
end
```

For scalar hyperparameters, use `FT <: Real` rather than `Float64`. For config
objects that already have their own type parameters (`LorenzConfig{FT1,FT2}`,
`ObservationConfig{FT1,FT2}`), capture the full concrete type with a single
type parameter (`LC <: LorenzConfig`) so the struct stays concrete after
construction — this is what gives Julia its speed.

### Step B3 — Keep JLD2 save/load backward-compatible

When adding helpers that read or write JLD2 files (e.g. `save_preliminaries`,
`load_preliminaries`), match the exact key names that the existing calibrate and
leaderboard scripts already use. Downstream scripts load specific keys by name;
a silent rename breaks them.

`save_preliminaries` already handles atomic writes internally (tmp file → `mv`),
so callers must not add their own tmp-file logic around it. Just call
`save_preliminaries(pdc, filepath)` directly, and wrap the call in a `try/catch`
only when SLURM array tasks may race:

```julia
try
    save_preliminaries(pdc, prelim_file)
    @info "Saved computed quantities to $(prelim_file)"
catch
    @info "Prelim file already written by another task; discarding duplicate"
end
```

The `try/catch` handles the case where `mv` fails because a concurrent task
already renamed its own tmp file to `prelim_file` first. Do not `rm` the tmp
file in the catch — `save_preliminaries` cleans up after itself.

### Step B4 — Add `using` imports to the common/ file if needed

Check the file's existing imports before adding new ones. For example,
`Lorenz96.jl` did not originally import `JLD2` — adding `save_preliminaries`
required adding `using JLD2`. Keep imports at the top of the file.

### Step B5 — Smoke-test all experiments that include the modified file

After appending new code, verify the file still loads cleanly in every
experiment that includes it (not just the one you were working in):

```bash
find . -name "*.jl" | xargs grep -l 'common/forward_maps/Lorenz63' \
  | head -3   # spot-check a few callers
julia --project=... -e 'include("<caller>"); println("OK")'
```

---

## What belongs in common/ (decision guide)

**Go to common/** when:
- The same file exists verbatim in ≥2 experiment dirs.
- The code is meaningful to every (experiment × leaderboard_type) pair — e.g.
  the four forward maps, the two metric families (opt/uq), shared config helpers.
- Future methods will need it without modification.

**Stay experiment-local** when:
- The code is tightly coupled to one method's algorithmic structure (e.g. the
  CES emulate/sample pipeline, a method-specific plotting script).
- It has already diverged from the "identical" copy in another experiment.

**Add subdirs to common/ as needed** — e.g. `common/plotting/` for shared
simulation-output plots, `common/config/` for a unified `experiment_config.jl`
— but discuss with the user before creating new top-level subdirectories so the
layout stays intentional.

---

## Tracking common/ population state

Before starting any task, check what is already in common/:

```bash
find common -type f | sort
```

As of 2026-06-22, `forward_maps/` contains `Lorenz63.jl` and `Lorenz96.jl`
(each with `PerfectDataConfig`, `compute_perfect_data`, `save_preliminaries`,
and `load_preliminaries`); `opt_metrics/` and `uq_metrics/` are still empty.

Also note: `uq_experiments/calibrate_emulate_sample/experiment_config.jl` and
its `hpc-variant/experiment_config.jl` are intentionally kept as two separate
files — do not propose merging them.

---

## Using `compute_perfect_data` and `save_preliminaries` from experiment scripts

When updating a calibrate script to use `PerfectDataConfig` instead of its own
manual spin-up/R-estimation/IC-cov computation, follow this pattern:

### Flat scripts (non-hpc calibrate_l*.jl)

Replace the manual block with a single `compute_perfect_data(...)` call, then
unpack the fields you need, then save:

```julia
ny = 9   # or 2*nx for L96
t = 0.01
T_start = 30.0   # problem-specific
T_end = T
pdc = compute_perfect_data(
    truth_params, nx, ny,                              # L63 signature
    LorenzConfig(t, 1000.0), rand(rng_i, Normal(0, 1), nx),
    LorenzConfig(t, T), ObservationConfig(T_start, T_end),
)
# For L96, omit ny and pass R_inflation=inff as a keyword argument.
x0                     = pdc.x0
y                      = pdc.y
lorenz_config_settings = pdc.lorenz_config_settings
observation_config     = pdc.observation_config
R                      = pdc.R
R_inv_var              = pdc.R_inv_var
ic_cov_sqrt            = pdc.ic_cov_sqrt

prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
if !isfile(prelim_file)
    save_preliminaries(pdc, prelim_file)
    @info "Saved computed quantities to $(prelim_file)"
end
```

Do not duplicate `R_sqrt = sqrt(R)` — it is not saved to or loaded from the
prelim file, and downstream code does not use it.

### if/else load-or-compute patterns (e.g. non-hpc calibrate_l96.jl)

Call `save_preliminaries` inside the `else` branch only — `pdc` does not exist
in the `if` (load) branch and should not be returned from it:

```julia
if isfile(prelim_file)
    loaded_data = JLD2.load(prelim_file)
    x0 = loaded_data["x0"]
    # ... unpack the rest ...
else
    pdc = compute_perfect_data(phi, nx, ...; R_inflation = inff)
    x0 = pdc.x0; y = pdc.y; ...
    save_preliminaries(pdc, prelim_file)
    @info "Saved computed quantities to $(prelim_file)"
end
```

### Functions that return NamedTuples (hpc-variant build_setup)

When `compute_perfect_data` is called inside a function that returns a
NamedTuple, include `pdc` in the tuple. The caller needs it to call
`save_preliminaries` — it cannot reconstruct `pdc` from the unpacked fields:

```julia
function build_setup(cfg)
    ...
    pdc = compute_perfect_data(...)
    x0 = pdc.x0; y = pdc.y; ...
    return (; nx, nu, ny, x0, y, R, R_inv_var, ic_cov_sqrt,
              lorenz_config_settings, observation_config, ..., pdc)
end

function main()
    setup = build_setup(cfg)
    ...
    if !isfile(prelim_file)
        try
            save_preliminaries(setup.pdc, prelim_file)
            @info "Saved computed quantities to $(prelim_file)"
        catch
            @info "Prelim file already written by another task; discarding duplicate"
        end
    end
end
```

For hpc-variant l96, where `build_setup` has an if/else load-or-compute block,
call `save_preliminaries(pdc, prelim_file)` with the `try/catch` inside the
`else` branch — do not return `pdc` from `build_setup` in that case, since
`pdc` doesn't exist in the load branch.

---

## Final step — improve this skill

After finishing, offer to improve the **common-handler** skill itself using
skill-creator: "Would you like to improve the **common-handler** skill via
skill-creator? You can share suggestions, or I can analyse what happened this
session — e.g. new workflows encountered, Julia conventions that came up, or
steps that felt incomplete — to refine the skill for next time."
