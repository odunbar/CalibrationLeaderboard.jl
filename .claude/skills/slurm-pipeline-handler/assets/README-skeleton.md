# <METHOD_NAME> — <opt|uq> experiment

<One-line description of the method and what leaderboard type it targets.>

## One-time setup

The run date is set automatically: `submit_*.sh` computes `RUN_DATE` once and threads
it through every job in the pipeline; `experiment_config.jl` reads it from the
environment and falls back to `today()` for local runs — no manual pinning needed.
See "Pin the run date via RUN_DATE" in the `slurm-pipeline-handler` skill.

Precompile once before array jobs:

```bash
bash submit_precompile.sh [EXP_ID]
```

## Pipeline

### L63
```
<STAGE1>  ─(afterok)→  <STAGE2>  ─(afterany)→  <STAGE3>
```

### L96 (const / vec / flux)
```
<STAGE1>  ─(afterok)→  <STAGE2>  ─(afterany)→  <STAGE3>
```

## Standalone (serial / local)

```bash
# L63
julia --project=. <MAIN_SCRIPT_L63>.jl

# L96 — set EXPERIMENT env var
EXPERIMENT=l96_const julia --project=. <MAIN_SCRIPT_L96>.jl
EXPERIMENT=l96_vec   julia --project=. <MAIN_SCRIPT_L96>.jl
EXPERIMENT=l96_flux  julia --project=. <MAIN_SCRIPT_L96>.jl

# Single cell (pass 1-based task index):
julia --project=. <MAIN_SCRIPT_L63>.jl 1
EXPERIMENT=l96_const julia --project=. <MAIN_SCRIPT_L96>.jl 3
```

## HPC (Caltech Resnick cluster, SLURM)

### Submission (recommended)

```bash
bash submit_precompile.sh [EXP_ID]
bash submit_l63.sh        [EXP_ID]
bash submit_l96_const.sh  [EXP_ID]
bash submit_l96_vec.sh    [EXP_ID]
bash submit_l96_flux.sh   [EXP_ID]
```

All four cases can run simultaneously — output paths are case-specific.

### Manual submission

```bash
# L63
RUN_DATE=$(date +%Y-%m-%d)
STAGE1_JID=$(sbatch --parsable -A esm \
             --export=ALL,SCRIPT=<SCRIPT_L63>,RUN_DATE=${RUN_DATE} <STAGE1>.sbatch)
sbatch -A esm \
       --dependency=afterok:${STAGE1_JID} \
       --kill-on-invalid-dep=yes \
       --export=ALL,RUN_DATE=${RUN_DATE} \
       <STAGE2>.sbatch
```

### Sbatch files reference

| File | Type | Description |
|---|---|---|
| `<STAGE1>.sbatch` | array (1–<N_TASKS>) | <Description> |
| `<STAGE2>.sbatch` | single job | <Description> |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

### Adjusting array size

The sbatch files default to `--array=1-<N_TASKS>` (`length(N_ens_sizes) * n_repeats`).
If you change either in `experiment_config.jl`, update the upper bound in every
array sbatch file.

### Smoke test

Before a full submission, run a single task to verify file resolution:

```bash
sbatch --array=1-1 --export=ALL,SCRIPT=<SCRIPT_L63> <STAGE1>.sbatch
```
