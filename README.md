# CalibrationLeaderboard.jl

Simulations and pipelines that generate the results shown on the
[Calibration Benchmark leaderboard](https://huggingface.co/spaces/calibrationcomp/calibration_benchmark/)
on Hugging Face. This repository is the source of truth for how each entry on
the leaderboard is produced: given a forward map (e.g., Lorenz63, Lorenz96) and
a calibration/UQ method, it runs the experiment, computes the agreed-upon
metrics, and exports results in the format the leaderboard app consumes.

## Structure

- `common/` — shared code used across experiments: forward maps
  (`forward_maps/`), optimization metrics (`opt_metrics/`), and UQ metrics
  (`uq_metrics/`). Anything more than one experiment needs should live here
  rather than being duplicated.
- `opt_experiments/` — one directory per optimization method (e.g. Adam,
  Levenberg-Marquardt, Ensemble Kalman Processes, Consensus-Based
  Optimization), each following a common local/HPC layout.
- `uq_experiments/` — one directory per UQ method (currently
  Calibrate-Emulate-Sample), each following the same calibrate → emulate/sample
  → pushforward → diagnostics/leaderboard pipeline shape, with local and
  `hpc-variant` versions.
- `src/` — the `CalibrationLeaderboard` package itself.

## Designed to be extensible

This repo is built so that adding a new method or a new experiment case is a
mechanical, low-risk operation rather than a one-off reinvention: new methods
plug into the same `common/` forward maps and metrics, and follow the same
local + HPC pipeline conventions as existing ones. The goal is that the
benchmark grows by *composition* (new experiment directories) rather than by
duplicating or forking existing code.

## Claude Code skills

This is a first attempt at making the repository itself Claude Code-native.
Clone the repo and run `claude` inside it — three skills under
`.claude/skills/` capture the conventions for extending the benchmark:

- **`new-method-builder`** — scaffolds a new `opt_experiments/<method>/` or
  `uq_experiments/<method>/` directory with the established structure, wired
  up to the shared forward maps and metrics in `common/`.
- **`common-handler`** — manages `common/`: migrating duplicated code
  (e.g. forward maps) into it, adding new shared structs/helpers, and fixing
  experiment include paths to point at the shared code instead of local
  copies.
- **`slurm-pipeline-handler`** — creates and maintains the local + HPC (SLURM)
  variants of an experiment's pipeline, including sbatch files, job
  dependency chains, and array sizes for sweeps over ensemble sizes/repeats.

### Suggested skill pipeline

A typical flow for adding a new method to the leaderboard:

1. **`new-method-builder`** — scaffold the new experiment directory so the
   algorithm has a correctly structured home from the start.
2. **`common-handler`** — as the method takes shape, make sure it's actually
   reusing shared forward maps/metrics (and push any new reusable pieces into
   `common/`) so results stay comparable and repeatable across methods.
3. **`slurm-pipeline-handler`** — once the method runs locally, add/adapt the
   SLURM pipeline for your own HPC system to scale up to the number of
   repeats/ensemble sizes needed for leaderboard-quality results.
