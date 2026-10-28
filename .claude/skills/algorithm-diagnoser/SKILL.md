---
name: algorithm-diagnoser
description: >-
  Investigates an existing leaderboard method in opt_experiments/ or
  uq_experiments/ of CalibrationLeaderboard.jl and adds cheap convergence
  diagnostics that explain why runs score badly. It records per-iteration
  quantities the algorithm already computes (and saves objects it currently
  throws away, e.g. a GNKI ekpobj), combines them across all seeds into one
  file per experiment, and writes experiment_diagnosis_visualization.jl to
  plot successful vs failed seeds. Use this skill whenever the user wants to:
  diagnose why a method fails, returns NaN, or scores badly on the leaderboard;
  add convergence or per-iteration diagnostics to run_*.jl or calibrate_*.jl;
  check gradient or Jacobian magnitude and direction quality (adam, LM,
  score-based); inspect ensemble collapse, spread or misfit (GNKI, EKI, CES);
  check whether history matching or BOED rules out the truth; save
  ekpobj, n_tried, GP hyperparameters, or other discarded state; combine
  diagnostics across seeds; compare a good seed with a bad one; or compare
  diagnostics between methods or variants (e.g. exact vs GFDT Jacobians).
  Trigger even when the user says "why does adam fail", "why is LM NaN on
  l96_vec", "debug convergence", "is the Jacobian garbage", "track the
  gradient", "save the ekpobj", "plot good vs bad seeds", "what goes wrong
  across the iterations", or "diagnose the leaderboard score".
---

# Algorithm Diagnoser

Take one existing method (e.g. `opt_experiments/adam/`,
`uq_experiments/KalmanSampler/`), find out *why* its runs do or
don't reach the leaderboard target, and leave behind:

| Deliverable | Where |
|---|---|
| Per-run diagnostics: a `"diagnostics"` key in each existing result JLD2 (plus any discarded object worth saving, e.g. `"ekpobj"`) | additive edits to `run_*.jl` / `calibrate_*.jl` |
| One combined file per experiment, all seeds: `diagnostics_<tag>_<model>[_<fc>][_N<n>]_<date>.jld2` | `<method>/diagnostics/combine_diagnostics.jl` → `diagnostics/output/` |
| One diagnosis figure per experiment (PNG + PDF) | `<method>/diagnostics/experiment_diagnosis_visualization.jl` → `diagnostics/output/` |

The combine and figure scripts are thin: a method states only its tag, slots and
`PANELS`, and the shared code in `common/diagnostics/` does the rest. `*.jld2`,
`*.png`, `*.pdf` are gitignored. `references/worked-examples.md` walks through
adam, LM, GNKI and HM.

## Where diagnostics files go

| Scope | Location |
|---|---|
| Code several methods use as is: the recorder, the combine and plot cores, the file grammar, a diagnostics class two methods share | `common/diagnostics/` (`references/repo-layout.md` § Cross-method comparison directories lists it) |
| One algorithm: its `*_diagnostics.jl` (own call signature and keys), thin `combine_diagnostics.jl` and `experiment_diagnosis_visualization.jl`, `output/` | `<opt\|uq>_experiments/<method>/diagnostics/` |
| Compares two or more methods or variants (cross-method tables, overlays, regime studies) | `<opt\|uq>_experiments/diagnostics/<topic>/` (+ `output/`) |

**One copy of shared code.** Before writing a diagnostics file, look in
`common/diagnostics/` and in the sibling methods' `diagnostics/`. If another
method already has it, include the `common/` copy; if only a sibling has it,
and your method would copy it, move it to `common/diagnostics/` with the
`common-handler` skill (merge near-copies into the superset). A copy per method
drifts: the four recorders had already diverged from the template when this
was consolidated.

**Comparisons** live in a topic directory (create `uq_experiments/diagnostics/`
the first time a uq comparison needs it), named for what is compared, e.g.
`opt_experiments/diagnostics/jacobian_exact_and_gfdt_comparison/`. They read each
method's combined file from `<method>/diagnostics/output/` through
`DiagFiles.find_combined_by_tag` and `resolve_series`, write only to their own
`output/`, and locate the family root with `dirname(dirname(@__DIR__))`. Don't
copy combined files into the topic directory, and don't put comparison scripts
in a method's `diagnostics/`. Usage comments give the path from the family root,
e.g. `julia --project=levenberg_marquardt diagnostics/<topic>/compare_table.jl`.

**One-off studies** (a regime map, a derivative-problems histogram) may use their
own panels and output names. They still take file names from `DiagFiles` and
read the combined file rather than the per-seed results.

## Rules

**Don't change the run.** A diagnosis is only useful if it describes the run
the leaderboard scored, so:
- no extra forward-model evaluations (GP/NN predictions on *saved* emulators
  are fine; a "true derivative" reference belongs in an offline script such as
  `score_based_lm/validate_jacobian_l63.jl`);
- no RNG draws (including EKP accessors that sample), no in-place mutation of
  algorithm state, no tidying of algorithm lines;
- don't touch `conv_score`, `cost_accum`, `n_fwd_actual`, budgets or iteration
  counts, and compute anything non-trivial only on recorded iterations;
- add keys, never rename or overwrite them (leaderboard readers load single
  keys, so extras are invisible); `DIAG=0` turns recording off;
- no new packages in run scripts or in the files they include
  (`common/diagnostics/diag_recorder.jl` and the `*_diagnostics.jl` classes are
  Base + LinearAlgebra + Statistics; a new dependency forces a cluster
  re-precompile); `hpc-variant/` changes go through slurm-pipeline-handler.

**Storage cap.** Opt runs can last 10,000 iterations:
- ≤ 500 records per run (`DIAG_MAX_RECORDS`, ceiling 1000): dense for the
  first 50 iterations, log-spaced after, final iteration kept.
  `common/diagnostics/diag_recorder.jl` does this; its window accumulators
  (`_wmean`, `_wmin`, `_wmax`) keep rates between records.
- Float32 scalars; per-component vectors only when nu ≤ 10; never
  per-iteration Jacobians or ensembles.
- ~50 KB per opt run file, ≤ 20% growth of a uq cell file, ≤ 50 MB per combined file.

## 1. Investigate and assess the cause

Read the method's README, run/calibrate loop, `experiment_config.jl`,
leaderboard script, and any `from-hpc/**/notes.md` or `VALIDATION.md`
(`references/repo-layout.md` says where each lives). From the latest
`from-hpc/` nc files and `output/`, build a **failure map**: experiment × slot
(target or N_ens) → fraction failed and the per-seed outcome (opt: `metric`
NaN; uq: graded, see repo-layout § Status). If `git log` shows the code changed
since that HPC run, use the map only to choose seeds. Note what the loop
computes and discards, and what is already saved.

Then commit to the best-supported cause (a second only if the evidence is
genuinely split) and tell the user, briefly:
1. **The cause**, with the evidence behind it.
2. **The diagnostic that, had it been saved, would identify it**: name,
   formula, where in the loop its inputs already exist, and the value that
   confirms the cause versus a healthy run.

Keep it to 3–5 keys: the outcome (`rmse`), the quantity carrying the cause, and
the algorithm's consequence (e.g. step size). They become the figure's panels.
Formulas and blind spots are in `references/catalog-opt.md` /
`references/catalog-uq.md`.

| Cause | Diagnostic that identifies it |
|---|---|
| derivative magnitude invalid (AD through chaos) | `log10_J` ≫ 3, `jac_plaus` ≫ 3 |
| derivative direction is noise | `cos_truth_wmean` / `cos_prev_wmean` ≈ 0 |
| Adam frozen by one old gradient spike in v̂ | `vhat_ratio_wmean` ≫ 0 with `step_w_wmean` → 0 (even when the gradient is sane) |
| step below Float64 resolution / over-damped (LM) | `step_quant` ≠ 1, `log10_lambda` at cap |
| uq ensemble overconfident | `out_z` ≫ 1, `spread_w` collapsing faster than misfit |
| HM rules out the truth | `I2_truth_over_thr` > 1 at some wave |

If the smoke run contradicts the assessment, revise it and say so (adam did:
magnitude looked like the cause, the runs showed Adam's step freezing).

## 2. Instrument

Get each diagnostic from the cheapest source:
- **Already saved** (ABC `pool_sizes`, CES `ekpobj`, HM `waves`): compute it in
  the combine script and rerun nothing.
- **Built, then thrown away**: save the object if it has per-iteration history,
  fits the storage cap (`Base.summarysize`) and loads with packages already in
  the Manifest (GNKI `ekpobj`), or just its small summary fields (GP
  hyperparameters, `n_tried`, BOED `log_ev`).
- **Overwritten each iteration** (adam/LM's J̃, g, r̃; anything needing the
  previous iteration): record values.

To record, include `common/diagnostics/diag_recorder.jl` (Base only; never copy
it, so a recorder fix reaches every method). `record!` point values before the
convergence check, `accumulate!` per-step values every iteration, `finalize!` at the convergence break and after
the loop, then add `"diagnostics", diagnostics_dict(rec)` to the existing
`JLD2.save`. `assets/instrument-snippets.jl` shows the placement for adam, LM,
the GNKI `ekpobj` save and HM `n_tried`, plus patterns to avoid. When several
run scripts share diagnostics, put them in one helper in `<method>/diagnostics/`
(adam: `adam_diagnostics.jl`) that includes the recorder from `common/`.
**Name keys from the vocabulary** (`references/catalog-opt.md` § Canonical keys):
if the quantity is already there, record it under that name and unit, so
panels and comparisons work unchanged across methods.

Before running, check the added lines:
`git diff -U0 -- <method> | grep '^+' | grep -E 'rand|randn|lorenz_forward|G_func\(|ForwardDiff\.'`
should print nothing.

## 3. Smoke run: one failing, one successful seed

Run the instrumented method on **one failing and one successful seed** from
the failure map, in the cheapest case that has both (usually l63). If outcomes
can't be determined, run 2 seeds and use whatever they turn out to be. Map nc
seeds to CLI task indices with repo-layout § Task indexing, and use a
`RUN_DATE` / `CALIBRATE_DATE` no real run uses (e.g. `2099-01-01`).

Check each result file has a `"diagnostics"` key whose last `iter` is the
final iteration, within the storage cap. The full multi-seed picture comes
from an HPC rerun, not more local seeds.

## 4. Combine and plot

Copy `assets/combine_diagnostics_opt.jl` (or `_uq.jl`) to
`diagnostics/combine_diagnostics.jl` and fill in its `METHOD-SPECIFIC` block
(tag, slots, where the result files are). Copy
`assets/experiment_diagnosis_visualization.jl` to
`diagnostics/experiment_diagnosis_visualization.jl` and set `PANELS` to the
diagnostics chosen in § 1, using canonical keys where they exist (adam: RMSE,
‖J̃‖, direction, step). Both take environment variables only (`EXPERIMENT`,
`RUN_DATE` / `CALIBRATE_DATE`); the combined layout is documented in
`common/diagnostics/combine_core.jl`, the figure options in
`diagnosis_plot_core.jl`. With two smoke seeds the figure has one line per seed;
with many seeds it switches to per-status bands. A method whose single output
file holds several variants (ensemble_kalman_processes: TEKI/ETKI/GNKI/UKI) maps
each variant to a slot with `DiagCombine.write_combined` and plots with
`DIAG_MODE=slots`, which overlays the slots.

**Check that each panel discriminates, from the data and the plot script.**
For every `PANELS` key, compare the values the script will draw for the failing
and successful seeds (median and range over the iterations both run, plus the
final record; combine prints per-status medians). Keep a panel when:
- the seeds separate: typical values differ by more than each seed's own
  spread, or a reference line in `refs` falls between them;
- the values suit the scale: the script drops non-finite values, and
  non-positive ones on a log axis; extremes (log10(floatmin) clamps, overflow
  spikes) must not set the range, so clip it or floor exact zeros (adam's copy
  of the plot script adds `ylims` and `floor` panel fields);
- both seeds leave a series: window means are NaN at the first record, and a
  seed that converged at iteration 1 gives a single point, so choose another.

Rescale or drop panels that fail.

Before handing over, check the method fits in with the others:
- file names follow the grammar in `common/diagnostics/diag_files.jl`
  (`diagnostics_<tag>_<model>[_<fc>][_N<n>]_<date>.jld2`, the figure with
  `diagnosis_` instead; the `_N<n>` token only if the method has an N_ens axis);
- every key is a canonical one or documented as method-specific in the
  recorder's `describe!` text;
- `meta` has `method`, `case`, `date`, `git_hash`, `descriptions`;
- the method includes the shared recorder and has no copy of a file that exists
  in `common/diagnostics/`.

## 5. Summarise and hand over

Keep it short:
1. **The cause**, revised by the smoke run if needed, with the numbers that
   show it (e.g. "failing seed: log10‖J̃‖ ≈ 16, step ≈ 1e-11 after iteration
   130; successful seed converged at iteration 35") and each proxy's blind spot
   (e.g. cos to θ_true is weak far from the optimum).
2. **What the figure shows**: one line per panel, how the two seeds differ.
3. **Files and commands** to regenerate the combined file and the figure.

Then invite the user to change panels, seeds or scales, run more seeds, or
check the figure, and make those updates as they ask.

## Cross-skill references

- `slurm-pipeline-handler`: chain the combine (and optional plot) job after the
  leaderboard job in `hpc-variant/`; copy the combined JLD2 into
  `from-hpc/<date>/` next to the nc.
- `common-handler`: moves identical diagnostic code from methods into
  `common/diagnostics/` and rewrites the includes; any change to that directory
  goes through it.
- `new-method-builder`: new methods can include the recorder from the start.
- `dataviz`: the plot template's colours follow its palette (slots 1–3) plus
  line style.

## Final step — improve this skill

Offer: "Would you like to improve the **algorithm-diagnoser** skill using
skill-creator? You can share suggestions, or I can analyse what came up this
session (an uninformative proxy, a diagnostic that nailed the failure, a plot
pattern worth templating) to refine the skill for next time."
