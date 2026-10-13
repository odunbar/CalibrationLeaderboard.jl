# combine_diagnostics.jl — <METHOD> (opt) — merge per-seed "diagnostics" into ONE file per experiment.
#
# Template: .claude/skills/algorithm-diagnoser/assets/combine_diagnostics_opt.jl
# Lives in <method>/diagnostics/. States only what is specific to <METHOD>; the merge, the file layout and the
# summary are common/diagnostics/combine_core.jl (layout documented there).
#
# Local (from the method directory):
#   julia --project=. diagnostics/combine_diagnostics.jl                         # every case with files, RUN_DATE or today
#   EXPERIMENT=l63 RUN_DATE=2099-01-02 julia --project=. diagnostics/combine_diagnostics.jl
#   DIAG_SOURCE=/path/to/results  ...                                            # read result JLD2s from elsewhere
#
# Env-only CLI on purpose: experiment_config.jl's l96_experiment() reads ARGS[2].
#
# Output: diagnostics/output/diagnostics_<tag>_<model>[_<fc>]_<date>.jld2

include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "combine_core.jl"))
using .DiagFiles, .DiagCombine

const RESULT_DIR = get(ENV, "DIAG_SOURCE", joinpath(@__DIR__, "..", "output"))

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
const METHOD_TAG = "<tag>"            # e.g. "adam", "lm" (as in result/nc filenames)

# Slots = everything in a task except rng_idx. adam/LM/RL: rmse_target. score_based_*: fix N_ens in `cell_file`,
# add `const N_ENS = parse(Int, get(ENV, "DIAG_N_ENS", "1"))`, and pass `n_ens = N_ENS` (+ extra_meta) to combine_opt —
# see score_based_lm/diagnostics/combine_diagnostics.jl. A different closeness (default: window-min RMSE)
# or theta_true_note can be passed to combine_opt.
slot_name(cfg) = "rmse_target"
slot_values(cfg) = cfg.rmse_targets
cell_file = (cfg, slot, rng_idx) -> joinpath(RESULT_DIR, result_filename(cfg, slot, rng_idx))
# ╚══════════════════════════════════════════════════════════════════════════╝

DiagCombine.combine_opt(; tag = METHOD_TAG, cases = env_cases(experiments), config = experiment_config, nc_name = nc_filename,
                       result_dir = RESULT_DIR, out_dir = joinpath(@__DIR__, "output"), slot_name, slot_values, cell_file,
                       git_dir = @__DIR__)
