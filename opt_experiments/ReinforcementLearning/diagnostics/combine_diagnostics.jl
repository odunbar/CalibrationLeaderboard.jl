# combine_diagnostics.jl — ReinforcementLearning (opt) — merge per-seed "diagnostics" into ONE file per experiment.
#
# Template: .claude/skills/algorithm-diagnoser/assets/combine_diagnostics_opt.jl
# Lives in <method>/diagnostics/. States only what is specific to RL; the merge, the file layout and the
# summary are common/diagnostics/combine_core.jl (layout documented there).
#
# One combined file per (ALGORITHM, UPDATE_FREQUENCY, case): the tag is rl-<algorithm>-<update_frequency>, as in the nc names.
#
# Local (from the method directory):
#   ALGORITHM=ddpg UPDATE_FREQUENCY=short julia --project=. diagnostics/combine_diagnostics.jl     # every case with files
#   EXPERIMENT=l96_vec RUN_DATE=2099-01-01 ALGORITHM=ddpg UPDATE_FREQUENCY=short julia --project=. diagnostics/combine_diagnostics.jl
#   DIAG_SOURCE=/path/to/results  ...                                            # read result JLD2s from elsewhere
#
# Env-only CLI on purpose: experiment_config.jl's l96_experiment() reads ARGS[2].
#
# Output: diagnostics/output/diagnostics_rl-<alg>-<freq>_<model>[_<fc>]_<date>.jld2

include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "combine_core.jl"))
using .DiagFiles, .DiagCombine

const RESULT_DIR = get(ENV, "DIAG_SOURCE", joinpath(@__DIR__, "..", "output"))

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
# experiment_config(case) alone defaults to (:long, :ddpg); the arm comes from the same env vars as the run scripts.
rl_config(case) = experiment_config(case, rl_update_frequency(), rl_algorithm())
const METHOD_TAG = algorithm_name(rl_config(:l63))      # rl-<algorithm>-<update_frequency>

# Slots = everything in a task except rng_idx: rmse_target here. Closeness = window-min RMSE (default); "iter" is the RL step.
slot_name(cfg) = "rmse_target"
slot_values(cfg) = cfg.rmse_targets
cell_file = (cfg, slot, rng_idx) -> joinpath(RESULT_DIR, result_filename(cfg, slot, rng_idx))
# ╚══════════════════════════════════════════════════════════════════════════╝

DiagCombine.combine_opt(; tag = METHOD_TAG, cases = env_cases(experiments), config = rl_config, nc_name = nc_filename,
                       result_dir = RESULT_DIR, out_dir = joinpath(@__DIR__, "output"), slot_name, slot_values, cell_file,
                       git_dir = @__DIR__)
