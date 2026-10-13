# experiment_diagnosis_visualization.jl — levenberg_marquardt — one diagnosis figure per experiment.
#
# Template: .claude/skills/algorithm-diagnoser/assets/experiment_diagnosis_visualization.jl
# Lives in <method>/diagnostics/. Edit only the PANELS list; the drawing is common/diagnostics/diagnosis_plot_core.jl.
#
# Reads diagnostics/output/diagnostics_lm_<model>[_<fc>]_<date>.jld2 (from combine_diagnostics.jl) and writes
# diagnosis_<same name>.{png,pdf} next to it.
#
# Local (from the method directory):
#   julia --project=. diagnostics/experiment_diagnosis_visualization.jl          # every case, latest date found
#   EXPERIMENT=l63 RUN_DATE=2099-01-02 julia --project=. diagnostics/experiment_diagnosis_visualization.jl
# Options (env; DIAG_FILE, DIAG_SLOT, DIAG_MODE, DIAG_SEEDS) are documented in diagnosis_plot_core.jl.

include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diagnosis_plot_core.jl"))
using .DiagFiles, .DiagPlot

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
# One entry per panel (≤ 9). `key` is a stored key or a canonical one (DiagFiles.KEY_ALIASES).
# yscale: :identity or :log10 (store log10 values for quantities spanning many decades).
# refs: horizontal reference lines; the symbol :slot draws the slot value (e.g. the RMSE target).
# ylims (optional): clip the y axis, e.g. to hide floor values such as log10(floatmin).
# floor (optional): draw values ≤ 0 at this level instead of dropping them on a log axis.
const PANELS = [
    (key = "rmse",            label = "whitened RMSE (dotted: target)",          yscale = :identity, refs = Any[:slot]),
    (key = "log10_J",         label = "magnitude: log10 ‖J̃‖ (dotted: 3)",     yscale = :identity, refs = Any[3.0]),
    (key = "cos_truth",       label = "alignment: cos(−g, θ_true − θ)",         yscale = :identity, refs = Any[0.0]),
    (key = "lin_cos_wmean",   label = "alignment at step: cos(Δr̃_actual, −J̃Δθ)", yscale = :identity, refs = Any[0.0]),
]
# Default mode: draw only one successful and one failed seed.
const DEFAULT_MODE = "highlight"
# ╚══════════════════════════════════════════════════════════════════════════╝

run_diagnosis(; panels = PANELS, default_mode = DEFAULT_MODE, cases = env_cases(experiments), config = experiment_config,
              nc_name = nc_filename, diag_dir = joinpath(@__DIR__, "output"))
