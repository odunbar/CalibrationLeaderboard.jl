# experiment_diagnosis_visualization.jl — score_based_lm — one diagnosis figure per experiment.
#
# Template: .claude/skills/algorithm-diagnoser/assets/experiment_diagnosis_visualization.jl
# Lives in <method>/diagnostics/. Edit only the PANELS list; the drawing is common/diagnostics/diagnosis_plot_core.jl.
#
# Reads diagnostics/output/diagnostics_sblm_<model>[_<fc>][_N<n>]_<date>.jld2 (from combine_diagnostics.jl) and writes
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
# score_based_lm tasks are (N_ens, rmse_target, rng_idx); one combined file per N_ens, slot = rmse_target.
# N_ens = 1 is the like-for-like comparison with levenberg_marquardt. Override: DIAG_N_ENS=10.
const N_ENS = parse(Int, get(ENV, "DIAG_N_ENS", "1"))
# One entry per panel (≤ 9). `key` is a stored key or a canonical one (DiagFiles.KEY_ALIASES).
# yscale: :identity or :log10 (store log10 values for quantities spanning many decades).
# refs: horizontal reference lines; the symbol :slot draws the slot value (e.g. the RMSE target).
# ylims (optional): clip the y axis, e.g. to hide floor values such as log10(floatmin).
# floor (optional): draw values ≤ 0 at this level instead of dropping them on a log axis.
const PANELS = [
    (key = "rmse",                label = "whitened RMSE (dotted: target)",                       yscale = :identity, refs = Any[:slot]),
    (key = "log10_rho_over_rhoH", label = "L63 regime: log10 ρ/ρ_H(β) (< 0: C± stable)",        yscale = :identity, refs = Any[0.0]),
    (key = "log10_varmin",        label = "window regime: log10 min var (< −2: collapsed)",      yscale = :identity, refs = Any[-2.0]),
    (key = "lin_cos_wmean",       label = "derivative: cos(Δr̃_actual, −J̃Δθ) (1 = valid)",      yscale = :identity, refs = Any[0.0]),
    (key = "log10_step_wmean",    label = "consequence: log10 ‖Lp⁻¹Δθ‖ (LM step)",              yscale = :identity, refs = Any[]),
]
# Default mode: draw only one successful and one failed seed.
const DEFAULT_MODE = "highlight"
# ╚══════════════════════════════════════════════════════════════════════════╝

run_diagnosis(; panels = PANELS, default_mode = DEFAULT_MODE, cases = env_cases(experiments), config = experiment_config,
              nc_name = nc_filename, diag_dir = joinpath(@__DIR__, "output"), n_ens = N_ENS)
