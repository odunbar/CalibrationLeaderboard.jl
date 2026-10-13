# experiment_diagnosis_visualization.jl — ensemble_kalman_processes — estimated-gradient diagnosis figure.
#
# Reads the combined files written by combine_diagnostics.jl (diagnostics/output/diagnostics_ekp_<model>[_<fc>]_N<N_ens>_<date>.jld2)
# and draws, per N_ens, the four EKP variants (TEKI, ETKI, GNKI, UKI) overlaid (DIAG_MODE=slots): thin line per seed,
# bold line = median over seeds. The gradient is estimated from the ensemble itself (J̃ = R^{-1/2} C_Gu C_uu⁺,
# g = −J̃ᵀr̃); see ekp_diagnostics.jl. The drawing is common/diagnostics/diagnosis_plot_core.jl.
#
# Local (from the method directory; it has no Manifest, e.g. use the adam environment):
#   julia --project=../adam diagnostics/experiment_diagnosis_visualization.jl
#   EXPERIMENT=l96_const julia --project=../adam diagnostics/experiment_diagnosis_visualization.jl
# Options (env): EXPERIMENT=l63|l96_const|l96_vec|l96_flux (default l63), DIAG_N_ENS=<n> (default: every N_ens),
#   DIAG_FILE=<path>; the others are documented in diagnosis_plot_core.jl.
#
# Output: diagnostics/output/diagnosis_ekp_<model>[_<fc>]_N<N_ens>_<date>.{png,pdf}

include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diagnosis_plot_core.jl"))
using .DiagFiles, .DiagPlot

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
const PANELS = [
    (key = "rmse",       label = "whitened RMSE at G(ū) (dotted: target)",       yscale = :log10,    refs = Any[:target]),
    (key = "log10_J",    label = "magnitude: log10 ‖J̃‖ (ensemble estimate)",     yscale = :identity, refs = Any[3.0]),
    (key = "cos_truth",  label = "alignment: cos(−g, u_true − ū)",                yscale = :identity, refs = Any[0.0]),
    (key = "cos_prev",   label = "alignment: cos(g_t, g_{t−1})",                  yscale = :identity, refs = Any[0.0]),
    (key = "spread_w",   label = "ensemble spread (prior = 1)",                   yscale = :log10,    refs = Any[]),
    (key = "step_w",     label = "mean step ‖Lp⁻¹(ū_t − ū_{t−1})‖",               yscale = :log10,    refs = Any[]),
]
const DEFAULT_MODE = "slots"   # the four variants side by side, as in the original figure
# ╚══════════════════════════════════════════════════════════════════════════╝

function main()
    haskey(ENV, "DIAG_FILE") && return visualize_file(ENV["DIAG_FILE"]; panels = PANELS, default_mode = DEFAULT_MODE)
    dir = joinpath(@__DIR__, "output")
    case = get(ENV, "EXPERIMENT", "l63")
    re = Regex("^diagnostics_ekp_$(case_stem(case))_N(\\d+)_(\\d{4}-\\d{2}-\\d{2})\\.jld2\$")
    hits = [(f, parse(Int, m.captures[1]), m.captures[2]) for f in (isdir(dir) ? readdir(dir) : String[]) for m in (match(re, f),) if m !== nothing]
    isempty(hits) && error("No combined file for $case in $dir — run combine_diagnostics.jl first")
    date = maximum(h[3] for h in hits)
    for (f, n, dt) in sort(hits; by = h -> h[2])
        dt == date && (!haskey(ENV, "DIAG_N_ENS") || n == parse(Int, ENV["DIAG_N_ENS"])) &&
            visualize_file(joinpath(dir, f); panels = PANELS, default_mode = DEFAULT_MODE)
    end
end

main()
