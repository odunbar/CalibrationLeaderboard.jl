# experiment_diagnosis_visualization.jl — BayesianOptimalExperimentalDesign — one diagnosis figure per experiment.
#
# Lives in BayesianOptimalExperimentalDesign/diagnostics/. Edit only the PANELS list; the drawing is common/diagnostics/diagnosis_plot_core.jl.
# uq plots the middle N_ens slot (DIAG_SLOT=<index> to choose another).
#
# Reads diagnostics/output/diagnostics_<variant key>_<model>[_<fc>]_<date>.jld2 (from combine_diagnostics.jl) and writes
# diagnosis_<same name>.{png,pdf} next to it. Set BOED_VARIANT for the variant (it picks the file name through method_key).
#
# Local (from the method directory):
#   BOED_VARIANT=iekf julia --project=. diagnostics/experiment_diagnosis_visualization.jl          # every case, latest date found
#   BOED_VARIANT=iekf EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-09 julia --project=. diagnostics/experiment_diagnosis_visualization.jl
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
# Panels a variant never records (tmcmc_* keys for the IEKF variants and vice versa; eig_* keys for the no-EIG variants) are skipped.
# Reading guide: the first four say whether the emulator + posterior are right, the next two whether the sampler converged, the last two what the acquisition did.
const PANELS = [
    (key = "cov_err",                label = "coverage error mean|S(q)−q|",                       yscale = :identity, refs = Any[0.0]),
    (key = "misfit_post",            label = "emulator misfit in posterior (≈1 ok)", yscale = :log10, refs = Any[1.0]),
    (key = "ll_truth_minus_post",    label = "loglik(truth) − loglik(posterior)",             yscale = :identity, refs = Any[0.0]),
    (key = "truth_dist_over_spread", label = "truth distance / posterior spread", yscale = :identity, refs = Any[1.0]),
    (key = "loo_msse_med",           label = "emulator LOO msse (≫1 overconfident)",                     yscale = :log10,    refs = Any[1.0]),
    (key = "tmcmc_log_ev",           label = "TMCMC log-evidence",                    yscale = :identity, refs = Any[]),
    (key = "iekf_misfit_flat",       label = "IEKF last-step misfit change (→0 plateau)", yscale = :log10, refs = Any[0.01]),
    (key = "eig_gain_over_postdraw", label = "EIG(batch) − EIG(posterior draws)",  yscale = :identity, refs = Any[0.0]),
    (key = "cand_mahal_med",         label = "batch Mahalanobis radius (>1 = tails)", yscale = :identity, refs = Any[1.0]),
]
# Default mode: draw only one successful and one failed seed (highlight | auto | all | representative | slots).
const DEFAULT_MODE = "highlight"
# ╚══════════════════════════════════════════════════════════════════════════╝

run_diagnosis(; panels = PANELS, default_mode = DEFAULT_MODE, cases = env_cases(experiments), config = experiment_config,
              nc_name = nc_filename, diag_dir = joinpath(@__DIR__, "output"))
