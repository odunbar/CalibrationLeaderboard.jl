# test3_eig_placement.jl — TEST 3: where does EIG place the new points, relative to the posterior?
#
# Reads the combined files of the variants (combine_diagnostics.jl), takes one N_ens slot, and shows median ± IQR over seeds of the acquisition keys per iteration.
# The no-EIG variants (tmcmc, iekf; dashed) draw their batch from the posterior, so they are the baseline for 'posterior-like'.
#
#   1  cand_mahal_med (dotted: post_mahal_med)   posterior Mahalanobis radius of the batch; 1 ≈ posterior draws, > 1 = in the tails
#   2  cand_r_over_ball                          max ‖z‖ / ball radius: ≈ 1 means the optimiser pinned points to the search-region boundary (an artefact, not a preference)
#   3  cand_mindist_med (dotted: post_mindist)   distance of the batch to the nearest existing training point; dotted = where the posterior itself is
#   4  batch_mindist_med                         nearest-neighbour distance WITHIN the batch (≈ 0 = collapse onto one spot)
#   5  cand_gp_std (dotted: post_gp_std)         GP predictive std at the batch vs inside the posterior (is EIG chasing uncertainty?)
#   6  eig_batch                                 EIG reached; compare with the columns below in the printed table (flat EIG landscape ⇒ small gain)
#   7  eig_gain_over_postdraw                    EIG(batch) − EIG(plain posterior draws); ≈ 0 = the optimiser adds nothing over sampling the posterior
#   8  cand_move                                 how far EIG moved the batch from its starting posterior draws (z-units per √k)
#   9  eig_opt_calls                             optimiser effort (limit: cfg.eig_call_limit); with eig_hit_limit the fraction of iterations stopped by the limit
#
# Local (from the method directory), after combine_diagnostics.jl for each variant:
#   EXPERIMENT=l96_vec julia --project=. diagnostics/test3_eig_placement.jl
# Output: diagnostics/output/test3_eig_placement_<case>_N<N>.png and a printed table.

include(joinpath(@__DIR__, "compare_common.jl"))

case = compare_case()
vars = load_variants(case)
j = chosen_slot(vars)
N = first(values(vars))["slot_values"][j]
println("test3: $case, N_ens=$N; variants with data: ", join(sort(collect(keys(vars))), ", "))

panels = [
    variant_band_panel(vars, "cand_mahal_med", j; title = "batch posterior-Mahalanobis radius", baseline_key = "post_mahal_med", ref = 1.0, legend = :topright),
    variant_band_panel(vars, "cand_r_over_ball", j; title = "max ‖z‖ / ball radius (≈1: pinned)", ref = 1.0),
    variant_band_panel(vars, "cand_mindist_med", j; title = "distance to nearest training point (dotted: posterior)", baseline_key = "post_mindist_med", yscale = :log10),
    variant_band_panel(vars, "batch_mindist_med", j; title = "within-batch nearest-neighbour distance", yscale = :log10),
    variant_band_panel(vars, "cand_gp_std", j; title = "GP std at batch (dotted: in posterior)", baseline_key = "post_gp_std"),
    variant_band_panel(vars, "eig_batch", j; title = "EIG of the batch"),
    variant_band_panel(vars, "eig_gain_over_postdraw", j; title = "EIG(batch) − EIG(posterior draws)", ref = 0.0),
    variant_band_panel(vars, "cand_move", j; title = "move from starting posterior draws"),
    variant_band_panel(vars, "eig_opt_calls", j; title = "EIG optimiser objective evaluations"),
]
fig = plot(panels...; layout = (3, 3), size = (1400, 1000), plot_title = "test3: EIG placement — $case N_ens=$N (median ± IQR over seeds; dashed = no-EIG baseline)", plot_titlefontsize = 10, dpi = 130,
           left_margin = 3Plots.mm, bottom_margin = 3Plots.mm)
mkpath(COMPARE_OUT)
out = joinpath(COMPARE_OUT, "test3_eig_placement_$(CASE_STEMS[case])_N$(N).png")
savefig(fig, out)

# ---- table: medians over seeds and iterations k ≥ 2 ----
tablekeys = ["cand_mahal_med", "post_mahal_med", "cand_r_over_ball", "cand_at_ball_frac", "cand_mindist_med", "post_mindist_med", "batch_mindist_med", "cand_gp_std", "post_gp_std",
             "eig_batch", "eig_init", "eig_postdraw", "eig_priordesign", "eig_gain_over_postdraw", "eig_gain_over_init", "cand_move", "eig_converged", "eig_hit_limit", "eig_opt_calls", "eig_opt_secs"]
println("\nmedian over seeds and iterations k ≥ 2 (N_ens=$N):")
vlist = sort(collect(keys(vars)))
println(rpad("  key", 26), join([rpad(v, 12) for v in vlist]))
for key in tablekeys
    row = String[]
    for v in vlist
        M = seed_matrix(vars[v], key, j)
        push!(row, rpad(M === nothing ? "—" : string(round(finite_med(M[2:end, :]); sigdigits = 3)), 12))
    end
    all(==(rpad("—", 12)), row) || println("  ", rpad(key, 24), join(row))
end
println("\nReading: cand_mahal_med ≫ post_mahal_med = EIG pushes points into the posterior tails; ≈ equal = posterior-like.",
        "\n         cand_r_over_ball ≈ 1 or cand_at_ball_frac ≫ 0 = the 'tails' are the search-region boundary.",
        "\n         eig_gain_over_postdraw ≈ 0 = the optimiser adds no EIG over drawing from the posterior (flat objective / under-converged optimiser; check eig_converged, eig_hit_limit, eig_opt_calls).")
println("Saved $out")
