# test4_eig_helpfulness.jl — TEST 4: does the EIG stage help?  Paired comparison of each EIG variant with its no-EIG baseline.
#
# Pairs: tmcmc_eig vs tmcmc, iekf_eig vs iekf (same posterior sampler, only the acquisition differs). For every (seed, N_ens) present in BOTH combined files:
#   d(k) = metric_eig(k) − metric_noeig(k)   for k = 2…K     (k = 1 is the shared LHS design, no acquisition yet)
# and per cell: dmean = mean_k d(k), dfinal = d(K). Across cells: mean, standard error, the fraction of cells where the EIG variant is lower, and a two-sided sign-test p-value.
# 'Lower is better' is stated only where it is unambiguous (coverage error, GP std in the posterior, truth distance); the other keys are reported signed.
#
# Also reported: the leaderboard score ("metric" = median budget-to-target over reached quantiles; lower = fewer forward evaluations) and the wall-clock price of the EIG stage.
#
# Pairing caveat: both variants share the seed's initial LHS design (the rng draws it first), so k = 1 is paired exactly for the IEKF pair. ST-MCMC draws from Julia's GLOBAL RNG,
# so the tmcmc pair separates from the first posterior onward (even two runs of the SAME tmcmc variant differ): read that pair as paired by (seed, N_ens) only.
#
# Local (from the method directory), after combine_diagnostics.jl for the variants:
#   EXPERIMENT=l96_vec julia --project=. diagnostics/test4_eig_helpfulness.jl
# Output: diagnostics/output/test4_eig_helpfulness_<case>.png, .jld2 and a printed table.

using Distributions
include(joinpath(@__DIR__, "compare_common.jl"))

case = compare_case()
vars = load_variants(case)
PAIRS = [("tmcmc_eig", "tmcmc"), ("iekf_eig", "iekf")]
# (key, lower-is-better?: true / false / nothing = ambiguous, meaning)
METRICS = [
    ("cov_err", true, "leaderboard coverage error mean|S(q)−q|"),
    ("truth_dist_over_spread", true, "truth distance / posterior spread"),
    ("gp_std_post", true, "GP predictive std inside the posterior"),
    ("misfit_post", nothing, "emulator misfit in the posterior (≈1 ok)"),
    ("ll_truth_minus_post", nothing, "loglik(truth) − loglik(posterior median)"),
    ("loo_msse_med", nothing, "emulator LOO msse (≈1 ok)"),
    ("post_spread_w", nothing, "posterior spread (prior = 1)"),
]
signp(nneg, n) = n == 0 ? NaN : (b = Binomial(n, 0.5); min(1.0, 2 * min(cdf(b, nneg), ccdf(b, nneg - 1))))

# [record] series of one (seed, slot) cell, trimmed to the cell's records
cell_series(c, key, i, j) = haskey(c, key) ? Float64.(c[key][1:c["n_records"][i, j], i, j]) : nothing

rows = Dict{String, Any}()
println("test4: $case — paired EIG vs no-EIG (d = EIG variant − baseline; cells = (seed, N_ens))")
for (ve, vb) in PAIRS
    (haskey(vars, ve) && haskey(vars, vb)) || (println("\n[$ve vs $vb] skipped: need both combined files (have: ", join(sort(collect(keys(vars))), ", "), ")"); continue)
    ce, cb = vars[ve], vars[vb]
    slots = intersect(ce["slot_values"], cb["slot_values"])
    seeds = intersect(ce["seeds"], cb["seeds"])
    cells = [(se, sb, ne) for ne in slots for s in seeds for (se, sb) in [(findfirst(==(s), ce["seeds"]), findfirst(==(s), cb["seeds"]))]
             if ce["n_records"][se, findfirst(==(ne), ce["slot_values"])] > 1 && cb["n_records"][sb, findfirst(==(ne), cb["slot_values"])] > 1]
    println("\n[$ve vs $vb] $(length(cells)) paired cells over N_ens ∈ $(sort(unique(ne for (_, _, ne) in cells)))")
    isempty(cells) && continue
    println(rpad("  metric", 26), rpad("mean d", 11), rpad("± s.e.", 10), rpad("P(eig lower)", 14), rpad("sign p", 9), "| final-iteration d: mean   P(lower)")
    for (key, lowbetter, meaning) in METRICS
        dm, df = Float64[], Float64[]
        for (se, sb, ne) in cells
            je, jb = findfirst(==(ne), ce["slot_values"]), findfirst(==(ne), cb["slot_values"])
            a, b = cell_series(ce, key, se, je), cell_series(cb, key, sb, jb)
            (a === nothing || b === nothing) && continue
            K = min(length(a), length(b))
            d = (a[1:K] .- b[1:K])[2:end]
            fd = filter(isfinite, d)
            isempty(fd) && continue
            push!(dm, mean(fd))
            isfinite(d[end]) && push!(df, d[end])
        end
        isempty(dm) && continue
        n = length(dm)
        plow = mean(dm .< 0)
        tag = lowbetter === true ? "  (lower = better)" : lowbetter === false ? "  (higher = better)" : ""
        println("  ", rpad(key, 24), rpad(round(mean(dm); sigdigits = 3), 11), rpad(round(std(dm) / sqrt(n); sigdigits = 2), 10), rpad(round(plow; digits = 2), 14), rpad(round(signp(count(<(0), dm), n); sigdigits = 2), 9),
                "| ", rpad(round(mean(df); sigdigits = 3), 11), rpad(round(mean(df .< 0); digits = 2), 6), tag)
        rows["$(ve)_vs_$(vb)_$(key)"] = (; dm, df)
    end
    # leaderboard score and cost
    sc_e = [ce["metric"][se, findfirst(==(ne), ce["slot_values"])] for (se, sb, ne) in cells]
    sc_b = [cb["metric"][sb, findfirst(==(ne), cb["slot_values"])] for (se, sb, ne) in cells]
    ok = findall(i -> isfinite(sc_e[i]) && isfinite(sc_b[i]), eachindex(sc_e))
    isempty(ok) || println("  leaderboard budget-to-target (median over quantiles; lower = better): EIG ", round(mean(sc_e[ok]); sigdigits = 4), " vs baseline ", round(mean(sc_b[ok]); sigdigits = 4),
                           "  | EIG better in ", round(100 * mean(sc_e[ok] .< sc_b[ok]); digits = 0), "%, tied in ", round(100 * mean(sc_e[ok] .== sc_b[ok]); digits = 0), "% of ", length(ok), " cells")
    secs = [finite_med(cell_series(ce, "eig_opt_secs", se, findfirst(==(ne), ce["slot_values"]))[2:end]) for (se, sb, ne) in cells if cell_series(ce, "eig_opt_secs", se, 1) !== nothing]
    isempty(secs) || println("  wall-clock price of the EIG optimiser: median ", round(finite_med(secs); sigdigits = 3), " s per iteration (this is what the baseline does not pay)")
end

# ---- figure: per-iteration paired difference (median ± IQR over cells) for four metrics, one line per pair ----
fig_keys = [("cov_err", "Δ coverage error"), ("gp_std_post", "Δ GP std in posterior"), ("truth_dist_over_spread", "Δ truth distance / spread"), ("misfit_post", "Δ emulator misfit in posterior")]
Nfig = first(values(vars))["slot_values"][chosen_slot(vars)]   # the figure shows one N_ens; the table pools all
panels = []
for (key, ttl) in fig_keys
    p = plot(; title = ttl * " (EIG − baseline)", titlefontsize = 9, xlabel = "GBOED iteration k", legend = key == "cov_err" ? :topright : false)
    for (ve, vb) in PAIRS
        (haskey(vars, ve) && haskey(vars, vb)) || continue
        ce, cb = vars[ve], vars[vb]
        je, jb = findfirst(==(Nfig), ce["slot_values"]), findfirst(==(Nfig), cb["slot_values"])
        (je === nothing || jb === nothing) && continue
        Me, Mb = seed_matrix(ce, key, je), seed_matrix(cb, key, jb)
        (Me === nothing || Mb === nothing) && continue
        K, ns = min(size(Me, 1), size(Mb, 1)), min(size(Me, 2), size(Mb, 2))
        med, lo, hi = band_stats(Me[1:K, 1:ns] .- Mb[1:K, 1:ns])
        ok = findall(isfinite, med)
        isempty(ok) && continue
        plot!(p, ok, med[ok]; ribbon = (med[ok] .- lo[ok], hi[ok] .- med[ok]), fillalpha = 0.15, color = VARIANT_COLORS[ve], lw = 2, label = "$ve − $vb")
    end
    hline!(p, [0.0]; color = :gray, ls = :dash, label = false)
    push!(panels, p)
end
fig = plot(panels...; layout = (2, 2), size = (1200, 800), plot_title = "test4: EIG vs no-EIG, paired by seed — $case", plot_titlefontsize = 10, dpi = 130, left_margin = 3Plots.mm, bottom_margin = 3Plots.mm)
mkpath(COMPARE_OUT)
out = joinpath(COMPARE_OUT, "test4_eig_helpfulness_$(CASE_STEMS[case])")
savefig(fig, out * ".png")
JLD2.save(out * ".jld2", Dict(rows))
println("\nSaved $out.{png,jld2}")
