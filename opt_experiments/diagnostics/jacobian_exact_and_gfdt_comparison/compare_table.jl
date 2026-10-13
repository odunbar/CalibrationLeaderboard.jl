# compare_table.jl — summary table of the gradient diagnostics: plain method vs score-based variant,
# one row per (case, method), for every case that has combined diagnostics files.
#
# Reads the same combined files as compare_diagnostics.jl (via its find_combined) and runs nothing.
# Each cell is the median over seeds of a per-seed statistic, with the interquartile range in brackets
# (min–max when fewer than 4 seeds). Per-seed statistic = median over that seed's recorded iterations
# (equal weight per seed), or the last recorded value for the "final" columns.
#
#   julia --project=levenberg_marquardt diagnostics/jacobian_exact_and_gfdt_comparison/compare_table.jl
# Options (env): PAIR=lm|adam   EXPERIMENT=l63|l96_const|l96_vec|l96_flux   DIAG_N_ENS=<n>   DIAG_SLOT=<i>
#
# Output: output/compare_table.md and output/compare_table.csv. Columns:
#   seeds / converged      seeds with diagnostics and how many reached the target
#   N_ens                  ensemble size of the file (plain methods: 1)
#   iters                  median last recorded iteration per seed (how long the seeds ran)
#   seeds with non-finite J̃  seeds (of those shown) with any non-finite Jacobian entry; recorded by the LM scripts only (— for Adam).
#                          Those seeds are absent from every magnitude / alignment cell, which use finite values only.
#   log10‖J̃‖, max, jac_plaus  magnitude of the whitened Jacobian: per-seed median, per-seed max (catches early
#                          blow-ups that a long run's median hides), and log10(‖J̃ Lp‖/‖r̃‖) (>3 implausible)
#   cos truth / cos prev   alignment of −g with the direction to θ_true / with the previous gradient
#                          (NaN-free medians; nu = 1 and 2 give coarse cosines; flux has no truth)
#   log10 step             log10 ‖Lp⁻¹Δθ‖ of the optimiser step
#   final log10 λ | v̂      LM: damping at the last record; Adam: log10(√v̂ / rms g) at the last record
#   cost (conv.)           median leaderboard cost (conv_score) of the converged seeds

include(joinpath(@__DIR__, "compare_diagnostics.jl"))

# Runs that crashed leave no result file, so the combined diagnostics cannot show them. Recorded by hand from the
# run logs (seeds 1-5, target 1.0, fake run date 2099-01-01); update when the runs are redone.
const CRASHED = Dict(
    ("l96_flux", "score-based Adam") => 3,   # seeds 1, 2, 4: ArgumentError "matrix contains Infs or NaNs" in KGMMScore
)
const RUN_NOTES = [
    "**score-based Adam, l96_flux (N_ens=10):** 3 of 5 seeds (1, 2, 4) crashed with `ArgumentError: matrix contains Infs or NaNs` " *
    "inside `KGMMScore` (`score_gfdt.jl`, called from `refresh_score!`): a matrix in the KGMM fit was non-finite (cause not diagnosed; plausibly non-finite or degenerate window states as θ moves). " *
    "A crash saves no result file, so those seeds have no diagnostics and are not counted in the seeds column (shown as \"+n crashed\"). " *
    "The call sits before the added diagnostics lines in that iteration, so the diagnostics did not cause it. Earlier HPC runs had about 85% failed seeds on flux (cause not checked).",
    "**score-based LM, l96_flux (N_ens=10):** 3 of 5 seeds (1, 2, 4) used the ForwardDiff fallback on all 50 iterations; the Jacobian it returned was non-finite every time, " *
    "so the LM step was NaN, λ ran to its cap and θ never moved. These are failed (not crashed) runs and are counted, but they contribute nothing to the finite-only magnitude and alignment cells " *
    "(see the non-finite J̃ column): the score-based LM flux magnitude is therefore from its 2 converged seeds only.",
    "**Adam, l96_flux:** not run (10,000 ForwardDiff iterations through the NN were too long locally), so the Adam pair has only its score-based row for flux.",
    "**l96_flux alignment:** no usable θ_true (NN weights are non-identifiable), so the cos-truth cells are —.",
]

# (header, key (stored or canonical, see DiagFiles.KEY_ALIASES), transform, stat) ; stat = :med (median over the seed's records) | :max | :last
function table_columns(pair)
    opt_key, opt_hdr = pair.name == "lm" ? ("log10_lambda", "final log10 λ") : ("vhat_ratio_wmean", "final log10(√v̂/rms g)")
    return [
        ("log10‖J̃‖ (finite only)", "log10_J",                         identity, :med),
        ("max log10‖J̃‖",          "log10_J",                          identity, :max),
        ("seeds with non-finite J̃", "nonfinite_J",                    identity, :nonfinite),
        ("jac_plaus",             "jac_plaus",                        identity, :med),
        ("cos truth",             "cos_truth",                        identity, :med),
        ("cos prev",              "cos_prev",                         identity, :med),
        ("log10 step",            "log10_step",                       identity, :med),
        (opt_hdr,                 opt_key,                            identity, :last),
    ]
end

finite(v) = filter(isfinite, v)
fmt_int(x) = isfinite(x) ? string(round(Int, x)) : "—"
fmt(x) = isfinite(x) ? (abs(x) >= 100 || (x != 0 && abs(x) < 0.01) ? @sprintf("%.2g", x) : @sprintf("%.2f", x)) : "—"

# "med [q1, q3]" over per-seed values (min–max when < 4 seeds)
function fmt_spread(vals)
    v = finite(vals)
    isempty(v) && return "—"
    lo, hi = length(v) >= 4 ? (quantile(v, 0.25), quantile(v, 0.75)) : extrema(v)
    return "$(fmt(median(v))) [$(fmt(lo)), $(fmt(hi))]"
end

function per_seed(A, n_rec, seeds_idx, slot, f, stat)
    out = Float64[]
    for i in seeds_idx
        n = n_rec[i]
        n == 0 && (push!(out, NaN); continue)
        col = [isfinite(x) ? f(Float64(x)) : NaN for x in A[1:n, i, slot]]
        v = finite(col)
        push!(out, isempty(v) ? NaN : stat === :last ? v[end] : stat === :max ? maximum(v) : median(v))
    end
    return out
end

function table_row(pair, role, case, slot)
    m = getproperty(pair, role)
    hit = find_combined(m, case)
    hit === nothing && return nothing
    c = JLD2.load(hit.path)
    status = c["status"][:, slot]
    idx = findall(s -> s in ("converged", "failed"), status)
    isempty(idx) && return nothing
    n_rec = c["n_records"][:, slot]
    last_it = [Float64(c["iter"][n_rec[i], i, slot]) for i in idx if n_rec[i] > 0]
    cost = finite(Float64[c["metric"][i, slot] for i in idx if status[i] == "converged"])
    cells = String[]
    for (_, key, f, stat) in table_columns(pair)
        A = resolve_series(c, key)
        if A === nothing
            push!(cells, "—")      # key not recorded by this method's scripts
        elseif stat === :nonfinite
            nbad = count(i -> n_rec[i] > 0 && any(x -> isfinite(x) && x > 0, A[1:n_rec[i], i, slot]), idx)
            push!(cells, "$nbad / $(length(idx))")
        else
            push!(cells, fmt_spread(per_seed(A, n_rec, idx, slot, f, stat)))
        end
    end
    return (case = case, method = m.label, seeds = length(idx), conv = count(==("converged"), status[idx]),
            n_ens = hit.n === nothing ? 1 : hit.n, iters = isempty(last_it) ? NaN : median(last_it),
            cells = cells, cost = isempty(cost) ? "—" : fmt_int(median(cost)), date = hit.date)
end

function build_tables()
    pairs = haskey(ENV, "PAIR") ? filter(p -> p.name == ENV["PAIR"], PAIRS) : PAIRS
    cases = haskey(ENV, "EXPERIMENT") ? [ENV["EXPERIMENT"]] : CASES
    slot = parse(Int, get(ENV, "DIAG_SLOT", "1"))
    md = IOBuffer(); csv = IOBuffer()
    println(md, "# Gradient diagnostics: plain vs score-based\n")
    println(md, "Median over seeds of the per-seed median over recorded iterations, [interquartile range] (min–max when fewer than 4 seeds). ",
                "Rows without a combined diagnostics file are omitted. Cosines for nu = 1 (l96_const) and nu = 2 (l63) are coarse; l96_flux has no θ_true, so its alignment cells are —.\n")
    for pair in pairs
        hdr = [h for (h, _, _, _) in table_columns(pair)]
        cols = ["case", "method", "seeds (conv.)", "N_ens", "iters", hdr..., "cost (conv.)"]
        println(md, "## $(pair.plain.label) vs $(pair.score.label)\n")
        println(md, "| ", join(cols, " | "), " |"); println(md, "|", join(fill("---", length(cols)), "|"), "|")
        pair.name == pairs[1].name && println(csv, join(["pair"; cols; "date"], ","))
        nrows = 0
        for case in cases
            for role in (:plain, :score)
                r = table_row(pair, role, case, slot)
                r === nothing && continue
                nrows += 1
                row = [r.case, r.method, "$(r.seeds) ($(r.conv))" * (get(CRASHED, (r.case, r.method), 0) > 0 ? " +$(CRASHED[(r.case, r.method)]) crashed" : ""), string(r.n_ens), fmt_int(r.iters), r.cells..., r.cost]
                println(md, "| ", join(row, " | "), " |")
                println(csv, join(["\"$(x)\"" for x in [pair.name; row; r.date]], ","))
            end
        end
        nrows == 0 && println(md, "| (no data) |", repeat(" |", length(cols) - 1))
        println(md)
    end
    println(md, "## Notes\n")
    for n in RUN_NOTES
        println(md, "- ", n)
    end
    println(md)
    return String(take!(md)), String(take!(csv))
end

function main_table()
    md, csv = build_tables()
    mkpath(OUT_DIR)
    write(joinpath(OUT_DIR, "compare_table.md"), md)
    write(joinpath(OUT_DIR, "compare_table.csv"), csv)
    print(md)
    @info "Saved $(joinpath(OUT_DIR, "compare_table")).{md,csv}"
end

main_table()
