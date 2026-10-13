# compare_table_first100.jl — overview table of the gradient diagnostics over the first N_FIRST iterations,
# one block per experiment, rows = method × outcome (converged / failed seeds shown separately).
#
# Reads the same combined files as compare_diagnostics.jl (via its find_combined) and runs nothing.
# Per-seed statistic = mean over that seed's recorded iterations <= N_FIRST (each record weighted by the number of
# iterations it stands for, so Adam's log-spaced records are not over-counted; LM records every iteration, weight 1).
# Each cell is mean ± std (sample std, n-1) of that statistic over the seeds in the row; a seed that stopped early
# (converged) contributes the iterations it ran. NaN / non-finite records are dropped. "—" = key not recorded
# (cos to the truth on l96_flux: no θ_true) or fewer than 2 seeds for the std.
#
# Columns:
#   RMSE final / initial  whitened RMSE at the seed's last record within the window divided by its first record
#                         (per-seed ratio, then mean ± std; < 1 = progress, converged seeds stop at the target)
#   log10 ‖J̃‖            magnitude of the whitened Jacobian (healthy O(1-3); AD through chaos 9-22; ≫ 3 implausible)
#   cos truth             cos(−g, θ_true − θ) (≈ 0: direction is noise)
#
#   julia --project=levenberg_marquardt diagnostics/jacobian_exact_and_gfdt_comparison/compare_table_first100.jl
# Options (env): N_FIRST=100   DIAG_SLOT=<i> (rmse-target slot, default 1)   DIAG_N_ENS=<n>   EXPERIMENT=l63|...
#
# Output: output/compare_table_first100.{md,csv,tex}
#   .tex is a single booktabs tabular (\usepackage{booktabs}) with one block of rows per experiment.

include(joinpath(@__DIR__, "compare_diagnostics.jl"))

const N_FIRST = parse(Int, get(ENV, "N_FIRST", "100"))

const TABLE_CASES = ["l96_const", "l63", "l96_vec", "l96_flux"]   # block order of the table

const METHODS = [   # (pair, role, display name): plain methods first, the score-based (KGMM) rows together after
    (PAIRS[2], :plain, "ADAM"), (PAIRS[1], :plain, "LM"),
    (PAIRS[2], :score, "ADAM-KGMM"), (PAIRS[1], :score, "LM-KGMM"),
]
const COLUMNS = [   # (header, key (stored or canonical, see DiagFiles.KEY_ALIASES), transform, show ± std)
    ("RMSE final / initial", "rmse",                            :rmse_ratio, false),
    ("log10 ‖J̃‖",           "log10_J",                          :plain,      false),
    ("cos truth",            "cos_truth",                        :plain,      true),
]

# weighted mean over the records of one seed within the window; mode :rmse_ratio = last / first finite record
function seed_stat(c, A, i, slot, mode, target)
    n = c["n_records"][i, slot]
    n == 0 && return NaN
    its = Float64.(c["iter"][1:n, i, slot])
    vals = Float64.(A[1:n, i, slot])
    prev = vcat(0.0, its[1:end-1])
    w = its .- prev
    ok = isfinite.(vals) .& isfinite.(its) .& (its .<= N_FIRST) .& (w .> 0)
    any(ok) || return NaN
    mode === :rmse_ratio && return vals[findlast(ok)] / vals[findfirst(ok)]
    return sum(w[ok] .* vals[ok]) / sum(w[ok])
end

fmt_ms(v, with_std = true) = begin
    v = filter(isfinite, v)
    isempty(v) && return "—"
    m = sum(v) / length(v)
    with_std || return @sprintf("%.3g", m)
    length(v) < 2 && return @sprintf("%.3g ± —", m)
    return @sprintf("%.3g ± %.2g", m, std(v))
end

function group_rows(case, slot)
    rows = []
    for (pair, role, name) in METHODS
        m = getproperty(pair, role)
        hit = find_combined(m, case)
        hit === nothing && continue
        c = JLD2.load(hit.path)
        status = c["status"][:, slot]
        target = Float64(c["slot_values"][slot])
        for st in ("converged", "failed")
            idx = findall(==(st), status)
            isempty(idx) && continue
            cells = String[]
            for (_, key, mode, with_std) in COLUMNS
                A = resolve_series(c, key)
                push!(cells, A === nothing ? "—" : fmt_ms([seed_stat(c, A, i, slot, mode, target) for i in idx], with_std))
            end
            last_it = [Float64(min(N_FIRST, c["iter"][c["n_records"][i, slot], i, slot])) for i in idx if c["n_records"][i, slot] > 0]
            push!(rows, (method = name, n_ens = hit.n === nothing ? 1 : hit.n, status = st, seeds = length(idx),
                         iters = isempty(last_it) ? NaN : sum(last_it) / length(last_it), cells = cells,
                         date = hit.date, target = target))
        end
    end
    return rows
end

# ── LaTeX ───────────────────────────────────────────────────────────────────
const TEX_HDR = ["method", "outcome", "RMSE final / initial", "\\(\\log_{10}\\|\\tilde J\\|\\)", "\\(\\cos\\) truth"]
tex_cell(x) = replace(replace(replace(x, r"(\d)e([+-]?)0*(\d+)" => s"\1\\times 10^{\2\3}"), " ± —" => " \\pm \\text{--}"), " ± " => " \\pm ") |> y -> y == "—" ? "--" : "\\(" * y * "\\)"
tex_esc(x) = replace(x, "_" => "\\_")

function tex_table(blocks)
    io = IOBuffer()
    ncol = length(TEX_HDR)
    println(io, "% Requires \\usepackage{booktabs}. Cells: mean over seeds (\\(\\pm\\) std for cos truth only); RMSE = last/first record within the first $N_FIRST iterations, other columns = per-seed mean over those iterations.")
    println(io, "\\begin{tabular}{ll" * "r"^(ncol - 2) * "}")
    println(io, "\\toprule")
    println(io, join(TEX_HDR, " & "), " \\\\")
    for (case, target, rows) in blocks
        println(io, "\\midrule")
        println(io, "\\multicolumn{$ncol}{l}{\\textbf{", tex_esc(case), "}} \\\\")
        println(io, "\\addlinespace[2pt]")
        for r in rows
            row = [tex_esc(r.method), r.status, tex_cell.(r.cells)...]
            println(io, join(row, " & "), " \\\\")
        end
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    return String(take!(io))
end

function main_first100()
    cases = haskey(ENV, "EXPERIMENT") ? [ENV["EXPERIMENT"]] : TABLE_CASES
    slot = parse(Int, get(ENV, "DIAG_SLOT", "1"))
    hdr = ["method", "outcome", [h for (h, _, _, _) in COLUMNS]...]
    md = IOBuffer(); csv = IOBuffer()
    println(md, "# Gradient diagnostics over the first $N_FIRST iterations: converged vs failed seeds\n")
    println(md, "Mean over seeds (± std for cos truth only). RMSE final / initial = last over first record within the first $N_FIRST iterations; the other columns are the per-seed mean over iterations ≤ $N_FIRST (records weighted by the iterations they span; ",
                "seeds that converged earlier contribute the iterations they ran). Outcome = leaderboard status of the seed ",
                "(converged = reached the rmse target). Rows without a combined diagnostics file are omitted. ",
                "Cosines for nu = 1 (l96_const) and nu = 2 (l63) are coarse; l96_flux has no θ_true (cos truth —). ",
                "Adam rows are window means of its record points, LM rows are per-iteration values.\n")
    println(csv, join(["\"case\"", "\"date\"", ["\"$h\"" for h in hdr]...], ","))
    blocks = []
    for case in cases
        rows = group_rows(case, slot)
        isempty(rows) && continue
        push!(blocks, (case, string(rows[1].target), rows))
        println(md, "## $case\n")
        println(md, "| ", join(hdr, " | "), " |"); println(md, "|", join(fill("---", length(hdr)), "|"), "|")
        for r in rows
            row = [r.method, r.status, r.cells...]
            println(md, "| ", join(row, " | "), " |")
            println(csv, join(["\"$x\"" for x in [case; r.date; row]], ","))
        end
        println(md)
    end
    mkpath(OUT_DIR)
    write(joinpath(OUT_DIR, "compare_table_first100.md"), String(take!(md)))
    write(joinpath(OUT_DIR, "compare_table_first100.csv"), String(take!(csv)))
    write(joinpath(OUT_DIR, "compare_table_first100.tex"), tex_table(blocks))
    @info "Saved $(joinpath(OUT_DIR, "compare_table_first100")).{md,csv,tex}"
end

main_first100()
