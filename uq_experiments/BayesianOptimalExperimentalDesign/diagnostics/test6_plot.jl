# test6_plot.jl — figure for test6_eig_optimiser_refit.jl: does the EIG the optimiser gains on its own posterior points carry over to newly drawn points?
#
# One figure per case (no figure title), two panels sharing one y-axis, one dot per (seed, iteration) stage:
#   left  ("optimizer validation set")  % change in EIG, final vs start, on the optimiser's own n_eig points (in-sample)       (eig_final_opt − eig_start_opt) / |eig_start_opt|
#   right ("unseen validation set")     the same % change on newly drawn test points (mean over the held-out sets)             (eig_final_fresh − eig_start_fresh) / |eig_start_fresh|
# x axis: optimiser evaluations (forward passes of the objective, incl. those inside gradients) until it stopped, from 0 to the refit call limit.
# Colour = variant; filled circle = the optimiser's own criteria stopped it, open diamond = it hit the call limit (the legend lists only the variants). Dashed line = 0 (no change in EIG).
#
# Reads diagnostics/output/test6_eig_refit_<case>_<variant>_<date>[_<tag>].jld2 (both EIG variants if present; the latest date per variant).
#   EXPERIMENT=l96_vec,l96_flux (default: every case with a file)   DIAG_VARIANTS=iekf_eig (default: tmcmc_eig,iekf_eig)   DIAG_TAG=<tag> (default: files without a tag)   DIAG_DATES=iekf_eig=2026-10-19,tmcmc_eig=2026-10-16
#
#   julia --project=. diagnostics/test6_plot.jl
# Output: diagnostics/output/test6_eig_refit_<case>[_<variants, if DIAG_VARIANTS is set>][_<tag>].{png,pdf}
#         diagnostics/output/test6_eig_refit_table[_<variants>][_<tag>].tex — LaTeX table (booktabs) of the mean and std over stages of the % EIG gain, in-sample and out-of-sample,
#         one row per case with IEKF and TMCMC side by side ("--" where a variant has no data); the same statistics as the dots in the figures (every stage of the refit, all seeds and N_ens pooled).

using JLD2
using Plots
using Printf
using Statistics

const OUT = joinpath(@__DIR__, "output")
const CASES = Dict("l63" => "l63", "l96_const" => "l96_const-force", "l96_vec" => "l96_vec-force", "l96_flux" => "l96_flux-force")
const COLORS = Dict("tmcmc_eig" => "#2a78d6", "iekf_eig" => "#eb6834")
const TAG = get(ENV, "DIAG_TAG", "")
const VARIANTS = haskey(ENV, "DIAG_VARIANTS") ? String.(split(ENV["DIAG_VARIANTS"], ",")) : ["tmcmc_eig", "iekf_eig"]
const DATES = Dict{String, String}(String(first(kv)) => String(last(kv)) for kv in (split(p, "=") for p in split(get(ENV, "DIAG_DATES", ""), ","; keepempty = false)))

function find_file(case, variant)
    stem = CASES[case]
    re = Regex("^test6_eig_refit_$(stem)_$(variant)_(\\d{4}-\\d{2}-\\d{2})$(isempty(TAG) ? "" : "_" * TAG)\\.jld2\$")
    hits = [(m.captures[1], f) for f in readdir(OUT) for m in [match(re, f)] if m !== nothing]
    haskey(DATES, variant) && filter!(h -> h[1] == DATES[variant], hits)
    return isempty(hits) ? nothing : joinpath(OUT, sort(hits)[end][2])
end

const CASE_ORDER = ["l63", "l96_const", "l96_vec", "l96_flux"]
stage_pct(d, startkey, finalkey) = 100 .* (d[finalkey] .- d[startkey]) ./ abs.(d[startkey])   # % change in EIG per stage
table_rows = NamedTuple[]
table_settings = Dict{String, Any}()

cases = haskey(ENV, "EXPERIMENT") ? String.(split(ENV["EXPERIMENT"], ",")) : collect(keys(CASES))
for case in sort(cases; by = c -> something(findfirst(==(c), CASE_ORDER), length(CASE_ORDER) + 1))
    data = Dict{String, Dict}()
    for v in VARIANTS
        f = find_file(case, v)
        f === nothing || (data[v] = JLD2.load(f))
    end
    if isempty(data)
        @info "test6_plot: no test6 output for $case (tag '$TAG') in $OUT — run test6_eig_optimiser_refit.jl first"
        continue
    end
    call_limit = maximum(Int(d["settings"]["call_limit"]) for d in values(data))

    xstep = call_limit <= 10_000 ? 2000 : 5000
    function panel(startkey, finalkey; title, legend)
        p = plot(; title, titlefontsize = 15, xlabel = "total optimizer evaluations", guidefontsize = 16, tickfontsize = 12, legendfontsize = 12,
                 xlims = (0, 1.02 * call_limit), xticks = collect(0:xstep:call_limit), grid = false, legend, framestyle = :box)
        for (v, d) in sort(collect(data); by = first)
            y = stage_pct(d, startkey, finalkey)
            x = d["n_calls"]
            conv = d["converged"] .> 0.5
            ok = isfinite.(x) .& isfinite.(y)
            for (sel, marker, fill, lab) in ((ok .& conv, :circle, 1.0, v), (ok .& .!conv, :diamond, 0.0, nothing))
                any(sel) || continue
                scatter!(p, x[sel], y[sel]; color = COLORS[v], marker, markersize = 5, markerstrokewidth = 1.2, markerstrokecolor = COLORS[v],
                         markeralpha = fill == 1.0 ? 0.7 : 1.0, markercolor = fill == 1.0 ? COLORS[v] : :white, label = lab === nothing ? false : lab)
            end
        end
        hline!(p, [0.0]; color = :black, ls = :dash, lw = 1, label = false)
        return p
    end
    p1 = panel("eig_start_opt", "eig_final_opt"; title = "optimizer validation set", legend = :topleft)
    p2 = panel("eig_start_fresh", "eig_final_fresh"; title = "unseen validation set", legend = false)
    plot!(p1; ylabel = "% change in EIG")
    fig = plot(p1, p2; layout = (1, 2), link = :y, size = (1400, 600), dpi = 140, left_margin = 9Plots.mm, bottom_margin = 8Plots.mm, right_margin = 6Plots.mm, top_margin = 3Plots.mm)
    stem = "test6_eig_refit_$(CASES[case])$(haskey(ENV, "DIAG_VARIANTS") ? "_" * join(VARIANTS, "_") : "")$(isempty(TAG) ? "" : "_" * TAG)"
    savefig(fig, joinpath(OUT, stem * ".png"))
    savefig(fig, joinpath(OUT, stem * ".pdf"))
    for v in VARIANTS
        haskey(data, v) || continue
        d = data[v]
        ok = isfinite.(d["n_calls"])
        yin = filter(isfinite, stage_pct(d, "eig_start_opt", "eig_final_opt")[ok])
        yout = filter(isfinite, stage_pct(d, "eig_start_fresh", "eig_final_fresh")[ok])
        push!(table_rows, (; case, variant = v, n = length(yin), in_mean = mean(yin), in_std = std(yin), out_mean = mean(yout), out_std = std(yout)))
        merge!(table_settings, Dict("call_limit" => d["settings"]["call_limit"], "n_eig" => d["settings"]["n_eig"], "n_fresh" => d["settings"]["n_fresh"]))
    end
    println("Saved ", joinpath(OUT, stem), ".{png,pdf}  (", join(["$v: $(length(d["k"])) stages" for (v, d) in sort(collect(data); by = first)], ", "), ")")
end

# ---- LaTeX table: mean ± std over stages of the % EIG gain, in-sample and out-of-sample, IEKF and TMCMC posteriors side by side ----
if !isempty(table_rows)
    find_row(case, v) = (i = findfirst(r -> r.case == case && r.variant == v, table_rows); i === nothing ? nothing : table_rows[i])
    pm(r, a, b) = r === nothing ? "--" : @sprintf("\$%.1f \\pm %.1f\$", getfield(r, a), getfield(r, b))
    esc(x) = replace(x, "_" => "\\_")
    io = IOBuffer()
    println(io, "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lrrrr}\n\\toprule")
    println(io, " & \\multicolumn{2}{c}{in-sample (optimizer validation set)} & \\multicolumn{2}{c}{out-of-sample (unseen validation set)} \\\\")
    println(io, "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    println(io, "Case & IEKF & TMCMC & IEKF & TMCMC \\\\\n\\midrule")
    for case in unique(r.case for r in table_rows)
        ri, rt = find_row(case, "iekf_eig"), find_row(case, "tmcmc_eig")
        println(io, esc(case), " & ", pm(ri, :in_mean, :in_std), " & ", pm(rt, :in_mean, :in_std), " & ", pm(ri, :out_mean, :out_std), " & ", pm(rt, :out_mean, :out_std), " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
    println(io, "\\caption{Percentage change in expected information gain (EIG) between the start and the end of each batch optimization, mean \$\\pm\$ standard deviation over all stages ",
            "(seeds, iterations and ensemble sizes pooled), with the optimizer limited to $(table_settings["call_limit"]) evaluations. ",
            "IEKF and TMCMC are the two posterior samplers feeding the EIG optimization. ",
            "In-sample: scored on the $(table_settings["n_eig"]) posterior points the optimizer used. Out-of-sample: mean over $(table_settings["n_fresh"]) newly drawn sets of $(table_settings["n_eig"]) posterior points.}")
    println(io, "\\label{tab:eig_refit_gain}\n\\end{table}")
    tex = String(take!(io))
    tex_path = joinpath(OUT, "test6_eig_refit_table$(haskey(ENV, "DIAG_VARIANTS") ? "_" * join(VARIANTS, "_") : "")$(isempty(TAG) ? "" : "_" * TAG).tex")
    write(tex_path, tex)
    println("\nSaved ", tex_path, "\n", tex)
end
