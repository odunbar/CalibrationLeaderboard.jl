# diagnosis_plot_core.jl — the diagnosis figure drawn from a combined file (opt and uq).
#
# A method's <method>/diagnostics/experiment_diagnosis_visualization.jl only states its PANELS (and the config
# accessors that locate the combined file) and calls `DiagPlot.run_diagnosis(...)` / `DiagPlot.visualize_file(...)`.
# Reads diagnostics/output/diagnostics_<tag>_<model>[_<fc>][_N<n>]_<date>.jld2 (combine_core.jl) and writes
# diagnosis_<...>.{png,pdf} next to it (names: diag_files.jl).
#
# Options (env; the same for every method):
#   DIAG_FILE=<path>          plot exactly this combined file
#   DIAG_SLOT=<index>         which slot (target / N_ens / variant) to plot; default tightest target (opt) or middle N_ens (uq)
#   DIAG_MODE=highlight|auto|all|representative|slots
#                             highlight = only one successful and one failed seed: the converged seed with the median
#                               metric (the lowest metric is often a lucky iteration-1 stop with no trajectory) and the
#                               typical failure; all = every seed as thin lines; representative = per-status
#                               median+IQR bands; auto = representative when more than 30 seeds have records;
#                             slots = every slot overlaid, one colour per slot (e.g. the EKP variants): thin line per
#                               seed, bold median over seeds
#   DIAG_SEEDS=3,17           highlight mode: plot exactly these seeds; other modes: extra seeds to highlight
#
# Panels (one NamedTuple each, ≤ 9): `key` = stored key or canonical key (DiagFiles.KEY_ALIASES), `label`,
# `yscale` (:identity | :log10), `refs` (horizontal reference lines; :slot draws the slot value, :target the
# `meta["target_rmse"]`), optional `ylims` (clip the axis, e.g. to hide log10(floatmin)) and `floor` (draw values ≤ 0
# at this level instead of dropping them on a log axis).
#
# Colours: dataviz reference palette slots 1–4 (blue/orange/aqua/amber), with line style as a second cue so status
# never relies on colour alone.

ENV["GKSwstype"] = "100"   # headless GR
isdefined(@__MODULE__, :DiagFiles) || include(joinpath(@__DIR__, "diag_files.jl"))

module DiagPlot

using JLD2
using Plots
using Statistics
using ..DiagFiles

export visualize_file, run_diagnosis

const STYLE = Dict(
    "converged" => (color = "#2a78d6", ls = :solid),
    "reached" => (color = "#2a78d6", ls = :solid),
    "partial" => (color = "#1baf7a", ls = :dot),
    "failed" => (color = "#eb6834", ls = :dash),
    "never" => (color = "#eb6834", ls = :dash),
)
const STATUS_ORDER = ["converged", "reached", "partial", "failed", "never"]
const SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
const SLOT_LINESTYLES = [:solid, :dash, :dot, :dashdot]

function default_slot(c)
    haskey(ENV, "DIAG_SLOT") && return parse(Int, ENV["DIAG_SLOT"])
    vals = c["slot_values"]
    return c["slot_name"] == "rmse_target" ? argmin(vals) : cld(length(vals), 2)
end

# series (x, y) for one seed, finite points only (and y > 0 on a log axis)
function seed_series(c, A, i, j, yscale; floor = nothing)
    n = c["n_records"][i, j]
    n == 0 && return Float64[], Float64[]
    x = Float64.(c["iter"][1:n, i, j])
    y = Float64.(A[1:n, i, j])
    floor === nothing || (y = [isfinite(v) && v <= 0 ? floor : v for v in y])
    keep = isfinite.(x) .& isfinite.(y)
    yscale === :log10 && (keep .&= y .> 0)
    return x[keep], y[keep]
end

# per-status median and IQR at each iteration value shared by ≥ 3 seeds still running
function band(c, A, idx, j, yscale; min_seeds = 3)
    pts = Dict{Float64, Vector{Float64}}()
    for i in idx
        x, y = seed_series(c, A, i, j, yscale)
        for (xi, yi) in zip(x, y)
            push!(get!(pts, xi, Float64[]), yi)
        end
    end
    xs = sort([x for (x, v) in pts if length(v) >= min_seeds])
    med = [median(pts[x]) for x in xs]
    lo = [quantile(pts[x], 0.25) for x in xs]
    hi = [quantile(pts[x], 0.75) for x in xs]
    return xs, med, lo, hi
end

# highlight mode: the converged seed with the (upper) median metric, and the typical failure
function highlight_seeds(c, j, status)
    metric = c["metric"][:, j]
    succ = sort(findall(==("converged"), status); by = i -> (metric[i], i))
    out = Dict{Int, String}()
    isempty(succ) || (out[succ[cld(length(succ) + 1, 2)]] = "median success")
    typ = c["selected_seeds"][2, j]
    typ > 0 && (out[typ] = "typical failure")
    return out
end

ref_value(r, c, j) = r === :slot ? c["slot_values"][j] : r === :target ? get(c["meta"], "target_rmse", NaN) : r

function style_panel(p, pi, c, max_iter, logx; legend = pi == 1 ? :best : false)
    plt = plot(; title = p.label, titlefontsize = 9, xlabel = "iteration", yscale = p.yscale,
               xscale = logx ? :log10 : :identity, legend, legendfontsize = 6, grid = :y, gridalpha = 0.15,
               framestyle = :box, tickfontsize = 7)
    hasproperty(p, :ylims) && ylims!(plt, p.ylims)
    xlims!(plt, (1, max_iter))
    return plt
end

function assemble(plots, c, ttl)
    n = length(plots)
    ncol = n == 4 ? 2 : min(n, 3)
    nrow = cld(n, ncol)
    return plot(plots...; layout = (nrow, ncol), size = (430 * ncol, 300 * nrow + 40),
                plot_title = ttl, plot_titlefontsize = 10, dpi = 150, left_margin = 3Plots.mm, bottom_margin = 3Plots.mm)
end

function usable_panels(c, panels)
    out = Tuple{Any, Any}[]
    for p in panels
        A = resolve_series(c, p.key)
        A === nothing ? @info("Panel key $(p.key) not in combined file — skipped") : push!(out, (p, A))
    end
    isempty(out) && error("None of the PANELS keys exist in the combined file")
    return out
end

function diagnosis_figure(c, j, panels; mode = get(ENV, "DIAG_MODE", "highlight"))
    status = c["status"][:, j]
    seeds = c["seeds"]
    with_rec = findall(i -> c["n_records"][i, j] > 0 && haskey(STYLE, status[i]), eachindex(seeds))
    mode == "auto" && (mode = length(with_rec) > 30 ? "representative" : "all")

    extra = haskey(ENV, "DIAG_SEEDS") ? [findfirst(==(s), seeds) for s in parse.(Int, split(ENV["DIAG_SEEDS"], ",")) if s in seeds] : Int[]
    if mode == "highlight"
        sel_name = isempty(extra) ? highlight_seeds(c, j, status) : Dict(i => status[i] for i in extra)
        sel = sort(collect(keys(sel_name)); by = i -> status[i] == "converged" ? 0 : 1)
    else
        sel = unique(vcat([i for i in c["selected_seeds"][:, j] if i > 0], extra))
        sel_name = Dict(i => n for (i, n) in zip(c["selected_seeds"][:, j], c["selected_names"]) if i > 0)
    end

    used = usable_panels(c, panels)
    max_iter = maximum(filter(isfinite, c["iter"][:, :, j]); init = 1.0)
    logx = max_iter >= 1000
    slot_val = c["slot_values"][j]

    plots = Plots.Plot[]
    for (pi, (p, A)) in enumerate(used)
        plt = style_panel(p, pi, c, max_iter, logx)
        for r in p.refs
            hline!(plt, [ref_value(r, c, j)]; color = :gray, ls = :dot, lw = 1, label = "")
        end
        for lab in (mode == "highlight" ? String[] : STATUS_ORDER)
            idx = [i for i in with_rec if status[i] == lab]
            isempty(idx) && continue
            st = STYLE[lab]
            if mode == "all"
                for (n, i) in enumerate(idx)
                    x, y = seed_series(c, A, i, j, p.yscale)
                    isempty(x) && continue
                    plot!(plt, x, y; color = st.color, ls = st.ls, lw = 0.8, alpha = 0.35,
                          label = n == 1 ? "$lab (n=$(length(idx)))" : "")
                end
            else
                xs, med, lo, hi = band(c, A, idx, j, p.yscale)
                isempty(xs) && continue
                plot!(plt, xs, med; ribbon = (med .- lo, hi .- med), fillalpha = 0.2, color = st.color,
                      ls = st.ls, lw = 1.5, label = "$lab median+IQR (n=$(length(idx)))")
            end
        end
        for i in sel
            haskey(STYLE, status[i]) || continue
            x, y = seed_series(c, A, i, j, p.yscale; floor = get(p, :floor, nothing))
            isempty(x) && continue
            st = STYLE[status[i]]
            name = replace(get(sel_name, i, "highlight"), "_" => " ")
            score = isfinite(c["metric"][i, j]) ? ", score $(round(Int, c["metric"][i, j]))" : ""
            plot!(plt, x, y; color = st.color, ls = st.ls, lw = mode == "highlight" ? 1.8 : 2.5,
                  label = "seed $(seeds[i]) ($name, $(status[i])$score)")
        end
        push!(plots, plt)
    end

    m = c["meta"]
    n_ok = count(in(("converged", "reached")), status)
    n_known = count(in(keys(STYLE)), status)
    ttl = "$(m["method"]) · $(m["case"]) · $(m["date"]) · $(c["slot_name"])=$(slot_val) · $(n_ok)/$(n_known) successful" * (mode == "highlight" ? "" : " · $(mode)")
    return assemble(plots, c, ttl)
end

# slots mode: every slot overlaid (thin line per seed, bold median over the seeds still running)
function slots_figure(c, panels)
    slots = c["slot_values"]
    seeds = c["seeds"]
    used = usable_panels(c, panels)
    max_iter = maximum(filter(isfinite, c["iter"]); init = 1.0)
    logx = max_iter >= 1000
    plots = Plots.Plot[]
    for (pi, (p, A)) in enumerate(used)
        plt = style_panel(p, pi, c, max_iter, logx; legend = pi == 1 ? :topright : false)
        for r in p.refs
            hline!(plt, [ref_value(r, c, 1)]; color = :gray, ls = :dot, lw = 1, label = "")
        end
        for (j, sv) in enumerate(slots)
            col, ls = SLOT_COLORS[mod1(j, 4)], SLOT_LINESTYLES[mod1(j, 4)]
            idx = [i for i in eachindex(seeds) if c["n_records"][i, j] > 0]
            for i in idx
                x, y = seed_series(c, A, i, j, p.yscale)
                isempty(x) && continue
                plot!(plt, x, y; color = col, ls = ls, lw = 1, alpha = 0.35, label = "")
            end
            xs, med, _, _ = band(c, A, idx, j, p.yscale; min_seeds = 1)
            nok = count(in(("converged", "reached")), c["status"][:, j])
            isempty(xs) || plot!(plt, xs, med; color = col, ls = ls, lw = 2.2, label = "$sv ($nok/$(length(idx)) converged)")
        end
        push!(plots, plt)
    end
    m = c["meta"]
    nens = haskey(m, "N_ens") ? " · N_ens=$(m["N_ens"])" : ""
    ttl = "$(m["method"]) · $(m["case"])$(nens) · $(m["date"]) · $(c["slot_name"])s overlaid, $(length(seeds)) seeds (bold = median over seeds)"
    return assemble(plots, c, ttl)
end

"""
    visualize_file(path; panels, default_mode = "highlight") -> figure stem

Draw the diagnosis figure of one combined file and save it (png + pdf) next to it.
"""
function visualize_file(path; panels, default_mode = "highlight")
    c = load_combined(path)
    mode = get(ENV, "DIAG_MODE", default_mode)
    fig, j = if mode == "slots"
        slots_figure(c, panels), nothing
    else
        j = default_slot(c)
        diagnosis_figure(c, j, panels; mode), j
    end
    stem = figure_stem(path)
    for ext in ("png", "pdf")
        savefig(fig, joinpath(dirname(path), "$stem.$ext"))
    end
    println("Saved $(joinpath(dirname(path), stem)).{png,pdf}")
    if j !== nothing
        for (i, name) in zip(c["selected_seeds"][:, j], c["selected_names"])
            i > 0 && println("  $(rpad(name, 16)) seed $(c["seeds"][i])  status=$(c["status"][i, j])  metric=$(c["metric"][i, j])")
        end
    end
    return stem
end

"""
    run_diagnosis(; panels, cases, config, nc_name, diag_dir, default_mode = "highlight", n_ens = nothing)

One figure per case (or just `ENV["DIAG_FILE"]`): locates the combined file from the leaderboard nc name,
using the exact date if RUN_DATE / CALIBRATE_DATE is set, otherwise the latest date available.
"""
function run_diagnosis(; panels, cases, config, nc_name, diag_dir, default_mode = "highlight", n_ens = nothing)
    if haskey(ENV, "DIAG_FILE")
        visualize_file(ENV["DIAG_FILE"]; panels, default_mode)
        return
    end
    pinned = haskey(ENV, "RUN_DATE") || haskey(ENV, "CALIBRATE_DATE")
    for case in cases
        path = find_combined(diag_dir, nc_name(config(case)); n_ens, exact = pinned)
        path === nothing ? (@info "No combined diagnostics file for $case — run combine_diagnostics.jl first") :
                           visualize_file(path; panels, default_mode)
    end
end

end # module DiagPlot
