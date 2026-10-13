# compare_overview.jl — one overview figure per experiment: plain vs score-based, Adam row and LM row.
#
#   row 1: Adam  vs score-based Adam          row 2: LM  vs score-based LM
#   columns: (1) whitened RMSE   (2) Jacobian magnitude log10 ‖J̃‖   (3) alignment cos(−g, θ_true − θ)
#
# These are the top-row panels of the per-pair figures from compare_diagnostics.jl, put together so the two
# optimiser families can be read side by side. Reads the same combined diagnostics files (via compare_diagnostics.jl's
# find_combined / load_method / panel_plot) and runs nothing. Experiments: l63, l96_const, l96_vec ("L96 spatial"),
# l96_flux ("L96 NN"). A pair with no combined file is shown as a labelled blank row; panels without finite data
# (alignment on l96_flux, which has no θ_true) are blank with the reason.
#
#   julia --project=levenberg_marquardt diagnostics/jacobian_exact_and_gfdt_comparison/compare_overview.jl
# Options (env): EXPERIMENT=l63|l96_const|l96_vec|l96_flux   DIAG_N_ENS=<n>   DIAG_SLOT=<i>
#                OVERVIEW_BY_STATUS=0   pool converged and failed seeds (default: draw them as separate groups)
#                OVERVIEW_MAX_ITER=<n>  x-axis limit for the status-split figures (default 50, linear, both rows)
#
# By default each method is split into its converged and its failed seeds (status from the leaderboard score: finite
# conv_score = converged): colour marks the method (see ROW_COLORS), line style the status (solid =
# converged, dotted = failed). Each group draws ONE representative seed (the one with finite data in the most panels,
# ties: lowest seed index). The seed's Jacobian may be non-finite at every recorded iteration,
# in which case it is not drawn in the magnitude panel and a note says so. No caption, no grid; top row has no x tick labels
# or x title, bottom row has no panel titles.
#
# Output: output/overview_<case>.{png,pdf}
# Colours: top (Adam) row blue + green, bottom (LM) row maroon + dark orange (ROW_COLORS); status by line style (see above).

include(joinpath(@__DIR__, "compare_diagnostics.jl"))

const OVERVIEW_PAIRS = ["adam", "lm"]   # row order
const CASE_TITLE = Dict("l63" => "L63", "l96_const" => "L96 const", "l96_vec" => "L96 spatial (vec-force)",
                        "l96_flux" => "L96 NN (flux-force)")
const OVERVIEW_PANELS = 3               # rmse, log10 ‖J̃‖, cos to truth = the first three panels of each pair

blank_panel(title, msg) = begin
    plt = plot(; title = title, titlefontsize = 14, framestyle = :box, legend = false, ticks = nothing)
    annotate!(plt, 0.5, 0.5, text(msg, 11, :center, :gray40))
    xlims!(plt, (0, 1)); ylims!(plt, (0, 1))
    plt
end

const BY_STATUS = get(ENV, "OVERVIEW_BY_STATUS", "1") != "0"
# Both rows share one linear x-axis over the first MAX_ITER outer iterations (Adam records every one of the first 50).
const MAX_ITER = parse(Int, get(ENV, "OVERVIEW_MAX_ITER", "50"))
const STATUS_LS = Dict("converged" => :solid, "failed" => :dot)
# colours per row (plain, score-based). Validated with the dataviz script (all-pairs, light surface):
# blue/green pass every check; maroon/dark orange pass all but contrast of the orange (2.98:1, just under 3:1),
# which the legend labels cover. (Pure CSS maroon #800000 is too dark for the validator's lightness band.)
const ROW_COLORS = Dict("adam" => (plain = "#2a78d6", score = "#008300"),
                        "lm"   => (plain = "#8c2f39", score = "#e0780a"))

# seeds of `d` with the given status (nothing if there are none)
function subset_status(d::MethodData, status::String)
    idx = findall(==(status), d.status)
    isempty(idx) && return nothing
    return MethodData(d.label, d.n_ens, d.date, d.seeds[idx], d.status[idx], d.iter[:, idx],
                      Dict(k => v[:, idx] for (k, v) in d.series), d.target, count(==("converged"), d.status[idx]), nothing)
end

# "score-based Adam" -> "Adam-KGMM", "score-based LM" -> "LM-KGMM"
kgmm_name(label) = replace(label, r"[Ss]core-based (\w+)" => s"\1-KGMM")

# the single seed drawn for a category: the one with finite data in the most panels (ties: lowest seed index)
function representative(d::MethodData)
    nfin = [count(Y -> any(isfinite, Y[:, j]), values(d.series)) for j in axes(d.iter, 2)]
    return argmax(nfin)   # argmax returns the first maximum
end

# one panel with up to four groups: (plain | score) × (converged | failed). `groups` = [(role, status, MethodData, label)]
function group_panel(p, groups, target; first_col, colors)
    plt = plot(; title = replace(p.label, " (dotted: 3)" => ""), titlefontsize = 14, xlabel = "iteration", yscale = p.yscale, xscale = :identity,
               legend = first_col ? :best : false, legendfontsize = 9, grid = false, framestyle = :box, tickfontsize = 11,
               xguidefontsize = 12, yguidefontsize = 12)
    vals = Float64[]
    undrawn = String[]
    for (role, status, d, label) in groups
        haskey(d.series, p.label) || continue
        Y = d.series[p.label]
        col, ls = getproperty(colors, role), STATUS_LS[status]
        nbad = count(j -> !any(isfinite, Y[:, j]), axes(Y, 2))
        nbad > 0 && push!(undrawn, "$(d.label) $status: $nbad/$(size(Y, 2)) seeds non-finite, not drawn")
        nbad == size(Y, 2) && continue
        if length(d.seeds) >= MIN_SEEDS_FOR_BAND
            s = summarize(d, p.label)
            keep = (s.x .<= MAX_ITER) .& (p.yscale == :log10 ? s.q1 .> 0 : trues(length(s.x)))
            s = (x = s.x[keep], q1 = s.q1[keep], med = s.med[keep], q3 = s.q3[keep], n = s.n[keep])
            isempty(s.x) && continue
            plot!(plt, s.x, s.med; ribbon = (s.med .- s.q1, s.q3 .- s.med), fillalpha = 0.15, color = col, lw = 2, ls = ls, label = label)
            append!(vals, s.q1, s.q3)
        else
            first_drawn = true
            for j in axes(Y, 2)
                ok = isfinite.(Y[:, j]) .& isfinite.(d.iter[:, j]) .& (d.iter[:, j] .<= MAX_ITER) .& (p.yscale == :log10 ? Y[:, j] .> 0 : true)
                any(ok) || continue
                plot!(plt, d.iter[ok, j], Y[ok, j]; color = col, ls = ls, lw = 1.6, alpha = 0.85, label = first_drawn ? label : "")
                first_drawn = false
                append!(vals, Y[ok, j])
            end
        end
    end
    for r in p.refs
        p.key == "log10_J" && continue   # no reference line on the Jacobian magnitude panel
        hline!(plt, [r === :target ? target : r]; color = :gray, ls = :dot, lw = 1, label = "")
    end
    if !isempty(vals)
        lo, hi = extrema(vals)
        p.yscale == :log10 ? ylims!(plt, (lo / 1.3, hi * 1.3)) :
            ylims!(plt, (lo - 0.05 * max(hi - lo, 1e-9), hi + 0.05 * max(hi - lo, 1e-9)))
    end
    xlims!(plt, (1, MAX_ITER))
    p.key == "cos_truth" && ylims!(plt, (-1, 1))   # alignment is a cosine: fixed range
    if !isempty(undrawn) && p.key == "log10_J"
        xl, yl = xlims(plt), ylims(plt)
        annotate!(plt, xl[1] + 0.03 * (xl[2] - xl[1]), yl[1] + 0.04 * (yl[2] - yl[1]),
                  text(join(undrawn, "\n"), 9, :left, :gray30))
    end
    return plt
end

# data for both methods of a pair, or nothing (+ reason)
function load_pair(pair, case, slot)
    hits = Dict(role => find_combined(getproperty(pair, role), case) for role in (:plain, :score))
    if any(isnothing, values(hits))
        names = [getproperty(pair, r).label for r in (:plain, :score) if hits[r] === nothing]
        return nothing, "no combined diagnostics for $(join(names, ", "))\n(not run for this experiment)"
    end
    data = Dict{Symbol, MethodData}()
    for role in (:plain, :score)
        d = load_method(getproperty(pair, role), hits[role], pair.panels[1:OVERVIEW_PANELS], slot)
        d === nothing && return nothing, "no usable seeds for $(getproperty(pair, role).label)"
        data[role] = d
    end
    return data, ""
end

function overview(case)
    slot = parse(Int, get(ENV, "DIAG_SLOT", "1"))
    plots = Plots.Plot[]
    ok_any = false
    target = NaN
    for pname in OVERVIEW_PAIRS
        pair = first(filter(p -> p.name == pname, PAIRS))
        data, why = load_pair(pair, case, slot)
        specs = pair.panels[1:OVERVIEW_PANELS]
        if data === nothing
            for p in specs
                push!(plots, blank_panel(p.label, why))
            end
            continue
        end
        ok_any = true
        target = data[:plain].target
        max_iter = maximum(maximum(filter(isfinite, vec(data[r].iter))) for r in (:plain, :score))
        logx = max_iter >= 1000
        for (i, p) in enumerate(specs)
            has = any(haskey(data[r].series, p.label) && any(isfinite, data[r].series[p.label]) for r in (:plain, :score))
            if !has
                push!(plots, blank_panel(p.label, p.key == "cos_truth" ?
                      "no θ_true for this experiment\n(NN weights are non-identifiable)" : "no data"))
                continue
            end
            if BY_STATUS
                groups = Tuple[]
                for role in (:plain, :score), status in ("converged", "failed")
                    sub = subset_status(data[role], status)
                    sub === nothing && continue
                    rep_j = representative(sub)
                    sub = MethodData(sub.label, sub.n_ens, sub.date, sub.seeds[rep_j:rep_j], sub.status[rep_j:rep_j],
                                     sub.iter[:, rep_j:rep_j], Dict(k => v[:, rep_j:rep_j] for (k, v) in sub.series),
                                     sub.target, count(==("converged"), sub.status[rep_j:rep_j]), nothing)
                    nm = kgmm_name(sub.label)
                    push!(groups, (role, status, sub, "$nm · $status"))
                end
                plt = group_panel(p, groups, data[:plain].target; first_col = i == 1, colors = ROW_COLORS[pname])
            else
                plt = panel_plot(p, data, [:plain, :score]; first_panel = i == 1, logx)   # legend only in the first column
                plot!(plt; xguidefontsize = 12, yguidefontsize = 12)
            end
            if pname == OVERVIEW_PAIRS[1]
                plot!(plt; xlabel = "", xformatter = _ -> "")   # top row: no x tick labels / title
            else
                plot!(plt; title = "")                            # bottom row: no panel titles
            end
            if i == 1
                plot!(plt; legend = pname == "adam" ? :topleft : :bottomright, ylabel = "$(pair.plain.label) vs\n$(kgmm_name(pair.score.label))")
            end
            push!(plots, plt)
        end
    end
    ok_any || (@info "overview/$case: no data for either pair — skipped"; return nothing)
    ttl = "$(CASE_TITLE[case]) · rmse_target=$(isfinite(target) ? target : "—")"
    fig = plot(plots...; layout = (2, 3), size = (1250, 700), plot_title = ttl, plot_titlefontsize = 18,
               left_margin = 6Plots.mm, bottom_margin = 3Plots.mm)
    mkpath(OUT_DIR)
    base = joinpath(OUT_DIR, BY_STATUS ? "overview_$(case)" : "overview_pooled_$(case)")
    savefig(fig, base * ".png"); savefig(fig, base * ".pdf")
    @info "Saved $(base).{png,pdf}"
    return base
end

function main_overview()
    cases = haskey(ENV, "EXPERIMENT") ? [ENV["EXPERIMENT"]] : CASES
    for case in cases
        overview(case)
    end
end

main_overview()
