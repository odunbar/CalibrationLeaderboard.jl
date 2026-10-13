# compare_diagnostics.jl — compare per-iteration diagnostics of a plain method with its score-based variant:
#
#   LM   (levenberg_marquardt)  vs  score-based LM   (score_based_lm)
#   Adam (adam)                 vs  score-based Adam (score_based_adam)
#
# Reads the combined files written by <method>/diagnostics/combine_diagnostics.jl
# (<method>/diagnostics/output/diagnostics_<tag>_<model>[_<fc>][_N<n>]_<date>.jld2); runs nothing.
# One figure per (pair, case) with both methods in each panel: median over the seeds still running
# (solid = plain, dashed = score-based) with an interquartile band; with fewer than 4 seeds the
# individual seeds are drawn instead. Panels are the gradient magnitude / alignment keys both
# methods share (rmse, log10‖J̃‖, cos to the truth, cos to the previous gradient) plus two keys
# specific to the optimiser (LM: step size, λ; Adam: step size, v̂ memory).
#
# Run from opt_experiments/ with any experiment project that has JLD2 + Plots, e.g.
#   julia --project=levenberg_marquardt diagnostics/jacobian_exact_and_gfdt_comparison/compare_diagnostics.jl
# Options (env):
#   PAIR=lm|adam                  only this pair (default both)
#   EXPERIMENT=l63|l96_const|l96_vec|l96_flux   only this case (default: every case where both files exist)
#   DIAG_N_ENS=<n>                score-based file to use (default: smallest N_ens available)
#   DIAG_SLOT=<i>                 rmse-target slot (default 1, the tightest target)
#
# Output: diagnostics/jacobian_exact_and_gfdt_comparison/output/compare_<pair>_<case>.{png,pdf} and a _summary.txt table of the
# values drawn (median at the first, middle and last common record).
#
# Colours: dataviz reference palette slots 1 (blue) and 3 (aqua), validated together (all pairs);
# aqua is below 3:1 contrast on the light surface, so line style (dashed) and the legend carry identity too.

ENV["GKSwstype"] = "100"   # headless GR
using JLD2
using Plots
using Printf
using Statistics

const OPT_DIR = dirname(dirname(@__DIR__))
include(joinpath(dirname(OPT_DIR), "common", "diagnostics", "diag_files.jl"))   # file grammar, canonical keys, lookup
using .DiagFiles: find_combined_by_tag, resolve_series
const OUT_DIR = joinpath(@__DIR__, "output")

const CASES = ["l63", "l96_const", "l96_vec", "l96_flux"]

# ╔══ PAIRS AND PANELS ══════════════════════════════════════════════════════╗
# key: a stored key or a canonical key (DiagFiles.KEY_ALIASES resolves point vs window-mean names and units:
# the LM scripts record point values, the Adam scripts window means). f transforms the value before plotting.
# rmse on a linear axis for both Adam and LM
rmse_panel(yscale) = (key = "rmse", label = "whitened RMSE (dotted: target)", yscale = yscale, refs = Any[:target], f = identity)
const SHARED_PANELS = [
    (key = "log10_J",                         label = "magnitude: log10 ‖J̃‖ (dotted: 3)",           yscale = :identity, refs = Any[3.0],     f = identity),
    (key = "cos_truth",                       label = "alignment: cos(−g, θ_true − θ)",              yscale = :identity, refs = Any[0.0],     f = identity),
    (key = "cos_prev",                        label = "alignment: cos(g_t, g_{t−1})",                yscale = :identity, refs = Any[0.0],     f = identity),
]
const PAIRS = [
    (name = "lm",
     plain = (label = "LM", dir = "levenberg_marquardt", tag = "lm"),
     score = (label = "score-based LM", dir = "score_based_lm", tag = "sblm"),
     panels = vcat([rmse_panel(:identity)], SHARED_PANELS, [
         (key = "log10_step",        label = "LM step: log10 ‖Lp⁻¹Δθ‖",             yscale = :identity, refs = Any[], f = identity),
         (key = "log10_lambda",      label = "LM damping: log10 λ (cap 8)",          yscale = :identity, refs = Any[8.0], f = identity),
     ])),
    (name = "adam",
     plain = (label = "Adam", dir = "adam", tag = "adam"),
     score = (label = "score-based Adam", dir = "score_based_adam", tag = "sbadam"),
     panels = vcat([rmse_panel(:identity)], SHARED_PANELS, [
         (key = "log10_step",         label = "Adam step: log10 ‖Lp⁻¹Δθ‖",                    yscale = :identity, refs = Any[], f = identity),
         (key = "vhat_ratio_wmean",   label = "Adam memory: log10(√v̂ / rms g)",               yscale = :identity, refs = Any[0.0], f = identity),
     ])),
]
# ╚══════════════════════════════════════════════════════════════════════════╝

const STYLE = Dict(:plain => (color = "#2a78d6", ls = :solid), :score => (color = "#1baf7a", ls = :dash))
const MIN_SEEDS_FOR_BAND = 4   # fewer seeds: draw each seed; otherwise median + IQR while >= 3 seeds remain
const MIN_ALIVE = 3

# ── file discovery ──────────────────────────────────────────────────────────
# Returns (path, N_ens or nothing, date) of the best combined file for this method and case, or nothing: the
# smallest N_ens (like-for-like with the N_ens = 1 plain methods), then the latest date; DIAG_N_ENS pins N_ens.
find_combined(m, case) = find_combined_by_tag(joinpath(OPT_DIR, m.dir, "diagnostics", "output"), m.tag, case;
                                              n_ens = haskey(ENV, "DIAG_N_ENS") ? parse(Int, ENV["DIAG_N_ENS"]) : nothing)

# ── data ────────────────────────────────────────────────────────────────────
struct MethodData
    label::String
    n_ens::Union{Nothing, Int}
    date::String
    seeds::Vector{Int}
    status::Vector{String}
    iter::Matrix{Float64}                  # record × seed
    series::Dict{String, Matrix{Float64}}  # panel label => record × seed (transformed)
    target::Float64
    n_converged::Int
    n_nonfinite_J::Union{Nothing, Int}     # seeds with any non-finite J̃ entry (key recorded by the LM scripts only)
end

function load_method(m, hit, panels, slot)
    c = JLD2.load(hit.path)
    status = c["status"][:, slot]
    keep = findall(s -> s in ("converged", "failed"), status)
    isempty(keep) && return nothing
    nrec = maximum(c["n_records"][keep, slot])
    iter = Float64.(c["iter"][1:nrec, keep, slot])
    series = Dict{String, Matrix{Float64}}()
    for p in panels
        A = resolve_series(c, p.key)
        A === nothing && continue
        raw = Float64.(A[1:nrec, keep, slot])
        series[p.label] = map(x -> isfinite(x) ? p.f(x) : NaN, raw)
    end
    nf = haskey(c, "nonfinite_J") ?
         count(j -> any(x -> isfinite(x) && x > 0, c["nonfinite_J"][1:nrec, keep[j], slot]), eachindex(keep)) : nothing
    return MethodData(m.label, hit.n, hit.date, collect(c["seeds"][keep]), status[keep], iter, series,
                      Float64(c["slot_values"][slot]), count(==("converged"), status[keep]), nf)
end

# median / quartiles over seeds at each iteration present in >= MIN_ALIVE seeds
function summarize(d::MethodData, label)
    Y = d.series[label]
    grid = sort(unique(filter(isfinite, vec(d.iter))))
    xs, q1, med, q3, nalive = Float64[], Float64[], Float64[], Float64[], Int[]
    # index rows by iteration per seed
    lookup = [Dict(Int(d.iter[r, s]) => Y[r, s] for r in axes(d.iter, 1) if isfinite(d.iter[r, s])) for s in axes(Y, 2)]
    for it in grid
        vals = [v for l in lookup for v in (get(l, Int(it), NaN),) if isfinite(v)]
        length(vals) >= MIN_ALIVE || continue
        push!(xs, it); push!(nalive, length(vals))
        push!(q1, quantile(vals, 0.25)); push!(med, median(vals)); push!(q3, quantile(vals, 0.75))
    end
    return (x = xs, q1 = q1, med = med, q3 = q3, n = nalive)
end

# ── plotting ────────────────────────────────────────────────────────────────
function method_name(d, n_ens_note)
    nm = n_ens_note && d.n_ens !== nothing ? "$(d.label) (N_ens=$(d.n_ens))" : d.label
    extra = d.n_nonfinite_J !== nothing && d.n_nonfinite_J > 0 ? ", $(d.n_nonfinite_J) non-finite J̃" : ""
    return "$nm · $(length(d.seeds)) seeds, $(d.n_converged) converged$extra"
end

function panel_plot(p, data, role_order; first_panel, logx, ylim_pad = 0.05)
    plt = plot(; title = p.label, titlefontsize = 9, xlabel = "iteration", yscale = p.yscale,
               xscale = logx ? :log10 : :identity, legend = first_panel ? :topright : false,
               legendfontsize = 6, grid = :y, gridalpha = 0.15, framestyle = :box, tickfontsize = 7)
    allvals = Float64[]
    for role in role_order
        d = data[role]
        haskey(d.series, p.label) || continue
        st = STYLE[role]
        name = method_name(d, role == :score)
        many = length(d.seeds) >= MIN_SEEDS_FOR_BAND
        if many
            s = summarize(d, p.label)
            isempty(s.x) && continue
            if p.yscale == :log10
                keep = s.q1 .> 0
                s = (x = s.x[keep], q1 = s.q1[keep], med = s.med[keep], q3 = s.q3[keep], n = s.n[keep])
            end
            plot!(plt, s.x, s.med; ribbon = (s.med .- s.q1, s.q3 .- s.med), fillalpha = 0.18, color = st.color,
                  linewidth = 2, ls = st.ls, label = name)
            append!(allvals, s.q1, s.q3)
        else
            Y = d.series[p.label]
            for j in axes(Y, 2)
                ok = isfinite.(Y[:, j]) .& isfinite.(d.iter[:, j]) .& (p.yscale == :log10 ? Y[:, j] .> 0 : true)
                any(ok) || continue
                plot!(plt, d.iter[ok, j], Y[ok, j]; color = st.color, linewidth = 1.6, ls = st.ls, alpha = 0.85,
                      label = j == 1 ? name : "")
                append!(allvals, Y[ok, j])
            end
        end
    end
    for r in p.refs
        val = r === :target ? data[role_order[1]].target : r
        hline!(plt, [val]; color = :gray, ls = :dot, lw = 1, label = "")
    end
    if !isempty(allvals)
        lo, hi = extrema(allvals)
        if p.yscale == :log10
            ylims!(plt, (lo / 1.3, hi * 1.3))
        else
            pad = max(hi - lo, 1e-9) * ylim_pad
            ylims!(plt, (lo - pad, hi + pad))
        end
    end
    return plt
end

function summary_rows(pair, case, data, panels)
    io = IOBuffer()
    println(io, "pair=$(pair.name) case=$case; median over seeds still running (≥$MIN_ALIVE) at the first, middle and last common record")
    for role in (:plain, :score)
        d = data[role]
        println(io, "  $(method_name(d, role == :score))   file date $(d.date)")
    end
    for p in panels
        for role in (:plain, :score)
            d = data[role]
            haskey(d.series, p.label) || continue
            s = summarize(d, p.label)
            isempty(s.x) && continue
            idx = unique([1, cld(length(s.x), 2), length(s.x)])
            vals = join([@sprintf("it %d: %.3g", s.x[i], s.med[i]) for i in idx], "   ")
            println(io, @sprintf("  %-34s %-18s %s", p.label, d.label, vals))
        end
    end
    return String(take!(io))
end

function compare(pair, case)
    hits = Dict(role => find_combined(getproperty(pair, role), case) for role in (:plain, :score))
    if any(isnothing, values(hits))
        missing_roles = [string(getproperty(pair, r).dir) for r in (:plain, :score) if hits[r] === nothing]
        @info "$(pair.name)/$case: no combined file for $(join(missing_roles, ", ")) — skipped"
        return nothing
    end
    slot = parse(Int, get(ENV, "DIAG_SLOT", "1"))
    data = Dict{Symbol, MethodData}()
    for role in (:plain, :score)
        d = load_method(getproperty(pair, role), hits[role], pair.panels, slot)
        d === nothing && (@info "$(pair.name)/$case: no usable seeds for $(getproperty(pair, role).dir) — skipped"; return nothing)
        data[role] = d
    end
    # drop panels with no finite data in either method (e.g. cos-to-truth on flux, which has no truth)
    panels = [p for p in pair.panels if any(haskey(data[r].series, p.label) && any(isfinite, data[r].series[p.label]) for r in (:plain, :score))]
    max_iter = maximum(maximum(filter(isfinite, vec(data[r].iter))) for r in (:plain, :score))
    logx = max_iter >= 1000
    role_order = [:plain, :score]
    plots = Plots.Plot[panel_plot(p, data, role_order; first_panel = i == 1, logx) for (i, p) in enumerate(panels)]

    foot = "median (line) and interquartile range (band) over seeds still running, shown while ≥$MIN_ALIVE remain; " *
           "fewer than $MIN_SEEDS_FOR_BAND seeds: one line per seed.  x = outer iteration (iteration costs differ: plain methods " *
           "spend nu+1 evaluations, score-based N_ens)."
    cap = plot(; framestyle = :none, legend = false, ticks = nothing)
    annotate!(cap, 0.5, 0.5, text(foot, 6, :center, :gray30))
    ncol = 3
    nrow = cld(length(plots), ncol)
    while length(plots) < nrow * ncol    # keep the grid full so the caption row stays the last layout cell
        push!(plots, plot(; framestyle = :none, legend = false, ticks = nothing))
    end
    lay = @layout [grid(nrow, ncol); c{0.06h}]
    ttl = "$(pair.plain.label) vs $(pair.score.label) · $case · rmse_target=$(data[:plain].target)"
    fig = plot(plots..., cap; layout = lay, size = (1250, 380 * nrow + 40), plot_title = ttl, plot_titlefontsize = 11,
               left_margin = 4Plots.mm, bottom_margin = 3Plots.mm)
    mkpath(OUT_DIR)
    base = joinpath(OUT_DIR, "compare_$(pair.name)_$(case)")
    savefig(fig, base * ".png")
    savefig(fig, base * ".pdf")
    txt = summary_rows(pair, case, data, panels)
    write(base * "_summary.txt", txt)
    print(txt)
    @info "Saved $(base).{png,pdf,_summary.txt}"
    return base
end

function main()
    pairs = haskey(ENV, "PAIR") ? filter(p -> p.name == ENV["PAIR"], PAIRS) : PAIRS
    cases = haskey(ENV, "EXPERIMENT") ? [ENV["EXPERIMENT"]] : CASES
    for pair in pairs, case in cases
        compare(pair, case)
    end
end

# lets compare_table.jl include this file for PAIRS / CASES / find_combined without drawing figures
abspath(PROGRAM_FILE) == (@__FILE__) && main()
