# experiment_diagnosis_visualization.jl — ReinforcementLearning — short vs long, one row of four panels per case.
#
# Replaces the generic common/diagnostics/diagnosis_plot_core.jl figure for this method: each panel overlays ddpg (long) and
# ddpg-short against the number of STATISTICS WINDOWS of simulated data since the FIRST CRITIC UPDATE, (RL step − u0)/steps_per_window
# with u0 = max(n_warmup, batch_size) (the step ddpg_update! first runs; linear axis).  Earlier warm-up records are dropped, so
# every panel starts at 0.  A long step is one full window, a short step is 1/steps_per_window of one, so short has ~10x more
# records per window.  Spin-up is excluded for both arms (long pays it every step, short once at the start).
#
# Row = one case; the four panels are
#   (1) RMSE of the latest window under the applied action (the leaderboard measurement; thin grey line: target)
#   (2) ‖θn − θn₀‖·√n / path length of the actor (1 = random walk, ≫ 1 = directed motion; key path_vs_rw)
#   (3) signal relative to truth: window-mean cos(−g, θn_true − θn) of the actor step (blank for flux; key cos_truth_wmean)
#   (4) fraction of dead critic units (worse of the two hidden layers; units exactly zero over the whole minibatch; key dead_frac)
#
# Reads diagnostics/output/diagnostics_rl-<alg>-{long,short}_<model>[_<fc>]_<date>.jld2 (combine_diagnostics.jl, run once
# per UPDATE_FREQUENCY) and writes diagnosis_rl-<alg>-short-vs-long_<model>[_<fc>]_<date>.{png,pdf} next to them, plus one
# stacked figure of all cases found (…_all_<date>).
#
# Local (from the method directory):
#   RUN_DATE=2099-01-01 ALGORITHM=ddpg julia --project=. diagnostics/experiment_diagnosis_visualization.jl
#   EXPERIMENT=l96_vec RUN_DATE=2099-01-01 julia --project=. diagnostics/experiment_diagnosis_visualization.jl
# Options (env):
#   DIAG_SLOT=<index>    which rmse_target slot (default: tightest)
#   DIAG_SEED=<int>      plot this seed for both arms (default: lowest seed with records in both arms)
#   DIAG_MODE=bands      many seeds: median and IQR over all seeds with records, interpolated on a common window grid
#                        (needs ≥ 3 seeds at a grid point); default draws the single seed above.

ENV["GKSwstype"] = "100"   # headless GR
include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_files.jl"))
using .DiagFiles
using JLD2, Plots, Statistics

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
const ARMS = [(freq = :long,  label = "ddpg",  color = "#2a78d6", ls = :solid),
              (freq = :short, label = "ddpg-short",   color = "#1baf7a", ls = :solid)]
# key, panel title, y reference line, fixed ylims (nothing = auto), transform of the stored value (cfg → value → plotted)
const PANELS = [
    (key = "rmse",            title = "RMSE",     ref = :target, ylims = nothing,      f = (cfg, v) -> v),
    (key = "path_vs_rw",      title = "displacement·√n / path", ref = 1.0, ylims = nothing,      f = (cfg, v) -> v),
    (key = "cos_truth_wmean", title = "cos to truth",        ref = 0.0,     ylims = (-1.0, 1.0), f = (cfg, v) -> v),
    (key = "dead_frac",       title = "dead critic fraction", ref = nothing, ylims = (0.0, 1.0), f = (cfg, v) -> v),
]
# ╚══════════════════════════════════════════════════════════════════════════╝

const CASE_LABELS = Dict(:l63 => "L63", :l96_const => "L96 (const)", :l96_vec => "L96 (spatial)", :l96_flux => "L96 (NN)")
const ALL_FSCALE = 3.0   # font scale of the stacked "all" figure (per-case figures use 1.0)
pinned = haskey(ENV, "RUN_DATE")
diag_dir = get(ENV, "DIAG_DIR", joinpath(@__DIR__, "output"))   # DIAG_DIR: read/write somewhere else (e.g. a preview)
cfg_of(case, freq) = experiment_config(case, freq, rl_algorithm())

# combined file for (case, freq), or nothing
function load_arm(case, freq)
    cfg = cfg_of(case, freq)
    path = find_combined(diag_dir, nc_filename(cfg); exact = pinned)
    path === nothing ? nothing : (cfg = cfg, c = JLD2.load(path), path = path)
end

# RL step of the first critic update (rl_run.jl: step >= n_warmup && buf.n >= batch_size, buf.n = step barring blown-up resets)
first_update_step(cfg) = max(cfg.n_warmup, cfg.batch_size)

# did this arm converge within the budget? single seed: finite conv_score; bands: at least half of the plotted seeds
converged(c, seeds, j) = mean([isfinite(c["metric"][i, j]) for i in seeds]) >= 0.5

# (x, y) of one seed: statistics windows since the first critic update on x, finite points only
function seed_xy(c, key, i, j, cfg, f)
    n = c["n_records"][i, j]
    (n == 0 || !haskey(c, key)) && return Float64[], Float64[]
    x = (Float64.(c["iter"][1:n, i, j]) .- first_update_step(cfg)) ./ cfg.steps_per_window; y = f.(Ref(cfg), Float64.(c[key][1:n, i, j]))
    keep = isfinite.(x) .& isfinite.(y) .& (x .>= 0)
    return x[keep], y[keep]
end

# median and IQR over seeds on a common window grid (linear interpolation inside each seed's support)
function band_xy(c, key, seeds, j, cfg, f; ngrid = 300)
    xy = [seed_xy(c, key, i, j, cfg, f) for i in seeds]
    xmax = maximum([isempty(x) ? 0.0 : x[end] for (x, _) in xy]; init = 0.0)
    g = collect(range(0.0, xmax; length = ngrid))
    M = fill(NaN, ngrid, length(seeds))
    for (s, (x, y)) in enumerate(xy), (k, gk) in enumerate(g)
        (isempty(x) || gk < x[1] || gk > x[end]) && continue
        m = searchsortedlast(x, gk); m == length(x) && (M[k, s] = y[end]; continue)
        w = (gk - x[m]) / (x[m + 1] - x[m]); M[k, s] = (1 - w) * y[m] + w * y[m + 1]
    end
    ok = [count(isfinite, M[k, :]) >= 3 for k in 1:ngrid]
    q(p) = [quantile(filter(isfinite, M[k, :]), p) for k in findall(ok)]
    return g[ok], q(0.5), q(0.25), q(0.75)
end

# one row of four panels for `case`; returns the list of subplots (nothing if neither arm has a file)
function case_row(case, slot_choice, mode, seed_choice; first_row = true, last_row = true, fscale = 1.0, legend_panel = first_row ? 1 : 0)
    arms = [(a, load_arm(case, a.freq)) for a in ARMS]
    any(x -> x[2] !== nothing, arms) || return nothing
    ref_arm = first(x for x in arms if x[2] !== nothing)[2]
    c0 = ref_arm.c
    j = slot_choice === nothing ? argmin(c0["slot_values"]) : slot_choice
    target = Float64(c0["slot_values"][j])
    seeds_of(c) = [i for i in eachindex(c["seeds"]) if c["n_records"][i, j] > 0]
    common = intersect([seeds_of(x[2].c) for x in arms if x[2] !== nothing]...)
    seed = seed_choice !== nothing ? seed_choice : (isempty(common) ? nothing : first(common))
    note = join(["$(a.freq): " * (x === nothing ? "n/a" : (seed !== nothing && isfinite(x.c["metric"][seed, j]) ? "converged" : "failed")) for (a, x) in arms], ", ")
    @info "$(case): plotting " * (mode == "bands" ? "median/IQR over seeds" : "seed $(seed)") * " ($(note))"
    row = Plots.Plot[]
    for (p, P) in enumerate(PANELS)
        pl = plot(; title = first_row ? P.title : "", titlefontsize = 9fscale, xlabel = last_row ? "windows since first update" : "",
                  ylabel = p == 1 ? get(CASE_LABELS, case, String(case)) * (mode == "bands" ? " (median/IQR)" : "") : "", guidefontsize = 8fscale, legend = p == legend_panel ? (legend_panel == 1 ? :topright : :right) : false,
                  legendfontsize = 7fscale, tickfontsize = 8fscale, grid = false, left_margin = (p == 1 ? 14 : 6) * fscale * Plots.mm, bottom_margin = 4fscale * Plots.mm, right_margin = 4fscale * Plots.mm)
        xmax = 0.0
        for (a, x) in arms
            x === nothing && continue
            if mode == "bands"
                gx, med, lo, hi = band_xy(x.c, P.key, seeds_of(x.c), j, x.cfg, P.f)
                isempty(gx) && continue
                plot!(pl, gx, med; ribbon = (med .- lo, hi .- med), fillalpha = 0.18, color = a.color, ls = converged(x.c, seeds_of(x.c), j) ? :solid : :dot, lw = 4.6, label = a.label)
                xmax = max(xmax, gx[end])
            else
                seed === nothing && continue
                xs, ys = seed_xy(x.c, P.key, seed, j, x.cfg, P.f)
                isempty(xs) && continue
                plot!(pl, xs, ys; color = a.color, ls = converged(x.c, [seed], j) ? :solid : :dot, lw = 4.4, label = a.label)
                xmax = max(xmax, xs[end])
            end
        end
        if p == legend_panel   # line-style key (empty proxy series)
            plot!(pl, [NaN], [NaN]; color = :gray30, ls = :solid, lw = 4.4, label = "converged")
            plot!(pl, [NaN], [NaN]; color = :gray30, ls = :dot, lw = 4.4, label = "failed")
        end
        ref = P.ref === :target ? target : P.ref
        ref === nothing || hline!(pl, [ref]; color = :gray60, ls = :solid, lw = 1, label = "")
        xlims!(pl, 0, xmax > 0 ? xmax : 1)
        if P.ylims === :hidden
            ylims!(pl, 0, ref_arm.cfg.hidden)
        elseif P.ylims !== nothing
            ylims!(pl, P.ylims...)
        end
        push!(row, pl)
    end
    return row
end

function save_fig(plots, nrows, stem, title; fscale = 1.0)
    # canvas grows with the fonts so titles and labels fit
    fig = plot(plots...; layout = (nrows, 4), size = round.(Int, (1800, 330 * nrows + 60) .* (1 + 0.5 * (fscale - 1))), plot_title = title, plot_titlefontsize = 11fscale)
    for ext in ("png", "pdf")
        savefig(fig, joinpath(diag_dir, "$(stem).$(ext)"))
    end
    @info "Saved $(joinpath(diag_dir, stem)).{png,pdf}"
end

function main()
    slot_choice = haskey(ENV, "DIAG_SLOT") ? parse(Int, ENV["DIAG_SLOT"]) : nothing
    seed_choice = haskey(ENV, "DIAG_SEED") ? parse(Int, ENV["DIAG_SEED"]) : nothing
    mode = get(ENV, "DIAG_MODE", "single")
    cases = [c for c in env_cases(experiments) if any(f -> load_arm(c, f) !== nothing, (:long, :short))]
    isempty(cases) && return @info "No combined diagnostics files found — run combine_diagnostics.jl for UPDATE_FREQUENCY=long and short first"
    date = string(experiment_config(first(cases), :long, rl_algorithm()).run_date)
    tag = "rl-$(rl_algorithm())-short-vs-long"
    rows = Dict(c => case_row(c, slot_choice, mode, seed_choice) for c in cases)
    for c in cases
        save_fig(rows[c], 1, "diagnosis_$(tag)_$(case_stem(c))_$(date)", "$(tag) · $(case_stem(c)) · $(date)")
    end
    if length(cases) > 1   # stacked overview: titles on the first row, x labels on the last
        stacked = reduce(vcat, [case_row(c, slot_choice, mode, seed_choice; first_row = k == 1, last_row = k == length(cases), fscale = ALL_FSCALE,
                                                 legend_panel = k == min(2, length(cases)) ? 4 : 0) for (k, c) in enumerate(cases)])
        save_fig(stacked, length(cases), "diagnosis_$(tag)_all_$(date)", "$(tag) · $(date)"; fscale = ALL_FSCALE)
    end
end

main()
