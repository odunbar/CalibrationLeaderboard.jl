# critic_activation_comparison.jl — ReinforcementLearning — one line per critic activation on the four diagnostics panels.
#
# One-off comparison (CRITIC_ACT experiments on vec seed 1; relu = leaderboard baseline).  Same four panels, x axis, fonts and
# line conventions as experiment_diagnosis_visualization.jl: x = statistics windows since the first critic update; dotted =
# did not converge within the budget, solid = converged; thin grey lines are references (RMSE target, 1 = random walk, cos = 0).
# The RMSE panel of every row shares the y-range of the first row (diverged runs in later rows run off the top).
# One row per arm that has >= 2 activations with a combined file (long now; short once the gelu/tanh short runs exist).
#
# Each activation is a run date (one RUN_DATE per CRITIC_ACT, see VARIANTS); combine_diagnostics.jl must have been run for
# that date and UPDATE_FREQUENCY.  Note the dead-unit fraction is not the same quantity across activations: relu = units
# exactly zero over the batch, smooth activations = units with activation derivative < 0.01 over the batch (tanh: saturated).
#
# Local (from the method directory):
#   ALGORITHM=ddpg EXPERIMENT=l96_vec julia --project=. diagnostics/critic_activation_comparison.jl
# Options (env): DIAG_SEED (default: lowest seed with records in every variant), DIAG_SLOT (default: tightest target).
# Output: diagnostics/output/diagnosis_rl-<alg>-critic-act_<model>[_<fc>]_<last date>.{png,pdf}

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_files.jl"))
using .DiagFiles
using JLD2, Plots, Statistics

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
const CASE = :l96_vec
const VARIANTS = [(label = "relu",     date = "2099-01-01", color = "#2a78d6"),
                  (label = "softplus", date = "2099-01-06", color = "#eb6834"),
                  (label = "gelu",     date = "2099-01-07", color = "#1baf7a"),
                  (label = "tanh",     date = "2099-01-08", color = "#eda100")]
const ARM_LABEL = Dict(:long => "ddpg", :short => "ddpg-short")
const PANELS = [(key = "rmse", title = "RMSE", ref = :target, ylims = nothing),
                (key = "path_vs_rw", title = "displacement·√n / path", ref = 1.0, ylims = nothing),
                (key = "cos_truth_wmean", title = "cos to truth", ref = 0.0, ylims = (-1.0, 1.0)),
                (key = "dead_frac", title = "dead critic fraction", ref = nothing, ylims = (0.0, 1.0))]
const FSCALE = 2.0
# ╚══════════════════════════════════════════════════════════════════════════╝

diag_dir = get(ENV, "DIAG_DIR", joinpath(@__DIR__, "output"))
cfg_of(freq, date) = experiment_config(CASE, freq, rl_algorithm())

function load_variant(freq, v)
    cfg = experiment_config(CASE, freq, rl_algorithm())
    name = diag_filename(nc_filename(cfg))                       # …_<run_date>.jld2 of the config's own date
    name = replace(name, r"\d{4}-\d{2}-\d{2}\.jld2$" => "$(v.date).jld2")
    path = joinpath(diag_dir, name)
    isfile(path) ? (cfg = cfg, c = JLD2.load(path)) : nothing
end

first_update_step(cfg) = max(cfg.n_warmup, cfg.batch_size)
converged(c, i, j) = isfinite(c["metric"][i, j])

function series(c, key, i, j, cfg)
    n = c["n_records"][i, j]
    (n == 0 || !haskey(c, key)) && return Float64[], Float64[]
    x = (Float64.(c["iter"][1:n, i, j]) .- first_update_step(cfg)) ./ cfg.steps_per_window
    y = Float64.(c[key][1:n, i, j])
    keep = isfinite.(x) .& isfinite.(y) .& (x .>= 0)
    return x[keep], y[keep]
end

function arm_row(freq, last_row, first_row)
    loaded = [(v, load_variant(freq, v)) for v in VARIANTS]
    loaded = [(v, l) for (v, l) in loaded if l !== nothing]
    length(loaded) >= 2 || return nothing
    c0 = first(loaded)[2].c
    j = haskey(ENV, "DIAG_SLOT") ? parse(Int, ENV["DIAG_SLOT"]) : argmin(c0["slot_values"])
    target = Float64(c0["slot_values"][j])
    have(c) = [i for i in eachindex(c["seeds"]) if c["n_records"][i, j] > 0]
    common = intersect([have(l.c) for (_, l) in loaded]...)
    seed = haskey(ENV, "DIAG_SEED") ? parse(Int, ENV["DIAG_SEED"]) : (isempty(common) ? nothing : first(common))
    seed === nothing && return nothing
    @info "$(freq): seed $(seed), variants " * join(["$(v.label)" * (converged(l.c, seed, j) ? " (converged)" : " (failed)") for (v, l) in loaded], ", ")
    row = Plots.Plot[]
    fs = FSCALE
    for (p, P) in enumerate(PANELS)
        pl = plot(; title = first_row ? P.title : "", titlefontsize = 9fs, xlabel = last_row ? "windows since first update" : "",
                  ylabel = p == 1 ? ARM_LABEL[freq] : "", guidefontsize = 8fs, tickfontsize = 8fs, legendfontsize = 7fs, grid = false,
                  legend = (first_row && p == 4) ? :right : false, left_margin = (p == 1 ? 16 : 6) * fs * Plots.mm,
                  bottom_margin = (last_row ? 8 : 4) * fs * Plots.mm, right_margin = 4fs * Plots.mm)
        xmax = 0.0
        for (v, l) in loaded
            xs, ys = series(l.c, P.key, seed, j, l.cfg)
            isempty(xs) && continue
            plot!(pl, xs, ys; color = v.color, ls = converged(l.c, seed, j) ? :solid : :dot, lw = 4.4, label = v.label)
            xmax = max(xmax, xs[end])
        end
        ref = P.ref === :target ? target : P.ref
        ref === nothing || hline!(pl, [ref]; color = :gray60, ls = :solid, lw = 1, label = "")
        xlims!(pl, 0, xmax > 0 ? xmax : 1)
        P.ylims === nothing || ylims!(pl, P.ylims...)
        push!(row, pl)
    end
    return row
end

function main()
    arms = [:long, :short]
    rows = Dict(a => arm_row(a, a == :short, a == :long) for a in arms)   # placeholder flags fixed below
    present = [a for a in arms if rows[a] !== nothing]
    isempty(present) && return @info "Need combined files for at least two activations — run combine_diagnostics.jl for each date"
    # rebuild with the right first/last-row flags now that we know which rows exist
    rows = [arm_row(a, k == length(present), k == 1) for (k, a) in enumerate(present)]
    # every row's RMSE panel uses the y-range of the first (long) row, so a diverged run in a later row cannot squash the others
    yr = Plots.ylims(rows[1][1])
    for k in 2:length(rows)
        ylims!(rows[k][1], yr...)
    end
    fig = plot(reduce(vcat, rows)...; layout = (length(rows), 4), size = (3300, 640 * length(rows) + 200),
               plot_title = "rl-$(rl_algorithm()) critic activation · $(CASE)", plot_titlefontsize = 11FSCALE)
    stem = "diagnosis_rl-$(rl_algorithm())-critic-act_$(case_stem(CASE))_$(VARIANTS[end].date)"
    for ext in ("png", "pdf")
        savefig(fig, joinpath(diag_dir, "$(stem).$(ext)"))
    end
    @info "Saved $(joinpath(diag_dir, stem)).{png,pdf}"
end

main()
