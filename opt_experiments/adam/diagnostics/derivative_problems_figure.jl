# derivative_problems_figure.jl — adam — what is wrong with the ForwardDiff derivative, pooled over all seeds.
#
# Reads diagnostics/output/diagnostics_adam_<model>[_<fc>]_<date>.jld2 (combine_diagnostics.jl) and writes
# derivative_problems_adam_<model>[_<fc>]_<date>.{png,pdf} next to it. Every recorded iteration of every
# seed is one sample, split by regime: fluctuating (for l63: chaotic) window vs window collapsed to a fixed point (log10_statvar < -2).
#
#   (a) magnitude   log10 ‖J̃‖
#   (b) validity    log10(actual / J̃-predicted residual change), first 100 iterations (steps still non-negligible)
#   (c) direction   cos(−g, θ_true − θ)
#   (d) consequence Adam's step vs log10(√v̂ / |g|): one old gradient spike in v̂ freezes the step
#
# Local (from the method directory):
#   EXPERIMENT=l63 RUN_DATE=2099-02-02 julia --project=. diagnostics/derivative_problems_figure.jl
#   DIAG_FILE=<path> julia --project=. diagnostics/derivative_problems_figure.jl

ENV["GKSwstype"] = "100"
using JLD2
using Plots
using Statistics

include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_files.jl"))
using .DiagFiles: diag_filename

const DIAG_DIR = joinpath(@__DIR__, "output")
const REGIMES = [
    (name = "fluctuating window", test = v -> v >= -2, color = "#eb6834"),
    (name = "collapsed window",   test = v -> v < -2,  color = "#2a78d6"),
]
# finite-difference reference ‖J‖ at θ_true (score_based_lm/VALIDATION.md, l63 only)
const FD_REF = Dict("l63" => log10(221.0))

# pooled finite values of `key` over records whose regime passes `test` (and `itest` on the iteration)
function pooled(c, key, test; j = 1, itest = _ -> true)
    it, sv, x = c["iter"], c["log10_statvar"], c[key]
    vals = Float64[]
    for i in axes(it, 2), r in 1:c["n_records"][i, j]
        (isfinite(sv[r, i, j]) && test(sv[r, i, j]) && itest(it[r, i, j]) && isfinite(x[r, i, j])) || continue
        push!(vals, x[r, i, j])
    end
    return vals
end

function hist_panel(c, key, title, xlabel; refs = Float64[], itest = _ -> true, bins = 40)
    plt = plot(; title, xlabel, ylabel = "fraction of records", titlefontsize = 9, legendfontsize = 7,
               grid = :y, gridalpha = 0.15, framestyle = :box, tickfontsize = 7, legend = :top)
    for rg in REGIMES
        v = pooled(c, key, rg.test; itest)
        isempty(v) && continue
        stephist!(plt, v; bins, normalize = :probability, color = rg.color, lw = 2,
                  label = "$(rg.name): median $(round(median(v); sigdigits = 3)) (n=$(length(v)))")
    end
    vline!(plt, refs; color = :gray, ls = :dot, lw = 1.5, label = "")
    return plt
end

function step_panel(c)
    plt = plot(; title = "(d) consequence: Adam step frozen by v̂ memory", titlefontsize = 9,
               xlabel = "log10(√v̂ / |g|)  (decades of old-spike memory; x clipped at 99.5th pct)", ylabel = "log10 whitened step ‖Lp⁻¹Δθ‖",
               legendfontsize = 7, grid = :y, gridalpha = 0.15, framestyle = :box, tickfontsize = 7, legend = :bottomleft)
    it, sv, vr, sw = c["iter"], c["log10_statvar"], c["vhat_ratio_wmean"], c["step_w_wmean"]
    allx = filter(isfinite, vec(vr))
    isempty(allx) || xlims!(plt, (min(minimum(allx), 0.0) - 0.5, quantile(allx, 0.995) + 0.5))   # clip rare overflow-scale spikes
    for rg in REGIMES
        x, y = Float64[], Float64[]
        for i in axes(it, 2), r in 1:c["n_records"][i, 1]
            (isfinite(sv[r, i, 1]) && rg.test(sv[r, i, 1]) && isfinite(vr[r, i, 1]) && isfinite(sw[r, i, 1]) && sw[r, i, 1] > 0) || continue
            push!(x, vr[r, i, 1]); push!(y, log10(sw[r, i, 1]))
        end
        scatter!(plt, x, y; ms = 2, msw = 0, alpha = 0.35, color = rg.color, label = rg.name)
    end
    return plt
end

function make_figure(path)
    c = JLD2.load(path)
    m = c["meta"]
    model = m["case"] == "l63" ? "l63" : m["case"]
    refJ = haskey(FD_REF, model) ? [FD_REF[model]] : Float64[]
    pa = hist_panel(c, "log10_J", "(a) magnitude: ForwardDiff ‖J̃‖" * (isempty(refJ) ? "" : " (dotted: FD truth ≈ 221)"),
                    "log10 ‖J̃‖"; refs = refJ)
    pb = hist_panel(c, "lin_ratio_wmean", "(b) validity: actual / J̃-predicted Δr̃ (iterations ≤ 100)",
                    "log10(‖Δr̃_actual‖ / ‖J̃Δθ‖)   (0 = derivative predicts the response)"; refs = [0.0], itest = it -> it <= 100)
    pc = hist_panel(c, "cos_truth_wmean", "(c) direction: does −g point toward θ_true?", "cos(−Lpᵀg, Lp⁻¹(θ_true − θ))"; refs = [0.0], bins = 30)
    pd = step_panel(c)
    n_ok = count(==("converged"), c["status"][:, 1])
    n_all = count(in(("converged", "failed")), c["status"][:, 1])
    ttl = "adam · $(m["case"]) · $(m["date"]) · derivative diagnosis pooled over $(n_all) seeds ($(n_ok) converged)"
    fig = plot(pa, pb, pc, pd; layout = (2, 2), size = (1250, 850), plot_title = ttl, plot_titlefontsize = 11,
               dpi = 150, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
    stem = replace(basename(path), r"^diagnostics_" => "derivative_problems_", r"\.jld2$" => "")
    for ext in ("png", "pdf")
        savefig(fig, joinpath(dirname(path), "$stem.$ext"))
    end
    println("Saved $(joinpath(dirname(path), stem)).{png,pdf}")
end

function main()
    if haskey(ENV, "DIAG_FILE")
        make_figure(ENV["DIAG_FILE"])
        return
    end
    cases = haskey(ENV, "EXPERIMENT") ? [Symbol(ENV["EXPERIMENT"])] : experiments
    for case in cases
        path = joinpath(DIAG_DIR, diag_filename(nc_filename(experiment_config(case))))
        isfile(path) ? make_figure(path) : (@info "No combined diagnostics file $path — run combine_diagnostics.jl first")
    end
end

main()
