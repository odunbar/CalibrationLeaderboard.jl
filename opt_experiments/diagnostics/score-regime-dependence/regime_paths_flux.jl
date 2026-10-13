# regime_paths_flux.jl — does the score-based LM (GFDT) fail on l96_flux when θ starts in, or passes through, the region where
# the run's collapse flag (min over sites of the temporal variance < 1e-2 ⇒ ForwardDiff fallback) fires?
#
# L63 has a 2-D θ, so regime_paths_l63.jl draws paths over a (log ρ, log β) map. Here θ is the 61 weights of the forcing NN, so the
# regime is read off two scalars that the run already records at every iteration (regime_diagnostics.jl):
#   F_min          min over sites of the forcing F(x; θ)       (the physical regime parameter, see score_based_lm/diagnostics/regime_map_flux.jl)
#   log10_varmin   log10 of the smallest per-site temporal variance in the run's window   (flag fires below −2)
# Three panels:
#   A  regime plane: OFFLINE prior-draw map (grey/orange dots from regime_map_flux.jl: 200 draws, flagged or not) with each run's
#      recorded (F_min, log10_varmin) path over it. Blue solid = converged, orange dashed = failed; open circle = start, filled = end.
#   B  log10_varmin against iteration, with the flag threshold (−2).
#   C  RMSE against iteration with the target; iterations whose window is flagged are marked ×.
# Printed table: per seed, whether the run STARTED flagged, how many recorded iterations were flagged, final RMSE, outcome and
# ad_iters / gfdt_iters.
#
# Caveats: (i) build_l96_problem("flux-force") trains the prior-mean network with the unseeded global RNG, so each process has a different
# prior mean; a "seed" is therefore one draw of (problem instance, θ_init), not a reproducible start. (ii) The recorded flag value
# is the run's own window (min over the N_ens branches), the map's is one IC, so the two agree only up to that spread.
#
# Runs are made by the method's own run script (N_ens = 10, rmse_target = 1.0, kgmm / parallel), task index = seed:
#   EXPERIMENT=l96_flux SCORE_KIND=kgmm BUDGET_MODE=parallel RUN_DATE=2099-01-06 julia --project=../score_based_lm run_l96_sblm.jl <seed>
# Background map: julia --project=. diagnostics/regime_map_flux.jl   (from score_based_lm/)
#
#   julia --project=../../score_based_lm regime_paths_flux.jl        # from this directory
# Env: PATH_DATES=2099-01-04,2099-01-06 (result dates to pool), PATH_N=10, PATH_SEEDS=1,...,16 (default: every file found)

ENV["GKSwstype"] = "100"
using JLD2, Plots, Statistics

const OPT = joinpath(@__DIR__, "..", "..")
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)
const RESDIR = joinpath(OPT, "score_based_lm", "output")
const STEM = "l96_sblm_kgmm_parallel_result_flux-force"
const N_ENS = get(ENV, "PATH_N", "10")
const THRESH = -2.0                                   # log10(1e-2), the run's own collapse threshold

dates = split(get(ENV, "PATH_DATES", "2099-01-06"), ",")
seed_filter = haskey(ENV, "PATH_SEEDS") ? parse.(Int, split(ENV["PATH_SEEDS"], ",")) : nothing

# one entry per (date, seed) file with recorded regime keys
runs = NamedTuple[]
for dt in dates, s in 1:200
    seed_filter !== nothing && !(s in seed_filter) && continue
    fn = joinpath(RESDIR, "$(STEM)_$(N_ENS)_1p0_$(s)_$(dt).jld2")
    isfile(fn) || continue
    r = load(fn); d = r["diagnostics"]
    all(haskey(d, k) for k in ("iter", "rmse", "log10_varmin", "F_min")) || (@warn "no regime keys: $fn"; continue)
    push!(runs, (; seed = s, date = dt, it = Float64.(d["iter"]), rmse = Float64.(d["rmse"]), lv = Float64.(d["log10_varmin"]),
                   Fmin = Float64.(d["F_min"]), ok = isfinite(r["conv_score"]), ad = r["ad_iters"], gfdt = r["gfdt_iters"]))
end
isempty(runs) && error("no result files with regime keys for dates $(dates) in $RESDIR")
label(r) = length(dates) > 1 ? "$(r.seed)@$(r.date[end-4:end])" : string(r.seed)

# ---- table
println("seed  start_flagged  n_flagged/n_rec  min_log10var  final_rmse  outcome    ad/gfdt")
for r in runs
    fl = r.lv .< THRESH
    println(rpad(label(r), 6), rpad(fl[1], 15), rpad("$(sum(fl))/$(length(fl))", 17), rpad(round(minimum(r.lv); digits = 2), 14),
            rpad(round(r.rmse[end]; digits = 2), 12), rpad(r.ok ? "converged" : "FAILED", 11), "$(r.ad)/$(r.gfdt)")
end
groups = (("start flagged", r -> r.lv[1] < THRESH), ("start unflagged, passes through flag", r -> r.lv[1] >= THRESH && any(r.lv .< THRESH)),
          ("never flagged", r -> all(r.lv .>= THRESH)))
println("\ngroup                                   n   converged")
for (name, f) in groups
    g = filter(f, runs)
    println(rpad(name, 40), rpad(length(g), 4), count(r -> r.ok, g))
end

# ---- figure
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a9a"
bg = joinpath(OPT, "score_based_lm", "diagnostics", "output", "regime_map_flux_draws.jld2")
fs = (titlefontsize = 17, guidefontsize = 14, tickfontsize = 11, legend_font_pointsize = 10, background_color_legend = RGBA(1, 1, 1, 0.9))

pA = plot(; xlabel = "F_min at θ", ylabel = "log10 min-site variance", title = "regime plane", legend = :bottomright,
          left_margin = 10Plots.mm, bottom_margin = 8Plots.mm, fs...)
if isfile(bg)
    d = load(bg); c = d["collapsed"]
    scatter!(pA, d["Fmin"][.!c], d["lvmin"][.!c]; color = GREY, alpha = 0.35, markerstrokewidth = 0, markersize = 4, label = "prior draws, not flagged")
    scatter!(pA, d["Fmin"][c], d["lvmin"][c]; color = ORANGE, alpha = 0.25, markerstrokewidth = 0, markersize = 4, label = "prior draws, flagged")
else
    @warn "no background map at $bg (run score_based_lm/diagnostics/regime_map_flux.jl)"
end
hline!(pA, [THRESH]; color = :black, ls = :dot, label = "flag threshold")
pB = plot(; xlabel = "LM iteration", ylabel = "log10 min-site variance", title = "window regime along the run", legend = false,
          left_margin = 10Plots.mm, bottom_margin = 8Plots.mm, fs...)
hline!(pB, [THRESH]; color = :black, ls = :dot)
pC = plot(; xlabel = "LM iteration", ylabel = "RMSE", title = "progress to target (× = flagged window)", yscale = :log10, legend = :topright,
          left_margin = 10Plots.mm, bottom_margin = 8Plots.mm, fs...)
hline!(pC, [1.0]; color = :black, ls = :dash, label = "target 1.0")

shown = Set{Bool}()
for r in runs
    col = r.ok ? BLUE : ORANGE; ls = r.ok ? :solid : :dash
    lab = r.ok in shown ? "" : (r.ok ? "converged run" : "failed run"); push!(shown, r.ok)
    plot!(pA, r.Fmin, r.lv; color = col, ls, lw = 1.6, label = lab)
    scatter!(pA, [r.Fmin[1]], [r.lv[1]]; color = :white, markerstrokecolor = col, markerstrokewidth = 2, markersize = 6, label = "")
    scatter!(pA, [r.Fmin[end]], [r.lv[end]]; color = col, markerstrokecolor = col, markersize = 5, label = "")
    plot!(pB, r.it, r.lv; color = col, ls, lw = 1.6, label = "")
    plot!(pC, r.it, r.rmse; color = col, ls, lw = 1.6, label = lab)
    fl = r.lv .< THRESH
    any(fl) && scatter!(pC, r.it[fl], r.rmse[fl]; marker = :xcross, color = :black, markersize = 4, label = "")
end
fig = plot(pA, pB, pC; layout = (1, 3), size = (1900, 600), top_margin = 6Plots.mm)
savefig(fig, joinpath(OUT, "regime_paths_flux.png")); savefig(fig, joinpath(OUT, "regime_paths_flux.pdf"))
println("wrote ", joinpath(OUT, "regime_paths_flux.png"))
