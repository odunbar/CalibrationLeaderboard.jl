# regime_paths_l63.jl — does the score-based (GFDT) optimiser fail where the parameter lands in a non-chaotic regime?
#
# One figure, two panels (LM-KGMM | Adam-KGMM), same background in both: the OFFLINE map of how chaotic L63 is over the
# (log ρ, log β) prior region. For each grid θ the forward model the experiments use is integrated from M perturbed ICs;
# an IC counts as chaotic if the smallest state-component variance over the SECOND HALF of the statistics window is
# ≥ 1e-2 (the run's own collapse threshold, applied to the second half so a decaying transient is not counted as chaos).
# Background = fraction of ICs that pass ("probability of chaos"): 0 fixed point for every IC, 1 chaotic for every IC.
# Overlaid per panel: the recorded θ path of every seed (blue solid = converged, orange dashed = failed; open circle =
# start, filled = end), the true θ (red +), and the prior 1σ / 2σ contours (dotted).
#
# The runs themselves are made by the two methods' own run scripts (N_ens = 1, rmse_target = 1.0, kgmm / parallel):
#   RUN_DATE=2099-01-05 SCORE_KIND=kgmm BUDGET_MODE=parallel julia --project=../score_based_lm run_l63_sblm.jl  <seed>
#   RUN_DATE=2099-01-05 SCORE_KIND=kgmm BUDGET_MODE=parallel julia --project=../score_based_lm run_l63_sbadam.jl <seed>
# (score_based_adam/ has the same Project.toml as score_based_lm/ but no Manifest, so the LM environment is reused.)
#
#   julia --project=../../score_based_lm regime_paths_l63.jl        # from this directory
# Env: REGIME_M=24 (ICs per grid point), REGIME_N=31 (grid size per axis), PATH_DATE=2099-01-05,
#      PATH_SEEDS=1,2,3,4,5,7,8,9,10,...,19

ENV["GKSwstype"] = "100"
using JLD2, LinearAlgebra, Plots, Random, Statistics

const OPT = joinpath(@__DIR__, "..", "..")
include(joinpath(OPT, "..", "common", "forward_maps", "Lorenz63.jl"))
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# (panel title, result directory, file stem) — filename: <stem>_result_<N_ens>_<target>_<seed>_<date>.jld2
const METHODS = [
    ("LM-KGMM",   joinpath(OPT, "score_based_lm", "output"),   "l63_sblm_kgmm_parallel"),
    ("Adam-KGMM", joinpath(OPT, "score_based_adam", "output"), "l63_sbadam_kgmm_parallel"),
]

ld = load_preliminaries(joinpath(OPT, "score_based_lm", "output", "l63_computed_preliminaries.jld2"))
M = parse(Int, get(ENV, "REGIME_M", "24")); NG = parse(Int, get(ENV, "REGIME_N", "31"))
logρ = range(log(14.0), log(40.0), length = NG)
logβ = range(log(1.0), log(12.0), length = NG)
gridfile = joinpath(OUT, "regime_map_l63_chaosfrac_M$(M)_N$(NG).jld2")

win = stats_window_indices(ld.lorenz_config_settings, ld.observation_config)
nh = length(win) ÷ 2
if isfile(gridfile)
    frac_chaotic = load(gridfile, "frac_chaotic")
else
    frac_chaotic = zeros(NG, NG)          # [β index, ρ index]
    rng = MersenneTwister(2026)
    x0s = [ld.x0 .+ ld.ic_cov_sqrt * randn(rng, 3) for _ in 1:M]
    Threads.@threads for ib in 1:NG
        for ir in 1:NG
            θ = [logρ[ir], logβ[ib]]
            chaotic = map(x0s) do x0
                X = lorenz_forward_with_states(EnsembleMemberConfig(exp.(θ)), x0, ld.lorenz_config_settings, ld.observation_config).states[:, win]
                minimum(var(X[:, (nh + 1):end], dims = 2)) >= 1e-2
            end
            frac_chaotic[ib, ir] = mean(chaotic)
        end
    end
    jldsave(gridfile; frac_chaotic, logρ = collect(logρ), logβ = collect(logβ), M)
end

# ---- overlays
prior_mean = [3.3, 1.2]; prior_sd = [0.15, 0.5]
ell(k) = (prior_mean[1] .+ k * prior_sd[1] * cos.(range(0, 2π, 100)), prior_mean[2] .+ k * prior_sd[2] * sin.(range(0, 2π, 100)))
θ_true = log.([28.0, 8.0 / 3.0])
pdate = get(ENV, "PATH_DATE", "2099-01-05")
seeds = parse.(Int, split(get(ENV, "PATH_SEEDS", "1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19"), ","))

# recorded θ paths of one method: seed => (theta_1, theta_2, converged?)
function load_paths(dir, stem)
    paths = Dict{Int, Any}()
    for s in seeds
        fn = joinpath(dir, "$(stem)_result_1_1p0_$(s)_$(pdate).jld2")
        isfile(fn) || continue
        r = load(fn); d = r["diagnostics"]
        haskey(d, "theta_1") && (paths[s] = (d["theta_1"], d["theta_2"], isfinite(r["conv_score"])))
    end
    return paths
end

BLUE, ORANGE = "#2a78d6", "#eb6834"
function regime_panel(title, paths; show_ylabel)
    p = heatmap(collect(logρ), collect(logβ), frac_chaotic; xlabel = "log ρ", ylabel = show_ylabel ? "log β" : "", clims = (0, 1),
                color = cgrad(:Greys), colorbar = false,
                title = "L63: $title paths", xlims = (logρ[1], logρ[end]), ylims = (logβ[1], logβ[end]),
                legend = :bottomleft, legend_font_pointsize = 12, titlefontsize = 21, guidefontsize = 17, tickfontsize = 12, background_color_legend = RGBA(1, 1, 1, 0.9), label = "",
                left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 4Plots.mm, top_margin = 8Plots.mm)
    for k in (1, 2)
        plot!(p, ell(k)...; color = :gray40, ls = :dot, lw = 4.8, label = k == 1 ? "prior 1σ, 2σ" : "")
    end
    shown = Set{Bool}()
    for s in sort(collect(keys(paths)))
        t1, t2, ok = paths[s]
        lab = ok in shown ? "" : (ok ? "converged run" : "failed run")
        push!(shown, ok)
        plot!(p, t1, t2; color = ok ? BLUE : ORANGE, lw = 2, ls = ok ? :solid : :dash, label = lab)
        scatter!(p, [t1[1]], [t2[1]]; color = :white, markerstrokecolor = ok ? BLUE : ORANGE, markerstrokewidth = 2, markersize = 6, label = "")
        scatter!(p, [t1[end]], [t2[end]]; color = ok ? BLUE : ORANGE, markerstrokecolor = ok ? BLUE : ORANGE, markersize = 5, label = "")
    end
    scatter!(p, [θ_true[1]], [θ_true[2]]; marker = :cross, color = :red, markerstrokecolor = :red, markerstrokewidth = 4, markersize = 16, label = "true θ")
    return p
end

panels = Any[]
for (i, (title, dir, stem)) in enumerate(METHODS)
    paths = load_paths(dir, stem)
    nok = count(v -> v[3], values(paths))
    println(rpad(title, 10), " seeds with paths: ", length(paths), "   converged: ", nok, "   failed: ", length(paths) - nok)
    push!(panels, regime_panel(title, paths; show_ylabel = i == 1))
end
# one colorbar shared by both panels: an empty plot whose only content is the colorbar
cbar = scatter([NaN], [NaN]; zcolor = [NaN], clims = (0, 1), color = cgrad(:Greys), label = "", framestyle = :none,
               xlims = (0, 1), ylims = (0, 1), ticks = false, grid = false, colorbar = :right, colorbar_title = "probability of chaos", colorbar_titlefontsize = 17, tickfontsize = 12,
               right_margin = 6Plots.mm)
fig = plot(panels..., cbar; layout = grid(1, length(panels) + 1; widths = [fill(0.46, length(panels)); 0.08]), size = (1700, 760))
savefig(fig, joinpath(OUT, "regime_paths_l63.png")); savefig(fig, joinpath(OUT, "regime_paths_l63.pdf"))
println("wrote ", joinpath(OUT, "regime_paths_l63.png"))
