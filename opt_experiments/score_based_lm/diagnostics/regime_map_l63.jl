# regime_map_l63.jl — OFFLINE map of how chaotic L63 is over the (ρ, β) prior region, with the per-seed θ paths on top.
#
# Tests: "the score-based Jacobian fails where the parameter lands in a different (non-mixing / multistable)
# regime". Offline, so it may spend forward evaluations the runs may not. For each grid θ = (log ρ, log β) it
# integrates the same forward model the experiment uses from M perturbed ICs and asks, per IC, whether the statistics
# window is still chaotic at its END: the smallest state-component variance over the SECOND HALF of the window is
# ≥ 1e-2 (the run's own collapse threshold, applied to the second half so that a decaying transient that has not yet
# reached a fixed point does not count as chaos). Background = the fraction of ICs that pass:
#   0 (white)  never chaotic: every IC decays onto a fixed point
#   1 (black)  chaotic for every IC
#   between    multistable: the same θ gives chaos for some ICs and a fixed point for others
# Overlaid: the per-seed θ paths (blue solid = converged, orange dashed = failed; open circle = start, filled = end),
# a large red + at the true value, and the prior 1σ/2σ contours (dotted).
#
#   julia --project=. diagnostics/regime_map_l63.jl
# Env: REGIME_M=24 (ICs per grid point), REGIME_N=31 (grid size per axis),
#      PATH_DATE=2099-01-05 (run date whose per-seed diagnostics supply θ paths), PATH_SEEDS=1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19

ENV["GKSwstype"] = "100"
using JLD2, LinearAlgebra, Plots, Random, Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include(joinpath(@__DIR__, "..", "experiment_config.jl"))
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

ld = load_preliminaries(joinpath(@__DIR__, "..", "output", "l63_computed_preliminaries.jld2"))
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

# per-seed θ paths (recorded theta_1, theta_2)
paths = Dict{Int, Any}()
pdate = get(ENV, "PATH_DATE", "2099-01-05")
for s in parse.(Int, split(get(ENV, "PATH_SEEDS", "1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19"), ","))
    fn = joinpath(@__DIR__, "..", "output", "l63_sblm_kgmm_parallel_result_1_1p0_$(s)_$(pdate).jld2")
    isfile(fn) || continue
    r = load(fn); d = r["diagnostics"]
    haskey(d, "theta_1") && (paths[s] = (d["theta_1"], d["theta_2"], isfinite(r["conv_score"])))
end

BLUE, ORANGE = "#2a78d6", "#eb6834"
p = heatmap(collect(logρ), collect(logβ), frac_chaotic; xlabel = "log ρ", ylabel = "log β", clims = (0, 1), color = cgrad(:Greys),
            colorbar_title = "probability of chaos",
            title = "L63: LM-KGMM paths",
            xlims = (logρ[1], logρ[end]), ylims = (logβ[1], logβ[end]), size = (900, 680), margin = 6Plots.mm, legend = :outerbottom,
            legend_column = 4, label = "")
for k in (1, 2)
    plot!(p, ell(k)...; color = :gray40, ls = :dot, lw = 1.2, label = k == 1 ? "prior 1σ, 2σ" : "")
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
savefig(p, joinpath(OUT, "regime_map_l63.png")); savefig(p, joinpath(OUT, "regime_map_l63.pdf"))
println("wrote ", joinpath(OUT, "regime_map_l63.png"))
