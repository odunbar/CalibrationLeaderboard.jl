# regime_map_flux.jl — OFFLINE: which prior draws of the l96_flux parameters land in a regime where the run's
# `is_collapsed_window` flag (min over the 100 sites of the temporal variance < 1e-2) fires?
#
# Why it matters: when the flag fires, the run abandons GFDT for ForwardDiff through the chaotic L96 window
# (log10‖J̃‖ ≈ 35 in the diagnostics run) and, in that run, θ never moved. The flag was designed for L63 (a 3-D
# fixed point); on L96 with a spatially varying forcing it can fire because a FEW sites with low / negative
# forcing are quiet while the rest of the domain is chaotic.
#
# Per prior draw θ ~ N(prior_mean, prior_cov) (same prior as the run) integrate the same forward model once and report
#   F_min / F_mean / F_max  forcing profile at θ
#   log10 min_i var_i       the quantity the flag thresholds (log10(1e-2) = −2)
#   frac_quiet_sites        fraction of sites with var_i < 1e-2
#   collapsed               flag fires (⇒ AD fallback in the run)
# NOTE: `build_l96_problem("flux-force")` trains its networks with the *global, unseeded* RNG, so the prior mean θ is a
# different weight vector in every process; this map is therefore one problem instance (seeded draws below are over θ only).
#
#   julia --project=. diagnostics/regime_map_flux.jl      # env: FLUX_K=200 draws, FLUX_SEED=1
ENV["GKSwstype"] = "100"
using Distributions, Flux, JLD2, LinearAlgebra, Plots, Random, Statistics

const _COMMON = joinpath(@__DIR__, "..", "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "score_gfdt.jl"))
include(joinpath(@__DIR__, "..", "l96_problem.jl"))
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

K = parse(Int, get(ENV, "FLUX_K", "200"))
prob = build_l96_problem("flux-force", joinpath(@__DIR__, "..", "output"))
(; x0, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx, prior_mean, prior_cov) = prob
win = stats_window_indices(lorenz_cfg, obs_cfg)
prior = MvNormal(prior_mean, Matrix(Symmetric(prior_cov)))

rng = MersenneTwister(parse(Int, get(ENV, "FLUX_SEED", "1")))
θs = [rand(rng, prior) for _ in 1:K]
x0s = [x0 .+ ic_cov_sqrt * randn(rng, nx) for _ in 1:K]
Fmin = zeros(K); Fmean = zeros(K); Fmax = zeros(K); lvmin = zeros(K); quiet = zeros(K); vars = Vector{Vector{Float64}}(undef, K); Fs = Vector{Vector{Float64}}(undef, K)
Threads.@threads for k in 1:K
    emc = make_emc(prob, θs[k])
    Fs[k] = Float64.(forcing(emc, x0))
    X = lorenz_forward_with_states(emc, x0s[k], lorenz_cfg, obs_cfg).states[:, win]
    v = vec(var(X, dims = 2)); vars[k] = v
    Fmin[k], Fmean[k], Fmax[k] = minimum(Fs[k]), mean(Fs[k]), maximum(Fs[k])
    lvmin[k] = log10(max(minimum(v), floatmin(Float64))); quiet[k] = mean(v .< 1e-2)
end
collapsed = lvmin .< -2
println("K=$K prior draws: flag fires (min-site var < 1e-2) for $(sum(collapsed)) = $(round(100 * mean(collapsed); digits = 1))%")
println("  F_min  of flagged draws: median $(round(median(Fmin[collapsed]); digits = 2)); unflagged: $(round(median(Fmin[.!collapsed]); digits = 2))")
println("  quiet-site fraction in flagged draws: median $(round(median(quiet[collapsed]); digits = 3)) (a few sites, not the whole domain)")
jldsave(joinpath(OUT, "regime_map_flux_draws.jld2"); Fmin, Fmean, Fmax, lvmin, quiet, collapsed)

p1 = scatter(Fmin, lvmin; group = collapsed, xlabel = "F_min at θ", ylabel = "log10 min-site variance (flag: < −2)", label = ["not flagged" "flagged → AD fallback"],
             color = [:"#2a78d6" :"#eb6834"], title = "L96 flux prior draws ($K)")
hline!(p1, [-2.0]; color = :black, ls = :dot, label = "")
p2 = scatter(Fmean, quiet; group = collapsed, xlabel = "F_mean at θ", ylabel = "fraction of sites with var < 1e-2", label = "", color = [:"#2a78d6" :"#eb6834"], title = "how much of the domain is quiet")
kbad = findfirst(collapsed)
p3 = if kbad === nothing
    plot(title = "no flagged draw")
else
    plot(1:nx, Fs[kbad]; label = "F(site) at flagged θ", xlabel = "site", color = "#eb6834", lw = 2, ylabel = "forcing",
         title = "flagged draw: forcing and site variance")
    plot!(twinx(), 1:nx, log10.(max.(vars[kbad], 1e-12)); label = "log10 var_i", color = "#2a78d6", lw = 2, ylabel = "log10 var_i", legend = :bottomright)
end
fig = plot(p1, p2, p3; layout = (1, 3), size = (1650, 500), margin = 6Plots.mm)
savefig(fig, joinpath(OUT, "regime_map_flux.png")); savefig(fig, joinpath(OUT, "regime_map_flux.pdf"))
println("wrote ", joinpath(OUT, "regime_map_flux.png"))
