# Shared helpers for the forward-map visualisations (model-agnostic).
#
# Include this AFTER a forward map (common/forward_maps/Lorenz63.jl or Lorenz96.jl):
# the functions below call `lorenz_forward_with_states` / `stats_window_indices`,
# which are resolved at call time from whichever forward map has been included.
# (The two forward maps define clashing names, so only one can be loaded per session.)
#
# A `prob` (built by `l63_setup()` / `l96_setup(case)`) is a NamedTuple with fields
#   name, nx, ny, dt, T, config, obs, phi_truth, truth_θ, prior, build(θ), x0, y, R, ic_cov_sqrt,
#   param_names, output_names
# Nothing here touches an optimisation algorithm: it only runs the simulator.

get!(ENV, "GKSwstype", "100")  # headless GR: we only ever savefig, never display

using Plots
using StatsPlots
using Statistics
using LinearAlgebra
using Random
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

const OUTPUT_DIR = joinpath(@__DIR__, "output")

########################################################################
############################ Style #####################################
########################################################################
# First three slots of the validated dataviz palette (safe to use together),
# plus neutral ink/gray. One colour per *entity*, reused in every figure.
const INK_COLOR = "#0b0b0b"          # truth / data
const ALT_PARAM_COLOR = "#eb6834"    # same IC, different parameter
const ALT_IC_COLOR = "#1baf7a"       # same parameter, different initial condition
const PRIOR_COLOR = "#2a78d6"        # prior samples / pushforwards
const WINDOW_COLOR = "#eda100"       # statistics-window highlight (background wash)
const MUTED_COLOR = "#8a8984"        # context (pre-window segments, IC cloud, noise bands)

const FONT_SCALE = 3  # multiplier on the base font sizes below (base: tick 10, guide 11, title 12, legend 9)
const LW_SCALE = 1.5  # multiplier on every line width (use lwidth(x) for explicit widths)
lwidth(x) = LW_SCALE * x
const SIZE_SCALE = 2  # figures are drawn on a canvas this much bigger so the large fonts still fit
figsize(w, h) = (SIZE_SCALE * w, SIZE_SCALE * h)

default(
    framestyle = :box,
    grid = false,
    linewidth = LW_SCALE,
    tickfontsize = 10 * FONT_SCALE,
    guidefontsize = 11 * FONT_SCALE,
    titlefontsize = 12 * FONT_SCALE,
    legendfontsize = 9 * FONT_SCALE,
    plot_titlefontsize = 16 * FONT_SCALE,
    titlelocation = :left,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.75),
    left_margin = 26Plots.mm,
    bottom_margin = 16Plots.mm,
    top_margin = 10Plots.mm,
    dpi = 60,  # layout and font proportions are dpi-independent; lower dpi = fewer pixels = smaller PNGs
)

function save_figure(p, name::AbstractString)
    mkpath(OUTPUT_DIR)
    path = joinpath(OUTPUT_DIR, name)
    savefig(p, path)
    @info "Saved figure" path
    return path
end

########################################################################
########################## Time / window helpers ########################
########################################################################
time_axis(xn::AbstractMatrix, dt) = range(0.0, step = dt, length = size(xn, 2))

# Time span covered by the statistics window. Column j of `lorenz_solve` holds
# time (j-1)*dt, and `stats_window_indices` is the exact column set `stats` averages.
window_times(idx::AbstractUnitRange, dt) = ((first(idx) - 1) * dt, (last(idx) - 1) * dt)

shade_window!(p, t0, t1; label = false) =
    vspan!(p, [t0, t1]; color = WINDOW_COLOR, alpha = 0.18, linewidth = 0, label = label)

# Trajectory drawn faint before the statistics window and bold inside it, in the dims `dims`
# (2 dims -> 2D phase portrait).
function window_segments!(p, xn, idx, dims; color, label, lw = 1.6)
    pre = 1:first(idx)
    win = idx
    plot!(p, (xn[d, pre] for d in dims)...; color = color, alpha = 0.3, linewidth = lwidth(0.6), label = false)
    plot!(p, (xn[d, win] for d in dims)...; color = color, linewidth = lwidth(lw), label = label)
end

########################################################################
############################ Statistics #################################
########################################################################
row_quantile(M::AbstractMatrix, q) = [quantile(r, q) for r in eachrow(M)]

# Median + 25-75% (line + inner ribbon) and 5-95% (outer ribbon) across the columns of M.
function band!(p, x, M::AbstractMatrix; color, label)
    q05, q25, q50, q75, q95 = (row_quantile(M, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95))
    plot!(p, x, q50; ribbon = (q50 .- q05, q95 .- q50), fillalpha = 0.15, color = color, linewidth = 0, label = "$(label) 5–95%")
    plot!(p, x, q50; ribbon = (q50 .- q25, q75 .- q50), fillalpha = 0.30, color = color, linewidth = lwidth(2), label = "$(label) median, 25–75%")
end

########################################################################
############################ Simulations ################################
########################################################################
# Constrained-space prior draws, one column per sample.
sample_prior_constrained(rng, prior, n) =
    transform_unconstrained_to_constrained(prior, construct_initial_ensemble(rng, prior, n))

run_forward(prob, phi, x0) = lorenz_forward_with_states(phi, x0, prob.config, prob.obs)

# The three simulations of plot 1 (+ a cloud of IC-perturbed statistics that shows the chaos noise):
#   truth       : true parameter, spun-up initial condition x0 (so G == y exactly)
#   alt_param   : a different parameter (first prior draw accepted by `alt_accept`), same x0
#   alt_ic      : the true parameter, x0 + ic_cov_sqrt * randn  (the perturbation the
#                 leaderboard's methods use), so the difference is pure chaos
#   cloud_G     : statistics of `n_ic_cloud` more IC-perturbed runs at the true parameter
function run_scenarios(prob; seed = 1, n_ic_cloud = 30, alt_accept = θ -> true)
    rng = MersenneTwister(seed)
    perturbed_x0() = prob.x0 .+ prob.ic_cov_sqrt * randn(rng, prob.nx)

    # first prior draw (in the seeded stream) that `alt_accept` allows, e.g. θ -> θ[2] < truth β
    θ_alt = nothing
    for _ in 1:10_000
        θ = sample_prior_constrained(rng, prob.prior, 1)[:, 1]
        alt_accept(θ) && (θ_alt = θ; break)
    end
    θ_alt === nothing && error("no prior draw accepted by `alt_accept` in 10000 tries")
    phi_alt = prob.build(θ_alt)
    x0_alt = perturbed_x0()

    truth = run_forward(prob, prob.phi_truth, prob.x0)
    alt_param = run_forward(prob, phi_alt, prob.x0)
    alt_ic = run_forward(prob, prob.phi_truth, x0_alt)
    cloud_G = n_ic_cloud == 0 ? zeros(prob.ny, 0) : hcat((run_forward(prob, prob.phi_truth, perturbed_x0()).G for _ in 1:n_ic_cloud)...)
    return (; truth, alt_param, alt_ic, θ_alt, phi_alt, x0_alt, cloud_G)
end

# Push `n` prior draws through the forward map (all from the same spun-up x0).
# Keeps the raw states of the first `n_traj` finite runs. Non-finite runs are dropped.
function prior_pushforward(prob; n = 100, n_traj = 8, seed = 2)
    rng = MersenneTwister(seed)
    θ = sample_prior_constrained(rng, prob.prior, n)
    keep = Int[]
    Gs = Vector{Vector{Float64}}()
    trajs = Matrix{Float64}[]
    for j in 1:n
        r = run_forward(prob, prob.build(θ[:, j]), prob.x0)
        all(isfinite, r.G) || continue
        push!(keep, j)
        push!(Gs, r.G)
        length(trajs) < n_traj && push!(trajs, r.states)
    end
    n_dropped = n - length(keep)
    n_dropped > 0 && @warn "Dropped $(n_dropped) prior samples with non-finite statistics"
    return (; θ = θ[:, keep], phis = [prob.build(θ[:, j]) for j in keep], G = reduce(hcat, Gs), trajs, n_dropped)
end

# Add head-room above the data so an in-panel legend does not sit on top of it.
function pad_top!(p; frac = 0.3)
    lo, hi = ylims(p)
    ylims!(p, lo, hi + frac * (hi - lo))
end
