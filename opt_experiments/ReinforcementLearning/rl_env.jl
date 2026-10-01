# ReinforcementLearning — environment: the Lorenz forward map viewed as an RL environment.
#
# Include AFTER common/forward_maps/Lorenz{63,96}.jl (uses lorenz_solve, stats,
# stats_window_indices, lorenz_forward_with_states, LorenzConfig, ObservationConfig,
# which have identical signatures in both files).
#
# Required fields of `prob`:  x0, y, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx, ny,
#                             make_forcing (F::Vector -> forcing object)
#
# One trajectory is followed continuously, x_{k+1} being the final state of step k.
#
#   :long   each step is ONE full forward evaluation (LorenzConfig length T, statistics
#           over [T_start, T_end]) started at x_k.   cost = 1 unit / step.
#   :short  after one initial full evaluation, each step advances the trajectory by
#           stride = W/steps_per_window time units under F_k (W = length of the statistics
#           window) and recomputes the SAME statistics G over the trailing W time units
#           (sliding window, same number of samples as the standard window).  Consecutive
#           windows overlap, so there are steps_per_window RL updates per W of simulated
#           time.   cost = stride/T units / step, T = full forward-evaluation length.
#
# Φ_k = ‖R^{-1/2}(y − G_k)‖² = ny · RMSE_k², i.e. the whitened misfit used by every other
# experiment's convergence test.

mutable struct RLEnv
    x::Vector{Float64}                         # current state x_k
    window::Union{Nothing, Matrix{Float64}}    # :short only — trailing statistics window
end

RLEnv(x::AbstractVector) = RLEnv(Vector{Float64}(x), nothing)

# Fresh IC: perturbation of the truth-attractor state, as in every other experiment.
reset_env!(env::RLEnv, prob, rng) = begin
    env.x      = prob.x0 .+ prob.ic_cov_sqrt * randn(rng, prob.nx)
    env.window = nothing
    env
end

window_columns(prob) = length(stats_window_indices(prob.lorenz_cfg, prob.obs_cfg))
stride_steps(prob, cfg) = max(1, window_columns(prob) ÷ cfg.steps_per_window)

whitened_misfit(prob, G) = (r = prob.R_inv_var * (prob.y .- G); dot(r, r))

"""
    env_step!(env, prob, cfg, F) -> (; x, x_next, G, Φ, cost)

Advance the environment one RL step under action `F` (raw units).  `cost` is in units of
one full forward evaluation.
"""
function env_step!(env::RLEnv, prob, cfg, F)
    params = prob.make_forcing(F)
    x_k    = copy(env.x)

    if cfg.update_frequency === :long || env.window === nothing
        # full standard forward evaluation (for :short this initialises the window)
        out = lorenz_forward_with_states(params, env.x, prob.lorenz_cfg, prob.obs_cfg)
        env.x = out.states[:, end]
        if cfg.update_frequency === :short
            env.window = out.states[:, stats_window_indices(prob.lorenz_cfg, prob.obs_cfg)]
        end
        G, cost = out.G, 1.0
    else
        dt = prob.lorenz_cfg.dt
        nΔ = stride_steps(prob, cfg)
        Nw = size(env.window, 2)
        # T = (nΔ - 1/2) dt so that ceil(T/dt) = nΔ robustly (no floating-point overshoot)
        xn = lorenz_solve(params, env.x, LorenzConfig(dt, (nΔ - 0.5) * dt))
        env.x      = xn[:, end]
        env.window = hcat(env.window[:, (nΔ + 1):end], xn[:, 2:end])
        # all Nw columns: indices ceil(0.5)=1 .. ceil(Nw-0.5)=Nw
        G    = stats(env.window, prob.lorenz_cfg, ObservationConfig(0.5 * dt, (Nw - 0.5) * dt))
        cost = nΔ * dt / prob.lorenz_cfg.T
    end
    return (; x = x_k, x_next = copy(env.x), G, Φ = whitened_misfit(prob, G), cost)
end
