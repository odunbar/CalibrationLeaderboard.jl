# rl_diagnostics.jl — critic-training diagnostics for the RL (DDPG/TD3) method.
#
# Included by rl_run.jl.  Base + LinearAlgebra + Statistics + Flux only (Flux is already a dependency of
# ddpg_core.jl); the recorder is the shared common/diagnostics/diag_recorder.jl.
#
# Nothing here draws from an RNG, runs the Lorenz model or mutates agent/buffer state.  Cheap per-step
# scalars are accumulated every step; the heavier probe (one extra critic backward pass on the minibatch
# the update already drew, one linear regression on the replay buffer) runs only on recorded steps.
#
# Question these answer: does the critic learn a usable ∇_F q (the actor's only information), and if not,
# is it because the critic fails, the data carry no recoverable action signal, or the actor moves anyway?

include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_recorder.jl"))

mutable struct RLDiag
    rec::DiagRecorder
    θn0::Vector{Float64}                  # initial actor (whitened coordinates)
    θn_true::Union{Nothing, Vector{Float64}}   # whitened truth, only if the problem has one
    g_prev::Union{Nothing, Vector{Float32}}
    path_len::Float64                     # Σ ‖Δθn‖ over actor updates
    frac::Float64                         # exploration std (prior-std units), reference scale for act_dist_rel
end

function RLDiag(n_steps_max::Integer, θn0::Vector{Float64}, θn_true, frac::Real)
    # n_steps_max is only an upper bound on the schedule length; the schedule is by RL step index
    rec = DiagRecorder(n_steps_max)
    describe!(rec,
        "rmse"            => "whitened RMSE of the window under the APPLIED action (policy + exploration noise), at the record",
        "cost"            => "cumulative simulation budget (forward-evaluation units, as in conv_score) at the record: x axis for comparing short and long",
        "rmse_wmin"       => "window minimum of RMSE since the last record",
        "critic_unexpl"   => "window mean of (pre-update TD loss on a fresh minibatch) / var(target): ≈ 1 = critic explains none of the target variance, ≪ 1 healthy; NaN-safe ratio",
        "zt_std_wmean"    => "window mean of the std of the TD targets in the batch (critic-output scale)",
        "g_norm_wmean"    => "window mean of ‖∇_θ actor loss‖ on actor-update steps (whitened units)",
        "cos_prev_wmean"  => "window mean of cos(g_t, g_{t-1}) of consecutive actor gradients; ≈ 0 = direction is noise",
        "step_pol_wmean"  => "window mean of ‖Δθn‖ per actor update (Adam ≈ actor_lr·√nu when the gradient is sign noise)",
        "cos_truth_wmean" => "window mean of cos(−g, θn_true − θn_pre) over actor updates: does the actor step (ascent of q) point at the truth; NaN where no truth (flux)",
        "policy_disp"     => "‖θn − θn_initial‖ (prior-std units): net actor displacement",
        "path_eff"        => "net displacement / total path length of the actor; ≈ 1/√n_updates = random walk, → 1 = steady drift",
        "path_vs_rw"      => "path_eff·√(n_actor_updates): ≈ 1 = random walk, ≫ 1 = directed motion",
        "dist_truth"      => "prior-whitened distance ‖θn − θn_true‖ of the noise-free policy (NaN where no truth: flux)",
        "grad_snr"        => "‖ḡ‖ / (rms per-sample deviation / √B): signal-to-noise of the batch-mean critic action gradient; ≲ 1 = noise",
        "grad_consist"    => "mean over batch of cos(∇_a q(x_i,θ), ḡ): do states agree on the action gradient direction",
        "log10_gnorm"     => "log10 ‖ḡ‖ (batch-mean critic action gradient, whitened action units)",
        "q_std_full"      => "std over batch of q(x_i, F_i) (buffer actions)",
        "q_std_state"     => "std over batch of q(x_i, θ) at the fixed policy action: state-driven spread of q",
        "sens_act_state"  => "rms ∂q/∂a per action component / rms ∂q/∂x per state component, at (x_i,θ); ≪ 1 = critic ignores the action",
        "dead_frac"       => "max over hidden layers of the fraction of units inactive for the whole batch: exactly zero (relu) or activation derivative < 0.01 (smooth activations)",
        "act_dist_rel"    => "median ‖θn − F_i‖ over batch / (explore_frac·√nu): ≈ 1 = actor still inside the exploration cloud of the data, ≫ 1 = critic extrapolating",
        "phi_r2_act"      => "adjusted R² of a linear regression of buffer Φ on buffer whitened actions: recoverable action signal at this sample size",
        "phi_r2_state"    => "adjusted R² of a linear regression of buffer Φ on normalised states",
        "phi_r2_both"     => "adjusted R² of Φ on [state, action] (linear)",
        "cos_critic_lin"  => "cos(ḡ, −β_act): critic action gradient vs the action slope of the linear Φ fit (both-fit)",
        "log10_phi_scale" => "log10 of the target scale φ_scale at the record (constant unless RENORM_EVERY > 0)",
        "n_per_param"     => "buffer size / (nx + nu + 1): samples per regression parameter",
    )
    return RLDiag(rec, copy(θn0), θn_true, nothing, 0.0, Float64(frac))
end

_tof32(x) = Float32(x)

# Adjusted R² of ordinary least squares (tiny ridge) of y on the columns of D (n×p, intercept included).
function _adj_r2(D::Matrix{Float64}, y::Vector{Float64})
    n, p = size(D)
    n > p + 1 || return NaN, zeros(p)
    β = (D' * D + (1e-6 * n) * I) \ (D' * y)
    res = y - D * β
    sst = sum(abs2, y .- mean(y))
    sst > 0 || return NaN, β
    r2 = 1 - sum(abs2, res) / sst
    return 1 - (1 - r2) * (n - 1) / (n - p), β
end

# Fraction of hidden units that are inactive for the whole batch.  ReLU: exactly zero output (as in earlier runs).
# Smooth activations never output exactly 0, so use the slope: a unit is dead if its activation derivative is < 0.01
# for every sample (softplus/elu/gelu/swish: pre-activation far negative; tanh: saturated).
function _dead_frac(layer, inp, h)
    if layer.σ === relu
        h === nothing && (h = layer(inp))
        return mean(all(h .== 0, dims = 2))
    end
    z = layer.weight * inp .+ layer.bias
    δ = 1.0f-3
    d = (layer.σ.(z .+ δ) .- layer.σ.(z .- δ)) ./ (2δ)
    return mean(all(abs.(d) .< 0.01f0, dims = 2))
end

"""
    rl_critic_probe(a, b, X, Fb, θ32, frac) -> Dict{String,Float64}

Heavy diagnostics at a recorded step.  `X`, `Fb` are the (normalised) minibatch the update just used, `θ32` the
current actor.  Uses critic 1 as the actor does.
"""
function rl_critic_probe(a, b, X, Fb, θ32, frac)
    c1 = a.critics[1]
    nx, B = size(X)
    nu = length(θ32)
    out = Dict{String, Float64}()

    A  = repeat(θ32, 1, B)
    ip = vcat(X, A)
    gin = Flux.gradient(z -> sum(c1(z)), ip)[1]            # per-sample input gradients (samples independent)
    gA  = gin[(nx + 1):end, :]
    gX  = gin[1:nx, :]
    ḡ   = vec(mean(gA, dims = 2))
    dev = gA .- ḡ
    dev_rms = sqrt(mean(sum(abs2, dev, dims = 1)))
    out["grad_snr"]     = norm(ḡ) / max(dev_rms / sqrt(B), eps(Float32))
    out["log10_gnorm"]  = log10(max(norm(ḡ), 1e-30))
    cs = [diag_cos(view(gA, :, j), ḡ) for j in 1:B]
    out["grad_consist"] = mean(filter(isfinite, cs))
    rms_a = sqrt(mean(abs2, gA)); rms_x = sqrt(mean(abs2, gX))
    out["sens_act_state"] = rms_a / max(rms_x, eps(Float32))

    out["q_std_full"]  = std(vec(c1(vcat(X, Fb))))
    out["q_std_state"] = std(vec(c1(ip)))

    inp_b = vcat(X, Fb)
    h1 = c1.layers[1](inp_b)
    out["dead_frac"] = max(_dead_frac(c1.layers[1], inp_b, h1), _dead_frac(c1.layers[2], h1, nothing))

    out["act_dist_rel"] = median([norm(θ32 .- view(Fb, :, j)) for j in 1:B]) / (frac * sqrt(nu))

    # linear reference fits on the whole buffer: is there action signal in the data at all?
    n   = b.n
    Xs  = Float64.(norm_x(a, b.x[:, 1:n]))
    Fs  = Float64.(norm_F(a, b.F[:, 1:n]))
    y   = b.Φ[1:n] ./ a.φ_scale
    one_ = ones(n)
    r2a, _  = _adj_r2(hcat(Fs', one_), y)
    r2s, _  = _adj_r2(hcat(Xs', one_), y)
    r2b, βb = _adj_r2(hcat(Xs', Fs', one_), y)
    out["phi_r2_act"] = r2a; out["phi_r2_state"] = r2s; out["phi_r2_both"] = r2b
    out["cos_critic_lin"] = diag_cos(ḡ, -βb[(nx + 1):(nx + nu)])
    out["n_per_param"]    = n / (nx + nu + 1)
    return out
end

"""
    rl_diag_step!(dg, agent, buf, upd, step, RMSE, cost)

Call once per RL step after the (optional) update; `cost` is the cumulative budget so far.  `upd` is the Dict `ddpg_update!` filled this step, or
`nothing` if no update ran (warm-up).  Accumulates the cheap per-step scalars; on scheduled steps records a row
and, if an update ran, adds the critic probe.
"""
function rl_diag_step!(dg::RLDiag, agent, buf, upd, step::Integer, RMSE::Real, cost::Real)
    rec = dg.rec
    rec.enabled || return nothing
    accumulate!(rec, "rmse", RMSE; mode = :min)
    updated = upd !== nothing && haskey(upd, :loss)
    if updated
        zv = upd[:zt_var]
        zv > 0 && accumulate!(rec, "critic_unexpl", upd[:loss] / zv)
        accumulate!(rec, "zt_std", upd[:zt_std])
        g = get(upd, :g, nothing)
        if g !== nothing
            accumulate!(rec, "g_norm", norm(g))
            dg.g_prev === nothing || accumulate!(rec, "cos_prev", diag_cos(g, dg.g_prev))
            dg.g_prev = g                      # fresh Float32 array made by the hook; not aliased with agent state
            Δ = norm(upd[:θ_post] - upd[:θ_pre])
            dg.θn_true === nothing || accumulate!(rec, "cos_truth", diag_cos(-g, dg.θn_true .- upd[:θ_pre]))
            accumulate!(rec, "step_pol", Δ)
            dg.path_len += Δ
        end
    end
    want(rec, step) || return nothing
    vals = Dict{Symbol, Float64}(:rmse => RMSE, :cost => cost)
    disp = norm(agent.θn - dg.θn0)
    vals[:policy_disp] = disp
    vals[:path_eff]    = dg.path_len > 0 ? disp / dg.path_len : NaN
    vals[:log10_phi_scale] = log10(agent.φ_scale)
    vals[:path_vs_rw]  = dg.path_len > 0 ? vals[:path_eff] * sqrt(max(agent.n_updates ÷ agent.policy_delay, 1)) : NaN
    dg.θn_true === nothing || (vals[:dist_truth] = norm(agent.θn - dg.θn_true))
    if updated
        X, Fb, θ32 = upd[:batch]
        for (k, v) in rl_critic_probe(agent, buf, X, Fb, θ32, dg.frac)
            vals[Symbol(k)] = v
        end
    end
    record!(rec, step; pairs(vals)...)
    return nothing
end
