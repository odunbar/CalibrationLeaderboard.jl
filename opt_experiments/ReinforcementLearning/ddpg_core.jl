# ReinforcementLearning — DDPG / TD3 core (model-agnostic; shared by the L63 and L96 drivers)
# DDPG vs TD3 is purely configuration (n_critics, policy_delay, target_noise); see experiment_config.jl.
#
# Nothing here touches the Lorenz forward maps, so it is safe to include from both
# run_l63_ddpg.jl and run_l96_ddpg.jl.
#
# Mapping from the write-up (README.md, "Reinforcement learning formulation"):
#   actor   π_θ(x) = F_θ        state-independent: the parameter vector IS the policy.
#                               Stored as θn in PRIOR-WHITENED units, F = prior_mean + L θn with
#                               L = chol(prior_cov).L (L = diag(prior_std) for a diagonal prior).
#   critic  Q_ϑ(x, F)           small MLP (or several, for the TD3 min-target trick).
#   reward  r_k = Φ_{k-1} − Φ_k with Φ_k the whitened misfit of the statistics window.
#
# Structural identity used for the critic.  The return telescopes to
#       Q(x_k,F_k) = Φ_{k-1} + q(x_k,F_k),        q := −E[Φ_∞-type tail | x_k,F_k]
# and Φ_{k-1} is a scalar already observed at x_k that F_k cannot influence.  Substituting
# into the Bellman target  z_k = r_k + γ Q(x_{k+1}, π(x_{k+1}))  gives, EXACTLY,
#       q-target_k = −(1−γ) Φ_k + γ q(x_{k+1}, π(x_{k+1})),
# because Φ_{k-1} cancels.  So we learn q (never storing Φ_{k-1}) and ∇_F Q = ∇_F q.
# This avoids asking the network to regress a history-dependent offset it cannot see.
# At γ = 1 the reward term vanishes identically and q has no F-dependence to learn —
# hence gamma < 1 in experiment_config.jl.

using Flux
using LinearAlgebra
using Optimisers
using Random
using Statistics

########################################################################
###############  Replay buffer  #######################################
########################################################################

mutable struct ReplayBuffer
    x::Matrix{Float64}      # nx × cap   state x_k
    F::Matrix{Float64}      # nu × cap   applied action F_k (raw units)
    Φ::Vector{Float64}      # cap        misfit Φ_k of the window run under F_k
    x′::Matrix{Float64}     # nx × cap   next state x_{k+1}
    cap::Int
    n::Int                  # number of valid entries
    head::Int               # next write position
end

ReplayBuffer(nx::Int, nu::Int, cap::Int) =
    ReplayBuffer(zeros(nx, cap), zeros(nu, cap), zeros(cap), zeros(nx, cap), cap, 0, 1)

function push_transition!(b::ReplayBuffer, x, F, Φ, x′)
    b.x[:, b.head]  = x
    b.F[:, b.head]  = F
    b.Φ[b.head]     = Φ
    b.x′[:, b.head] = x′
    b.head = b.head == b.cap ? 1 : b.head + 1
    b.n    = min(b.n + 1, b.cap)
    return b
end

########################################################################
###############  Agent  ###############################################
########################################################################

# Hidden activation of the critic (cfg.critic_act; :relu is the leaderboard setting, the others are experiments).
critic_activation(name::Symbol) =
    name === :relu     ? relu :
    name === :softplus ? softplus :
    name === :elu      ? elu :
    name === :gelu     ? gelu :
    name === :swish    ? swish :
    name === :tanh     ? tanh :
    throw(ArgumentError("Unknown critic activation: $name (relu, softplus, elu, gelu, swish, tanh)"))

function build_critic(nin::Int, hidden::Int, rng::AbstractRNG, act = relu)
    init = (dims...) -> Flux.glorot_uniform(rng, dims...)
    return Chain(
        Dense(nin   => hidden, act; init),
        Dense(hidden => hidden, act; init),
        Dense(hidden => 1; init),
    )
end

mutable struct DDPGAgent
    θn::Vector{Float64}                  # actor parameters, prior-std units
    actor_state::Any                     # Optimisers state for θn
    critics::Vector{Any}
    targets::Vector{Any}
    critic_states::Vector{Any}
    # fixed normalisation (set by init_normalization! at the end of warm-up)
    F_mean::Vector{Float64}
    F_L::LowerTriangular{Float64, Matrix{Float64}}   # Cholesky factor of the prior covariance
    x_mean::Vector{Float64}
    x_std::Vector{Float64}
    φ_scale::Float64
    # hyperparameters
    γ::Float64
    polyak::Float64
    policy_delay::Int
    target_noise::Float64
    target_noise_clip::Float64
    batch_size::Int
    n_updates::Int
    renorm_every::Int                    # 0 = scalings frozen at the first update (default); k > 0 = re-estimate every k updates
end

function DDPGAgent(cfg, θ0_raw::Vector{Float64}, prior_mean, prior_cov, nx::Int, rng::AbstractRNG)
    nu  = length(θ0_raw)
    L   = LowerTriangular(Matrix{Float64}(cholesky(Symmetric(Matrix{Float64}(prior_cov))).L))
    θn  = L \ (θ0_raw .- prior_mean)
    actor_state = Optimisers.setup(
        Optimisers.Adam(cfg.actor_lr, (cfg.adam_beta1, cfg.adam_beta2), cfg.adam_eps), θn)
    critics = Any[build_critic(nx + nu, cfg.hidden, rng, critic_activation(cfg.critic_act)) for _ in 1:cfg.n_critics]
    targets = Any[deepcopy(c) for c in critics]
    cstates = Any[Flux.setup(Flux.Adam(cfg.critic_lr), c) for c in critics]
    return DDPGAgent(
        θn, actor_state, critics, targets, cstates,
        Vector{Float64}(prior_mean), L,
        zeros(nx), ones(nx), 1.0,
        cfg.gamma, cfg.polyak, cfg.policy_delay, cfg.target_noise, cfg.target_noise_clip,
        cfg.batch_size, 0, cfg.renorm_every,
    )
end

# Current (noise-free) policy output in raw units.
policy_action(a::DDPGAgent) = a.F_mean .+ a.F_L * a.θn

# Set the (fixed) input/output scalings from the warm-up data.
function init_normalization!(a::DDPGAgent, b::ReplayBuffer)
    xs = b.x[:, 1:b.n]
    a.x_mean  = vec(mean(xs, dims = 2))
    a.x_std   = max.(vec(std(xs, dims = 2)), 1e-3)
    a.φ_scale = max(median(b.Φ[1:b.n]), 1e-8)
    return a
end

# Re-estimate the input/target scalings from the current buffer WITHOUT changing the function the critics compute:
# the first-layer state columns/bias and the output layer of every critic and target critic are rescaled so that
# the output in real units (φ_scale · net output) is unchanged.  Used only when cfg.renorm_every > 0.
#   inputs  z = (x − m)/s  →  z′ = (x − m′)/s′ :   W[:, state] ← W[:, state]·diag(s′/s),  b ← b + W_old[:, state]·((m′ − m)/s)
#   outputs q_n = q/φ      →  q_n′ = q/φ′      :   W_out ← W_out·(φ/φ′),  b_out ← b_out·(φ/φ′)
# No RNG, no forward model.
function renormalize!(a::DDPGAgent, b::ReplayBuffer)
    nx   = length(a.x_mean)
    xs   = b.x[:, 1:b.n]
    m′   = vec(mean(xs, dims = 2))
    s′   = max.(vec(std(xs, dims = 2)), 1e-3)
    φ′   = max(median(b.Φ[1:b.n]), 1e-8)
    shiftv = Float32.((m′ .- a.x_mean) ./ a.x_std)
    ratio  = Float32.(s′ ./ a.x_std)
    ρ      = Float32(a.φ_scale / φ′)
    for net in vcat(a.critics, a.targets)
        l1 = net.layers[1]; lo = net.layers[end]
        l1.bias .+= l1.weight[:, 1:nx] * shiftv          # uses the OLD weights
        l1.weight[:, 1:nx] .*= reshape(ratio, 1, :)
        lo.weight .*= ρ
        lo.bias   .*= ρ
    end
    a.x_mean, a.x_std, a.φ_scale = m′, s′, φ′
    return a
end

norm_x(a::DDPGAgent, X) = Float32.((X .- a.x_mean) ./ a.x_std)
norm_F(a::DDPGAgent, F) = Float32.(a.F_L \ (F .- a.F_mean))   # whitened action = actor coordinates

# Gaussian exploration noise on the applied action, shaped by the prior Cholesky factor:
# noise = frac · L · randn (frac in prior-std units along the whitened directions).
explore_action(a::DDPGAgent, rng::AbstractRNG, frac::Real) =
    policy_action(a) .+ frac .* (a.F_L * randn(rng, length(a.θn)))

function polyak!(tgt, src, τ::Real)
    pt, re = Flux.destructure(tgt)
    ps, _  = Flux.destructure(src)
    return re((1 - τ) .* pt .+ τ .* ps)
end

########################################################################
###############  One gradient update  #################################
########################################################################

# One critic update on a minibatch, then (every policy_delay calls) an actor
# update ascending the critic and a Polyak update of the target critics.
# Returns (critic_loss, actor_updated::Bool).
# `diag` (default nothing): a Dict the caller passes to read back update internals for diagnostics
# (diagnostics/rl_diagnostics.jl); it is only written to, never read, and does not change the update.
function ddpg_update!(a::DDPGAgent, b::ReplayBuffer, rng::AbstractRNG; diag::Union{Nothing, Dict{Symbol, Any}} = nothing)
    a.renorm_every > 0 && a.n_updates > 0 && a.n_updates % a.renorm_every == 0 && renormalize!(a, b)
    B   = a.batch_size
    idx = rand(rng, 1:b.n, B)
    X   = norm_x(a, b.x[:, idx])
    X′  = norm_x(a, b.x′[:, idx])
    Fb  = norm_F(a, b.F[:, idx])
    Φb  = b.Φ[idx]
    nu  = length(a.θn)
    θ32 = Float32.(a.θn)

    # --- TD target (see header: Φ_{k-1} has cancelled) ---
    A′ = repeat(θ32, 1, B)
    if a.target_noise > 0
        A′ = A′ .+ clamp.(Float32(a.target_noise) .* randn(rng, Float32, nu, B),
                          -Float32(a.target_noise_clip), Float32(a.target_noise_clip))
    end
    in′ = vcat(X′, A′)
    q′  = reduce((u, v) -> min.(u, v), [vec(t(in′)) for t in a.targets])
    zt  = Float32.(-(1 - a.γ) .* Φb ./ a.φ_scale) .+ Float32(a.γ) .* q′

    # --- critic regression ---
    inp  = vcat(X, Fb)
    loss = 0.0f0
    for i in eachindex(a.critics)
        l, g = Flux.withgradient(m -> mean(abs2, vec(m(inp)) .- zt), a.critics[i])
        Flux.update!(a.critic_states[i], a.critics[i], g[1])
        loss += l
    end
    a.n_updates += 1

    if diag !== nothing
        empty!(diag)
        diag[:loss]    = Float64(loss) / length(a.critics)
        diag[:zt_var]  = Float64(var(zt))
        diag[:zt_std]  = Float64(std(zt))
        diag[:batch]   = (X, Fb, θ32)        # references to the existing minibatch arrays (not copied)
    end

    actor_updated = false
    if a.n_updates % a.policy_delay == 0
        # --- actor: ascend q_1(x, θ) averaged over buffer states ---
        c1 = a.critics[1]
        g  = Flux.gradient(θ -> -mean(vec(c1(vcat(X, repeat(θ, 1, B))))), θ32)[1]
        θ_pre = diag === nothing ? nothing : copy(a.θn)
        if g !== nothing && all(isfinite, g)
            a.actor_state, a.θn = Optimisers.update!(a.actor_state, a.θn, Float64.(g))
        end
        if diag !== nothing
            diag[:g]      = g === nothing ? nothing : Float32.(g)
            diag[:θ_pre]  = θ_pre
            diag[:θ_post] = copy(a.θn)
        end
        # --- target critics ---
        for i in eachindex(a.targets)
            a.targets[i] = polyak!(a.targets[i], a.critics[i], a.polyak)
        end
        actor_updated = true
    end
    return (Float64(loss) / length(a.critics), actor_updated)
end
