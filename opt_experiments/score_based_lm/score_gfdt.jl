# score_based_lm — model-agnostic score + GFDT machinery
#
# Nothing in this file dispatches on LorenzConfig / EnsembleMemberConfig, so it
# is safe to include from both run_l63_sblm.jl and run_l96_sblm.jl (Lorenz63.jl
# and Lorenz96.jl define clashing types and can never be loaded together).
#
# Contents:
#   ScoreModel interface       — GaussianScore (quasi-Gaussian FDT) and NeuralScore (DSM)
#   train_score! / score       — denoising score matching at a single fixed sigma
#   stein_recalibrate          — enforce -E[x s(x)'] = I on the calibration ensemble
#   response_integral          — tapered one-sided correlation integral

using Flux
using LinearAlgebra
using Optimisers
using Random
using Statistics

########################################################################
###############  Degenerate-attractor detection  ######################
########################################################################

"""
    is_collapsed_window(X; tol = 1e-2)

True when the trajectory window has collapsed onto a fixed point, so that the
invariant measure is a point mass.

GFDT is *undefined* there: `grad log p` does not exist, and in practice the
ridge-regularised score blows up and the estimator returns a large meaningless
Jacobian rather than zero.  For L63 roughly 30% of draws from the experiment's
prior land here (large beta kills the chaos), and ForwardDiff is perfectly
well-behaved on exactly those parameters -- no chaos means no tangent-linear
blow-up -- which is what motivates the AD fallback in the run scripts.

The separation is not delicate: measured `min(var)` is >= 5 on chaotic draws and
<= 6e-4 on collapsed ones, four orders of magnitude apart.

Caveat: this catches fixed points, not limit cycles (a periodic orbit has
non-zero variance but is still a singular measure in R^3).  The Stein residual
`stein_norm` returned by `gfdt_jacobian` is the backstop for those.
"""
is_collapsed_window(X::AbstractMatrix; tol::Real = 1e-2) =
    minimum(var(X, dims = 2)) < tol

########################################################################
###############  ScoreModel interface  ################################
########################################################################

abstract type ScoreModel end

# score(m, X) -> nx x n matrix, in PHYSICAL coordinates.
# X is nx x n (columns are states).

########################################################################
###############  Quasi-Gaussian score  ################################
########################################################################

# s(x) = -C^{-1} (x - mu).  Exact for a Gaussian invariant measure; used both as
# a published baseline (Leith / quasi-Gaussian FDT) and as a debugging aid that
# isolates GFDT-formula error from score-network error.
struct GaussianScore{VV <: AbstractVector, MM <: AbstractMatrix} <: ScoreModel
    mu::VV
    Cinv::MM
end

# `ridge` shrinks towards a scaled identity; needed for nx=100 with ~5000 samples.
function GaussianScore(X::AbstractMatrix; ridge::Real = 1e-2)
    mu = vec(mean(X, dims = 2))
    C  = cov(X, dims = 2)
    n  = size(C, 1)
    Creg = Matrix(C) + ridge * (tr(C) / n) * I
    return GaussianScore(mu, inv(Symmetric(Creg)))
end

score(m::GaussianScore, X::AbstractMatrix) = -m.Cinv * (X .- m.mu)

########################################################################
###############  Neural score (denoising score matching)  #############
########################################################################

# Trained by DSM at a SINGLE fixed noise level sigma (not an NCSN ladder):
#
#   z~ = z + sigma * eps,   eps ~ N(0, I)
#   L  = E || net(z~) - eps ||^2
#   s_z(z) = -net(z) / sigma
#
# The NCSN sigma-ladder exists so that annealed Langevin *sampling* mixes; we
# never sample, we only need p_sigma ~= p at one small sigma.  sigma is the
# mollification scale that makes grad log p exist at all for a singular SRB
# measure, so it is a physical regulariser, not a nuisance parameter -- sweep it.
mutable struct NeuralScore{NT, VV} <: ScoreModel
    net::NT
    # Deliberately Any: starts as `nothing` and is later assigned the Optimisers
    # state tree, whose type is not known at construction.
    opt_state::Any
    mu::VV           # per-component standardisation, refit every LM iteration
    sd::VV
    sigma::Float32
    trained::Bool
end

function NeuralScore(net; sigma::Real, nx::Int)
    return NeuralScore(net, nothing, zeros(nx), ones(nx), Float32(sigma), false)
end

# z = (x - mu) ./ sd  =>  p_x(x) = p_z(z) * prod(1/sd)  =>  s_x(x) = s_z(z) ./ sd
# Getting this rescaling wrong silently scales the entire Jacobian.
function standardise!(m::NeuralScore, X::AbstractMatrix)
    m.mu = vec(mean(X, dims = 2))
    m.sd = max.(vec(std(X, dims = 2)), 1e-8)
    return (X .- m.mu) ./ m.sd
end

normalise(m::NeuralScore, X::AbstractMatrix) = (X .- m.mu) ./ m.sd

function score(m::NeuralScore, X::AbstractMatrix)
    m.trained || error("NeuralScore used before training")
    Z  = Float32.(normalise(m, X))
    sz = -m.net(Z) ./ m.sigma        # score in normalised coordinates
    return Float64.(sz) ./ m.sd      # back to physical coordinates
end

"""
    train_score!(m::NeuralScore, X; epochs, lr, batch, rng, val_frac = 0.1)

Denoising score matching on the columns of `X` (nx x n, physical coordinates).
Refits the standardisation, then runs `epochs` passes of Adam.

Returns `(; train_loss, val_loss)` from the final epoch.  A val_loss that
diverges from train_loss means the net is overfitting the (very small) sample --
shrink the net rather than training longer.
"""
function train_score!(
    m::NeuralScore,
    X::AbstractMatrix;
    epochs::Int,
    lr::Real,
    batch::Int,
    rng::AbstractRNG,
    weight_decay::Real = 0.0,
    val_frac::Real = 0.1,
)
    Z = Float32.(standardise!(m, X))
    n = size(Z, 2)

    perm  = randperm(rng, n)
    n_val = max(1, round(Int, val_frac * n))
    Zval  = Z[:, perm[1:n_val]]
    Ztr   = Z[:, perm[(n_val + 1):end]]
    n_tr  = size(Ztr, 2)

    rule = weight_decay > 0 ?
        Optimisers.OptimiserChain(
            Optimisers.WeightDecay(Float32(weight_decay)),
            Optimisers.Adam(Float32(lr)),
        ) :
        Optimisers.Adam(Float32(lr))

    # Warm start: keep the existing optimiser state across LM iterations when the
    # net has already been trained, so the score is refined rather than relearned.
    if m.opt_state === nothing || !m.trained
        m.opt_state = Flux.setup(rule, m.net)
    else
        Optimisers.adjust!(m.opt_state, Float32(lr))
    end

    sigma = m.sigma
    bs    = min(batch, n_tr)
    train_loss = NaN

    for _ in 1:epochs
        order = randperm(rng, n_tr)
        tot, nb = 0.0, 0
        for i in 1:bs:n_tr
            idx = order[i:min(i + bs - 1, n_tr)]
            zb  = Ztr[:, idx]
            eps = randn(rng, Float32, size(zb))
            zt  = zb .+ sigma .* eps
            l, gs = Flux.withgradient(net -> mean(abs2, net(zt) .- eps), m.net)
            Flux.update!(m.opt_state, m.net, gs[1])
            tot += l; nb += 1
        end
        train_loss = tot / max(nb, 1)
    end

    eps_v = randn(rng, Float32, size(Zval))
    val_loss = mean(abs2, m.net(Zval .+ sigma .* eps_v) .- eps_v)

    m.trained = true
    return (; train_loss, val_loss = Float64(val_loss))
end

########################################################################
###############  KGMM score (k-means Gaussian-mixture estimator)  #####
########################################################################
#
# arXiv:2503.18054.  Alternative to single-sigma DSM for the sample-starved,
# low-to-moderate-dimension regime where DSM's mollification scale sigma has no
# good setting (VALIDATION.md: at n=1001 in d=3, tuned DSM still has ~15% score
# error vs 0.17% for fitting a Gaussian).  KGMM fits a K-component Gaussian
# mixture by k-means: each cluster supplies its own LOCAL mean/covariance, so the
# "bandwidth" is set by the local data spread rather than a single global sigma.
# No training loop -- fit cost is one k-means run plus K covariance estimates,
# both closed-form, so this is expected to be far cheaper than DSM per iteration.
#
# The exact score of a Gaussian mixture p(x) = sum_k w_k N(x; mu_k, Sigma_k) is
# the responsibility-weighted average of the per-component scores:
#   grad log p(x) = sum_k gamma_k(x) * grad log N_k(x),   gamma_k(x) = posterior
# so once the mixture is fit, the score follows exactly (no further approximation).

# Vectorised Lloyd's algorithm.  X is nx x n; returns (centroids :: nx x K,
# assign :: n-vector of cluster labels in 1:K).  Squared distances are formed via
# one BLAS matmul (||x-c||^2 = ||x||^2 - 2 x'c + ||c||^2) rather than a triple
# loop, so this comfortably handles the largest window here (nx=100, n=25000).
function kmeans_fit(X::AbstractMatrix, K::Int, rng::AbstractRNG; max_iter::Int = 50)
    nx, n = size(X)
    K = clamp(K, 1, n)
    C = Matrix{Float64}(X[:, randperm(rng, n)[1:K]])
    assign = zeros(Int, n)
    xn = vec(sum(abs2, X, dims = 1))                # n
    for _ in 1:max_iter
        cn = vec(sum(abs2, C, dims = 1))            # K
        D2 = cn .+ xn' .- 2 .* (C' * X)              # K x n
        new_assign = [argmin(@view(D2[:, j])) for j in 1:n]
        new_assign == assign && break
        assign = new_assign
        for k in 1:K
            members = findall(==(k), assign)
            isempty(members) || (C[:, k] = vec(mean(view(X, :, members), dims = 2)))
        end
    end
    return C, assign
end

struct KGMMScore{VM1 <: AbstractVector, VM2 <: AbstractVector, VV <: AbstractVector} <: ScoreModel
    mus::VM1               # K-vector of nx-vectors
    Cinvs::VM2             # K-vector of nx x nx precision matrices
    halflogdet::VV         # K-vector, 0.5*logdet(Cinv_k), precomputed
    logw::VV               # K-vector, log(n_k / n)
end

# `K` defaults to ~one cluster per (shrink_factor * nx) points, clamped to a
# sane range -- few enough clusters that each has more members than nx (so the
# raw per-cluster covariance is itself well-posed before shrinkage even helps),
# many enough that the mixture can resolve local structure.
function kgmm_default_k(n::Int, nx::Int; per_cluster::Int = 5)
    return clamp(round(Int, n / (per_cluster * nx)), 3, 60)
end

function KGMMScore(X::AbstractMatrix, rng::AbstractRNG; K::Int = kgmm_default_k(size(X, 2), size(X, 1)),
                   ridge::Real = 0.1)
    nx, n = size(X)
    Sigma_global = Matrix(cov(X, dims = 2)) + ridge * (tr(cov(X, dims = 2)) / nx) * I
    C, assign = kmeans_fit(X, K, rng)
    Kused = size(C, 2)

    mus    = Vector{Vector{Float64}}(undef, Kused)
    Cinvs  = Vector{Matrix{Float64}}(undef, Kused)
    hld    = zeros(Kused)
    logw   = zeros(Kused)

    for k in 1:Kused
        members = findall(==(k), assign)
        nk = length(members)
        logw[k] = log(max(nk, 1) / n)
        if nk <= nx
            # Too few points in this cluster to trust a local covariance at all;
            # fall back to the global one, centred on the (possibly empty) centroid.
            mus[k]  = nk > 0 ? vec(mean(view(X, :, members), dims = 2)) : vec(C[:, k])
            Sigma_k = Sigma_global
        else
            Xk      = view(X, :, members)
            mus[k]  = vec(mean(Xk, dims = 2))
            Sigma_k = Matrix(cov(Xk, dims = 2))
            # Shrink toward the global covariance; smaller clusters shrink more.
            alpha   = clamp(nx / nk, 0.0, 1.0)
            Sigma_k = (1 - alpha) .* Sigma_k .+ alpha .* Sigma_global
            Sigma_k = Sigma_k + ridge * (tr(Sigma_k) / nx) * I
        end
        Cinv_k     = inv(Symmetric(Sigma_k))
        Cinvs[k]   = Cinv_k
        hld[k]     = 0.5 * logdet(Symmetric(Cinv_k))
    end
    return KGMMScore(mus, Cinvs, hld, logw)
end

function score(m::KGMMScore, X::AbstractMatrix)
    nx, n = size(X)
    K = length(m.logw)
    S = zeros(nx, n)
    logp = zeros(K)
    for j in 1:n
        x = @view X[:, j]
        for k in 1:K
            d = x .- m.mus[k]
            logp[k] = m.logw[k] + m.halflogdet[k] - 0.5 * dot(d, m.Cinvs[k] * d)
        end
        w = exp.(logp .- maximum(logp))
        w ./= sum(w)
        sx = zeros(nx)
        for k in 1:K
            d = x .- m.mus[k]
            sx .+= w[k] .* (-(m.Cinvs[k] * d))
        end
        S[:, j] = sx
    end
    return S
end

########################################################################
###############  Stein recalibration  #################################
########################################################################

# The exact score satisfies -E[x s(x)'] = I.  Finite samples and network
# approximation introduce a defect: write -E[x s(x)'] = I - Xi and replace
# s by s (I - Xi)^{-1}.  (arXiv:2509.19660 App. B.3.)  Also returns ||Xi|| so
# callers can report how far off the raw score was.
#
# GATED: the recalibration is only valid as a SMALL correction.  When ||Xi|| is
# large the moment matrix M is far from I (a score network that has learned ~0
# gives M ~ 0 and ||Xi|| -> sqrt(nx)), and `M' \ S` then amplifies garbage rather
# than correcting it.  Above `max_defect` we leave the score alone and let the
# caller see the large ||Xi|| as the diagnostic it is.
function stein_recalibrate(S::AbstractMatrix, X::AbstractMatrix; max_defect::Real = 0.5)
    n  = size(X, 2)
    Xc = X .- mean(X, dims = 2)
    Sc = S .- mean(S, dims = 2)
    M  = -(Xc * Sc') / n              # nx x nx, equals I for an exact score
    nrm = norm(I - M)                 # ||Xi||, reported as a diagnostic
    nrm > max_defect && return S, nrm
    # Seeking S_new = W*S with -E[x S_new'] = I.  Since -E[x (W S)'] = M W',
    # we need W' = M^{-1}, i.e. W = M^{-T} and S_new = M' \ S.
    # (Equivalently the paper's row-vector form s (I - Xi)^{-1}.)
    S_new = try
        Matrix(M)' \ S
    catch
        @warn "Stein recalibration matrix singular; using raw score"
        S
    end
    return S_new, nrm
end

########################################################################
###############  Correlation integral  ################################
########################################################################

# Tukey window: flat over [0, a*L], cosine roll-off over [a*L, L].
# a = 0.5 measured best across tau_max on an OU process with a closed form:
# no taper degrades as tau_max grows (accumulated large-lag noise); a full Hann
# biases low at small tau_max (it shrinks the integrand multiplicatively).
function tukey(l::Real, L::Real, a::Real)
    a >= 1 && return 1.0
    l <= a * L && return 1.0
    return 0.5 * (1 + cos(pi * (l - a * L) / ((1 - a) * L)))
end

"""
    response_integral_unnormalised(dA, dB; dt, lag_stride, tau_max, tukey_alpha)

Returns `(num, n_eff)` with `num = sum_t A~[:,t] dB[:,t]'` where `A~` is the
tapered, trapezoid-weighted lag filter applied to `dA`.  Divide `num` by the
accumulated `n_eff` to get `dm/dtheta` (n_A x nu).

`dA` (n_A x nt) and `dB` (nu x nt) must already be centred.

Implementation note: only the *integral* is needed, so `dA` is filtered once and
a single gemm produces the whole n_A x nu block -- never materialise an
n_A x nu x n_lag array.  Worst case here (l96_flux: 200 x 61 x 81) is ~20 ms.
"""
function response_integral_unnormalised(
    dA::AbstractMatrix,
    dB::AbstractMatrix;
    dt::Real,
    lag_stride::Int,
    tau_max::Real,
    tukey_alpha::Real = 0.5,
)
    nt = size(dA, 2)
    @assert size(dB, 2) == nt "dA and dB must have the same number of columns"
    L = Int(round(tau_max / dt))
    L = min(L, nt - 2)                      # never ask for more lag than we have
    lags  = 0:lag_stride:L
    n_eff = nt - L
    n_eff <= 0 && error("tau_max too large for the available window")
    dtl = lag_stride * dt

    Atil = zeros(eltype(dA), size(dA, 1), n_eff)
    nl = length(lags)
    for (i, l) in enumerate(lags)
        trap = (i == 1 || i == nl) ? 0.5 : 1.0
        w = dtl * trap * tukey(l, L, tukey_alpha)
        @views Atil .+= w .* dA[:, (1 + l):(l + n_eff)]
    end
    @views num = Atil * transpose(dB[:, 1:n_eff])
    return num, n_eff
end

function response_integral(dA, dB; kwargs...)
    num, n_eff = response_integral_unnormalised(dA, dB; kwargs...)
    return num / n_eff
end

########################################################################
###############  Observable / conjugate accumulation  #################
########################################################################

# Accumulator so that :fair mode (N_ens independent base-length windows) and
# :ensemble mode (one window N_ens times longer) share one code path.  Each
# member is centred on its OWN finite-time mean before being folded in.
mutable struct ResponseAccumulator{MM <: AbstractMatrix}
    num::MM
    den::Int
end
ResponseAccumulator(n_A::Int, nu::Int) = ResponseAccumulator(zeros(n_A, nu), 0)

function accumulate_response!(
    acc::ResponseAccumulator,
    A::AbstractMatrix,
    B::AbstractMatrix;
    kwargs...,
)
    dA = A .- mean(A, dims = 2)
    dB = B .- mean(B, dims = 2)   # E[B]=0 exactly; centring is a free control variate
    num, n_eff = response_integral_unnormalised(dA, dB; kwargs...)
    acc.num .+= num
    acc.den += n_eff
    return acc
end

finish_response(acc::ResponseAccumulator) = acc.num / acc.den

########################################################################
###############  GFDT Jacobian driver  ################################
########################################################################

"""
    gfdt_jacobian(Xs, score_model, moment_fn, conj_fn, dphi_fn; dt, lag_stride,
                  tau_max, tukey_alpha, stein)

Assemble `J = (dphi/dm) * (dm/dtheta)` from one or more attractor windows.

* `Xs`        — vector of nx x M state matrices.  Length `N_ens` in `:fair` mode
                (independent base-length windows); length 1 in `:ensemble` mode
                (a single window `N_ens` times longer).
* `moment_fn(X)    -> n_A x M`   raw-moment observables whose time averages determine G
* `conj_fn(X, S)   -> nu x M`    conjugate variables B_j
* `dphi_fn(m)      -> ny x n_A`  Jacobian of G = phi(m) w.r.t. the raw moments

Returns `(; J, m, dmdtheta, stein_norm, b_mean_ratio)` where `b_mean_ratio` is
`|mean(B_j)| / std(B_j)` per parameter -- the Stein identity applied to the most
relevant test function, and the sharpest cheap check that the score is sane.
"""
function gfdt_jacobian(
    Xs::AbstractVector,
    score_model::ScoreModel,
    moment_fn,
    conj_fn,
    dphi_fn;
    dt::Real,
    lag_stride::Int,
    tau_max::Real,
    tukey_alpha::Real = 0.5,
    stein::Bool = true,
)
    As  = [moment_fn(X) for X in Xs]
    n_A = size(As[1], 1)

    # Pooled raw moments, from exactly the same columns the response uses.
    tot, ntot = zeros(n_A), 0
    for A in As
        tot .+= vec(sum(A, dims = 2))
        ntot += size(A, 2)
    end
    m = tot ./ ntot

    acc = nothing
    stein_norm = NaN
    b_sum = nothing
    b_sq = nothing
    for (A, X) in zip(As, Xs)
        S = score(score_model, X)
        if stein
            S, stein_norm = stein_recalibrate(S, X)
        end
        B = conj_fn(X, S)
        if acc === nothing
            acc   = ResponseAccumulator(n_A, size(B, 1))
            b_sum = zeros(size(B, 1))
            b_sq  = zeros(size(B, 1))
        end
        b_sum .+= vec(sum(B, dims = 2))
        b_sq  .+= vec(sum(abs2, B, dims = 2))
        accumulate_response!(acc, A, B; dt, lag_stride, tau_max, tukey_alpha)
    end

    b_mean = b_sum ./ ntot
    b_std  = sqrt.(max.(b_sq ./ ntot .- b_mean .^ 2, 0.0))
    b_mean_ratio = abs.(b_mean) ./ max.(b_std, 1e-300)

    dmdtheta = finish_response(acc)
    J = dphi_fn(m) * dmdtheta
    return (; J, m, dmdtheta, stein_norm, b_mean_ratio)
end
