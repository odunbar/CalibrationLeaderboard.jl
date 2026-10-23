# offline_tmcmc.jl — faster ST-MCMC for the offline tests (test2). Included by offline_common.jl after boed_core.jl.
#
# TransitionalMCMC.tmcmc evaluates the likelihood one point at a time through `pmap` (serial: there are no Distributed workers), and its Metropolis
# step re-evaluates the target at the current state. For the high-dimensional cases (l96_vec/flux: 71 GPs with ~600 training points) that is ~17 ms per
# call and hours per cell. Three implementations of the SAME algorithm (same tempering bisection, weights, resampling, shrinkage covariance and
# random-walk Metropolis moves), selected with DIAG_TMCMC:
#   package    TransitionalMCMC.tmcmc via boed_core's run_tmcmc (reference; serial)
#   threaded   a copy of tmcmc with `pmap` replaced by a threaded map over particles (same calls, run on Threads.nthreads() threads)
#   batched    (default) all particles advance in lock-step: one batched GP prediction (threaded over output modes) per Metropolis step, and the
#              current state's log-likelihood is carried instead of recomputed
# Notes on `batched`: the package's chain per particle has length thin*1 + burnin and keeps the state at index burnin+1, i.e. after `burnin` Metropolis steps,
# so `thin` has no effect on the sample and only `burnin` steps are taken here. It draws from the explicit `rng` (reproducible per seed), unlike the package.
# validate_offline_tmcmc.jl checks the three against each other on a frozen GP (log-evidence, energy distance vs the package's own MC floor).
# These are used by the OFFLINE tests only; the production run (boed_core.run_tmcmc) is unchanged.

const _TM = TransitionalMCMC
const _CE = TransitionalMCMC.CovarianceEstimation
const _SB = TransitionalMCMC.StatsBase
const TMCMC_VERBOSE = get(ENV, "DIAG_TMCMC_VERBOSE", "0") == "1"   # per-stage progress for the batched sampler
const TMCMC_IMPL = Symbol(get(ENV, "DIAG_TMCMC", "batched"))
TMCMC_IMPL in (:package, :threaded, :batched) || error("DIAG_TMCMC must be package, threaded or batched")

_tmcmc_cov(θ::AbstractMatrix) = cov(_CE.LinearShrinkage(_CE.DiagonalUnitVariance(), :lw), θ)   # θ: n x k, as in the package

# Log-likelihood of every column of Zb (k_in x M) under `gps`, the form of boed_loglik (with_var=false drops the GP predictive variance: the 'novar' sampler).
# Threaded over output modes; each mode makes ONE predict_y call for all M points.
function boed_loglik_cols(prob::BOEDProblem, gps::BOEDGPs, Zb::AbstractMatrix; with_var::Bool = true)
    M, nm = size(Zb, 2), length(gps.gps)
    parts = zeros(M, nm)
    Threads.@threads :dynamic for j in 1:nm
        mu, v = GaussianProcesses.predict_y(gps.gps[j], Zb)
        yj = prob.y_whitened[j]
        for i in 1:M
            vi = with_var ? 1.0 + v[i] : 1.0
            parts[i, j] = -0.5 * (mu[i] - yj)^2 / vi - 0.5 * log(2π * vi)
        end
    end
    return vec(sum(parts; dims = 2))
end

# Same ordered result as `map(f, rows)`, over threads (the package's pmap is serial without workers).
function _tmap(f, rows)
    out = Vector{Any}(undef, length(rows))
    Threads.@threads :dynamic for i in eachindex(rows)
        out[i] = f(rows[i])
    end
    return out
end

# TransitionalMCMC.tmcmc (v. of the Manifest) with `pmap` -> `_tmap`; returns (θ n x k, log evidence).
function tmcmc_threaded(log_fD_T::Function, log_fT::Function, sample_fT::Function, Nsamples::Integer, burnin::Integer, thin::Integer, beta2::Float64 = 0.01)
    βj = 0.0
    θ_j = sample_fT(Nsamples)
    Log_ev = 0.0
    while βj < 1
        Lp_j = Float64.(reduce(vcat, _tmap(log_fD_T, collect(eachrow(θ_j)))))
        Lp_adjust = maximum(Lp_j)
        βj1, w_j = _TM._beta_and_weights(βj, Lp_j .- Lp_adjust)
        Log_ev = log(mean(w_j)) + (βj1 - βj) * Lp_adjust + Log_ev
        prop = mu -> _TM.proprnd(mu, Σ_j, log_fT)
        target = x -> log_fD_T(x) .* βj1 .+ log_fT(x)
        wn_j = w_j ./ sum(w_j)
        indices = _SB.sample(1:Nsamples, _SB.Weights(wn_j), Nsamples; replace = true)
        θ_j1 = θ_j[indices, :]
        Σ_j = beta2 * _tmcmc_cov(θ_j1)
        θ_j1 = _tmap(x -> _TM.metropolis_hastings_simple(target, prop, x, 1, burnin, thin)[1], collect(eachrow(θ_j1)))
        θ_j = reduce(vcat, θ_j1)
        βj = βj1
    end
    return θ_j, Log_ev
end

# Lock-step ST-MCMC. ll_cols(X) -> M log-likelihoods for the columns of the k x M matrix X; logprior_cols likewise. `sample_fT(n)` returns n x k.
# returns (θ n x k, log evidence, number of tempering stages, mean Metropolis acceptance rate)
function tmcmc_batched(ll_cols::Function, logprior_cols::Function, sample_fT::Function, Nsamples::Integer, burnin::Integer, rng::AbstractRNG; beta2::Float64 = 0.01)
    θ = sample_fT(Nsamples)                  # n x k
    k = size(θ, 2)
    X = Matrix(θ')                           # k x n
    Lp = ll_cols(X)
    βj, Log_ev = 0.0, 0.0
    stages, t0, n_acc, n_prop = 0, time(), 0, 0
    while βj < 1
        stages += 1
        Lp_adjust = maximum(Lp)
        βj1, w_j = _TM._beta_and_weights(βj, Lp .- Lp_adjust)
        Log_ev += log(mean(w_j)) + (βj1 - βj) * Lp_adjust
        idx = _SB.sample(rng, 1:Nsamples, _SB.Weights(w_j ./ sum(w_j)), Nsamples; replace = true)
        Xc = X[:, idx]
        Lc = Lp[idx]
        Lchol = cholesky(Symmetric(beta2 * _tmcmc_cov(Matrix(Xc')))).L
        Tc = βj1 .* Lc .+ logprior_cols(Xc)
        for _ in 1:burnin
            P = Xc .+ Lchol * randn(rng, k, Nsamples)
            Lq = ll_cols(P)
            Tq = βj1 .* Lq .+ logprior_cols(P)
            acc = min.(0.0, Tq .- Tc) .>= log.(rand(rng, Nsamples))
            Xc[:, acc] = P[:, acc]
            Tc[acc] = Tq[acc]
            n_acc += count(acc); n_prop += Nsamples
            Lc[acc] = Lq[acc]
        end
        X, Lp, βj = Xc, Lc, βj1
        TMCMC_VERBOSE && @info "tmcmc_batched: stage $stages β=$(round(βj; sigdigits = 4)) elapsed_s=$(round(time() - t0; digits = 1))"
    end
    return Matrix(X'), Log_ev, stages, n_acc / max(n_prop, 1)
end

"""
    offline_tmcmc(prob, gps, n, rng; burnin, thin, impl = TMCMC_IMPL, with_var = true, diag = nothing) -> k_in x n

Posterior sample from the GP surrogate by ST-MCMC with the chosen implementation. `with_var = false` drops the GP predictive variance from the likelihood
(test2's `tmcmc_novar`). Fills `diag["tmcmc_log_ev"]`.
"""
function offline_tmcmc(prob::BOEDProblem, gps::BOEDGPs, n::Int, rng::AbstractRNG; burnin::Int, thin::Int, beta2::Float64 = 0.01, impl::Symbol = TMCMC_IMPL, with_var::Bool = true,
                       diag::Union{Nothing, AbstractDict} = nothing)
    k = prob.prior_basis.k_R
    logprior(z) = sum(logpdf.(Normal(), z))
    priorRnd(m) = Matrix(lhs_standard_normal_sample(k, m, rng)')
    if impl === :package || k == 1   # k == 1 takes the package's scalar code path
        if with_var && beta2 == 0.01
            return run_tmcmc(prob, gps, n, rng; burnin = burnin, thin = thin, diag = diag)
        end
        ll = z -> (mu = predict_boed(gps, z isa Real ? [z] : z)[1]; sum(-0.5 .* (mu .- prob.y_whitened) .^ 2 .- 0.5 * log(2π)))
        samps, lev = TransitionalMCMC.tmcmc(ll, logprior, priorRnd, n, burnin, thin, beta2)
    elseif impl === :threaded
        ll = z -> with_var ? boed_loglik(prob, gps, z) : (mu = predict_boed(gps, z)[1]; sum(-0.5 .* (mu .- prob.y_whitened) .^ 2 .- 0.5 * log(2π)))
        samps, lev = tmcmc_threaded(ll, logprior, priorRnd, n, burnin, thin, beta2)
    else
        lpc(X) = -0.5 .* vec(sum(abs2, X; dims = 1)) .- 0.5 * size(X, 1) * log(2π)
        samps, lev, stages, acc = tmcmc_batched(X -> boed_loglik_cols(prob, gps, X; with_var = with_var), lpc, priorRnd, n, burnin, MersenneTwister(rand(rng, UInt)); beta2 = beta2)
    end
    diag === nothing || (diag["tmcmc_log_ev"] = Float64(lev))
    diag === nothing || impl !== :batched || k == 1 || (diag["tmcmc_stages"] = stages; diag["tmcmc_accept"] = acc)
    return Matrix(reshape(samps, n, k)')
end
