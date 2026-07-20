# R-whitened PCA output-space coverage metric — shared math extracted from the
# (until now, independently duplicated) exp_to_leaderboard.jl coverage
# computation in uq_experiments/calibrate_emulate_sample/,
# uq_experiments/GaussNewtonKalmanInversion/, and uq_experiments/HistoryMatching/.
#
# Treating each raw output dimension as an independent trial for marginal
# coverage double-counts correlated components whenever the observation-noise
# covariance R has off-diagonal structure. Decorrelating via R's eigenbasis
# (R = VΛVᵀ), retaining the top modes needed to capture `retain_var` of R's
# variance, and computing coverage of the whitened truth against whitened
# pushforward samples avoids that.
#
# This file only extracts the shared MATH (whitening basis, marginal
# coverage, budget-to-target). Each experiment's exp_to_leaderboard.jl still
# owns its own netcdf schema, since the surrounding dims/semantics (e.g. what
# "k_iter" means) are method-specific.
#
# `WhitenedPCABasis`/`whitened_pca_basis`/`whiten_vector`/`whiten_samples` are
# generic (they just truncated-whiten against any covariance matrix), so
# uq_experiments/HistoryMatching/history_matching_core.jl also reuses them for
# GP input/output preprocessing (whitening against the prior and observation
# covariances before fitting), not only for the leaderboard metric.

using LinearAlgebra
using Statistics

struct WhitenedPCABasis
    V::Matrix{Float64}         # n_output x k_R, top eigenvectors of R
    λ::Vector{Float64}         # k_R, corresponding eigenvalues
    k_R::Int                   # number of retained modes
    cum_var::Vector{Float64}   # cumulative variance fraction, length n_output (pre-truncation)
end

# Eigendecompose R, sort descending, keep the smallest number of modes whose
# cumulative variance exceeds `retain_var`.
function whitened_pca_basis(R::AbstractMatrix, retain_var::Real)
    eig_R = eigen(Symmetric(Matrix(R)))
    ord = sortperm(eig_R.values; rev = true)
    λ_all = eig_R.values[ord]
    V_all = eig_R.vectors[:, ord]
    cum_var = cumsum(λ_all) ./ sum(λ_all)
    k_R = something(findfirst(>=(retain_var), cum_var), length(λ_all))
    return WhitenedPCABasis(V_all[:, 1:k_R], λ_all[1:k_R], k_R, cum_var)
end

# x: n_output vector -> k_R vector
whiten_vector(basis::WhitenedPCABasis, x::AbstractVector) = basis.V' * x ./ sqrt.(basis.λ)

# X: n_samples x n_output -> n_samples x k_R
whiten_samples(basis::WhitenedPCABasis, X::AbstractMatrix) = Matrix((basis.V' * Matrix(X')) ./ sqrt.(basis.λ))'

# whitened_samples: n_samples x k_R ; whitened_truth: k_R vector.
# Returns marginal coverage fraction at each quantile in `quantile_probs`.
function marginal_coverage(whitened_samples::AbstractMatrix, whitened_truth::AbstractVector, quantile_probs)
    k_R = length(whitened_truth)
    return [mean(whitened_truth[d] <= quantile(whitened_samples[:, d], q) for d in 1:k_R) for q in quantile_probs]
end

# coverage_by_k[k]: coverage vector (one entry per quantile in
# `quantile_probs`) at budget step k. n_y is the effective (whitened) output
# dimension, used for the quantile's sampling-noise tolerance. For each
# scaling c AND each quantile independently, returns the smallest N_ens*k / k
# at which |coverage_k(q) - q| <= c*sqrt(q(1-q)/n_y), or NaN if never reached
# — matching the per-quantile-independent semantics already established in
# calibrate_emulate_sample's and GaussNewtonKalmanInversion's
# exp_to_leaderboard.jl (each quantile gets its own budget/iters, not a single
# "all quantiles pass at once" threshold). Returns (budget, iters) as
# (n_scalings, n_quantiles) matrices.
function budget_to_target(coverage_by_k::AbstractVector, N_ens::Int, quantile_probs, scalings, n_y::Int)
    K = length(coverage_by_k)
    n_q = length(quantile_probs)
    budget = fill(NaN, length(scalings), n_q)
    iters = fill(NaN, length(scalings), n_q)
    for (si, c) in enumerate(scalings)
        tol = c .* sqrt.(quantile_probs .* (1 .- quantile_probs) ./ n_y)
        for qi in 1:n_q
            for k in 1:K
                s = coverage_by_k[k][qi]
                isnan(s) && continue
                if abs(s - quantile_probs[qi]) <= tol[qi]
                    budget[si, qi] = N_ens * k
                    iters[si, qi] = k
                    break
                end
            end
        end
    end
    return budget, iters
end
