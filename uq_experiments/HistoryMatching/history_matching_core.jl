# HistoryMatching — shared wave / GP / rejection-sampling core.
#
# Included (via an @__DIR__-relative path) by calibrate_l63.jl and
# calibrate_l96.jl. Implements the Bayesian History Matching algorithm
# translated from
# Dropbox/Caltech/RobRebecca/Calibration-Race-Bayesian-methods/calib_race_hm_l63.py:
# fit one independent GP per output statistic on a wave's ensemble, then use
# the chi-squared implausibility of ALL waves fitted so far to define the
# Not-Ruled-Out-Yet (NROY) region that the next wave (or the final "posterior")
# is rejection-sampled from.
#
# GP fitting uses raw GaussianProcesses.jl (GPE, ARD squared-exponential
# kernel) — this method has no dependency on CalibrateEmulateSample.jl.
#
# Both the GP's input and output spaces are truncated-whitened before
# fitting, reusing common/uq_metrics/coverage_metrics.jl's generic
# WhitenedPCABasis machinery (the same math already used there for the
# leaderboard coverage metric):
#   - Output space is always whitened against the OBSERVATION covariance R —
#     fixed for the whole cell, known from preliminaries before any wave runs.
#     This also simplifies implausibility to a plain weighted sum (no matrix
#     solve), since after whitening R's contribution is exactly the identity.
#   - Input space is always whitened against the PRIOR covariance — likewise
#     fixed for the whole cell and known before any data exists. For
#     l96_vec's correlated 40-D prior this alone captures most of its
#     effective (much lower) dimensionality.
#   - l96_flux ADDITIONALLY applies a wave-LOCAL PCA on top of the
#     prior-whitened coordinates, fit fresh from that wave's own ensemble —
#     this is the one thing that cannot be known before seeing forward
#     evaluations (the NN-weight symmetry collapse is a property of the
#     forward map, not of the prior), so it must stay per-wave, using only
#     that wave's own data (never a later wave's).

using GaussianProcesses
using LinearAlgebra
using Statistics
using Random
using Distributions
using PDMats

include(joinpath(@__DIR__, "..", "..", "common", "uq_metrics", "coverage_metrics.jl"))

# GaussianProcesses.jl 0.12's `LinearAlgebra.ldiv!(cK::PDMat, x)` and PDMats.jl's
# generic `ldiv!(A::AbstractPDMat, B::AbstractVecOrMat)` are mutually ambiguous
# for `ldiv!(::PDMat, ::Matrix)` — this silently breaks `optimize!`'s internal
# solves (it throws a MethodError that our try/catch below would otherwise
# swallow, leaving every GP at its un-optimized initial hyperparameters).
# Both packages compute the same thing here; resolve the ambiguity explicitly.
LinearAlgebra.ldiv!(cK::PDMats.PDMat, x::AbstractVecOrMat) = LinearAlgebra.ldiv!(cK.chol, x)

########################################################################
###############  Log-normal moment matching (const-force / L63)  ######
########################################################################

# Given the desired mean/std of X = exp(mu + sigma*Z), Z ~ N(0,1), returns
# (mu, sigma). Reproduces the same constrained-space mean/std as
# EnsembleKalmanProcesses.jl's `constrained_gaussian(name, mean, std, 0, Inf)`
# bounded-below moment matching, without depending on EKP for it.
function lognormal_params_from_moments(mean_x::Real, std_x::Real)
    var_ratio = (std_x / mean_x)^2
    sigma2 = log(1 + var_ratio)
    mu = log(mean_x) - sigma2 / 2
    return mu, sqrt(sigma2)
end

########################################################################
###############  HMProblem: fixed-per-cell whitening setup  ############
########################################################################

# Everything about a (N_ens, rng_idx) cell's whitening that does NOT change
# wave-to-wave — built once (before the wave loop) from the prior and
# observation covariances, which are both available before any forward
# evaluation happens.
struct HMProblem
    output_basis::WhitenedPCABasis   # R-based output whitening + truncation
    y_whitened::Vector{Float64}      # observation, whitened into output_basis's coords (length output_basis.k_R)
    prior_mean::Vector{Float64}      # prior mean, in the space where the prior IS Gaussian (log-space for lognormal cases)
    prior_basis::WhitenedPCABasis    # prior-covariance-based input whitening + truncation
    inverse_transform::Function      # maps constrained theta -> the space where the prior is Gaussian (log or identity)
end

function make_hm_problem(
    y::AbstractVector, R::AbstractMatrix,
    prior_mean::AbstractVector, prior_cov::AbstractMatrix,
    inverse_transform::Function;
    retain_var_output::Real, retain_var_input::Real,
)
    output_basis = whitened_pca_basis(R, retain_var_output)
    y_whitened = whiten_vector(output_basis, y)
    prior_basis = whitened_pca_basis(prior_cov, retain_var_input)
    return HMProblem(output_basis, y_whitened, Vector{Float64}(prior_mean), prior_basis, inverse_transform)
end

# theta_batch: D x M (raw/constrained) -> k_R_prior x M (prior-whitened + truncated)
function to_prior_whitened(prob::HMProblem, theta_batch::AbstractMatrix)
    u = prob.inverse_transform.(theta_batch) .- prob.prior_mean
    return whiten_samples(prob.prior_basis, Matrix(u'))'
end

########################################################################
###############  WaveGPs: one wave's fitted emulators  #################
########################################################################

struct WaveGPs
    gps::Vector{GaussianProcesses.GPE}            # one per whitened+truncated output mode
    ensemble_pca::Union{WhitenedPCABasis, Nothing} # wave-local PCA on top of prior-whitened coords (l96_flux only)
end

# theta_ens: D x N (raw) ; results: N x n_out (raw). `ensemble_retain_var`
# triggers the additional wave-local PCA step (pass cfg.retain_var_input for
# l96_flux, `nothing` for every other case).
function fit_wave_gps(prob::HMProblem, theta_ens::AbstractMatrix, results::AbstractMatrix; ensemble_retain_var::Union{Real, Nothing} = nothing)
    Z = to_prior_whitened(prob, theta_ens)   # k_R_prior x N
    ensemble_pca = ensemble_retain_var === nothing ? nothing : whitened_pca_basis(cov(Z, dims = 2), ensemble_retain_var)
    Xfit = ensemble_pca === nothing ? Z : whiten_samples(ensemble_pca, Matrix(Z'))'   # k_in x N

    Yfit = whiten_samples(prob.output_basis, results)   # N x k_R_out
    k_R_out = prob.output_basis.k_R

    gps = Vector{GaussianProcesses.GPE}(undef, k_R_out)
    for j in 1:k_R_out
        yj = Yfit[:, j]
        sy = std(yj)
        sy = sy > 0 ? sy : 1.0
        ll = log.(vec(std(Xfit, dims = 2)) .+ 1e-8)
        kernel = GaussianProcesses.SEArd(ll, log(sy))
        gp = GaussianProcesses.GPE(Xfit, yj, GaussianProcesses.MeanZero(), kernel, -2.0)
        try
            GaussianProcesses.optimize!(gp)
        catch err
            @warn "GP hyperparameter optimization failed for whitened output $j; keeping initial hyperparameters." exception = err
        end
        gps[j] = gp
    end
    return WaveGPs(gps, ensemble_pca)
end

# theta_batch: D x M (raw) -> (mu: M x k_R_out, var: M x k_R_out), both in
# prob.output_basis's whitened coordinates.
function predict_wave(prob::HMProblem, wave::WaveGPs, theta_batch::AbstractMatrix)
    Z = to_prior_whitened(prob, theta_batch)
    Xp = wave.ensemble_pca === nothing ? Z : whiten_samples(wave.ensemble_pca, Matrix(Z'))'
    k_R_out = length(wave.gps)
    m = size(Xp, 2)
    mu = zeros(m, k_R_out)
    var = zeros(m, k_R_out)
    for j in 1:k_R_out
        # Very small N_ens (esp. 1-D const-force theta) can drive unconstrained
        # GP hyperparameter MLE toward a near-singular training covariance,
        # which then fails at *prediction* time with a LAPACK error rather
        # than at fit time. Treat a failed prediction as maximally uncertain
        # (a huge but finite variance, so the implausibility sum stays
        # well-defined without Inf/NaN) rather than crashing — this
        # downweights an unreliable output instead of it being an ad hoc
        # accept/reject override.
        try
            muj, varj = GaussianProcesses.predict_y(wave.gps[j], Xp)
            mu[:, j] = muj
            var[:, j] = varj
        catch err
            @warn "GP prediction failed for whitened output $j; treating these candidates as maximally uncertain on this output." exception = err
            var[:, j] .= 1e12
        end
    end
    return mu, var
end

########################################################################
###############  Implausibility & NROY rejection sampling  #############
########################################################################

# mu, y_whitened, var are all in prob.output_basis's whitened coordinates, so
# R's contribution is exactly the identity there — implausibility reduces to
# a plain weighted sum, no matrix solve needed.
implausibility_sq_whitened(mu::AbstractVector, y_whitened::AbstractVector, var::AbstractVector) =
    sum((mu .- y_whitened) .^ 2 ./ (1 .+ var))

# theta_batch: D x M (raw) -> BitVector (true = still NROY, i.e. passes every
# wave's implausibility test at the given chi-squared `threshold`, which
# should use output_basis.k_R degrees of freedom — the actual number of
# whitened output dimensions implausibility is computed over, not the raw
# output dimension).
function is_nroy(prob::HMProblem, theta_batch::AbstractMatrix, waves::Vector{WaveGPs}, threshold::Real)
    m = size(theta_batch, 2)
    nroy = trues(m)
    isempty(waves) && return nroy
    for wave in waves
        mu, var = predict_wave(prob, wave, theta_batch)
        for i in 1:m
            nroy[i] || continue
            nroy[i] &= (implausibility_sq_whitened(mu[i, :], prob.y_whitened, var[i, :]) < threshold)
        end
    end
    return nroy
end

# prior_sampler(rng, n) -> D x n matrix of iid prior draws (in constrained
# parameter space). Draws candidates in batches (not one-at-a-time, for
# GP-predict efficiency — mirrors the notebook's vmapped rejection sampler),
# evaluates `is_nroy` on each batch, accumulates survivors until `n_wanted`
# reached or `max_samples` exceeded. Returns (samples::Matrix (D x k),
# ok::Bool); k == n_wanted iff ok, else k < n_wanted (whatever was found
# before hitting max_samples).
function rejection_sample_nroy(
    prob::HMProblem,
    prior_sampler,
    D::Int,
    waves::Vector{WaveGPs},
    threshold::Real,
    n_wanted::Int,
    max_samples::Int,
    batch_size::Int,
    rng::AbstractRNG,
)
    if isempty(waves)
        return prior_sampler(rng, n_wanted), true
    end
    accepted = Matrix{Float64}(undef, D, 0)
    n_tried = 0
    while size(accepted, 2) < n_wanted && n_tried < max_samples
        this_batch = min(batch_size, max_samples - n_tried)
        cand = prior_sampler(rng, this_batch)
        n_tried += this_batch
        keep = is_nroy(prob, cand, waves, threshold)
        any(keep) && (accepted = hcat(accepted, cand[:, keep]))
    end
    if size(accepted, 2) >= n_wanted
        return accepted[:, 1:n_wanted], true
    else
        return accepted, false
    end
end

########################################################################
###############  Latin-hypercube prior sampling (wave 1)  ##############
########################################################################

# Generalizes calib_race_hm_l63.py's `latin_hypercube_sample` +
# `transform_to_lognorm` (independent-margins only) to arbitrary Gaussian
# priors: stratify each whitened coordinate via LHS, correlate via
# `prior_cov_sqrt` (e.g. a Cholesky factor), shift by `prior_mean`, then apply
# the per-case `constraint_transform` elementwise (e.g. `exp` for
# bounded-below/log-normal priors, `identity` for unconstrained ones).
# Needed because `l96_vec`'s prior is a correlated 40-D Gaussian and
# `l96_flux`'s is a 61-D Gaussian centered on a pretrained network's weights
# — independent per-dimension LHS margins don't generalize to either as-is.
# Operates in "prior-native" space (raw constrained theta via
# constraint_transform), independent of the HMProblem whitening machinery
# above, which governs only how the GP later consumes theta.
function lhs_prior_sample(prior_mean::AbstractVector, prior_cov_sqrt::AbstractMatrix, N::Int, rng::AbstractRNG; constraint_transform = identity)
    D = length(prior_mean)
    Z = zeros(D, N)
    for d in 1:D
        perm = randperm(rng, N)
        jitter = rand(rng, N)
        u = (perm .- 1 .+ jitter) ./ N            # stratified U(0,1)
        Z[d, :] = quantile.(Normal(), u)
    end
    theta = prior_mean .+ prior_cov_sqrt * Z
    return constraint_transform.(theta)
end
