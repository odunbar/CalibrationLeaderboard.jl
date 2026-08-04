# BayesianOptimalExperimentalDesign — shared GBOED (Goal-Oriented Bayesian
# Optimal Experimental Design) core.
#
# Included (via an @__DIR__-relative path) by calibrate_l63.jl and
# calibrate_l96.jl. Implements Algorithm 2 ("Goal-Oriented Bayesian Optimal
# Experimental Design (GBOED) with batched acquisitions") from Holthuijzen,
# Chakraborty, Krath, Catanach (2026), "Surrogate-based Bayesian calibration
# methods for chaotic systems" (arXiv:2508.13071):
#
#   1. Initial LHS design in the truncated, whitened prior space; forward-
#      evaluate it; fit one independent GP per whitened output mode.
#   2. Repeat for cfg.max_iters - 1 acquisition batches:
#        a. Draw approximate posterior samples X' from the current GP fit via
#           ST-MCMC (TransitionalMCMC.jl's `tmcmc`, matching the paper's own
#           cited ST-MCMC reference).
#        b. Jointly optimize a whole candidate batch X_cand (size = N_ens) to
#           maximize the closed-form EIG (paper Eq. 8) against X', via
#           Optim.jl's Fminbox(LBFGS()) + Zygote reverse-mode autodiff,
#           holding the GP's currently-fitted hyperparameters fixed.
#        c. Forward-evaluate the optimized batch, augment the CUMULATIVE
#           training set (unlike HistoryMatching, which fits one GP per wave
#           on that wave's ensemble alone, GBOED refits on the full growing
#           dataset every iteration), and refit.
#
# Like history_matching_core.jl, both the GP's input and output spaces are
# truncated-whitened before fitting, reusing
# common/uq_metrics/coverage_metrics.jl's WhitenedPCABasis machinery:
#   - Output space is whitened against the OBSERVATION covariance R — fixed
#     for the whole cell.
#   - Input space is whitened against the PRIOR covariance — likewise fixed
#     for the whole cell. Unlike HistoryMatching (which only ever whitens
#     forward), GBOED must also DECODE (unwhiten) optimized/sampled
#     candidates back to raw parameter space before calling the forward map,
#     since the LHS/TMCMC/EIG steps operate entirely in the truncated
#     whitened space (see `from_prior_whitened` below and
#     common/uq_metrics/coverage_metrics.jl's `unwhiten_samples`).
#
# GP fitting uses raw GaussianProcesses.jl (GPE) with the library's built-in
# `SEArd` (squared-exponential ARD) kernel — matching HistoryMatching's own
# kernel choice (history_matching_core.jl's `fit_wave_gps`), rather than the
# paper's own fixed-smoothness Matérn-7/2 (dropped in favor of consistency
# with the rest of this repo's GP-based methods and the library's own
# battle-tested, ForwardDiff-friendly implementation).

using GaussianProcesses
using LinearAlgebra
using Statistics
using Random
using Distributions
using PDMats
using Optim
using Zygote
using TransitionalMCMC

include(joinpath(@__DIR__, "..", "..", "common", "uq_metrics", "coverage_metrics.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "uq_metrics", "prior_transforms.jl"))

# Same GaussianProcesses.jl 0.12 / PDMats.jl `ldiv!` ambiguity fix
# history_matching_core.jl needs — this method uses the identical
# GaussianProcesses+PDMats combo, so the same silent-failure mode applies:
# without this, `GaussianProcesses.optimize!`'s internal solves throw a
# MethodError that the try/catch below would otherwise swallow, leaving every
# GP at its un-optimized initial hyperparameters.
LinearAlgebra.ldiv!(cK::PDMats.PDMat, x::AbstractVecOrMat) = LinearAlgebra.ldiv!(cK.chol, x)

# fit_boed_gps/boed_loglik below parallelize their per-output-GP loops across
# Julia threads; pin BLAS to 1 thread to avoid oversubscribing on top of that
# when multiple Julia threads are available (same reasoning as HistoryMatching).
Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)

########################################################################
###############  Progress/timing diagnostics  ##########################
########################################################################
# calibrate_l63.jl/calibrate_l96.jl's per-iteration loop calls several
# stages (forward-map evaluation, GP fitting, EIG optimization, ST-MCMC
# sampling) that can each individually run for a long time with NO
# built-in progress output, which from the outside looks identical to a
# genuine hang. `timed_stage` wraps a stage in start/finish @info lines
# (with elapsed wall-clock time) so it's visible which stage is actually
# running; `fit_boed_gps`/`run_tmcmc`/`optimize_batch` below additionally
# emit finer-grained progress WITHIN a stage (per-GP, per-loglik-call,
# per-LBFGS-iteration respectively).
function timed_stage(f::Function, label::AbstractString)
    @info "GBOED: starting $label"
    t0 = time()
    result = f()
    @info "GBOED: finished $label" elapsed_s = round(time() - t0; digits = 2)
    return result
end

########################################################################
###############  BOEDProblem: fixed-per-cell whitening setup  ##########
########################################################################
# Everything about a (N_ens, rng_idx) cell's whitening that does NOT change
# iteration-to-iteration — built once (before the acquisition loop) from the
# prior and observation covariances, exactly like HistoryMatching's
# HMProblem. Unlike HMProblem, this also carries `forward_transform` (the
# inverse of `inverse_transform`), since GBOED must decode LHS/TMCMC/EIG-
# optimized candidates in whitened space back to raw θ before calling the
# forward map — HistoryMatching never needs to go in that direction.
struct BOEDProblem
    output_basis::WhitenedPCABasis   # R-based output whitening + truncation
    y_whitened::Vector{Float64}      # observation, whitened into output_basis's coords
    prior_mean::Vector{Float64}      # prior mean, in "prior-native" (Gaussian) space
    prior_basis::WhitenedPCABasis    # prior-covariance-based input whitening + truncation
    inverse_transform::Function      # constrained theta -> prior-native (e.g. log)
    forward_transform::Function      # prior-native -> constrained theta (e.g. exp)
end

function make_boed_problem(
    y::AbstractVector, R::AbstractMatrix,
    prior_mean::AbstractVector, prior_cov::AbstractMatrix,
    inverse_transform::Function, forward_transform::Function;
    retain_var_output::Real, retain_var_input::Real,
)
    output_basis = whitened_pca_basis(R, retain_var_output)
    y_whitened = whiten_vector(output_basis, y)
    prior_basis = whitened_pca_basis(prior_cov, retain_var_input)
    return BOEDProblem(output_basis, y_whitened, Vector{Float64}(prior_mean), prior_basis, inverse_transform, forward_transform)
end

# theta_batch: D x M (raw/constrained) -> k_R_prior x M (prior-whitened + truncated)
function to_prior_whitened(prob::BOEDProblem, theta_batch::AbstractMatrix)
    u = prob.inverse_transform.(theta_batch) .- prob.prior_mean
    return whiten_samples(prob.prior_basis, Matrix(u'))'
end

# Z: k_R_prior x M (prior-whitened + truncated) -> D x M (raw/constrained).
# Exact inverse of to_prior_whitened up to the truncation's information loss
# (discarded prior-whitened modes are reconstructed at exactly zero, i.e.
# held at the prior mean) — needed because GBOED, unlike History Matching,
# samples/optimizes candidates directly in the whitened space and must decode
# them before evaluating the (raw-parameter-space) forward map.
function from_prior_whitened(prob::BOEDProblem, Z::AbstractMatrix)
    u = unwhiten_samples(prob.prior_basis, Matrix(Z'))'   # D x M, in "prior-native" space
    theta_native = u .+ prob.prior_mean
    return prob.forward_transform.(theta_native)
end

########################################################################
###############  LHS in the truncated whitened space  ##################
########################################################################
# Since PCA-whitening makes the retained coordinates' marginal prior EXACTLY
# standard normal (a property of Gaussian marginals, independent of
# truncation), one primitive serves both the initial LHS design (step 1) and
# the ST-MCMC prior sampler (`sample_fT` below) — this is
# history_matching_core.jl's `lhs_prior_sample` with the mean-shift/
# correlate/constraint-transform steps stripped, since whitening already IS
# that transform.
function lhs_standard_normal_sample(k::Int, N::Int, rng::AbstractRNG)
    Z = zeros(k, N)
    for d in 1:k
        perm = randperm(rng, N)
        jitter = rand(rng, N)
        u = (perm .- 1 .+ jitter) ./ N
        Z[d, :] = quantile.(Normal(), u)
    end
    return Z   # k x N, each row ~ stratified N(0,1)
end

########################################################################
###############  BOEDGPs: the cumulative-dataset GP fit  ################
########################################################################
# Unlike HistoryMatching's WaveGPs (one independent GP set per wave, never
# refit on later data), BOEDGPs is refit on the FULL cumulative dataset every
# iteration (paper Algorithm 2, step "Refit GP hyperparameters using the
# updated dataset D").
struct BOEDGPs
    gps::Vector{GaussianProcesses.GPE}   # one per whitened + truncated output mode
end

# Z: k_R_prior x N (prior-whitened + truncated inputs, CUMULATIVE across all
# iterations so far) ; results: N x n_out (raw outputs, CUMULATIVE).
#
# Bounds below are all RELATIVE to each GP's own data-driven initial guess
# (ll0's per-dimension empirical std of Z, `sy`'s empirical std of the
# whitened response) rather than a hardcoded absolute magnitude — this keeps
# a single set of defaults sane across every experiment case despite Z/Yfit's
# natural scale differing between them (Z is prior-whitened, ~unit variance;
# Yfit is R-whitened, whose scale depends on the signal-to-observation-noise
# ratio, which varies by case). Left fully unconstrained (the previous
# behavior), `optimize!`'s unconstrained MLE can drive the length scale
# arbitrarily short and/or the noise arbitrarily close to zero when fitting
# on very few cumulative points (small N_ens, early iterations) — both
# produce a near-singular training covariance and hence spuriously
# overconfident (near-zero-variance) predictions in extrapolated regions.
# That overconfidence is exactly what `boed_loglik` (used by ST-MCMC) can
# latch onto: unlike HistoryMatching's implausibility (a pure threshold
# statistic), `boed_loglik`'s -0.5*log(2π·v) term rewards low predictive
# variance regardless of whether the mean is actually close to the
# observation, and ST-MCMC's importance-weighted resampling can then
# collapse the whole particle population onto that one spurious point.
# Bounding length scale and noise away from their degenerate extremes removes
# the mechanism that creates that false confidence in the first place.
function fit_boed_gps(
    prob::BOEDProblem, Z::AbstractMatrix, results::AbstractMatrix;
    gp_lengthscale_log10_range::Real = 2.0,
    gp_signal_std_log10_range::Real = 2.0,
    gp_min_noise_std_frac::Real = 1e-3,
    gp_max_noise_std_frac::Real = 1.0,
)
    Yfit = whiten_samples(prob.output_basis, results)   # N x k_R_out
    k_R_out = prob.output_basis.k_R

    gps = Vector{GaussianProcesses.GPE}(undef, k_R_out)
    ll0 = log.(vec(std(Z, dims = 2)) .+ 1e-8)   # same for every j; hoisted out of the loop below
    ll_lo = ll0 .- gp_lengthscale_log10_range * log(10)
    ll_hi = ll0 .+ gp_lengthscale_log10_range * log(10)
    N = size(Z, 2)
    @info "fit_boed_gps: fitting $k_R_out GP(s) on N=$N cumulative points across $(Threads.nthreads()) thread(s)"
    Threads.@threads for j in 1:k_R_out
        t0 = time()
        yj = Yfit[:, j]
        sy = std(yj)
        sy = sy > 0 ? sy : 1.0
        lsy = log(sy)

        # Kernel params, per SEArd.get_params, are [ll (d-vector); lσ (scalar)].
        kernbounds = (
            vcat(ll_lo, lsy - gp_signal_std_log10_range * log(10)),
            vcat(ll_hi, lsy + gp_signal_std_log10_range * log(10)),
        )
        noise_lo = log(gp_min_noise_std_frac) + lsy
        noise_hi = log(gp_max_noise_std_frac) + lsy
        noisebounds = (noise_lo, noise_hi)

        kernel = GaussianProcesses.SEArd(ll0, lsy)
        # Centered exactly at the noise bounds' midpoint (rather than a fixed
        # -2.0) so the starting point is always strictly interior to
        # noisebounds regardless of sy's scale — Fminbox requires that.
        gp = GaussianProcesses.GPE(Z, yj, GaussianProcesses.MeanZero(), kernel, (noise_lo + noise_hi) / 2)
        try
            GaussianProcesses.optimize!(gp; kernbounds = kernbounds, noisebounds = noisebounds)
        catch err
            @warn "GP hyperparameter optimization failed for whitened output $j; keeping initial hyperparameters." exception = err
        end
        gps[j] = gp
        @info "fit_boed_gps: output mode $j/$k_R_out fit done" thread = Threads.threadid() elapsed_s = round(time() - t0; digits = 2)
    end
    return BOEDGPs(gps)
end

# z: k_R_prior vector (a single candidate, in prior-whitened+truncated space)
# -> (mu::Vector, var::Vector), both length k_R_out, in output_basis's
# whitened coordinates.
function predict_boed(gps::BOEDGPs, z::AbstractVector)
    Zmat = reshape(z, :, 1)
    k_R_out = length(gps.gps)
    mu = zeros(k_R_out)
    var = zeros(k_R_out)
    for j in 1:k_R_out
        try
            muj, varj = GaussianProcesses.predict_y(gps.gps[j], Zmat)
            mu[j] = muj[1]
            var[j] = varj[1]
        catch err
            @warn "GP prediction failed for whitened output $j; treating this candidate as maximally uncertain on this output." exception = err
            var[j] = 1e12
        end
    end
    return mu, var
end

########################################################################
###############  ST-MCMC posterior sampling (TransitionalMCMC.jl)  #####
########################################################################
# Paper Eq. 5: p(y_obs | θ, GP(D)) = MVN(y_obs; μ_GP(θ), Γ_GP(θ) + Γ_obs). In
# the R-whitened output space Γ_obs ≈ I (same simplification
# history_matching_core.jl's implausibility metric uses), so the per-output-
# mode variance is `1 + GP-predictive-variance`. Unlike HistoryMatching's
# implausibility (a pure threshold statistic, so it drops the normalizing
# log(2π·v) term), TMCMC needs a properly normalized (unnormalized-up-to-a-
# constant is fine, but internally consistent) log-density to target via
# tempering, so the full Gaussian log-density is required here.
function boed_loglik(prob::BOEDProblem, gps::BOEDGPs, z::AbstractVector)
    mu, var = predict_boed(gps, z)
    ll = 0.0
    for j in eachindex(mu)
        v = 1.0 + var[j]
        ll += -0.5 * (mu[j] - prob.y_whitened[j])^2 / v - 0.5 * log(2 * pi * v)
    end
    return ll
end

# Draws `n_samples` approximate posterior samples in the prior-whitened +
# truncated space via TransitionalMCMC.jl's `tmcmc` (Ching & Chen 2007
# Transitional MCMC — the paper's cited "ST-MCMC"). NOTE: `tmcmc`'s own
# internal convention is samples-as-ROWS (an Nsamples x k_R_prior matrix, the
# opposite of this file's own columns-as-samples convention used everywhere
# else) — verified directly against the installed package's source
# (TransitionalMCMC/src/tmcmc.jl), not just its README. Transpose at the
# boundary so callers of `run_tmcmc` see the usual k_R_prior x n_samples
# shape.
#
# When k_R_prior == 1 (a genuinely 1-D case, e.g. l96_const's single-parameter
# prior, or an aggressively truncated multi-D one), TransitionalMCMC.jl's own
# `metropolis_hastings_simple` (src/mcmc.jl) deliberately special-cases
# dims==1 and mutates a flat vector of scalars rather than length-1 vectors —
# confirmed directly against its source, not inferred — so `loglik`/`logprior`
# below must accept a bare `Real` as well as an `AbstractVector`.
function boed_loglik(prob::BOEDProblem, gps::BOEDGPs, z::Real)
    boed_loglik(prob, gps, [z])
end

# `burnin`/`thin` are exposed (rather than left at tmcmc's own defaults of
# 20/3) because they directly multiply the per-tempering-stage cost: tmcmc
# evaluates the GP-based log-likelihood roughly
# n_samples * (burnin + thin) * 2 times per stage via Distributed.pmap, which
# does NOT parallelize across Julia threads (Threads.@threads is a different
# parallelism model) — without extra worker processes from `addprocs()`, this
# is effectively serial with real per-call scheduling overhead. Empirically,
# even n_samples=200 at the library's own defaults made a single run_tmcmc
# call take many minutes; smaller burnin/thin (and a smaller n_samples, see
# experiment_config.jl) are the cheap first lever before reaching for
# addprocs() or a from-scratch sampler.
function run_tmcmc(prob::BOEDProblem, gps::BOEDGPs, n_samples::Int, rng::AbstractRNG; burnin::Int = 5, thin::Int = 1)
    k_R_prior = prob.prior_basis.k_R
    # TransitionalMCMC.jl's own per-tempering-stage "β_i = ..." @info lines are
    # the only built-in progress signal — WITHIN a stage (the
    # n_samples*(burnin+thin)*2 loglik calls described above) there is none,
    # so a long-running stage looks indistinguishable from a hang. This
    # counter reports progress within a stage too.
    n_calls = Ref(0)
    t_start = time()
    report_every = max(50, n_samples)
    function loglik(z)
        n_calls[] += 1
        if n_calls[] % report_every == 0
            @info "run_tmcmc: $(n_calls[]) log-likelihood evaluations so far" elapsed_s = round(time() - t_start; digits = 2)
        end
        return boed_loglik(prob, gps, z)
    end
    logprior(z) = sum(logpdf.(Normal(), z))
    priorRnd(n) = Matrix(lhs_standard_normal_sample(k_R_prior, n, rng)')   # n x k_R_prior, per tmcmc's row-major convention
    samps, _log_ev = TransitionalMCMC.tmcmc(loglik, logprior, priorRnd, n_samples, burnin, thin)
    @info "run_tmcmc: done" total_loglik_evals = n_calls[] elapsed_s = round(time() - t_start; digits = 2)
    return Matrix(reshape(samps, n_samples, k_R_prior)')   # back to k_R_prior x n_samples
end

########################################################################
###############  Standalone, AD-safe SE-ARD kernel  #####################
########################################################################
# Deliberately decoupled from GaussianProcesses.jl's own kernel-evaluation
# internals — that library's AD-compatibility for the EIG optimization's
# gradient is unverified and shouldn't be relied on. Used ONLY inside
# `eig_objective`, with the GP's currently-fitted hyperparameters held fixed
# (plain Float64 constants, not part of the optimization).

# X1: d x M1, X2: d x M2, iℓ2: d-vector of inverse squared length scales ->
# M1 x M2 matrix of the ARD-weighted squared distance Σ_k iℓ2[k]·(X1[k,i]-X2[k,j])².
#
# Written via the expanded-norm identity (‖a-b‖²_w = ‖a‖²_w - 2⟨a,b⟩_w + ‖b‖²_w)
# and BLAS matmuls rather than the mathematically-equivalent element-wise
# double loop this replaces: the loop version mutated a preallocated `d2`
# in-place, which Zygote (this file's autodiff for `eig_objective`, replacing
# ForwardDiff — see that function's comment) cannot differentiate through.
# Rewriting as pure broadcasting + `sum`/`*` sidesteps that entirely, and is
# also markedly faster in plain Float64 (BLAS-backed matmul vs. a scalar
# Julia loop). Clamped at 0 to guard against ~-1e-15 cancellation noise on
# the diagonal (X1 === X2 columns), which would otherwise make
# `se_cov_matrix`'s exp(-0.5*d2) evaluate at a a hair above 1 instead of
# exactly 1 — harmless in exact arithmetic but avoided for cleanliness.
function ard_sqdist(X1::AbstractMatrix, X2::AbstractMatrix, iℓ2::AbstractVector)
    X1w = X1 .* iℓ2
    n1 = vec(sum(X1w .* X1; dims = 1))       # M1, weighted squared norm of each X1 column
    n2 = vec(sum(X2 .* (iℓ2 .* X2); dims = 1)) # M2, weighted squared norm of each X2 column
    cross = X1w' * X2                          # M1 x M2, Σ_k iℓ2[k]·X1[k,i]·X2[k,j]
    return max.(n1 .+ n2' .- 2 .* cross, 0)
end

# k(x,x') = τ²·exp(-d²/2), matching GaussianProcesses.jl's own SEArd
# convention (`cov(se, r) = se.σ2*exp(-r/2)` with `r` already the ARD-weighted
# SQUARED distance). Unlike the Matérn family, SE's covariance is a smooth
# function of d² directly (no square root anywhere in the formula), so —
# unlike the Matérn-7/2 version this replaces — no epsilon floor is needed to
# keep ForwardDiff's gradient well-defined at d²=0 (every diagonal entry of a
# self-covariance matrix).
function se_cov_matrix(X1::AbstractMatrix, X2::AbstractMatrix, iℓ2::AbstractVector, τ2::Real)
    d2 = ard_sqdist(X1, X2, iℓ2)
    return τ2 .* exp.(-0.5 .* d2)
end

########################################################################
###############  EIG objective (paper Eq. 8) & joint-batch optimization #
########################################################################

extract_hyperparams(gps::BOEDGPs) = [(copy(gp.kernel.iℓ2), gp.kernel.σ2) for gp in gps.gps]

# X_cand_flat: vec(k_R_prior x B) candidate design (the batch being
# optimized). X_post: k_R_prior x n_post current ST-MCMC posterior samples
# (the "X'" of paper Eq. 8 — goal-oriented target locations). hyperparams_by_dim
# is `extract_hyperparams(gps)`, HELD FIXED during this inner optimization.
#
#   EIG(X_cand) = (1/k_R_out) Σ_j 0.5·log(det(K_pp^(j)) / det(Σ'^(j)))
#   Σ'^(j) = K_pp^(j) - K_pc^(j)·(K_cc^(j))⁻¹·K_cp^(j)
#
# Since GaussianProcesses.jl fits independent GPs per whitened output
# dimension (paper Section 2), EIG is computed per dimension and averaged
# (not summed) so the objective's scale doesn't grow with k_R_out — an
# aggregation choice not pinned down by the paper, kept trivially swappable.
#
# A Schur-complement/matrix-determinant-lemma reformulation was tried here —
# rewriting logdet(Σ') via the joint covariance [K_pp K_pc; K_pc' K_cc] so
# only a B x B (rather than n_post x n_post) matrix needs factoring — and
# REJECTED after measuring it directly under ForwardDiff (see below for why
# the gradient is now computed with Zygote instead, but this reasoning
# predates and is independent of that switch): (1) no actual speedup (~1.0x
# on the dominated gradient cost; profiling showed Julia's generic
# `Cholesky{Float64} \ Matrix{Dual}` dispatch promotes the WHOLE factor to
# Dual before solving, rather than doing a cheap mixed-type solve against the
# fixed Float64 factor, so the assumed saving never materialized); (2) it
# requires inverting K_pp directly, and K_pp — built from n_post ST-MCMC
# posterior samples via a smooth SE kernel — is frequently near-singular in
# practice (measured eigenvalues down to ~1e-15 for realistic posterior
# samples/length scales), making that inversion numerically unsound in
# exactly the regime this method already struggles with — a concern that
# applies regardless of autodiff backend. The original K_cc-based form below
# never inverts the large matrix (only takes ITS logdet via Cholesky, which
# stays well-behaved with a small jitter even when near-singular), which is
# why it's the right form to keep despite being the more expensive one.
function eig_objective(
    x_cand_flat::AbstractVector, X_post::AbstractMatrix,
    hyperparams_by_dim, k_R_prior::Int, B::Int; jitter::Real = 1e-6,
)
    X_cand = reshape(x_cand_flat, k_R_prior, B)
    total = zero(eltype(x_cand_flat))
    for (iℓ2, τ2) in hyperparams_by_dim
        K_pp = se_cov_matrix(X_post, X_post, iℓ2, τ2)
        K_pc = se_cov_matrix(X_post, X_cand, iℓ2, τ2)
        K_cc = se_cov_matrix(X_cand, X_cand, iℓ2, τ2) + jitter * τ2 * I
        Sigma_post = Symmetric(K_pp - K_pc * (K_cc \ K_pc'))
        logdet_pp = logdet(cholesky(Symmetric(K_pp) + jitter * τ2 * I))
        logdet_post = logdet(cholesky(Sigma_post + jitter * τ2 * I))
        total += 0.5 * (logdet_pp - logdet_post)
    end
    return total / length(hyperparams_by_dim)
end

# Draws the starting batch for the joint EIG optimization (`cfg.batch_init_strategy`):
#   :posterior_subsample — subsample B points from the current ST-MCMC posterior X'.
#   :fresh_lhs           — a fresh LHS draw in the truncated whitened space.
function init_candidate_batch(X_post::AbstractMatrix, B::Int, rng::AbstractRNG; strategy::Symbol = :posterior_subsample)
    if strategy === :posterior_subsample
        n_post = size(X_post, 2)
        idx = n_post >= B ? randperm(rng, n_post)[1:B] : rand(rng, 1:n_post, B)
        return X_post[:, idx]
    elseif strategy === :fresh_lhs
        return lhs_standard_normal_sample(size(X_post, 1), B, rng)
    else
        error("Unknown batch_init_strategy: $strategy")
    end
end

# Jointly optimizes the WHOLE candidate batch (all B points at once, paper
# Algorithm 2) via Optim.jl's Fminbox(LBFGS()), maximizing eig_objective
# (implemented here as minimizing its negation). Box bounds are ±bound_std
# standard deviations in the whitened Z-space, keeping candidates within the
# GP's trust region.
#
# Gradient is computed with Zygote (reverse-mode), not ForwardDiff
# (forward-mode) or ReverseDiff. ForwardDiff was the original choice but is
# the wrong complexity class here: its cost scales with the NUMBER OF INPUTS
# being differentiated (k_R_prior*B, e.g. 5*90=450 — one dual "lane" group
# per chunk of ~12 inputs, so dozens of forward passes per gradient), whereas
# a reverse-mode gradient of a scalar objective costs ~1 forward pass
# regardless of input count. ReverseDiff was tried next and rejected: it has
# no built-in adjoint for `cholesky`/`logdet`/`\`, so it traces the raw
# elementwise LAPACK-style algorithm operating on `TrackedReal`s, building an
# O(n_post^3 + B^3)-sized instruction tape PER gradient call (thousands to
# ~10^6 tracked nodes for this problem's n_post/B) — this is what exhausted
# memory. Zygote sidesteps this entirely: via ChainRules it has efficient,
# constant-size adjoint rules for `cholesky`, `logdet`, and `\` built in, so
# differentiating through `eig_objective` never touches their internals. The
# one piece that DID need rewriting for Zygote is `ard_sqdist` (see its own
# comment) — Zygote can't differentiate through in-place mutation, which the
# original loop-based version relied on.
# Optim.jl's `f_calls_limit`/`g_calls_limit` do NOT reliably bound total
# evaluations here — verified directly: Fminbox restarts a fresh inner LBFGS
# solve at every outer barrier round, and f_calls_limit/g_calls_limit are
# enforced PER INNER SOLVE, not cumulatively across the whole Fminbox run.
# `call_limit=2000` with `outer_iterations=20` still let a real run reach its
# full ~39,000-evaluation budget in testing — the option silently doesn't do
# what its name suggests under Fminbox. This works around that by tracking
# our OWN cumulative evaluation count and the best point seen so far, and
# throwing (caught just below) once `call_limit` is hit, returning that best
# point rather than trusting Optim to stop itself.
#
# Also applies the `g_tol`/`f_reltol` convergence-tolerance loosening
# discussed in experiment_config.jl's eig_g_tol/eig_f_reltol comments.
struct EIGCallLimitReached <: Exception end

function run_fminbox_with_call_limit(
    neg_eig_raw::Function, x_init::AbstractVector, lo::AbstractVector, hi::AbstractVector;
    iters::Int, outer_iters::Int, g_tol::Real, f_reltol::Real, call_limit::Int, label::AbstractString,
)
    n_calls = Ref(0)
    best_val = Ref(Inf)
    best_x = Ref(copy(x_init))
    t0 = time()
    function neg_eig(x)
        val = neg_eig_raw(x)
        n_calls[] += 1
        if val < best_val[]
            best_val[] = val
            best_x[] = copy(x)
        end
        if n_calls[] % 100 == 0
            @info "$label: $(n_calls[]) EIG objective evaluations so far" elapsed_s = round(time() - t0; digits = 2)
        end
        n_calls[] >= call_limit && throw(EIGCallLimitReached())
        return val
    end
    function neg_eig_grad!(G::AbstractVector, x::AbstractVector)
        G .= only(Zygote.gradient(neg_eig, x))
        return G
    end
    # Built as an explicit OnceDifferentiable (rather than passing `neg_eig`
    # straight to `Optim.optimize` with an `autodiff = ...` keyword) because
    # that keyword only recognizes `:finite`/`:forward` in the installed
    # Optim version — plugging in an arbitrary gradient function (here,
    # Zygote's) requires constructing the differentiable object by hand.
    od = Optim.OnceDifferentiable(neg_eig, neg_eig_grad!, x_init)
    opts = Optim.Options(
        iterations = iters, outer_iterations = outer_iters,
        g_abstol = g_tol, outer_g_abstol = g_tol,
        f_reltol = f_reltol, outer_f_reltol = f_reltol,
    )
    converged, outer_rounds, hit_limit = false, 0, false
    try
        res = Optim.optimize(od, lo, hi, x_init, Fminbox(LBFGS()), opts)
        converged = Optim.converged(res)
        outer_rounds = Optim.iterations(res)
    catch e
        e isa EIGCallLimitReached || rethrow()
        hit_limit = true
    end
    @info "$label: done" converged outer_rounds objective_evals = n_calls[] elapsed_s = round(time() - t0; digits = 2)
    if hit_limit
        @warn "$label: hit the EIG evaluation call_limit ($call_limit) before Optim's own convergence criteria were satisfied — the returned point(s) may be under-optimized. Consider raising cfg.eig_call_limit if this happens often." objective_evals = n_calls[] elapsed_s = round(time() - t0; digits = 2)
    elseif !converged && outer_rounds >= outer_iters
        @warn "$label: Fminbox's outer barrier loop hit its outer_iterations cap ($outer_iters) without converging — the returned point(s) may be under-optimized. Consider raising cfg.eig_outer_iters if this happens often." objective_evals = n_calls[] elapsed_s = round(time() - t0; digits = 2)
    end
    return best_x[], n_calls[]
end

# Jointly optimizes the WHOLE candidate batch (all B points at once, paper
# Algorithm 2) via Optim.jl's Fminbox(LBFGS()), maximizing eig_objective
# (implemented here as minimizing its negation). Box bounds are ±bound_std
# standard deviations in the whitened Z-space, keeping candidates within the
# GP's trust region.
#
# Gradient is computed with Zygote (reverse-mode), not ForwardDiff
# (forward-mode) or ReverseDiff. ForwardDiff was the original choice but is
# the wrong complexity class here: its cost scales with the NUMBER OF INPUTS
# being differentiated (k_R_prior*B, e.g. 5*90=450 — one dual "lane" group
# per chunk of ~12 inputs, so dozens of forward passes per gradient), whereas
# a reverse-mode gradient of a scalar objective costs ~1 forward pass
# regardless of input count. ReverseDiff was tried next and rejected: it has
# no built-in adjoint for `cholesky`/`logdet`/`\`, so it traces the raw
# elementwise LAPACK-style algorithm operating on `TrackedReal`s, building an
# O(n_post^3 + B^3)-sized instruction tape PER gradient call (thousands to
# ~10^6 tracked nodes for this problem's n_post/B) — this is what exhausted
# memory. Zygote sidesteps this entirely: via ChainRules it has efficient,
# constant-size adjoint rules for `cholesky`, `logdet`, and `\` built in, so
# differentiating through `eig_objective` never touches their internals. The
# one piece that DID need rewriting for Zygote is `ard_sqdist` (see its own
# comment) — Zygote can't differentiate through in-place mutation, which the
# original loop-based version relied on.
function optimize_batch(
    X_init::AbstractMatrix, X_post::AbstractMatrix, hyperparams_by_dim, k_R_prior::Int, B::Int;
    bound_std::Real, iters::Int, outer_iters::Int = 20, jitter::Real,
    g_tol::Real = 1e-3, f_reltol::Real = 1e-6, call_limit::Int = 5_000,
)
    lo = fill(-bound_std, k_R_prior * B)
    hi = fill(bound_std, k_R_prior * B)
    neg_eig_raw(x) = -eig_objective(x, X_post, hyperparams_by_dim, k_R_prior, B; jitter = jitter)
    x_star, _ = run_fminbox_with_call_limit(
        neg_eig_raw, vec(X_init), lo, hi;
        iters = iters, outer_iters = outer_iters, g_tol = g_tol, f_reltol = f_reltol,
        call_limit = call_limit, label = "optimize_batch",
    )
    return reshape(x_star, k_R_prior, B)
end

# PROTOTYPE — greedy/sequential alternative to `optimize_batch`'s joint batch
# optimization, NOT currently wired into calibrate_l63.jl/calibrate_l96.jl.
#
# `optimize_batch` jointly optimizes all B points at once (paper Algorithm 2),
# a k_R_prior*B-dimensional decision variable (e.g. ~2340 for l96_vec's
# largest N_ens). Profiling traced the 10,000s-to-80,000+ EIG-evaluation
# blowup to THIS joint dimensionality, not to the kernel matrices themselves
# (K_pp/K_cc are at most n_post x n_post / B x B — already cheap to factor at
# this problem's actual n_post/B scale; a Nyström/inducing-point approximation
# of THOSE would target a bottleneck this codebase doesn't currently have).
#
# This instead selects the batch ONE POINT AT A TIME: at step i, the
# previously selected i-1 points are held FIXED (folded into `eig_objective`
# as part of X_cand) and only the new point — a k_R_prior-dimensional
# decision variable, not k_R_prior*B — is optimized to maximize the EIG of
# the resulting i-point batch so far. This is the standard greedy forward-
# selection relaxation of joint Gaussian mutual-information batch
# maximization (e.g. Krause/Singh/Guestrin-style greedy GP sensor placement):
# provably near-optimal under (approximate) submodularity of this objective,
# and here mainly valuable for collapsing the per-step decision space by a
# factor of B. It reuses `eig_objective` verbatim (called with a growing
# candidate count i = 1, ..., B) rather than duplicating the EIG math.
#
# A genuine departure from the paper's specified joint optimization — kept
# separate from `optimize_batch` rather than replacing it so the two can be
# compared before deciding whether to switch production experiments over.

# Optimizes ONE new candidate point (a k_R_prior-dim decision variable) to
# maximize the EIG of the batch formed by appending it to the already-fixed
# `X_fixed` columns (X_fixed held constant, not part of the gradient). Shared
# by `optimize_batch_greedy` and `optimize_batch_hybrid`'s greedy prefix phase
# below — factored out because both need EXACTLY this one-point sub-solve,
# not because it's used more widely than that.
function optimize_one_point(
    x_init::AbstractVector, X_fixed::AbstractMatrix, X_post::AbstractMatrix,
    hyperparams_by_dim, k_R_prior::Int; bound_std::Real, iters::Int, outer_iters::Int, jitter::Real,
    g_tol::Real = 1e-3, f_reltol::Real = 1e-6, call_limit::Int = 5_000,
)
    lo = fill(-bound_std, k_R_prior)
    hi = fill(bound_std, k_R_prior)
    B_so_far = size(X_fixed, 2) + 1
    neg_eig_raw(x) = -eig_objective(vec(hcat(X_fixed, x)), X_post, hyperparams_by_dim, k_R_prior, B_so_far; jitter = jitter)
    return run_fminbox_with_call_limit(
        neg_eig_raw, x_init, lo, hi;
        iters = iters, outer_iters = outer_iters, g_tol = g_tol, f_reltol = f_reltol,
        call_limit = call_limit, label = "optimize_one_point",
    )
end

function optimize_batch_greedy(
    X_init::AbstractMatrix, X_post::AbstractMatrix, hyperparams_by_dim, k_R_prior::Int, B::Int;
    bound_std::Real, iters::Int, outer_iters::Int = 20, jitter::Real,
    g_tol::Real = 1e-3, f_reltol::Real = 1e-6, call_limit::Int = 5_000,
)
    X_sel = zeros(eltype(X_init), k_R_prior, 0)
    total_calls = 0
    t0 = time()
    report_every = max(1, B ÷ 10)
    for i in 1:B
        x_star, n_calls_i = optimize_one_point(
            X_init[:, i], X_sel, X_post, hyperparams_by_dim, k_R_prior;
            bound_std = bound_std, iters = iters, outer_iters = outer_iters, jitter = jitter,
            g_tol = g_tol, f_reltol = f_reltol, call_limit = call_limit,
        )
        X_sel = hcat(X_sel, x_star)
        total_calls += n_calls_i
        if i % report_every == 0 || i == B
            @info "optimize_batch_greedy: point $i/$B selected" objective_evals = n_calls_i cumulative_evals = total_calls elapsed_s = round(time() - t0; digits = 2)
        end
    end
    return X_sel
end

# PROTOTYPE — hybrid of `optimize_batch_greedy` and `optimize_batch`: greedily
# select points one at a time (cheap while the batch is still sparse) UNTIL a
# single greedy step's own evaluation count exceeds `greedy_eval_threshold` —
# empirically, comparing this against pure greedy on the same synthetic
# problem, EARLY greedy steps cost ~50-90 evaluations each while the whitened
# box (±bound_std) still has room, but LATE steps (once most of the box is
# already occupied) blow up to hundreds-to-thousands of evaluations each, as
# LBFGS struggles to squeeze one more point into an increasingly crowded,
# near-singular-K_cc configuration — greedy doesn't avoid this conditioning
# breakdown, it just relocates it into dozens of individually-hard one-point
# sub-problems, which is why pure greedy measured SLOWER overall than joint
# despite each step nominally being lower-dimensional.
#
# This hybrid tries to get the cheap part of greedy (fast early steps) without
# paying its expensive part (thrashing one point at a time through the
# crowded end-game): once the per-step cost signals that the remaining points
# no longer fit cheaply one at a time, it switches to a SINGLE joint
# optimization over all remaining slots at once (a k_R_prior*(B-K)-dimensional
# problem, smaller than the full k_R_prior*B joint problem) — jointly
# optimizing the remaining points together lets them mutually rearrange
# relative to each other and to the fixed prefix, rather than each one
# blindly hunting for room in isolation.
function optimize_batch_hybrid(
    X_init::AbstractMatrix, X_post::AbstractMatrix, hyperparams_by_dim, k_R_prior::Int, B::Int;
    bound_std::Real, iters::Int, outer_iters::Int = 20, jitter::Real, greedy_eval_threshold::Int = 300,
    g_tol::Real = 1e-3, f_reltol::Real = 1e-6, call_limit::Int = 5_000,
)
    X_sel = zeros(eltype(X_init), k_R_prior, 0)
    total_calls = 0
    t0 = time()
    i = 1
    while i <= B
        x_star, n_calls_i = optimize_one_point(
            X_init[:, i], X_sel, X_post, hyperparams_by_dim, k_R_prior;
            bound_std = bound_std, iters = iters, outer_iters = outer_iters, jitter = jitter,
            g_tol = g_tol, f_reltol = f_reltol, call_limit = call_limit,
        )
        X_sel = hcat(X_sel, x_star)
        total_calls += n_calls_i
        if n_calls_i > greedy_eval_threshold
            @info "optimize_batch_hybrid: greedy step $i/$B cost $n_calls_i evaluations (> threshold $greedy_eval_threshold) — switching remaining $(B - i) point(s) to a joint optimization" cumulative_evals = total_calls elapsed_s = round(time() - t0; digits = 2)
            i += 1
            break
        end
        i += 1
    end

    K = size(X_sel, 2)
    n_remaining = B - K
    if n_remaining > 0
        lo = fill(-bound_std, k_R_prior * n_remaining)
        hi = fill(bound_std, k_R_prior * n_remaining)
        neg_eig_joint_raw(x_free) = -eig_objective(
            vec(hcat(X_sel, reshape(x_free, k_R_prior, n_remaining))), X_post, hyperparams_by_dim, k_R_prior, B; jitter = jitter,
        )
        x_free_init = vec(X_init[:, (K + 1):B])
        x_free_star, n_calls_joint = run_fminbox_with_call_limit(
            neg_eig_joint_raw, x_free_init, lo, hi;
            iters = iters, outer_iters = outer_iters, g_tol = g_tol, f_reltol = f_reltol,
            call_limit = call_limit, label = "optimize_batch_hybrid (joint phase)",
        )
        total_calls += n_calls_joint
        X_sel = hcat(X_sel, reshape(x_free_star, k_R_prior, n_remaining))
        @info "optimize_batch_hybrid: joint phase for remaining $n_remaining point(s) done" cumulative_evals = total_calls elapsed_s = round(time() - t0; digits = 2)
    end
    return X_sel
end
