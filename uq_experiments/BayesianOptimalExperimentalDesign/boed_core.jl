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
#           Optim.jl's Fminbox(LBFGS()) + ForwardDiff autodiff, holding the
#           GP's currently-fitted hyperparameters fixed.
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
# GP fitting uses raw GaussianProcesses.jl (GPE), with a hand-written
# Matérn-7/2 ARD kernel (ν = 3.5, the paper's fixed smoothness — not one of
# GaussianProcesses.jl's three built-in Matérn types, 1/2/3/2/5/2).

using GaussianProcesses
using LinearAlgebra
using Statistics
using Random
using Distributions
using PDMats
using Optim
using ForwardDiff
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
###############  Custom Matérn-7/2 ARD kernel  #########################
########################################################################
# GaussianProcesses.jl's own `Matern(ν, ll, lσ)` constructor only supports
# ν ∈ {1/2, 3/2, 5/2} (throws ArgumentError otherwise) — ν = 3.5 (= 7/2, the
# paper's fixed smoothness) must be hand-written. Slots into the existing
# `GaussianProcesses.MaternARD` abstract type (gets the ARD chain-rule glue,
# `dKij_dθp`, for free), mirroring the library's own `Mat52Ard` exactly:
#
#   k(r) = σ²(1 + √7·r + 14r²/5 + 7√7·r³/15)·exp(-√7·r)
#
# `dk_dll`'s formula below was derived by hand (symbolic differentiation of
# k(r) w.r.t. a log-length-scale parameter, following the exact same
# derivation steps that reproduce Mat52Ard's documented
# `dk_dll = 5/3·σ²·wdiffp·(1+s)·exp(-s)` as a consistency check) and
# independently verified against Mat52Ard's known-correct result before use
# here.
mutable struct Mat72Ard{T <: Real} <: GaussianProcesses.MaternARD
    iℓ2::Vector{T}    # inverse squared length scales, one per input dim
    σ2::T             # signal variance τ²
    priors::Array
end
Mat72Ard(ll::Vector{T}, lσ::T) where {T} = Mat72Ard{T}(exp.(-2 .* ll), exp(2 * lσ), [])

function GaussianProcesses.set_params!(mat::Mat72Ard, hyp::AbstractVector)
    length(hyp) == GaussianProcesses.num_params(mat) ||
        throw(ArgumentError("Mat72Ard kernel has $(GaussianProcesses.num_params(mat)) parameters, received $(length(hyp))."))
    @views @. mat.iℓ2 = exp(-2 * hyp[1:(end - 1)])
    mat.σ2 = exp(2 * hyp[end])
end
GaussianProcesses.get_params(mat::Mat72Ard) = [-log.(mat.iℓ2) / 2; log(mat.σ2) / 2]
GaussianProcesses.get_param_names(mat::Mat72Ard) = [GaussianProcesses.get_param_names(mat.iℓ2, :ll); :lσ]
GaussianProcesses.num_params(mat::Mat72Ard) = length(mat.iℓ2) + 1

GaussianProcesses.cov(mat::Mat72Ard, r::Number) =
    (s = sqrt(7) * r; mat.σ2 * (1 + s + (2 / 5) * s^2 + (1 / 15) * s^3) * exp(-s))

GaussianProcesses.dk_dll(mat::Mat72Ard, r::Real, wdiffp::Real) =
    (s = sqrt(7) * r; (7 / 15) * mat.σ2 * wdiffp * (3 + 3 * s + s^2) * exp(-s))

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
function fit_boed_gps(prob::BOEDProblem, Z::AbstractMatrix, results::AbstractMatrix)
    Yfit = whiten_samples(prob.output_basis, results)   # N x k_R_out
    k_R_out = prob.output_basis.k_R

    gps = Vector{GaussianProcesses.GPE}(undef, k_R_out)
    ll0 = log.(vec(std(Z, dims = 2)) .+ 1e-8)   # same for every j; hoisted out of the loop below
    N = size(Z, 2)
    @info "fit_boed_gps: fitting $k_R_out GP(s) on N=$N cumulative points across $(Threads.nthreads()) thread(s)"
    Threads.@threads for j in 1:k_R_out
        t0 = time()
        yj = Yfit[:, j]
        sy = std(yj)
        sy = sy > 0 ? sy : 1.0
        kernel = Mat72Ard(ll0, log(sy))
        gp = GaussianProcesses.GPE(Z, yj, GaussianProcesses.MeanZero(), kernel, -2.0)
        try
            GaussianProcesses.optimize!(gp)
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
###############  Standalone, ForwardDiff-safe Matérn-7/2 kernel  #######
########################################################################
# Deliberately decoupled from GaussianProcesses.jl's own kernel-evaluation
# internals (Mat72Ard above) — that library's AD-compatibility for a
# ForwardDiff-driven optimization objective is unverified and shouldn't be
# relied on for the EIG gradient below. Used ONLY inside `eig_objective`,
# with the GP's currently-fitted hyperparameters held fixed (plain Float64
# constants, not part of the optimization).

# X1: d x M1, X2: d x M2, iℓ2: d-vector of inverse squared length scales ->
# M1 x M2 matrix of the ARD-weighted squared distance Σ_k iℓ2[k]·(X1[k,i]-X2[k,j])².
function ard_sqdist(X1::AbstractMatrix, X2::AbstractMatrix, iℓ2::AbstractVector)
    M1 = size(X1, 2)
    M2 = size(X2, 2)
    d2 = zeros(promote_type(eltype(X1), eltype(X2)), M1, M2)
    @inbounds for jj in 1:M2, ii in 1:M1
        s = zero(eltype(d2))
        for kk in eachindex(iℓ2)
            s += iℓ2[kk] * (X1[kk, ii] - X2[kk, jj])^2
        end
        d2[ii, jj] = s
    end
    return d2
end

# eps_r2 avoids the ForwardDiff NaN-gradient pitfall at r=0: every diagonal
# entry of a self-covariance matrix has r ≡ 0 (and d(r²)/dx ≡ 0 there too),
# so the naive sqrt(0)-chain-rule (which involves 1/(2·0)) yields NaN even
# though the true derivative of r itself is well-defined off-diagonal only;
# a tiny floor keeps every entry (on- and off-diagonal) smoothly
# differentiable without materially changing the covariance values.
function matern72_cov_matrix(X1::AbstractMatrix, X2::AbstractMatrix, iℓ2::AbstractVector, τ2::Real; eps_r2::Real = 1e-12)
    d2 = ard_sqdist(X1, X2, iℓ2)
    r = sqrt.(d2 .+ eps_r2)
    s = sqrt(7) .* r
    return τ2 .* (1 .+ s .+ (2 / 5) .* s .^ 2 .+ (1 / 15) .* s .^ 3) .* exp.(-s)
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
function eig_objective(
    x_cand_flat::AbstractVector, X_post::AbstractMatrix,
    hyperparams_by_dim, k_R_prior::Int, B::Int; jitter::Real = 1e-6,
)
    X_cand = reshape(x_cand_flat, k_R_prior, B)
    total = zero(eltype(x_cand_flat))
    for (iℓ2, τ2) in hyperparams_by_dim
        K_pp = matern72_cov_matrix(X_post, X_post, iℓ2, τ2)
        K_pc = matern72_cov_matrix(X_post, X_cand, iℓ2, τ2)
        K_cc = matern72_cov_matrix(X_cand, X_cand, iℓ2, τ2) + jitter * τ2 * I
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
# Algorithm 2) via Optim.jl's Fminbox(LBFGS()) with ForwardDiff-computed
# gradients, maximizing eig_objective (implemented here as minimizing its
# negation). Box bounds are ±bound_std standard deviations in the whitened
# Z-space, keeping candidates within the GP's trust region.
function optimize_batch(
    X_init::AbstractMatrix, X_post::AbstractMatrix, hyperparams_by_dim, k_R_prior::Int, B::Int;
    bound_std::Real, iters::Int, outer_iters::Int = 20, jitter::Real,
)
    lo = fill(-bound_std, k_R_prior * B)
    hi = fill(bound_std, k_R_prior * B)
    # Fminbox(LBFGS())'s own progress is otherwise invisible from the outside
    # until it returns — count objective evaluations (ForwardDiff calls this
    # many times per gradient, on top of Fminbox's outer barrier iterations)
    # so a slow-converging optimization is visibly still working, not hung.
    n_calls = Ref(0)
    t0 = time()
    function neg_eig(x)
        n_calls[] += 1
        if n_calls[] % 100 == 0
            @info "optimize_batch: $(n_calls[]) EIG objective evaluations so far" elapsed_s = round(time() - t0; digits = 2)
        end
        return -eig_objective(x, X_post, hyperparams_by_dim, k_R_prior, B; jitter = jitter)
    end
    # `iterations` caps EACH inner LBFGS solve at a fixed barrier weight;
    # `outer_iterations` caps Fminbox's OWN barrier-shrinking loop, which
    # Optim.jl otherwise defaults to 1000 with an outer_g_abstol=1e-8
    # convergence tolerance that's tight enough for this EIG objective to
    # frequently not be met quickly — silently multiplying total work by up
    # to 1000x. Capped much lower here; see experiment_config.jl's
    # eig_outer_iters comment.
    res = Optim.optimize(
        neg_eig, lo, hi, vec(X_init), Fminbox(LBFGS()),
        Optim.Options(iterations = iters, outer_iterations = outer_iters); autodiff = :forward,
    )
    converged = Optim.converged(res)
    outer_rounds = Optim.iterations(res)
    @info "optimize_batch: done" converged objective_evals = n_calls[] outer_rounds elapsed_s = round(time() - t0; digits = 2)
    if !converged && outer_rounds >= outer_iters
        @warn "optimize_batch: Fminbox's outer barrier loop hit its outer_iterations cap ($outer_iters) without converging — the returned EIG-optimized batch is likely under-optimized. Consider raising cfg.eig_outer_iters if this happens often." objective_evals = n_calls[] elapsed_s = round(time() - t0; digits = 2)
    end
    return reshape(Optim.minimizer(res), k_R_prior, B)
end
