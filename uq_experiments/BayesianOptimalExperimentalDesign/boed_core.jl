# BayesianOptimalExperimentalDesign — shared GBOED core, implementing Algorithm 2 of
# Holthuijzen et al. (2026, arXiv:2508.13071). Included by calibrate_l63.jl/calibrate_l96.jl.

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

# Fixes a GaussianProcesses.jl 0.12 / PDMats.jl `ldiv!` ambiguity that would
# otherwise silently leave every GP at its un-optimized initial hyperparameters.
LinearAlgebra.ldiv!(cK::PDMats.PDMat, x::AbstractVecOrMat) = LinearAlgebra.ldiv!(cK.chol, x)

# Avoids oversubscribing BLAS threads on top of fit_boed_gps/boed_loglik's own Julia-thread parallelism.
Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)

########################################################################
###############  Progress/timing diagnostics  ##########################
########################################################################
# Wraps a long-running stage in start/finish @info timing so it's visible from the outside that it's still working, not hung.
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
# Fixed per-cell whitening setup built once before the acquisition loop; also carries `forward_transform` to decode candidates back to raw θ.
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

# Inverse of to_prior_whitened (discarded modes reconstruct at the prior mean); decodes whitened candidates before the forward map.
function from_prior_whitened(prob::BOEDProblem, Z::AbstractMatrix)
    u = unwhiten_samples(prob.prior_basis, Matrix(Z'))'   # D x M, in "prior-native" space
    theta_native = u .+ prob.prior_mean
    return prob.forward_transform.(theta_native)
end

########################################################################
###############  LHS in the truncated whitened space  ##################
########################################################################
# PCA-whitened coordinates are exactly standard normal, so this one sampler serves both the initial LHS design and the ST-MCMC prior sampler.
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
# Refit on the full cumulative dataset every iteration, unlike HistoryMatching's per-wave WaveGPs.
struct BOEDGPs
    gps::Vector{GaussianProcesses.GPE}   # one per whitened + truncated output mode
end

# Z: k_R_prior x N (cumulative prior-whitened inputs); results: N x n_out (cumulative raw outputs).
# Kernel/noise bounds are relative to each GP's own data-driven scale, avoiding degenerate near-zero-noise fits ST-MCMC's resampling can collapse onto.
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
        # Centered at the noise bounds' midpoint so the start is always strictly interior, as Fminbox requires.
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

# z (single candidate) -> (mu, var), both length k_R_out, in output_basis's whitened coordinates.
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
# Paper Eq. 5's GP-marginal log-likelihood (Γ_obs≈I in the whitened output
# space); keeps the normalizing log(2π·v) term since ST-MCMC needs a properly
# normalized density, unlike HistoryMatching's pure-threshold implausibility.
function boed_loglik(prob::BOEDProblem, gps::BOEDGPs, z::AbstractVector)
    mu, var = predict_boed(gps, z)
    ll = 0.0
    for j in eachindex(mu)
        v = 1.0 + var[j]
        ll += -0.5 * (mu[j] - prob.y_whitened[j])^2 / v - 0.5 * log(2 * pi * v)
    end
    return ll
end

# Accepts a bare Real too: TransitionalMCMC.jl's dims==1 code path (used when k_R_prior==1) mutates scalars rather than length-1 vectors.
function boed_loglik(prob::BOEDProblem, gps::BOEDGPs, z::Real)
    boed_loglik(prob, gps, [z])
end

# Draws posterior samples via TransitionalMCMC.jl's `tmcmc`; transposes at the boundary since tmcmc uses samples-as-rows internally.
# `burnin`/`thin` are exposed below tmcmc's own 20/3 defaults since they directly multiply its (effectively serial) per-stage cost.
function run_tmcmc(prob::BOEDProblem, gps::BOEDGPs, n_samples::Int, rng::AbstractRNG; burnin::Int = 5, thin::Int = 1)
    k_R_prior = prob.prior_basis.k_R
    # Reports progress within a tempering stage, since tmcmc's own logging is only per-stage.
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
# Decoupled from GaussianProcesses.jl's own kernel internals (unverified AD-compatibility); uses the GP's fitted hyperparameters as fixed constants.

# X1: d x M1, X2: d x M2, iℓ2: inverse squared length scales -> M1 x M2 ARD-weighted squared distance.
# Computed via the expanded-norm identity + BLAS matmuls (Zygote can't differentiate the mutating-loop form); clamped at 0 against diagonal cancellation noise.
function ard_sqdist(X1::AbstractMatrix, X2::AbstractMatrix, iℓ2::AbstractVector)
    X1w = X1 .* iℓ2
    n1 = vec(sum(X1w .* X1; dims = 1))       # M1, weighted squared norm of each X1 column
    n2 = vec(sum(X2 .* (iℓ2 .* X2); dims = 1)) # M2, weighted squared norm of each X2 column
    cross = X1w' * X2                          # M1 x M2, Σ_k iℓ2[k]·X1[k,i]·X2[k,j]
    return max.(n1 .+ n2' .- 2 .* cross, 0)
end

# SE-ARD covariance, matching GaussianProcesses.jl's own SEArd convention (τ²·exp(-d²/2)).
function se_cov_matrix(X1::AbstractMatrix, X2::AbstractMatrix, iℓ2::AbstractVector, τ2::Real)
    d2 = ard_sqdist(X1, X2, iℓ2)
    return τ2 .* exp.(-0.5 .* d2)
end

########################################################################
###############  EIG objective (paper Eq. 8) & joint-batch optimization #
########################################################################

extract_hyperparams(gps::BOEDGPs) = [(copy(gp.kernel.iℓ2), gp.kernel.σ2) for gp in gps.gps]

# X_cand_flat: vec(k_R_prior x B) batch being optimized; X_post: the ST-MCMC posterior "X'" of paper Eq. 8; hyperparams_by_dim is held fixed.
#   EIG(X_cand) = (1/k_R_out) Σ_j 0.5·log(det(K_pp^(j)) / det(Σ'^(j))), Σ'^(j) = K_pp^(j) - K_pc^(j)·(K_cc^(j))⁻¹·K_cp^(j)
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

# Draws the starting batch (`cfg.batch_init_strategy`): :posterior_subsample subsamples B points from the current posterior X',
# :fresh_lhs draws fresh in the truncated whitened space.
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

# Runs Fminbox(LBFGS()) with Zygote reverse-mode gradients, and enforces `call_limit` itself: Optim's own
# f_calls_limit/g_calls_limit are enforced per-inner-solve under Fminbox, so they don't cumulate across outer rounds.
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
    # An explicit OnceDifferentiable (rather than `autodiff = ...`) since that
    # keyword only recognizes `:finite`/`:forward` in the installed Optim version.
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
# Algorithm 2) to maximize eig_objective, in whitened space bounded by ±bound_std.
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

# PROTOTYPE, not wired into calibrate_l63.jl/calibrate_l96.jl: greedily selects the batch one point at a time
# (previous selections held fixed), collapsing the per-step decision space from k_R_prior*B down to k_R_prior.

# Optimizes one new point to maximize the EIG of appending it to the fixed `X_fixed` columns; shared by the greedy and hybrid prototypes below.
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

# PROTOTYPE: greedily selects points until one step's cost exceeds `greedy_eval_threshold`, then jointly optimizes
# the remaining slots at once — measured roughly break-even with pure joint optimization, not a clear win.
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
