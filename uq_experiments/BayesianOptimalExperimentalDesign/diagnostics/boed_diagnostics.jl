# boed_diagnostics.jl — per-iteration diagnostics for the GBOED family (:tmcmc_eig, :tmcmc, :iekf_eig, :iekf).
#
# Included by boed_core.jl (AFTER its `using` lines: it calls GaussianProcesses.predict_y and uses boed_loglik / eig_objective /
# extract_hyperparams / ball_radius, which boed_core.jl defines). Adds no packages. Includes the shared recorder from common/.
#
# Rules this file follows (algorithm-diagnoser § Rules):
#   - no forward-model evaluations (GP predictions on the fitted GPs are fine), no RNG draws: every "subsample" is a fixed, evenly-spaced
#     column selection, and the EIG values are evaluated with eig_objective (a pure function) against that fixed subsample;
#   - nothing here mutates algorithm state; each function reads its inputs and returns new values;
#   - DIAG=0 turns recording off (the recorder is disabled and the calibrate scripts skip every call below).
#
# One record per GBOED iteration k (all iterations are recorded: max_iters <= 500). A row mixes three kinds of keys:
#   state_*   emulator + posterior at iteration k (GP_k, X_post_k)
#   sampler   how X_post_k was produced (TMCMC log-evidence / IEKF misfit trajectory)
#   acq       the batch that was ADDED to make GP_k, judged against the iteration k-1 posterior it was chosen from (NaN at k = 1)
# Whitened coordinates throughout: prior N(0, I_k), so a "spread" of 1 is the prior width.

include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_recorder.jl"))

const BOED_DIAG_N_SUB = 50   # fixed posterior subsample size for GP-prediction / EIG diagnostics

# Evenly spaced column indices (no RNG): n of the M columns.
_diag_idx(M::Int, n::Int) = M <= n ? collect(1:M) : unique(round.(Int, range(1, M; length = n)))

function boed_new_recorder(cfg)
    rec = DiagRecorder(cfg.max_iters)
    describe!(rec,
        # --- state: emulator at iteration k ---
        "n_train"            => "number of cumulative forward evaluations behind GP_k",
        "ll_truth"           => "GP-surrogate log-likelihood (boed_loglik) at the truth projected into the truncated whitened input space; NaN for flux (no identifiable truth)",
        "misfit_truth"       => "mean over output modes of (mu_j(z*) - y_j)^2 / (1 + var_j(z*)) at the projected truth; ≈ 1 when the emulator reproduces the observation at the truth",
        "gp_std_truth"       => "sqrt(mean_j var_j(z*)): GP predictive std at the truth",
        "ll_post_med"        => "median over a fixed 50-point posterior subsample of the surrogate log-likelihood",
        "misfit_post"        => "mean over the subsample and output modes of (mu_j - y_j)^2/(1+var_j)",
        "gp_std_post"        => "mean over the subsample of sqrt(mean_j var_j): GP predictive std inside the posterior",
        "gp_ls_med"          => "median over (output mode, input dim) of the fitted SE-ARD lengthscale (whitened units; prior std = 1)",
        "gp_ls_min"          => "smallest fitted lengthscale",
        "gp_ls_bound_frac"   => "fraction of (mode, dim) lengthscales within 1% of their optimizer bound (bounds = std(Z) * 10^±2): the emulator cannot resolve that direction",
        "gp_noise_ratio_med" => "median over modes of fitted noise std / std(y_j); → 1 means the GP treats the data as noise (noise at its upper bound)",
        "gp_sig_med"         => "median over modes of fitted signal std / std(y_j)",
        "gp_mll_sum"         => "sum over output modes of the GP log target (marginal log-likelihood)",
        "loo_msse_med"       => "median over modes of the leave-one-out mean standardised squared error; ≫ 1 = overconfident emulator",
        "loo_cov95_med"      => "median over modes of the LOO fraction of points inside the 95% interval (healthy ≈ 0.95)",
        # --- state: posterior at iteration k ---
        "post_spread_w"      => "sqrt(mean variance of the posterior in whitened coordinates); prior = 1",
        "post_mean_rms"      => "rms of the posterior mean in whitened coordinates (distance from the prior mean per dimension)",
        "post_eff_rank"      => "participation ratio (sum λ)^2 / sum λ^2 of the posterior covariance eigenvalues",
        "post_uniq_frac"     => "fraction of distinct posterior particles (sampler degeneracy: resampling collapse)",
        "truth_dist_over_spread" => "‖z* - mean‖ / (sqrt(k) * post_spread_w): ≫ 1 means the truth sits outside the posterior; NaN for flux",
        # --- sampler ---
        "tmcmc_log_ev"       => "TransitionalMCMC log-evidence (NaN for IEKF variants); jumps between iterations flag an unstable sampler",
        "tmcmc_loglik_calls" => "number of log-likelihood evaluations ST-MCMC used (≈ stages × particles × (burn-in + 1)); proxy for the number of tempering stages",
        "tmcmc_secs"         => "wall seconds in run_tmcmc",
        "iekf_misfit_first"  => "IEKF: mean over members of 0.5‖y - G(u)‖^2 on the GP mean, first step (NaN for TMCMC variants)",
        "iekf_misfit_last"   => "IEKF: the same at the last step",
        "iekf_misfit_flat"   => "IEKF: |m_last - m_prev| / m_prev over the last step; ≈ 0 = misfit has plateaued",
        "iekf_spread_ratio"  => "IEKF: ensemble spread at the last step / at the first step (prior-whitened)",
        "iekf_secs"          => "wall seconds in run_iekf",
        # --- acquisition (batch added at iteration k, judged against the k-1 posterior) ---
        "eig_batch"          => "EIG of the acquired batch against a FIXED 50-point subsample of the k-1 posterior (same hyperparameters as the optimiser)",
        "eig_init"           => "EIG of the batch the optimiser started from (= eig_batch for non-EIG variants)",
        "eig_postdraw"       => "EIG of a fixed evenly-spaced batch of posterior draws (the no-EIG reference batch)",
        "eig_priordesign"    => "EIG of the first LHS design (prior-space-filling control)",
        "eig_opt_final"      => "the optimiser's own final -objective (random 50-point subsample, so it differs slightly from eig_batch)",
        "eig_opt_calls"      => "EIG objective evaluations spent by the optimiser",
        "eig_opt_secs"       => "wall seconds in the EIG optimiser (the cost of the acquisition step)",
        "eig_converged"      => "1 if Optim reported convergence",
        "eig_hit_limit"      => "1 if the call limit stopped the optimiser first",
        "cand_r_mean"        => "mean ‖z‖ of the acquired batch (the prior shell is at sqrt(k))",
        "post_r_mean"        => "mean ‖z‖ of the k-1 posterior subsample",
        "cand_r_over_ball"   => "max ‖z‖ of the batch / ball radius R(eig_ball_quantile); → 1 = pinned to the search-region boundary",
        "cand_at_ball_frac"  => "fraction of batch points with ‖z‖ > 0.98 R",
        "cand_mahal_med"     => "median of ‖L⁻¹(z - μ_post)‖/sqrt(k) over the batch (posterior Mahalanobis radius; ≈ 1 for posterior draws, > 1 = in the tails)",
        "post_mahal_med"     => "the same for the posterior subsample itself (baseline)",
        "cand_move"          => "mean ‖z - z_init‖ / sqrt(k): how far EIG moved the batch from its starting posterior draws (0 for non-EIG)",
        "cand_mindist_med"   => "median over the batch of the distance to the nearest earlier training point / sqrt(k)",
        "post_mindist_med"   => "the same for the posterior subsample (baseline: how far the posterior is from existing data)",
        "batch_mindist_med"  => "median nearest-neighbour distance within the batch / sqrt(k): design collapse if ≈ 0",
        "cand_gp_std"        => "mean GP predictive std (GP_{k-1}) at the batch",
        "post_gp_std"        => "mean GP predictive std (GP_{k-1}) at the posterior subsample (baseline)",
    )
    return rec
end

########################################################################
###############  GP prediction helpers  ################################
########################################################################
# mu, var: k_R_out x M, in whitened output coordinates. Falls back to NaN on a failed prediction.
function _diag_gp_predict(gps, Zc::AbstractMatrix)
    M = size(Zc, 2)
    n = length(gps.gps)
    mu = fill(NaN, n, M)
    v = fill(NaN, n, M)
    for j in 1:n
        try
            m, s = GaussianProcesses.predict_y(gps.gps[j], Matrix{Float64}(Zc))
            mu[j, :] = m
            v[j, :] = s
        catch
        end
    end
    return mu, v
end

# Per-column surrogate log-likelihood (the same form as boed_loglik) from predictions.
function _diag_ll(mu, v, y_w)
    vv = 1 .+ v
    return vec(sum(-0.5 .* (mu .- y_w) .^ 2 ./ vv .- 0.5 .* log.(2π .* vv); dims = 1))
end

########################################################################
###############  state diagnostics (emulator + posterior at k)  ########
########################################################################
function _diag_gp_summary(gps, Z)
    k_in = size(Z, 1)
    ll0 = log.(vec(std(Z; dims = 2)) .+ 1e-8)
    lo, hi = ll0 .- 2 * log(10), ll0 .+ 2 * log(10)   # fit_boed_gps default gp_lengthscale_log10_range = 2
    ls, nr, sg, mll, msse, cov95 = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    nbound, ntot = 0, 0
    for gp in gps.gps
        llog = -0.5 .* log.(gp.kernel.iℓ2)
        append!(ls, exp.(llog))
        for d in 1:k_in
            ntot += 1
            tol = 0.01 * (hi[d] - lo[d])
            (llog[d] - lo[d] < tol || hi[d] - llog[d] < tol) && (nbound += 1)
        end
        sy = std(gp.y)
        sy = sy > 0 ? sy : 1.0
        push!(nr, exp(gp.logNoise.value) / sy)
        push!(sg, sqrt(gp.kernel.σ2) / sy)
        push!(mll, gp.target)
        if length(gp.y) <= 1500
            try
                Kinv_d = diag(inv(gp.cK))
                z2 = (gp.alpha .^ 2) ./ Kinv_d          # LOO standardised squared error
                push!(msse, mean(z2))
                push!(cov95, mean(z2 .< 1.959964^2))
            catch
            end
        end
    end
    med(x) = isempty(x) ? NaN : median(x)
    return (;
        gp_ls_med = med(ls), gp_ls_min = minimum(ls), gp_ls_bound_frac = nbound / max(ntot, 1),
        gp_noise_ratio_med = med(nr), gp_sig_med = med(sg), gp_mll_sum = sum(mll),
        loo_msse_med = med(msse), loo_cov95_med = med(cov95),
    )
end

function _diag_post_summary(X_post, z_true)
    k, n = size(X_post)
    mu = vec(mean(X_post; dims = 2))
    spread = sqrt(mean(var(X_post; dims = 2)))
    ev = try
        eigvals(Symmetric(cov(X_post; dims = 2)))
    catch
        fill(NaN, k)
    end
    ev = max.(ev, 0)
    uniq = length(Set(hash(view(X_post, :, i)) for i in 1:n)) / n
    tds = z_true === nothing ? NaN : norm(z_true .- mu) / (sqrt(k) * spread)
    return (;
        post_spread_w = spread, post_mean_rms = norm(mu) / sqrt(k), post_eff_rank = sum(ev)^2 / max(sum(abs2, ev), eps()),
        post_uniq_frac = uniq, truth_dist_over_spread = tds,
    )
end

"""
    boed_state_diag(prob, gps, Z, X_post, z_true) -> NamedTuple

Emulator and posterior summaries at iteration k. `Z`: k_in x N_train (cumulative design); `z_true`: projected truth in whitened input
coordinates or `nothing`. GP predictions only (no forward evaluations, no RNG).
"""
function boed_state_diag(prob, gps, Z::AbstractMatrix, X_post::AbstractMatrix, z_true)
    y_w = prob.y_whitened
    idx = _diag_idx(size(X_post, 2), BOED_DIAG_N_SUB)
    Xs = X_post[:, idx]
    mu, v = _diag_gp_predict(gps, Xs)
    llp = _diag_ll(mu, v, y_w)
    truth = if z_true === nothing
        (; ll_truth = NaN, misfit_truth = NaN, gp_std_truth = NaN)
    else
        mt, vt = _diag_gp_predict(gps, reshape(Float64.(z_true), :, 1))
        (; ll_truth = _diag_ll(mt, vt, y_w)[1], misfit_truth = mean((mt .- y_w) .^ 2 ./ (1 .+ vt)), gp_std_truth = sqrt(mean(vt)))
    end
    return (;
        n_train = size(Z, 2),
        truth...,
        ll_post_med = median(filter(isfinite, llp)),
        misfit_post = mean((mu .- y_w) .^ 2 ./ (1 .+ v)),
        gp_std_post = mean(sqrt.(vec(mean(v; dims = 1)))),
        _diag_gp_summary(gps, Z)...,
        _diag_post_summary(X_post, z_true)...,
    )
end

########################################################################
###############  sampler diagnostics  ##################################
########################################################################
# `sdiag`: the Dict filled by run_tmcmc / run_iekf through their `diag` keyword (empty for the other sampler).
function boed_sampler_diag(sdiag::AbstractDict)
    g(k) = Float64(get(sdiag, k, NaN))
    return (;
        tmcmc_log_ev = g("tmcmc_log_ev"), tmcmc_loglik_calls = g("tmcmc_loglik_calls"), tmcmc_secs = g("tmcmc_secs"),
        iekf_misfit_first = g("iekf_misfit_first"), iekf_misfit_last = g("iekf_misfit_last"), iekf_misfit_flat = g("iekf_misfit_flat"),
        iekf_spread_ratio = g("iekf_spread_ratio"), iekf_secs = g("iekf_secs"),
    )
end

########################################################################
###############  acquisition diagnostics  ##############################
########################################################################
function _nn_dist(A::AbstractMatrix, B::AbstractMatrix; skip_self::Bool = false)
    # distance from each column of A to its nearest column of B (skip_self: A === B, ignore the zero self-distance)
    out = fill(Inf, size(A, 2))
    for i in 1:size(A, 2), j in 1:size(B, 2)
        skip_self && i == j && continue
        d = norm(view(A, :, i) .- view(B, :, j))
        d < out[i] && (out[i] = d)
    end
    return out
end

"""
    boed_acq_diag(prob, gps_prev, X_post_prev, Z_prev, X_cand, raw, cfg) -> NamedTuple

Judges the batch `X_cand` acquired at iteration k against the state it was chosen from (GP_{k-1}, the k-1 posterior, the k-1 design).
`raw` is the Dict that `acquire_batch(...; diag = raw)` filled ("X_cand0", "eig_*"); empty for the no-EIG variants.
"""
function boed_acq_diag(prob, gps_prev, X_post_prev::AbstractMatrix, Z_prev::AbstractMatrix, X_cand::AbstractMatrix, raw::AbstractDict, cfg)
    k, B = size(X_cand)
    sk = sqrt(k)
    idx = _diag_idx(size(X_post_prev, 2), BOED_DIAG_N_SUB)
    Xs = X_post_prev[:, idx]
    hyp = extract_hyperparams(gps_prev)
    X0 = get(raw, "X_cand0", X_cand)
    # fixed no-RNG reference batches
    idxB = _diag_idx(size(X_post_prev, 2), B)
    Xdraw = X_post_prev[:, idxB]
    Xprior = size(Z_prev, 2) >= B ? Z_prev[:, 1:B] : Z_prev
    eig(X) = size(X, 2) == B ? eig_objective(vec(Matrix(X)), Xs, hyp, k, B; jitter = cfg.eig_jitter) : NaN
    # radius / ball
    R = ball_radius(k, cfg.eig_ball_quantile)
    rc = vec(sqrt.(sum(abs2, X_cand; dims = 1)))
    rp = vec(sqrt.(sum(abs2, Xs; dims = 1)))
    # posterior Mahalanobis radius
    mu_p = vec(mean(X_post_prev; dims = 2))
    mahal = try
        L = cholesky(Symmetric(cov(X_post_prev; dims = 2)) + 1e-6 * I).L
        f(X) = vec(sqrt.(sum(abs2, L \ (X .- mu_p); dims = 1))) ./ sk
        (; c = median(f(X_cand)), p = median(f(Xs)))
    catch
        (; c = NaN, p = NaN)
    end
    # nearest-neighbour distances
    cand_nn = _nn_dist(X_cand, Z_prev) ./ sk
    post_nn = _nn_dist(Xs, Z_prev) ./ sk
    batch_nn = B > 1 ? _nn_dist(X_cand, X_cand; skip_self = true) ./ sk : [NaN]
    # GP std at candidates vs posterior
    mc, vc = _diag_gp_predict(gps_prev, X_cand)
    mp, vp = _diag_gp_predict(gps_prev, Xs)
    return (;
        eig_batch = eig(X_cand), eig_init = eig(X0), eig_postdraw = eig(Xdraw), eig_priordesign = eig(Xprior),
        eig_opt_final = Float64(get(raw, "eig_opt_final", NaN)), eig_opt_calls = Float64(get(raw, "eig_opt_calls", NaN)),
        eig_converged = Float64(get(raw, "eig_converged", NaN)), eig_hit_limit = Float64(get(raw, "eig_hit_limit", NaN)), eig_opt_secs = Float64(get(raw, "eig_opt_secs", NaN)),
        cand_r_mean = mean(rc), post_r_mean = mean(rp), cand_r_over_ball = maximum(rc) / R, cand_at_ball_frac = mean(rc .> 0.98 * R),
        cand_mahal_med = mahal.c, post_mahal_med = mahal.p,
        cand_move = mean(vec(sqrt.(sum(abs2, X_cand .- X0; dims = 1)))) / sk,
        cand_mindist_med = median(cand_nn), post_mindist_med = median(post_nn), batch_mindist_med = median(batch_nn),
        cand_gp_std = mean(sqrt.(vec(mean(vc; dims = 1)))), post_gp_std = mean(sqrt.(vec(mean(vp; dims = 1)))),
    )
end

# The acquisition keys for iteration 1 (no acquisition yet).
boed_acq_nan() = (;
    eig_batch = NaN, eig_init = NaN, eig_postdraw = NaN, eig_priordesign = NaN, eig_opt_final = NaN, eig_opt_calls = NaN, eig_converged = NaN,
    eig_hit_limit = NaN, eig_opt_secs = NaN, cand_r_mean = NaN, post_r_mean = NaN, cand_r_over_ball = NaN, cand_at_ball_frac = NaN, cand_mahal_med = NaN, post_mahal_med = NaN,
    cand_move = NaN, cand_mindist_med = NaN, post_mindist_med = NaN, batch_mindist_med = NaN, cand_gp_std = NaN, post_gp_std = NaN,
)

########################################################################
###############  objects saved next to the scalars  ####################
########################################################################
# Everything an offline script needs to rebuild the emulator at any iteration and re-run the samplers on it (no forward runs):
# the cumulative design and whitened outputs, the fitted hyperparameters per iteration, a fixed posterior subsample per iteration,
# and the whitening bases. ~(k_in + k_out) * N_total * 4 B + small; see the storage note in diagnostics/README.md.
mutable struct BOEDDiagStore
    gp_hyper_by_k::Dict{Int, Matrix{Float64}}     # k_R_out x (k_in + 2): [iℓ2..., σ2, noise variance]
    X_post_sub_by_k::Dict{Int, Matrix{Float32}}
    X_cand0_by_k::Dict{Int, Matrix{Float32}}
end
BOEDDiagStore() = BOEDDiagStore(Dict{Int, Matrix{Float64}}(), Dict{Int, Matrix{Float32}}(), Dict{Int, Matrix{Float32}}())

function boed_store_iteration!(store::BOEDDiagStore, k::Int, gps, X_post, X_cand, raw::AbstractDict)
    store.gp_hyper_by_k[k] = reduce(vcat, [hcat(reshape(gp.kernel.iℓ2, 1, :), gp.kernel.σ2, exp(2 * gp.logNoise.value)) for gp in gps.gps])
    store.X_post_sub_by_k[k] = Float32.(X_post[:, _diag_idx(size(X_post, 2), 100)])
    # the acquired batch itself is Z[:, (k-1)N+1 : kN]; only the optimiser's starting batch (EIG variants) is not stored elsewhere
    X_cand !== nothing && haskey(raw, "X_cand0") && (store.X_cand0_by_k[k] = Float32.(raw["X_cand0"]))
    return store
end

"""
    boed_diagnostics_dict(rec, store, prob, Z, results, z_true, cfg) -> Dict{String,Any}

The value saved under the cell file's "diagnostics" key: the recorder's per-iteration scalars plus the objects above.
"""
function boed_diagnostics_dict(rec, store::BOEDDiagStore, prob, Z, results, z_true, cfg)
    d = diagnostics_dict(rec)
    rec.enabled || return d
    d["Z"] = Float32.(Z)
    d["Yw"] = Float32.(whiten_samples(prob.output_basis, results))
    d["y_whitened"] = Float64.(prob.y_whitened)
    d["z_true"] = z_true === nothing ? Float64[] : Float64.(z_true)
    d["gp_hyper_by_k"] = store.gp_hyper_by_k
    d["X_post_sub_by_k"] = store.X_post_sub_by_k
    d["X_cand0_by_k"] = store.X_cand0_by_k
    d["prior_V"] = prob.prior_basis.V
    d["prior_lambda"] = prob.prior_basis.λ
    d["prior_mean"] = prob.prior_mean
    d["output_V"] = prob.output_basis.V
    d["output_lambda"] = prob.output_basis.λ
    d["variant"] = String(cfg.variant)
    return d
end

# One row for iteration k. `acq` = boed_acq_nan() at k = 1, else boed_acq_diag(...) computed BEFORE gps/X_post/Z were updated.
# `X_cand`/`raw` (nothing/empty at k = 1) only feed the store.
function boed_record_iteration!(rec, store, k::Int, prob, gps, Z, X_post, z_true, sdiag, acq, X_cand, raw)
    rec.enabled || return nothing
    record!(rec, k; boed_state_diag(prob, gps, Z, X_post, z_true)..., boed_sampler_diag(sdiag)..., acq...)
    boed_store_iteration!(store, k, gps, X_post, X_cand, raw)
    return nothing
end

# Truth in raw parameter space for each case (nothing for flux: NN weights are not identifiable). Used only to project into z-space.
boed_truth_theta_l63() = [28.0, 8.0 / 3.0]
function boed_truth_theta_l96(force_case::AbstractString)
    force_case == "const-force" && return [8.0]
    if force_case == "vec-force"
        nx = 40
        return 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
    end
    return nothing
end
