# regime_diagnostics.jl — chaotic-regime / mixing diagnostics for the score-based (GFDT) run scripts.
#
# Included by score_based_lm and score_based_adam run_l63_*.jl / run_l96_*.jl (one shared copy).
# Base + LinearAlgebra + Statistics only. Uses only what the loop already holds (the state windows `Xs`,
# the GFDT result, θ); no forward evaluations, no RNG draws, no mutation of algorithm state.
# Compute these only on recorded iterations (`want(rec, it)`): they cost a few ACF sweeps.
#
# HYPOTHESIS UNDER TEST. The GFDT Jacobian J = (dφ/dm) · ∫₀^τmax C_AB(τ) dτ  (tapered; C_AB(τ) = ⟨A(x_τ) B(x_0)⟩,
# time averages over ONE window standing in for ensemble averages) is only valid if the window is a draw from a
# MIXING invariant measure with a smooth density. Written as checkable assumptions:
#
#   (M0) Chaotic, not collapsed/periodic: the window has non-trivial variance and a smooth (non-singular) measure.
#          → log10_varmin  (< -2: fixed point; is_collapsed_window → AD fallback)
#          → stein_norm    (‖Ξ‖, Stein defect of the fitted score: ≈ 0 smooth measure, → √nx score has learned nothing)
#          → b_ratio_*     (|mean B| / std B: Stein identity ⟨B⟩ = 0 for the conjugate variables)
#   (M1) Correlations DECAY inside the response cutoff: C_AB(τ) → 0 for τ ≲ τ_max, so truncating the integral loses
#        nothing. A limit cycle / slow regime keeps C_AB O(1) at τ_max.
#          → acf_tail      (max |ρ(τ)|, τ ∈ [τ_max/2, τ_max], state autocorrelation averaged over components)
#          → resp_tail     (‖R(τ_max) − R(τ_max/2)‖_F / ‖R(τ_max)‖_F: how much of the response integral lies in the
#                           upper half of the lag range; ≈ noise level if converged, O(1) if the integral is unfinished)
#   (M2) The window is LONG compared with the decorrelation time, so time averages ≈ ensemble averages (estimator
#        variance ∝ τ_int / T).
#          → tau_int, tau_int_ok (Sokal window closed inside the lag range), T_over_tau = T_total / tau_int
#   (M3) STATIONARY: the window does not drift (no transients, no regime switching within the window).
#          → drift_z       (|mean₁ − mean₂| of the energy E(t) between window halves, in units of its standard error)
#
# θ-location keys let the failures be mapped onto parameter space: theta_<i> (nu ≤ 10), theta_dist_init, and for
# the L96 forcings F_mean / F_min / F_max (the physical regime parameter).
#
# The only change to algorithm code is the optional `probe` callback in `gfdt_jacobian` (score_gfdt.jl), which is
# called with the raw (A, B) of each member and does not alter the result.

using LinearAlgebra
using Statistics

# Autocorrelation of the (member-pooled) state at lags 0:stride:L, averaged over components and over up to
# `max_members` members. Returns (lags_time, ρ). Each member is demeaned on its own window.
function _state_acf(Xs, dt, lag_stride, tau_max; max_members = 3)
    nt = size(Xs[1], 2)
    L = min(Int(round(tau_max / dt)), nt - 2)
    lags = 0:lag_stride:L
    acc = zeros(length(lags))
    nm = min(length(Xs), max_members)
    for m in 1:nm
        X = Xs[m] .- mean(Xs[m], dims = 2)
        v = vec(sum(abs2, X, dims = 2)) ./ nt
        ok = v .> 1e-30                                   # collapsed components carry no information
        any(ok) || continue
        Xo = X[ok, :]; vo = v[ok]
        for (i, l) in enumerate(lags)
            c = vec(sum(view(Xo, :, 1:(nt - l)) .* view(Xo, :, (1 + l):nt), dims = 2)) ./ (nt - l)
            acc[i] += mean(c ./ vo)
        end
    end
    return collect(lags) .* dt, acc ./ nm
end

# Sokal automatic-window integrated autocorrelation time (c = 5), in time units. τ_int(k) = dtl (½ + Σ_{j≤k} ρ_j).
# Returns (τ_int, closed::Bool); `closed` = the window closed inside the lag range (⇒ the ACF decayed in range).
function _tau_int(ρ, dtl; c = 5.0)
    s = 0.5
    for k in 2:length(ρ)
        s += ρ[k]
        s = max(s, 0.5)                                    # guards negative sums from noisy / oscillating tails
        if (k - 1) >= c * s
            return dtl * s, true
        end
    end
    return dtl * s, false
end

# Energy E(t) = mean over components of x², used for the stationarity check.
_energy(X) = vec(mean(abs2, X, dims = 1))

# State-window-only regime values (available on AD-fallback iterations too).
function regime_state_values(Xs, dt, lag_stride, tau_max)
    τ, ρ = _state_acf(Xs, dt, lag_stride, tau_max)
    dtl = lag_stride * dt
    τint, closed = _tau_int(ρ, dtl)
    nt = size(Xs[1], 2)
    T_total = length(Xs) * nt * dt
    half = findall(τ .>= tau_max / 2)
    acf_tail = isempty(half) ? NaN : maximum(abs, ρ[half])
    log10_varmin = log10(max(minimum(minimum(var(X, dims = 2)) for X in Xs), floatmin(Float64)))
    # drift: halves of the energy of each member, pooled; SE of a half mean = sqrt(var · 2τ_int / T_half)
    zs = Float64[]
    for X in Xs
        E = _energy(X); n2 = length(E) ÷ 2
        v = var(E)
        v > 1e-30 || continue
        se = sqrt(v * 2 * max(τint, dt) / (n2 * dt))
        push!(zs, abs(mean(E[1:n2]) - mean(E[(n2 + 1):(2n2)])) / (sqrt(2) * se))
    end
    return (
        log10_varmin = log10_varmin,
        tau_int      = τint,
        tau_int_ok   = closed,
        T_over_tau   = T_total / τint,
        acf_tail     = acf_tail,
        drift_z      = isempty(zs) ? NaN : maximum(zs),
    )
end

# Cumulative (flat-weight, untapered) response integral R(τ) = ∫₀^τ C_AB at `n_check` lag checkpoints, accumulated
# member by member through `gfdt_jacobian(...; probe = p)`. Mirrors response_integral_unnormalised (same lags, same
# n_eff columns) but keeps the partial integrals, so the convergence of the integral inside τ_max can be inspected.
mutable struct RespProbe
    dt::Float64
    lag_stride::Int
    tau_max::Float64
    n_check::Int
    num::Vector{Matrix{Float64}}
    den::Int
end
RespProbe(dt, lag_stride, tau_max; n_check = 8) = RespProbe(dt, lag_stride, tau_max, n_check, Matrix{Float64}[], 0)

function (p::RespProbe)(A::AbstractMatrix, B::AbstractMatrix)
    dA = A .- mean(A, dims = 2)
    dB = B .- mean(B, dims = 2)
    nt = size(dA, 2)
    L = min(Int(round(p.tau_max / p.dt)), nt - 2)
    lags = 0:p.lag_stride:L
    n_eff = nt - L
    dtl = p.lag_stride * p.dt
    nl = length(lags)
    checks = unique(round.(Int, range(1, nl, length = p.n_check + 1)[2:end]))   # lag-index checkpoints (> 1)
    checks = filter(>(1), checks)
    isempty(p.num) && (p.num = [zeros(size(dA, 1), size(dB, 1)) for _ in checks])
    S = zeros(size(dA, 1), n_eff)
    A0 = dA[:, 1:n_eff]
    @views for (i, l) in enumerate(lags)
        S .+= dA[:, (1 + l):(l + n_eff)]
        j = findfirst(==(i), checks)
        if j !== nothing
            Al = dA[:, (1 + l):(l + n_eff)]
            T = dtl .* (S .- 0.5 .* A0 .- 0.5 .* Al)         # trapezoid up to lag l
            p.num[j] .+= T * transpose(dB[:, 1:n_eff])
        end
    end
    p.den += n_eff
    return nothing
end

# resp_tail = ‖R(τ_max) − R(≈τ_max/2)‖_F / ‖R(τ_max)‖_F ; resp_over = max_j ‖R_j‖ / ‖R(τ_max)‖ (overshoot/oscillation).
function resp_values(p::Union{Nothing, RespProbe})
    (p === nothing || isempty(p.num) || p.den == 0) && return (resp_tail = NaN, resp_over = NaN)
    R = [n ./ p.den for n in p.num]
    nr = norm(R[end])
    nr > 0 || return (resp_tail = NaN, resp_over = NaN)
    mid = R[max(1, length(R) ÷ 2)]
    return (resp_tail = norm(R[end] - mid) / nr, resp_over = maximum(norm, R) / nr)
end

# Score-quality values from the GFDT result (all discarded by the run loop before).
gfdt_values(res) = res === nothing ? (stein_norm = NaN, b_ratio_max = NaN, b_ratio_med = NaN) :
    (stein_norm = res.stein_norm, b_ratio_max = maximum(res.b_mean_ratio), b_ratio_med = median(res.b_mean_ratio))

# θ-location values: θ itself when small, plus the prior-whitened distance from the starting point.
function theta_values(θ, θ_init, Lp)
    base = (theta_dist_init = norm(Lp \ (θ - θ_init)),)
    return length(θ) <= 10 ? (; theta = collect(θ), base...) : base
end

# Everything in one NamedTuple for `record!` / `finalize!`. `res` = the gfdt_jacobian result (nothing on AD iterations),
# `probe` = the RespProbe handed to gfdt_jacobian, `extra` = case-specific values (e.g. L96 forcing stats).
function regime_values(Xs, dt, cfg, θ, θ_init, Lp; res = nothing, probe = nothing, extra = (;))
    return (; regime_state_values(Xs, dt, cfg.lag_stride, cfg.tau_max)..., resp_values(probe)..., gfdt_values(res)...,
              theta_values(θ, θ_init, Lp)..., extra...)
end

# L63 only: where θ = (log ρ, log β) sits relative to the analytic regime boundary, no integration needed.
# The non-trivial fixed points C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1) are linearly stable below the Hopf value
#   ρ_H(β) = σ(σ+β+3)/(σ−β−1)   (β < σ−1; no Hopf, i.e. ρ_H = ∞, for β ≥ σ−1)
# and a window started there can either decay onto C± (collapsed, no invariant measure to differentiate) or ride a
# chaotic transient; the same θ then yields different regimes for different ICs. `fp_maxre` = max Re eig of the
# Jacobian at C±: < 0 means C± is attracting.
function l63_regime_values(θ; σ = 10.0)
    ρ, β = exp(θ[1]), exp(θ[2])
    ρH = β < σ - 1 ? σ * (σ + β + 3) / (σ - β - 1) : Inf
    fp_maxre = NaN
    if ρ > 1
        c = sqrt(β * (ρ - 1))
        Jf = [-σ σ 0.0; (ρ - (ρ - 1)) -1.0 -c; c c -β]
        fp_maxre = maximum(real, eigvals(Jf))
    end
    return (log10_rho_over_rhoH = log10(max(ρ / ρH, 1e-3)), fp_maxre = fp_maxre)
end

# L96: physical regime parameter, the forcing profile at θ. `forcing_fn(θ)` returns the per-site forcing vector.
function l96_regime_values(forcing_vec)
    return (F_mean = mean(forcing_vec), F_min = minimum(forcing_vec), F_max = maximum(forcing_vec))
end

describe_regime!(rec) = describe!(rec,
    "log10_rho_over_rhoH" => "L63: log10(ρ / ρ_H(β)), ρ_H = σ(σ+β+3)/(σ−β−1) the Hopf value (floored at −3): < 0 means the fixed points C± are stable and chaos is at best transient / metastable",
    "fp_maxre"         => "L63: max Re eig of the Jacobian at C±; < 0 = C± attracting (windows can collapse onto a fixed point, different ICs give different regimes)",
    "theta"            => "(nu ≤ 10) the parameter vector θ at the record, theta_<i>; the iterate's location in parameter space",
    "theta_dist_init"  => "prior-whitened distance ‖Lp⁻¹(θ − θ_init)‖ travelled from the starting point (region change)",
    "F_mean"           => "L96: mean forcing over sites at θ (physical regime parameter; flux: NN output on the sample range)",
    "F_min"            => "L96: min forcing over sites",
    "F_max"            => "L96: max forcing over sites",
    "log10_varmin"     => "(M0) log10 of the smallest state-component temporal variance in the window; < -2 = collapsed to a fixed point",
    "stein_norm"       => "(M0) Stein defect ‖I − (−E[x sᵀ])‖ of the fitted score on the last member: ≈ 0 smooth measure, → √nx the score learned nothing (NaN on AD iterations)",
    "b_ratio_max"      => "(M0) max_j |mean B_j| / std B_j: the Stein identity ⟨B⟩ = 0 for the conjugate variables; ≫ 0.1 means the score / mixing assumption fails",
    "b_ratio_med"      => "(M0) median_j |mean B_j| / std B_j",
    "acf_tail"         => "(M1) max |ρ(τ)| for τ ∈ [τ_max/2, τ_max] of the state autocorrelation (component- and member-averaged): ≈ 0 mixing within the cutoff, O(1) periodic / slow",
    "resp_tail"        => "(M1) ‖R(τ_max) − R(τ_max/2)‖/‖R(τ_max)‖ for the untapered cumulative response integral: ≈ noise if converged, O(1) if unfinished at the cutoff (NaN on AD iterations)",
    "resp_over"        => "(M1) max_τ ‖R(τ)‖ / ‖R(τ_max)‖ ≥ 1: > 1 means the response integral overshoots / oscillates before settling",
    "tau_int"          => "(M2) Sokal integrated autocorrelation time of the state (model time units)",
    "tau_int_ok"       => "(M2) 1 if the Sokal window closed inside the lag range (ACF decayed), 0 if not (τ_int is only a lower bound)",
    "T_over_tau"       => "(M2) total window length (all members) / τ_int: ≈ number of independent samples × 2",
    "drift_z"          => "(M3) |mean₁ − mean₂| of the energy between window halves in standard errors (max over members); ≫ 3 = non-stationary window",
)
