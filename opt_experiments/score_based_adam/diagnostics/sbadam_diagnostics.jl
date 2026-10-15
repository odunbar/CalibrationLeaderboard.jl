# sbadam_diagnostics.jl — derivative-quality diagnostics for the score_based_adam run scripts.
#
# Copy of opt_experiments/adam/diagnostics/adam_diagnostics.jl, adapted because here J̃ (GFDT, or AD on
# collapsed windows) is built AFTER the convergence check: observe! is called once J̃ is known, and
# converged! is called without J̃ (only rmse / dist_truth / statvar are recorded on the final row).
# observe! also takes method-specific point values (`extras`, e.g. cumulative AD-fallback count).
#
# Included by run_l63_sbadam.jl and run_l96_sbadam.jl. Uses only quantities the loop
# already computes (J̃, r̃, g, m̂, v̂, θ, and the forward output y - r_sum/N_ens):
# no forward-model evaluations, no RNG draws, no mutation of algorithm state.
# DIAG=0 disables recording (see diag_recorder.jl).
#
#   d = AdamDiag(cfg.N_iter, θ_true, prior_cov)          # before the loop
#   adam_diag_observe!(d, it, θ, J̃, r̃, statvar; extras...) # once J̃ is built (after the Jacobian block)
#   adam_diag_converged!(d, it, θ, r̃, statvar)           # inside the convergence branch, before break (no J̃ yet)
#   adam_diag_step!(d, θ_pre, θ, g, m̂, v̂, J̃, α, ε)       # after the Adam update
#   adam_diag_finish!(d, cfg.N_iter)                     # after the loop
#   diagnostics_dict(d.rec)                              # → "diagnostics" in the result JLD2

using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_recorder.jl"))

mutable struct AdamDiag
    rec::DiagRecorder
    θ_true::Union{Nothing, Vector{Float64}}   # nothing: no usable truth (flux)
    Lp::LowerTriangular{Float64, Matrix{Float64}}
    g_prev::Union{Nothing, Vector{Float64}}
    r_prev::Union{Nothing, Vector{Float64}}    # r̃ at the previous iteration
    dr_pred::Union{Nothing, Vector{Float64}}   # -J̃ Δθ predicted at the previous iteration
end

function AdamDiag(N_iter, θ_true, prior_cov)
    rec = DiagRecorder(N_iter)
    describe!(rec,
        "rmse"             => "whitened RMSE at θ before the step (single IC sample)",
        "rmse_wmin"        => "window minimum of the whitened RMSE",
        "log10_J"          => "log10 Frobenius norm of the whitened Jacobian J̃ (ForwardDiff through the ODE)",
        "log10_J_wmax"     => "window maximum of log10 ‖J̃‖: gradient spikes between records (these set Adam's v̂)",
        "jac_plaus"        => "log10(‖J̃ Lp‖ / ‖r̃‖): predicted residual change per prior std relative to the residual; ≫ 3 is physically implausible",
        "log10_statvar"    => "log10 of the forward output's variance statistic (l63: min of the 3 state variances; l96: mean over sites of std²); < -2 = window collapsed to a fixed point",
        "lin_ratio_wmean"  => "window mean of log10(‖r̃_t − r̃_{t−1}‖ / ‖J̃_{t−1} Δθ_{t−1}‖): actual vs linearly predicted residual change; ≈ 0 valid, ≪ 0 J̃ overstates the response. Floor: the IC changes each step, adding O(√(2ny)) to the actual change, so ≫ 0 just means the step is negligible",
        "cos_prev_wmean"   => "window mean of cos(g_t, g_{t−1}); ≈ 0 means the gradient direction is IC noise (nu = 1: 2·(sign-agreement fraction) − 1)",
        "cos_truth_wmean"  => "window mean of cos(−Lpᵀg, Lp⁻¹(θ_true − θ)); > 0 means the gradient points toward the truth",
        "vhat_ratio_wmean" => "window mean of log10(√mean(v̂) / rms(g)): decades by which Adam's second-moment memory exceeds the current gradient; ≫ 0 ⇒ step ≈ α·10^(−ratio), i.e. frozen by an old spike",
        "step_w_wmean"     => "window mean of the prior-whitened Adam step norm ‖Lp⁻¹ Δθ‖",
        "ad_iters_cum"     => "cumulative number of iterations that used the ForwardDiff fallback (collapsed window) instead of the GFDT/score Jacobian; a jump marks an AD iteration",
        "dist_truth"       => "prior-whitened distance ‖Lp⁻¹(θ − θ_true)‖",
    )
    Lp = cholesky(Symmetric(Matrix{Float64}(prior_cov))).L
    θt = θ_true === nothing ? nothing : Vector{Float64}(θ_true)
    return AdamDiag(rec, θt, Lp, nothing, nothing, nothing)
end

_dist_truth(d::AdamDiag, θ) = d.θ_true === nothing ? NaN : norm(d.Lp \ (θ - d.θ_true))

# After RMSE is known, before the convergence check. statvar: see "log10_statvar".
function adam_diag_observe!(d::AdamDiag, it, θ, J̃, r̃, statvar; extras...)
    rec = d.rec
    rec.enabled || return d
    accumulate!(rec, "rmse", norm(r̃) / sqrt(length(r̃)); mode = :min)
    accumulate!(rec, "log10_J", log10(norm(J̃)); mode = :max)
    d.dr_pred === nothing ||
        accumulate!(rec, "lin_ratio", log10(norm(r̃ - d.r_prev) / norm(d.dr_pred)))
    d.r_prev = copy(r̃)
    want(rec, it) && record!(rec, it; _point_values(d, θ, J̃, r̃, statvar)..., extras...)
    return d
end

_point_values(d::AdamDiag, θ, J̃, r̃, statvar) = (
    rmse          = norm(r̃) / sqrt(length(r̃)),
    log10_J       = log10(norm(J̃)),
    jac_plaus     = log10(norm(J̃ * d.Lp) / norm(r̃)),
    log10_statvar = log10(max(statvar, floatmin(Float64))),
    dist_truth    = _dist_truth(d, θ),
)

# Inside `if RMSE < rmse_target`, before `break`: forces a full row for the final iteration.
adam_diag_converged!(d::AdamDiag, it, θ, r̃, statvar; extras...) =
    finalize!(d.rec, it; rmse = norm(r̃) / sqrt(length(r̃)),
              log10_statvar = log10(max(statvar, floatmin(Float64))), dist_truth = _dist_truth(d, θ), extras...)

# After the loop; no-op if already finalized at convergence (N_iter is always a scheduled record).
adam_diag_finish!(d::AdamDiag, N_iter) = finalize!(d.rec, N_iter)

# After θ_new = θ_pre - α m̂ ./ (√v̂ .+ ε). g, J̃ are from θ_pre.
function adam_diag_step!(d::AdamDiag, θ_pre, θ_new, g, m̂, v̂, J̃, α, ε)
    rec = d.rec
    rec.enabled || return d
    d.g_prev === nothing || accumulate!(rec, "cos_prev", diag_cos(g, d.g_prev))
    d.g_prev = copy(g)
    d.θ_true === nothing || accumulate!(rec, "cos_truth", diag_cos(-(d.Lp' * g), d.Lp \ (d.θ_true - θ_pre)))
    accumulate!(rec, "vhat_ratio", log10(sqrt(sum(v̂) / length(v̂)) / sqrt(sum(abs2, g) / length(g))))
    Δθ = θ_new - θ_pre
    accumulate!(rec, "step_w", norm(d.Lp \ Δθ))
    d.dr_pred = -(J̃ * Δθ)
    return d
end

