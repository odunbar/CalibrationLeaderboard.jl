# lm_diagnostics.jl — gradient magnitude and alignment diagnostics for the LM run scripts.
#
# Included by run_l63_lm.jl / run_l96_lm.jl (levenberg_marquardt) and run_l63_sblm.jl /
# run_l96_sblm.jl (score_based_lm); one shared copy. Uses only quantities the loop already
# computes (J̃, r̃, θ, Δθ, θ_trial, r̃_trial, ρ, λ): no forward-model evaluations, no RNG
# draws, no mutation of algorithm state. DIAG=0 disables recording (see diag_recorder.jl).
#
# The "gradient" is g = −J̃ᵀr̃ = ∇(½‖r̃‖²) with r̃ = R^{-1/2}(y − G(θ)); LM never forms it,
# but J̃ and r̃ are exactly what LM solves with, so ‖g‖ and its direction describe the
# derivative information the step is built from.
#
#   d = LMDiag(cfg.N_iter, θ_true, cov(prior_dist))          # before the loop
#   lm_diag_observe!(d, it, θ, J̃, r̃, λ; extras...)          # once J̃ is known (point values at θ)
#   lm_diag_converged!(d, it, θ, r̃, λ, J̃)                   # inside the convergence branch, before break
#   lm_diag_step!(d, θ, θ_trial, Δθ, J̃, r̃, r̃_trial, ρ)      # after ρ, BEFORE the accept/λ update
#   lm_diag_finish!(d, N_iter)                               # after the loop
#   diagnostics_dict(d.rec)                                  # → "diagnostics" in the result JLD2
#
# Record semantics: *point* keys (rmse, log10_J, log10_g, cos_*, ...) are values at iteration
# `iter` before its step. *_wmean keys come from the step taken at the PREVIOUS iteration(s)
# since the last record (with N_iter ≤ 500 every iteration is recorded, so row t holds the
# step t−1 → t).

using LinearAlgebra

include(joinpath(@__DIR__, "diag_recorder.jl"))

mutable struct LMDiag
    rec::DiagRecorder
    θ_true::Union{Nothing, Vector{Float64}}   # nothing: no usable truth (flux)
    Lp::LowerTriangular{Float64, Matrix{Float64}}
    g_prev::Union{Nothing, Vector{Float64}}
end

function LMDiag(N_iter, θ_true, prior_cov)
    rec = DiagRecorder(N_iter)
    describe!(rec,
        "rmse"             => "whitened RMSE at θ before the step (IC sample of that iteration)",
        "log10_J"          => "MAGNITUDE: log10 Frobenius norm of the whitened Jacobian J̃; healthy O(1–3), AD through chaos gives 9–22",
        "jac_plaus"        => "MAGNITUDE: log10(‖J̃ Lp‖ / ‖r̃‖): predicted residual change per prior std relative to the residual; ≫ 3 is physically implausible",
        "log10_g"          => "MAGNITUDE: log10 ‖g‖, g = −J̃ᵀr̃ (gradient of ½‖r̃‖²)",
        "sigma_J"          => "‖J̃ᵀr̃‖/(‖J̃‖‖r̃‖) ∈ [0,1]: ≈ 0 means the residual lies outside range(J̃) (model cannot explain the misfit locally)",
        "cond_J"           => "log10 cond(J̃) (NaN for non-finite J̃)",
        "nonfinite_J"      => "number of non-finite entries in J̃",
        "cos_truth"        => "ALIGNMENT: cos(−Lpᵀg, Lp⁻¹(θ_true − θ)): > 0 means −gradient points toward the truth (nu = 1: ±1 sign agreement)",
        "cos_prev"         => "ALIGNMENT: cos(g_t, g_{t−1}) across iterations; ≈ 0 means the gradient direction is dominated by IC noise (needs a fresh IC each step; NaN at the first record)",
        "dist_truth"       => "prior-whitened distance ‖Lp⁻¹(θ − θ_true)‖",
        "log10_lambda"     => "log10 of the LM damping λ at the iteration (cap 1e8)",
        "cos_step_grad_wmean"  => "ALIGNMENT: window mean of cos(Δθ, −g): 1 = LM step is steepest descent (λ large), < 1 = Gauss-Newton rotation",
        "cos_step_truth_wmean" => "ALIGNMENT: window mean of cos(Lp⁻¹Δθ, Lp⁻¹(θ_true − θ))",
        "lin_cos_wmean"    => "ALIGNMENT at the step scale: window mean of cos(r̃_trial − r̃, −J̃Δθ) (same IC for both); ≈ 1 means the linearisation points the right way",
        "lin_ratio_wmean"  => "MAGNITUDE at the step scale: window mean of log10(‖r̃_trial − r̃‖ / ‖J̃Δθ‖); ≈ 0 valid, ≪ 0 means J̃ overstates the response",
        "log10_step_wmean" => "window mean of log10 ‖Lp⁻¹Δθ‖",
        "step_quant_wmean" => "window mean of ‖(θ+Δθ) − θ‖/‖Δθ‖; ≠ 1 means the step is below Float64 resolution of θ",
        "rho_clip_wmean"   => "window mean of ρ clamped to [−2, 2] (gain ratio)",
        "accepted_wmean"   => "fraction of steps in the window with ρ > 0",
    )
    Lp = cholesky(Symmetric(Matrix{Float64}(prior_cov))).L
    θt = θ_true === nothing ? nothing : Vector{Float64}(θ_true)
    return LMDiag(rec, θt, Lp, nothing)
end

_dist_truth(d::LMDiag, θ) = d.θ_true === nothing ? NaN : norm(d.Lp \ (θ - d.θ_true))

# point values that need J̃ (computed only on recorded iterations)
function _jac_values(d::LMDiag, θ, J̃, r̃)
    g = -(J̃' * r̃)
    gnorm = norm(g)
    cos_truth = d.θ_true === nothing ? NaN : diag_cos(-(d.Lp' * g), d.Lp \ (d.θ_true - θ))
    return (
        log10_J     = log10(norm(J̃)),
        jac_plaus   = log10(norm(J̃ * d.Lp) / norm(r̃)),
        log10_g     = log10(gnorm),
        sigma_J     = gnorm / (norm(J̃) * norm(r̃)),
        cond_J      = all(isfinite, J̃) ? log10(cond(J̃)) : NaN,
        nonfinite_J = count(!isfinite, J̃),
        cos_truth   = cos_truth,
        cos_prev    = d.g_prev === nothing ? NaN : diag_cos(g, d.g_prev),
    )
end

# Once J̃ and r̃ at θ are known. `extras` are method-specific point values (e.g. score_based_lm
# passes ad_fallback, stein_norm, b_mean_ratio).
function lm_diag_observe!(d::LMDiag, it, θ, J̃, r̃, λ; extras...)
    rec = d.rec
    rec.enabled || return d
    RMSE = norm(r̃) / sqrt(length(r̃))
    accumulate!(rec, "rmse", RMSE; mode = :min)
    if want(rec, it)
        record!(rec, it;
            rmse = RMSE, dist_truth = _dist_truth(d, θ), log10_lambda = log10(λ),
            _jac_values(d, θ, J̃, r̃)..., extras...)
    end
    d.g_prev = -(J̃' * r̃)    # always updated (a fresh array; never aliases the loop's state)
    return d
end

# Inside `if RMSE < rmse_target`, before `break`: forces a row for the final iteration.
# J̃ may be `nothing` (score_based_lm checks convergence before it builds J̃).
function lm_diag_converged!(d::LMDiag, it, θ, r̃, λ, J̃ = nothing; extras...)
    d.rec.enabled || return d
    vals = (rmse = norm(r̃) / sqrt(length(r̃)), dist_truth = _dist_truth(d, θ), log10_lambda = log10(λ), extras...)
    J̃ === nothing ? finalize!(d.rec, it; vals...) : finalize!(d.rec, it; vals..., _jac_values(d, θ, J̃, r̃)...)
    return d
end

# After ρ, before the accept / λ update. θ is still the pre-step value.
function lm_diag_step!(d::LMDiag, θ, θ_trial, Δθ, J̃, r̃, r̃_trial, ρ)
    rec = d.rec
    rec.enabled || return d
    Δr_act  = r̃_trial - r̃
    Δr_pred = -(J̃ * Δθ)
    accumulate!(rec, "lin_cos", diag_cos(Δr_act, Δr_pred))
    accumulate!(rec, "lin_ratio", log10(norm(Δr_act) / norm(Δr_pred)))
    accumulate!(rec, "cos_step_grad", diag_cos(Δθ, J̃' * r̃))
    d.θ_true === nothing || accumulate!(rec, "cos_step_truth", diag_cos(d.Lp \ Δθ, d.Lp \ (d.θ_true - θ)))
    accumulate!(rec, "log10_step", log10(norm(d.Lp \ Δθ)))
    accumulate!(rec, "step_quant", norm(θ_trial - θ) / norm(Δθ))
    accumulate!(rec, "rho_clip", clamp(ρ, -2.0, 2.0))
    accumulate!(rec, "accepted", ρ > 0)
    return d
end

# After the loop; no-op if already finalized at convergence.
lm_diag_finish!(d::LMDiag, N_iter) = finalize!(d.rec, N_iter)
