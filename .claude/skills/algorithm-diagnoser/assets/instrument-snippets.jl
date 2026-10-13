# instrument-snippets.jl — CORRECT / WRONG instrumentation patterns (not meant to run as-is).
#
# Every snippet is purely additive: new lines between existing algorithm lines,
# an extended return tuple, an extra JLD2.save key. Nothing here draws from an
# RNG, evaluates the forward model, or mutates algorithm state.
#
# Convention: point-in-time values are recorded with record!() at scheduled
# iterations (before the convergence check). Per-step values (cosines, step sizes,
# acceptance) go through accumulate!() every iteration and are flushed as window
# means at the next record, so thinning doesn't hide rates.

########################################################################
# 1. adam — opt_experiments/adam/run_l63_adam.jl (run_l96_adam.jl: same, other line numbers)
########################################################################

# ── includes (top of file, after the existing includes) ──
include(joinpath(_COMMON, "diagnostics", "diag_recorder.jl"))   # the single shared recorder; _COMMON = joinpath(@__DIR__, "..", "..", "common")

# ── in run_one, after prior_dist / θ are set and before the loop ──
θ_true = log.([28.0, 8 / 3])                       # from l63_preliminaries.jl (truth_params, log space)
Lp     = cholesky(Symmetric(cov(prior_dist))).L    # prior whitening: z = Lp \ (θ - μ)
rec    = DiagRecorder(cfg.N_iter)
describe!(rec,
    "rmse"           => "whitened RMSE at θ before the step (single IC sample)",
    "log10_J"        => "log10 Frobenius norm of the whitened Jacobian J̃",
    "jac_plaus"      => "log10(‖J̃ Lp‖ / ‖r̃‖): predicted residual change per prior std relative to the residual; ≫ 3 is physically implausible",
    "dist_truth"     => "prior-whitened distance ‖Lp \\ (θ - θ_true)‖",
    "nonfinite_J"    => "number of non-finite entries in J̃",
    "cos_prev_wmean" => "window mean of cos(g_t, g_{t-1}); ≈ 0 means the gradient direction is noise",
    "cos_truth_wmean"=> "window mean of cos(-Lpᵀg, Lp⁻¹(θ_true - θ)); > 0 means the step points toward the truth",
    "snr_wmean"      => "window mean of mean_i m̂ᵢ²/v̂ᵢ; noise floor (1-β₁)/(1+β₁) ≈ 0.053, near 1 = consistent gradient",
    "step_w_wmean"   => "window mean of the prior-whitened Adam step norm",
    "rmse_up_wmean"  => "fraction of iterations in the window where RMSE increased",
    "rmse_wmin"      => "window minimum of RMSE",
)
g_prev    = nothing
rmse_prev = NaN

# ── in the loop, right after `RMSE = norm(r̃) / sqrt(ny)` and BEFORE `if RMSE < rmse_target` ──
accumulate!(rec, "rmse", RMSE; mode = :min)
isnan(rmse_prev) || accumulate!(rec, "rmse_up", RMSE > rmse_prev)
rmse_prev = RMSE
if want(rec, outer_iter)
    record!(rec, outer_iter;
        rmse        = RMSE,
        log10_J     = log10(norm(J̃)),
        jac_plaus   = log10(norm(J̃ * Lp) / norm(r̃)),
        dist_truth  = norm(Lp \ (θ - θ_true)),
        nonfinite_J = count(!isfinite, J̃),
    )
end

# ── inside `if RMSE < rmse_target`, just before `break` ──
finalize!(rec, outer_iter; rmse = RMSE, dist_truth = norm(Lp \ (θ - θ_true)))

# ── right after `g = -(J̃' * r̃)` (θ is still the pre-step value here) ──
g_prev === nothing || accumulate!(rec, "cos_prev", diag_cos(g, g_prev))
accumulate!(rec, "cos_truth", diag_cos(-(Lp' * g), Lp \ (θ_true - θ)))
g_prev = copy(g)

# ── right after `θ = θ - α * m̂ ./ (sqrt.(v̂) .+ ε)` ──
accumulate!(rec, "snr", mean(m̂ .^ 2 ./ (v̂ .+ ε^2)))
accumulate!(rec, "step_w", norm(Lp \ (α * m̂ ./ (sqrt.(v̂) .+ ε))))

# ── after the loop ──
finalize!(rec, cfg.N_iter)                          # no-op if finalized at convergence
return (; conv_score, final_params, final_output, diagnostics = diagnostics_dict(rec))

# ── in main(), extend the existing JLD2.save ──
#     "rng_idx",      rng_idx,
#     "diagnostics",  result.diagnostics,

########################################################################
# 2. levenberg_marquardt — opt_experiments/levenberg_marquardt/run_l63_lm.jl
########################################################################
# Same setup as adam (θ_true, Lp, rec, rmse accumulators), then:

# ── record point, after RMSE and before the convergence check ──
if want(rec, outer_iter)
    record!(rec, outer_iter;
        rmse         = RMSE,
        log10_J      = log10(norm(J̃)),
        jac_plaus    = log10(norm(J̃ * Lp) / norm(r̃)),
        cond_J       = all(isfinite, J̃) ? log10(cond(J̃)) : NaN,   # recorded iterations only
        sigma_J      = norm(J̃' * r̃) / (norm(J̃) * norm(r̃)),       # residual outside range(J̃) when ≈ 0
        log10_lambda = log10(λ),
        dist_truth   = norm(Lp \ (θ - θ_true)),
    )
end

# ── right after `ρ = ...` and BEFORE `if ρ > 0` ──
# Free directional-derivative check: the trial step reuses the same IC sample, so
# r̃_trial - r̃ is the actual residual change and -J̃Δθ is the linear prediction.
Δr_act  = r̃_trial - r̃
Δr_pred = -(J̃ * Δθ)
accumulate!(rec, "lin_cos", diag_cos(Δr_act, Δr_pred))
accumulate!(rec, "lin_ratio", log10(norm(Δr_act) / norm(Δr_pred)))
accumulate!(rec, "rho_clip", clamp(ρ, -2.0, 2.0))
accumulate!(rec, "accepted", ρ > 0)
accumulate!(rec, "nonfinite_step", !all(isfinite, Δθ))
accumulate!(rec, "log10_step", log10(norm(Lp \ Δθ)))
accumulate!(rec, "step_quant", norm(θ_trial - θ) / norm(Δθ))      # ≠ 1: step below Float64 resolution
accumulate!(rec, "cos_step_truth", diag_cos(Lp \ Δθ, Lp \ (θ_true - θ)))
accumulate!(rec, "rmse_trial", norm(r̃_trial) / sqrt(ny); mode = :min)

########################################################################
# WRONG (all methods)
########################################################################
u  = randn(rng, nu)                                  # consumes the run's RNG → different trajectory (rule: don't change the run)
Jp = ForwardDiff.jacobian(G_func, θ + 1e-3 * u)      # extra forward evaluations (rule: don't change the run)
normalize!(g)                                        # mutates algorithm state (rule: don't change the run)
g_prev = g                                           # aliases; fine only because g is rebound each iteration — prefer copy(g)
record!(rec, outer_iter; J = vec(J̃))                 # full Jacobian per record (storage cap; recorder rejects length > 10)
conv_score = outer_iter * N_ens * (nu + 1) + diag_cost   # diagnostics never enter the metric (rule: don't change the run)
if outer_iter % 100 == 0                             # stride without finalize! loses the final iteration
    record!(rec, outer_iter; rmse = RMSE)
end

########################################################################
# 3. GNKI — save the ekpobj it already builds (uq_experiments/GaussNewtonKalmanInversion/calibrate_l{63,96}.jl)
########################################################################
# CORRECT: one extra key in the existing JLD2.save. Size ≈ (nu + ny) · N_ens · (N_iter + 1) · 8 B
# (≈ 4 MB at flux N_ens = 90); check Base.summarysize(ekpobj) on a smoke run against the storage cap.
JLD2.save(
    joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
    "phi_stored", ϕ_stored,
    # ... existing keys unchanged ...
    "sample_range", sample_range,
    "ekpobj", ekpobj,                                # NEW: enables misfit/spread/error_metrics post-hoc
)
# The combine script then reads it post-hoc (no rerun of anything else):
#   u = get_u(ekp)  (N_iter+1 matrices, unconstrained)   g = get_g(ekp)  (N_iter matrices)
#   get_error_metrics(ekp) if the EKP version provides it
# WRONG: calling a sampling accessor or update on the loaded ekpobj inside the run.

########################################################################
# 4. HistoryMatching — record n_tried without changing the sampler (history_matching_core.jl)
########################################################################
# CORRECT: optional keyword, default nothing → existing callers unaffected.
function rejection_sample_nroy(
    prob::HMProblem, prior_sampler, D::Int, waves::Vector{WaveGPs}, threshold::Real,
    n_wanted::Int, max_samples::Int, batch_size::Int, rng::AbstractRNG;
    early_stop_alpha::Real = 1e-6,
    early_stop_min_tried::Int = 5 * batch_size,
    diag::Union{Nothing, Dict{String, Any}} = nothing,      # NEW
)
    # ... body unchanged ...
    # NEW, immediately before the final `if size(accepted, 2) >= n_wanted`:
    if diag !== nothing
        diag["n_tried"] = n_tried
        diag["n_accepted"] = size(accepted, 2)
    end
    # ... unchanged returns ...
end
# Callers (calibrate_l63.jl and calibrate_l96.jl — both must keep running):
hm_wave_diagnostics = Dict{String, Any}[]                    # before the wave loop
wave_diag = Dict{String, Any}("wave" => w)
posterior, ok = rejection_sample_nroy(#= existing args =#; diag = wave_diag)
push!(hm_wave_diagnostics, wave_diag)
# and "hm_wave_diagnostics", hm_wave_diagnostics, in the JLD2.save — next to, not inside, "emulator_diagnostics".
