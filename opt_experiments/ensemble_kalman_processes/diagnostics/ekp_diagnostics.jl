# ekp_diagnostics.jl — estimated-gradient diagnostics for the ensemble_kalman_processes run scripts.
#
# Included by run_l63_example.jl and run_l96_example.jl. Uses only what the loop already holds at each
# iteration: the unconstrained ensemble u (get_u_final) and the forward outputs G_ens. No forward-model
# evaluations, no RNG draws, no mutation of the EKP object or of any loop state. DIAG=0 disables recording.
#
# Estimated gradient. The ensemble Kalman update is a statistical linearisation, G(u) ≈ Ḡ + Ĵ (u − ū) with
#       Ĵ = C_Gu C_uu⁺            (sample regression slope of the ensemble; pinv: Ne may be < nu + 1)
# and the data-misfit gradient it implies is  g = −J̃ᵀ r̃,  J̃ = R^{-1/2} Ĵ,  r̃ = R^{-1/2}(y − Ḡ).
# For Unscented (UKI) column 1 of the ensemble is the mean and the other columns are symmetric sigma points,
# so the regression uses the symmetric columns only (the centre point has zero deviation).
# Ḡ is the ensemble mean of G (UKI: the centre column), not the single-IC G(ū) used for the convergence check.
#
# Everything lives in prior-whitened coordinates where a scale is needed (Lp = chol(cov(prior)), unconstrained space).
#
#   ediag = EKPDiag(N_iter, n_methods, n_N, n_seeds, cov(prior), u_true)   # before the loops
#   ekp_diag_begin_cell!(ediag, kk, ee, rr)                               # per (method, N_ens, seed) cell
#   ekp_diag_observe!(ediag, i, get_u_final(ekpobj), G_ens, y, R_inv_var, uki, RMSE_e)   # after G_ens, before update
#   ekp_diag_converged!(ediag, i, get_u_final(ekpobj), RMSE_e, uki)       # in the convergence branch, before break
#   diagnostics_dict(ediag)                                               # → "diagnostics" in the output JLD2
#
# Arrays are Float32 [iteration, method, N_ens, seed], NaN where not recorded (N_iter ≤ 20 here, so every
# iteration is recorded; no thinning needed).

using LinearAlgebra
using Statistics

const EKP_DIAG_SCHEMA_VERSION = 1

const EKP_DIAG_DESCRIPTIONS = Dict(
    "rmse"        => "whitened RMSE at G(ensemble mean) for a single IC draw (the quantity the convergence check uses)",
    "log10_J"     => "MAGNITUDE: log10 ‖J̃‖_F, J̃ = R^{-1/2} C_Gu C_uu⁺ (ensemble-estimated whitened Jacobian)",
    "jac_plaus"   => "MAGNITUDE: log10(‖J̃ Lp‖ / ‖r̃‖): predicted residual change per prior std relative to the residual; ≫ 3 implausible",
    "log10_g"     => "MAGNITUDE: log10 ‖g‖, g = −J̃ᵀ r̃ (estimated gradient of ½‖r̃‖²)",
    "sigma_J"     => "‖J̃ᵀr̃‖/(‖J̃‖‖r̃‖) ∈ [0,1]: ≈ 0 means the residual lies outside the estimated range(J̃)",
    "cos_truth"   => "ALIGNMENT: cos(−Lpᵀg, Lp⁻¹(u_true − ū)): > 0 means the estimated descent direction points toward the truth (NaN for flux; nu = 1 gives ±1)",
    "cos_prev"    => "ALIGNMENT: cos(g_t, g_{t−1}) across iterations; ≈ 0 means the estimated gradient direction is noise",
    "dist_truth"  => "prior-whitened distance ‖Lp⁻¹(ū − u_true)‖ of the ensemble mean (NaN for flux)",
    "spread_w"    => "ensemble spread sqrt(tr(Lp⁻¹ C_uu Lp⁻ᵀ)/nu): 1 ≈ prior width; collapse faster than the misfit falls = overconfident (UKI: unweighted sigma-point estimate, approximate)",
    "step_w"      => "‖Lp⁻¹(ū_t − ū_{t−1})‖: prior-whitened movement of the ensemble mean since the previous iteration (NaN at the first)",
    "n_fail_G"    => "number of ensemble members whose forward output contained a non-finite value (excluded from the regression)",
)

mutable struct EKPDiag
    enabled::Bool
    N_iter::Int
    data::Dict{String, Array{Float32, 4}}
    Lp::LowerTriangular{Float64, Matrix{Float64}}
    u_true::Union{Nothing, Vector{Float64}}
    cell::NTuple{3, Int}
    g_prev::Union{Nothing, Vector{Float64}}
    ubar_prev::Union{Nothing, Vector{Float64}}
end

function EKPDiag(N_iter, n_methods, n_N, n_seeds, prior_cov, u_true; enabled::Bool = get(ENV, "DIAG", "1") != "0")
    data = Dict{String, Array{Float32, 4}}(k => fill(NaN32, N_iter, n_methods, n_N, n_seeds) for k in keys(EKP_DIAG_DESCRIPTIONS))
    Lp = cholesky(Symmetric(Matrix{Float64}(prior_cov))).L
    ut = u_true === nothing ? nothing : Vector{Float64}(u_true)
    return EKPDiag(enabled, N_iter, data, Lp, ut, (0, 0, 0), nothing, nothing)
end

function ekp_diag_begin_cell!(d::EKPDiag, kk, ee, rr)
    d.cell = (kk, ee, rr)
    d.g_prev = nothing
    d.ubar_prev = nothing
    return d
end

_set!(d::EKPDiag, key::String, it::Integer, val) = (d.data[key][it, d.cell...] = Float32(val); nothing)

# ensemble mean and (deviations) used for the regression; UKI: centre column is the mean, symmetric columns the points
function _ens_stats(U, G, uki::Bool)
    if uki
        ubar = U[:, 1]
        Ud = U[:, 2:end] .- ubar
        Gd = G[:, 2:end] .- mean(G[:, 2:end], dims = 2)
        Gbar = G[:, 1]
    else
        ubar = vec(mean(U, dims = 2))
        Ud = U .- ubar
        Gd = G .- mean(G, dims = 2)
        Gbar = vec(mean(G, dims = 2))
    end
    return ubar, Ud, Gd, Gbar
end

# point values that do not need G (spread, distance, step)
function _common!(d::EKPDiag, it, U, rmse, uki)
    ubar = uki ? U[:, 1] : vec(mean(U, dims = 2))
    Ud = uki ? U[:, 2:end] .- ubar : U .- ubar
    nu = size(U, 1)
    _set!(d, "rmse", it, rmse)
    if d.u_true !== nothing
        _set!(d, "dist_truth", it, norm(d.Lp \ (ubar - d.u_true)))
    end
    nd = max(size(Ud, 2) - (uki ? 0 : 1), 1)
    Cuu = (Ud * Ud') / nd
    W = d.Lp \ Cuu / d.Lp'
    _set!(d, "spread_w", it, sqrt(max(tr(W), 0.0) / nu))
    d.ubar_prev === nothing || _set!(d, "step_w", it, norm(d.Lp \ (ubar - d.ubar_prev)))
    return ubar
end

# at an iteration where G_ens exists (before the update)
function ekp_diag_observe!(d::EKPDiag, it, U, G, y, R_inv_var, uki::Bool, rmse)
    d.enabled || return d
    ubar = _common!(d, it, U, rmse, uki)
    good = [all(isfinite, view(G, :, j)) for j in 1:size(G, 2)]
    _set!(d, "n_fail_G", it, count(!, good))
    # UKI needs every sigma point; the ensemble methods regress on the finite members
    Gs, Us = uki ? (G, U) : (G[:, good], U[:, good])
    if size(Us, 2) >= 3 && all(isfinite, Gs)   # (UKI with a failed sigma point is skipped)
        _, Ud, Gd, Gbar = _ens_stats(Us, Gs, uki)
        n = max(size(Ud, 2) - (uki ? 0 : 1), 1)
        Cuu = (Ud * Ud') / n
        CGu = (Gd * Ud') / n
        J̃ = R_inv_var * (CGu * pinv(Cuu))
        r̃ = R_inv_var * (y - Gbar)
        g = -(J̃' * r̃)
        nJ, nr, ng = norm(J̃), norm(r̃), norm(g)
        _set!(d, "log10_J", it, log10(nJ))
        _set!(d, "jac_plaus", it, log10(norm(J̃ * d.Lp) / nr))
        _set!(d, "log10_g", it, log10(ng))
        _set!(d, "sigma_J", it, ng / (nJ * nr))
        d.u_true === nothing || _set!(d, "cos_truth", it, _cos(-(d.Lp' * g), d.Lp \ (d.u_true - ubar)))
        d.g_prev === nothing || _set!(d, "cos_prev", it, _cos(g, d.g_prev))
        d.g_prev = copy(g)
    else
        d.g_prev = nothing
    end
    d.ubar_prev = copy(ubar)
    return d
end

# convergence branch: G_ens was not built this iteration, so only the point values are recorded
function ekp_diag_converged!(d::EKPDiag, it, U, rmse, uki::Bool)
    d.enabled || return d
    _common!(d, it, U, rmse, uki)
    return d
end

function _cos(a, b)
    na, nb = norm(a), norm(b)
    (na > 0 && nb > 0 && isfinite(na) && isfinite(nb)) || return NaN
    return dot(a, b) / (na * nb)
end

function diagnostics_dict(d::EKPDiag)
    out = Dict{String, Any}("schema_version" => EKP_DIAG_SCHEMA_VERSION, "enabled" => d.enabled)
    d.enabled || return out
    out["descriptions"] = copy(EKP_DIAG_DESCRIPTIONS)
    out["dims"] = "iteration × method × N_ens × seed (same method / N_ens / seed order as conv_alg_iters)"
    for (k, a) in d.data
        out[k] = copy(a)
    end
    return out
end
