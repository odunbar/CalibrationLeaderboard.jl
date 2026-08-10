# score_based_lm — L96 raw moments, statistics map, and conjugate variables
#
# Must be included ALONGSIDE common/forward_maps/Lorenz96.jl only (see gfdt_l63.jl).

using ForwardDiff
using LinearAlgebra

########################################################################
###############  Raw moments A(x) and the map G = phi(m)  #############
########################################################################

# `stats` (common/forward_maps/Lorenz96.jl) returns [means; STDS] (not variances):
#   gt[1:nx]        = mean(x_i)
#   gt[nx+1:2nx]    = std(x_i)
#
#   A(x) = [x_1..x_nx, x_1^2..x_nx^2]
moment_observables_l96(X::AbstractMatrix) = vcat(X, X .^ 2)

function phi_l96(m::AbstractVector, nx::Int)
    mu = @view m[1:nx]
    m2 = @view m[(nx + 1):(2nx)]
    return vcat(collect(mu), sqrt.(max.(m2 .- mu .^ 2, 0.0)))
end

# dphi/dm, 2nx x 2nx:
#   [ I                  0              ]
#   [ diag(-mu_i/sd_i)   diag(1/(2 sd_i)) ]
function dphi_dm_l96(m::AbstractVector, nx::Int)
    mu = @view m[1:nx]
    m2 = @view m[(nx + 1):(2nx)]
    sd = sqrt.(max.(m2 .- mu .^ 2, 1e-12))
    P = zeros(2nx, 2nx)
    for i in 1:nx
        P[i, i] = 1.0
        P[nx + i, i]      = -mu[i] / sd[i]
        P[nx + i, nx + i] = 1.0 / (2 * sd[i])
    end
    return P
end

########################################################################
###############  Forcing-profile Jacobian (flux-force only)  ##########
########################################################################

# dF_i/dw_k for the 61-weight Chain(Dense(1=>20,tanh), Dense(20=>1)) evaluated at
# each point of `sample_range`.  This is a Jacobian of the FORCING PROFILE, not
# of a model run: 100 x 61, sub-millisecond, and costs ZERO forward-model
# evaluations.  Reuses exactly the build_forcing/forcing path that
# run_l96_lm.jl:115-117 already ForwardDiffs successfully.
function forcing_weight_jacobian(phi::FluxEMC, w::AbstractVector, phi_structure, sample_range)
    g = ww -> forcing(build_forcing(phi, ww, phi_structure, sample_range), nothing)
    return ForwardDiff.jacobian(g, w)
end

########################################################################
###############  Conjugate variables  #################################
########################################################################

# B_j(x) = -[ div(df/dtheta_j)(x) + (df/dtheta_j)(x) . s(x) ]
#
# For ALL THREE L96 cases the divergence term vanishes, because in this codebase
# the forcing is state-independent:
#   * forcing(::FluxEMC, x, i) ignores `x` and uses sample_range[i]
#     (common/forward_maps/Lorenz96.jl:56-57), and
#   * lorenz_solve evaluates the forcing ONCE before the time loop
#     (common/forward_maps/Lorenz96.jl:117),
# so F is a constant vector for the whole run and grad_x(dF/dtheta) == 0.
#
# WARNING: this is specific to this codebase, not a general fact.  A genuinely
# state-dependent neural closure F_i(x; w) has
#   div term = sum_i d/dx_i (dF_i/dw_k)  != 0,
# and dropping it would bias every column of J.  If the forcing is ever made
# state-dependent, reinstate the term here.
#
#   const-force (theta = log phi):  df/dphi = 1        =>  B = -sum_i s_i(x), times phi
#   vec-force   (theta = phi):      df/dphi_j = e_j    =>  B = -s(x)
#   flux-force  (theta = w):        df/dw_k            =>  B = -M' s(x),  M = dF/dw
function conjugate_variables_l96(
    case::String,
    X::AbstractMatrix,
    S::AbstractMatrix,
    theta::AbstractVector,
    prob,
)
    if case == "const-force"
        phi = exp(theta[1])                      # log chain rule folded in
        return phi .* (-sum(S, dims = 1))        # 1 x n
    elseif case == "vec-force"
        return -S                                # nx x n
    elseif case == "flux-force"
        M = forcing_weight_jacobian(prob.phi, theta, prob.phi_structure, prob.sample_range)
        return -(M' * S)                         # nu x n
    else
        throw(ArgumentError("Unknown L96 case: $case"))
    end
end
