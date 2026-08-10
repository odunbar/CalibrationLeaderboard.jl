# score_based_lm — L63 raw moments, statistics map, and conjugate variables
#
# Must be included ALONGSIDE common/forward_maps/Lorenz63.jl only: Lorenz63.jl and
# Lorenz96.jl define clashing LorenzConfig / EnsembleMemberConfig / ObservationConfig,
# so anything dispatching on them has to be per-model.

using LinearAlgebra

########################################################################
###############  Raw moments A(x) and the map G = phi(m)  #############
########################################################################

# `stats` (common/forward_maps/Lorenz63.jl) returns
#   gt[1:3] = mean(x, y, z)
#   gt[4:6] = var(x, y, z)
#   gt[7:8] = cov(x,y), cov(x,z)
#   gt[9]   = cov(y,z)
# which are NONLINEAR in time averages.  GFDT differentiates time averages, so we
# work with raw moments A and compose: G = phi(<A>).
#
#   A(x) = [x, y, z, x^2, y^2, z^2, xy, xz, yz]
moment_observables_l63(X::AbstractMatrix) = vcat(
    X,
    (@views X[1, :] .^ 2)', (@views X[2, :] .^ 2)', (@views X[3, :] .^ 2)',
    (@views X[1, :] .* X[2, :])', (@views X[1, :] .* X[3, :])', (@views X[2, :] .* X[3, :])',
)

function phi_l63(m::AbstractVector)
    return [
        m[1], m[2], m[3],
        m[4] - m[1]^2, m[5] - m[2]^2, m[6] - m[3]^2,
        m[7] - m[1] * m[2], m[8] - m[1] * m[3], m[9] - m[2] * m[3],
    ]
end

# dphi/dm, 9 x 9.
function dphi_dm_l63(m::AbstractVector)
    P = zeros(9, 9)
    P[1, 1] = 1.0
    P[2, 2] = 1.0
    P[3, 3] = 1.0
    P[4, 4] = 1.0; P[4, 1] = -2m[1]
    P[5, 5] = 1.0; P[5, 2] = -2m[2]
    P[6, 6] = 1.0; P[6, 3] = -2m[3]
    P[7, 7] = 1.0; P[7, 1] = -m[2]; P[7, 2] = -m[1]
    P[8, 8] = 1.0; P[8, 1] = -m[3]; P[8, 3] = -m[1]
    P[9, 9] = 1.0; P[9, 2] = -m[3]; P[9, 3] = -m[2]
    return P
end

########################################################################
###############  Conjugate variables  #################################
########################################################################

# B_j(x) = -[ div(df/dtheta_j)(x) + (df/dtheta_j)(x) . s(x) ]
#
# L63: f = [sigma(y-x), x(rho-z)-y, xy - beta z] with sigma = 10 fixed.
#
#   df/drho  = (0, x, 0)    div = 0     =>  B_rho  = -x * s_y(x)
#   df/dbeta = (0, 0, -z)   div = -1    =>  B_beta = 1 + z * s_z(x)
#
# Sanity (free, and asserted in validate_jacobian_l63.jl): <B> = 0 for both, by
# the Stein identity <x_i s_j> = -delta_ij.  <B_beta> = 1 + <z s_z> = 1 - 1 = 0.
#
# The run scripts optimise theta = (log rho, log beta), so the log chain rule is
# folded into B rather than post-multiplying J -- one line, and the correlation
# integral then comes out already in theta coordinates.
function conjugate_variables_l63(X::AbstractMatrix, S::AbstractMatrix, theta::AbstractVector)
    rho, beta = exp(theta[1]), exp(theta[2])
    n = size(X, 2)
    B = Matrix{Float64}(undef, 2, n)
    @views B[1, :] .= rho .* (-(X[1, :] .* S[2, :]))
    @views B[2, :] .= beta .* (1.0 .+ X[3, :] .* S[3, :])
    return B
end
