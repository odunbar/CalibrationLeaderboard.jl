# Shared prior-specification helpers — extracted from a byte-identical
# duplicate previously carried independently by
# uq_experiments/HistoryMatching/history_matching_core.jl and
# uq_experiments/BayesianOptimalExperimentalDesign/boed_core.jl.

# Given the desired mean/std of X = exp(mu + sigma*Z), Z ~ N(0,1), returns
# (mu, sigma). Reproduces the same constrained-space mean/std as
# EnsembleKalmanProcesses.jl's `constrained_gaussian(name, mean, std, 0, Inf)`
# bounded-below moment matching, without depending on EKP for it.
function lognormal_params_from_moments(mean_x::Real, std_x::Real)
    var_ratio = (std_x / mean_x)^2
    sigma2 = log(1 + var_ratio)
    mu = log(mean_x) - sigma2 / 2
    return mu, sqrt(sigma2)
end
