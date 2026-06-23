# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
struct EnsembleMemberConfig{VV <: AbstractVector}
    "rho, beta (unknowns)"
    u::VV
end

# This will change for different "Observations" of Lorenz
struct ObservationConfig{FT1 <: Real, FT2 <: Real}
    "initial time to gather statistics (T_start = N_start*dt)"
    T_start::FT1
    "end time to gather statistics (T_end = N_end*dt)"
    T_end::FT2
end

#########################################################################
############################ Model Functions ############################
#########################################################################

# Forward pass of forward model
# Inputs: 
# - params: structure with u (unknowns vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_forward(
    params::EnsembleMemberConfig,
    x0::VorM,
    config::LorenzConfig,
    observation_config::ObservationConfig,
) where {VorM <: AbstractVecOrMat}
    # run the Lorenz simulation
    xn = lorenz_solve(params, x0, config)
    # Get statistics
    gt = stats(xn, config, observation_config)
    return gt
end

#Calculates statistics for forward model output
# Inputs: 
# - xn: timeseries of states for length of simulation through Lorenz63
function stats(xn::VorM, config::LorenzConfig, observation_config::ObservationConfig) where {VorM <: AbstractVecOrMat}
    T_start = observation_config.T_start
    T_end = observation_config.T_end
    dt = config.dt
    N_start = Int(ceil(T_start / dt))
    N_end = Int(ceil(T_end / dt))
    xn_stat = xn[:, N_start:N_end]
    N_state = size(xn_stat, 1)
    gt = zeros(eltype(xn_stat), 9)  # Might want to switch to more general statement?
    gt[1:3] = mean(xn_stat, dims = 2)
    xn_stat_cov = cov(xn_stat, dims = 2)
    gt[4:6] = diag(xn_stat_cov)
    gt[7:8] = xn_stat_cov[1, 2:3]
    gt[9] = xn_stat_cov[2, 3]
    return gt
end

# Forward pass of the Lorenz 96 model
# Inputs: 
# - params: structure with u (unknowns vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    # Initialize    
    nstep = Int(ceil(config.T / config.dt))
    state_dim = isa(x0, AbstractVector) ? length(x0) : size(x0, 1)
    xn = zeros(promote_type(eltype(params.u), eltype(x0)), size(x0, 1), nstep + 1)
    xn[:, 1] = x0

    # March forward in time
    for j in 1:nstep
        xn[:, j + 1] = RK4(params, xn[:, j], config)
    end
    # Output
    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: 
# - params: structure with u (unknowns vector)
# - x: current state
function f(params::EnsembleMemberConfig, x::VorM) where {VorM <: AbstractVecOrMat}
    u = params.u
    N = length(x)
    f = zeros(promote_type(eltype(u), eltype(x)), N)

    f[1] = 10.0 * (x[2] - x[1])
    f[2] = x[1] * (u[1] - x[3]) - x[2]
    f[3] = x[1] * x[2] - u[2] * x[3]

    # Output
    return f
end

# RK4 solve
# Inputs: 
# - params: structure with F (state-dependent-forcing vector) 
# - xold: current state
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function RK4(params::EnsembleMemberConfig, xold::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    N = length(xold)
    dt = config.dt

    # Predictor steps (note no time-dependence is needed here)
    k1 = f(params, xold)
    k2 = f(params, xold + k1 * dt / 2.0)
    k3 = f(params, xold + k2 * dt / 2.0)
    k4 = f(params, xold + k3 * dt)
    # Step
    xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Output
    return xnew
end

#########################################################################
###################### PerfectDataConfig ###############################
#########################################################################

# Holds the full problem specification (config objects + computed quantities)
# for a Lorenz63 "perfect data" experiment.
# Construct via compute_perfect_data; persist via save_preliminaries / load_preliminaries.
struct PerfectDataConfig{
    EMC  <: EnsembleMemberConfig,
    LC   <: LorenzConfig,
    OC   <: ObservationConfig,
    VV1  <: AbstractVector,
    VV2  <: AbstractVector,
    VV3  <: AbstractVector,
    MM1  <: Union{AbstractMatrix, UniformScaling},
    MM2  <: Union{AbstractMatrix, UniformScaling},
    MM3  <: Union{AbstractMatrix, UniformScaling},
    FT   <: Real,
}
    # --- problem specification ---
    truth_params              :: EMC  # true (ρ, β) used to generate data
    nx                        :: Int  # state dimension (3)
    ny                        :: Int  # observation dimension (9)
    picking_initial_condition :: LC   # spin-up run config
    x_initial                 :: VV1  # seed state for spin-up
    lorenz_config_settings    :: LC   # config for data-generating run
    observation_config        :: OC   # statistics window [T_start, T_end]
    R_n_samples               :: Int  # non-overlapping windows for R estimate
    ic_cov_T                  :: FT   # integration time for IC-cov estimate
    ic_cov_scaling            :: FT   # multiplicative scaling on IC covariance
    # --- computed quantities ---
    x0          :: VV2  # spun-up initial condition
    y           :: VV3  # synthetic observation vector
    R           :: MM1  # observation noise covariance
    R_inv_var   :: MM2  # sqrt(inv(R)), for whitened-RMSE diagnostics
    ic_cov_sqrt :: MM3  # sqrt(ic_cov_scaling * Cov(long run))
end

function compute_perfect_data(
    truth_params              :: EnsembleMemberConfig,
    nx                        :: Int,
    ny                        :: Int,
    picking_initial_condition :: LorenzConfig,
    x_initial                 :: AbstractVector,
    lorenz_config_settings    :: LorenzConfig,
    observation_config        :: ObservationConfig;
    R_n_samples    :: Int  = 36,
    ic_cov_T       :: Real = 2000.0,
    ic_cov_scaling :: Real = 0.1,
)
    dt      = lorenz_config_settings.dt
    T_start = observation_config.T_start
    T_end   = observation_config.T_end

    # 1. Spin up to a point on the attractor
    x_spun_up = lorenz_solve(truth_params, x_initial, picking_initial_condition)
    x0 = x_spun_up[:, end]

    # 2. Synthetic observations
    y = lorenz_forward(truth_params, x0, lorenz_config_settings, observation_config)

    # 3. Observation noise covariance from R_n_samples non-overlapping windows
    window    = T_end - T_start
    T_R       = R_n_samples * window + T_start
    R_config  = LorenzConfig(dt, T_R)
    R_run     = lorenz_solve(truth_params, x_initial, R_config)
    R_samples = zeros(ny, R_n_samples)
    for ii in 1:R_n_samples
        local_obs = ObservationConfig(T_start + (ii - 1) * window, T_start + ii * window)
        R_samples[:, ii] = stats(R_run, R_config, local_obs)
    end
    R         = cov(R_samples, dims = 2)
    R_inv_var = sqrt(inv(Symmetric(R)))

    # 4. Initial-condition perturbation covariance
    cov_solve   = lorenz_solve(truth_params, x0, LorenzConfig(dt, ic_cov_T))
    ic_cov      = ic_cov_scaling * cov(cov_solve, dims = 2)
    ic_cov_sqrt = sqrt(Symmetric(ic_cov))

    return PerfectDataConfig(
        truth_params, nx, ny,
        picking_initial_condition, Vector{Float64}(x_initial),
        lorenz_config_settings, observation_config,
        R_n_samples, ic_cov_T, ic_cov_scaling,
        x0, y, R, R_inv_var, ic_cov_sqrt,
    )
end

function save_preliminaries(pdc::PerfectDataConfig, filepath::AbstractString)
    tmpfile = splitext(filepath)[1] * ".tmp.$(getpid()).jld2"
    JLD2.save(
        tmpfile,
        "x0",                     pdc.x0,
        "y",                      pdc.y,
        "lorenz_config_settings", pdc.lorenz_config_settings,
        "observation_config",     pdc.observation_config,
        "ic_cov_sqrt",            pdc.ic_cov_sqrt,
        "R",                      pdc.R,
        "R_inv_var",              pdc.R_inv_var,
    )
    mv(tmpfile, filepath)
end

function load_preliminaries(filepath::AbstractString)
    d = JLD2.load(filepath)
    return (
        x0                     = d["x0"],
        y                      = d["y"],
        lorenz_config_settings = d["lorenz_config_settings"],
        observation_config     = d["observation_config"],
        ic_cov_sqrt            = d["ic_cov_sqrt"],
        R                      = d["R"],
        R_inv_var              = d["R_inv_var"],
    )
end
