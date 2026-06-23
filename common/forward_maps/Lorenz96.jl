# Import modules
using LinearAlgebra
using Statistics
using Random, Distributions
using Flux
using JLD2



# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
abstract type EnsembleMemberConfig end
# struct EnsembleMemberConfig{FT}
#    val::FT
# end

# Sub-type of ensemble config for constant forcing
struct ConstantEMC{FT <: Real} <: EnsembleMemberConfig
    val::FT
end
build_forcing(::T, val::FT, args...) where {T <: ConstantEMC, FT <: Real} = ConstantEMC(val)
build_forcing(::T, val::FT, args...) where {T <: ConstantEMC, FT <: AbstractVector} = ConstantEMC(val[1])

# Sub-type of ensemble config for spatially-dependent forcing
struct VectorEMC{VV <: AbstractVector} <: EnsembleMemberConfig
    val::VV
end
build_forcing(::T, val::VV, args...) where {T <: VectorEMC, VV <: AbstractVector} = VectorEMC(val)

# Sub-type of ensemble config for spatially-dependent forcing with neural network approximation
struct FluxEMC{FC <: Flux.Chain, VV <: AbstractVector} <: EnsembleMemberConfig
    model::FC
    sample_range::VV
end
function build_forcing(::T, params, model, sample_range) where {T <: FluxEMC}
    _, reconstructor = Flux.destructure(model)
    return FluxEMC(reconstructor(params), Float32.(sample_range))
end

# Constant-global
forcing(params::ConstantEMC, x, i) = params.val
forcing(params::ConstantEMC, x) = repeat([params.val], length(x))

# Constant-vector
forcing(params::VectorEMC, x, i) = params.val[i]
forcing(params::VectorEMC, x) = params.val

# Flux
forcing(params::FluxEMC, x, i) = params.model([params.sample_range[i]])[1]
forcing(params::FluxEMC, x) = [params.model([sr])[1] for sr in params.sample_range]




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
# - params: structure with F 
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
# - xn: timeseries of states for length of simulation through Lorenz96
function stats(xn::VorM, config::LorenzConfig, observation_config::ObservationConfig) where {VorM <: AbstractVecOrMat}
    T_start = observation_config.T_start
    T_end = observation_config.T_end
    dt = config.dt
    N_start = Int(ceil(T_start / dt))
    N_end = Int(ceil(T_end / dt))
    xn_stat = xn[:, N_start:N_end]
    N_state = size(xn_stat, 1)
    gt = zeros(eltype(xn_stat), 2 * N_state)
    gt[1:N_state] = mean(xn_stat, dims = 2)
    gt[(N_state + 1):(2 * N_state)] = std(xn_stat, dims = 2)
    return gt
end

# Forward pass of the Lorenz 96 model
# Inputs: 
# - params: structure with F 
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    # Initialize    
    nstep = Int(ceil(config.T / config.dt))
    forcing_vec = forcing(params, x0)
    xn = zeros(promote_type(eltype(forcing_vec), eltype(x0)), size(x0, 1), nstep + 1)
    xn[:, 1] = x0

    # March forward in time
    for j in 1:nstep
        xn[:, j + 1] = RK4(forcing_vec, xn[:, j], config)
    end
    # Output
    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: 
# - params: structure with F 
# - x: current state
function f(forcing_vec::VV, x::VorM) where {VV <: AbstractVector, VorM <: AbstractVecOrMat}
    N = length(x)
    f = zeros(promote_type(eltype(forcing_vec), eltype(x)), N)
    # Loop over N positions
    for i in 3:(N - 1)
        f[i] = -x[i - 2] * x[i - 1] + x[i - 1] * x[i + 1] - x[i] + forcing_vec[i]
    end
    # Periodic boundary conditions
    f[1] = -x[N - 1] * x[N] + x[N] * x[2] - x[1] + forcing_vec[1]
    f[2] = -x[N] * x[1] + x[1] * x[3] - x[2] + forcing_vec[2]
    f[N] = -x[N - 2] * x[N - 1] + x[N - 1] * x[1] - x[N] + forcing_vec[N]

    # Output
    return f
end

# RK4 solve
# Inputs: 
# - params: structure with F 
# - xold: current state
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function RK4(forcing_vec::VV, xold::VorM, config::LorenzConfig) where {VV <: AbstractVector, VorM <: AbstractVecOrMat}
    N = length(xold)
    dt = config.dt

    # Predictor steps (note no time-dependence is needed here)
    k1 = f(forcing_vec, xold)
    k2 = f(forcing_vec, xold + k1 * dt / 2.0)
    k3 = f(forcing_vec, xold + k2 * dt / 2.0)
    k4 = f(forcing_vec, xold + k3 * dt)
    # Step
    xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Output
    return xnew
end


# Neural network functions 
function train_network(model, x_train, y_train)
    loss(model, x, y) = Flux.Losses.mse(model(x), y)

    # Reshape x_train and y_train for Flux compatibility
    x_train = reshape(x_train, 1, :)
    y_train = reshape(y_train, 1, :)
    x_train = Float32.(x_train)
    y_train = Float32.(y_train)

    opt = Flux.setup(Adam(), model)
    data = Flux.DataLoader((x_train, y_train), batchsize = 32, shuffle = true)  # train the model

    # Train the model over multiple epochs
    epochs = 5000
    for epoch in 1:epochs
        Flux.train!(loss, model, data, opt)
    end

    params, _ = Flux.destructure(model)
    return model, params
end

#########################################################################
###################### PerfectDataConfig ###############################
#########################################################################

# Holds the full problem specification (config objects + computed quantities)
# for a Lorenz96 "perfect data" experiment.
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
    truth_params              :: EMC  # true forcing (ConstantEMC / VectorEMC / FluxEMC)
    nx                        :: Int  # state dimension (40 or 100)
    ny                        :: Int  # observation dimension (2*nx)
    picking_initial_condition :: LC   # spin-up run config
    x_initial                 :: VV1  # seed state for spin-up
    lorenz_config_settings    :: LC   # config for data-generating run
    observation_config        :: OC   # statistics window [T_start, T_end]
    R_n_samples               :: Int  # non-overlapping windows for R estimate (10*ny)
    R_inflation               :: FT   # multiplicative inflation on R (e.g. 2.0 or 2.5)
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
    truth_params              :: EMC,
    nx                        :: Int,
    picking_initial_condition :: LorenzConfig,
    x_initial                 :: AbstractVector,
    lorenz_config_settings    :: LorenzConfig,
    observation_config        :: ObservationConfig;
    R_inflation    :: Real = 2.0,
    ic_cov_T       :: Real = 2000.0,
    ic_cov_scaling :: Real = 0.1,
) where {EMC <: EnsembleMemberConfig}
    dt      = lorenz_config_settings.dt
    T_start = observation_config.T_start
    T_end   = observation_config.T_end
    ny      = 2 * nx

    # 1. Spin up to a point on the attractor
    x_spun_up = lorenz_solve(truth_params, x_initial, picking_initial_condition)
    x0 = x_spun_up[:, end]

    # 2. Synthetic observations
    y = lorenz_forward(truth_params, x0, lorenz_config_settings, observation_config)

    # 3. Observation noise covariance from 10*ny non-overlapping windows, inflated
    window      = T_end - T_start
    R_n_samples = 10 * ny
    T_R         = R_n_samples * window + T_start
    R_config    = LorenzConfig(dt, T_R)
    R_run       = lorenz_solve(truth_params, x_initial, R_config)
    R_samples   = zeros(ny, R_n_samples)
    for ii in 1:R_n_samples
        local_obs = ObservationConfig(T_start + (ii - 1) * window, T_start + ii * window)
        R_samples[:, ii] = stats(R_run, R_config, local_obs)
    end
    R         = R_inflation * cov(R_samples, dims = 2)
    R_inv_var = sqrt(inv(Symmetric(R)))

    # 4. Initial-condition perturbation covariance
    cov_solve   = lorenz_solve(truth_params, x0, LorenzConfig(dt, ic_cov_T))
    ic_cov      = ic_cov_scaling * cov(cov_solve, dims = 2)
    ic_cov_sqrt = sqrt(Symmetric(ic_cov))

    return PerfectDataConfig(
        truth_params, nx, ny,
        picking_initial_condition, Vector{Float64}(x_initial),
        lorenz_config_settings, observation_config,
        R_n_samples, Float64(R_inflation), Float64(ic_cov_T), Float64(ic_cov_scaling),
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
