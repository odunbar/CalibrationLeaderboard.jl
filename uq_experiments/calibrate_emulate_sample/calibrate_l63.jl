# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Dates

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

include(joinpath(@__DIR__, "..", "..", "common", "forward_maps", "Lorenz63.jl")) # Contains Lorenz 96 source code
include("experiment_config.jl")

verbose_flag = false
save_all_ekp = true
########################################################################
############### Choose problem type and structure ######################
########################################################################

cfg         = experiment_config(:l63)
N_ens_sizes = cfg.N_ens_sizes
N_iter      = cfg.N_iter
n_repeats   = cfg.n_repeats
terminate_at = cfg.terminate_at
rng_seeds   = randperm(1_000_000)[1:n_repeats] # list of random seeds
@info "Running Lorenz 63 problem"
@info "Maximum number of EKI iterations: $N_iter"
configuration =
    Dict("N_iter" => N_iter, "N_ens_sizes" => N_ens_sizes, "terminate_at"=> terminate_at, "rng_seeds" => rng_seeds)

nx = 3  # dimensions of parameter vector
nu = 2
truth_params = EnsembleMemberConfig([28.0, 8.0 / 3.0])

#=
the following better encoder this prior:
prior_mean = [3.3, 1.2]
prior_cov = [
    0.15^2 0
    0 0.5^2
]
distribution = Parameterized(MvNormal(prior_mean, prior_cov))
constraint = repeat([no_constraint()], 2) # TODO: fix this... 
name = "l63_prior"
prior = ParameterDistribution(distribution, constraint, name)
=#

prior_r = constrained_gaussian("rho", exp(3.3), 4.153, 0, Inf)
prior_b = constrained_gaussian("beta", exp(1.2), 2.016, 0 ,Inf)
prior = combine_distributions([prior_r, prior_b])
#Creating prior distribution
T = 40.0



########################################################################
############################ Problem setup #############################
########################################################################
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)

output_dir = joinpath(@__DIR__, "output")
if !isdir(output_dir)
    mkdir(output_dir)
end

ny = 9
t = 0.01
T_start = 30.0
T_end = T
pdc = compute_perfect_data(
    truth_params, nx, ny,
    LorenzConfig(t, 1000.0), rand(rng_i, Normal(0.0, 1.0), nx),
    LorenzConfig(t, T), ObservationConfig(T_start, T_end),
)
x0                     = pdc.x0
y                      = pdc.y
lorenz_config_settings = pdc.lorenz_config_settings
observation_config     = pdc.observation_config
R                      = pdc.R
R_inv_var              = pdc.R_inv_var
ic_cov_sqrt            = pdc.ic_cov_sqrt

prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
if !isfile(prelim_file)
    save_preliminaries(pdc, prelim_file)
    @info "Saved computed quantities to $(prelim_file)"
end

########################################################################
########################### Running EKI Race ###########################
########################################################################

conv_alg_iters = fill(NaN, 4, length(N_ens_sizes), length(rng_seeds))
final_parameters = zeros(4, length(N_ens_sizes), length(rng_seeds), nu)
final_model_output = zeros(4, length(N_ens_sizes), length(rng_seeds), ny)

# method_names defined in experiment_config.jl


for (rr, rng_seed) in enumerate(rng_seeds)
    @info "Random seed: $(rng_seed)"
    rng = MersenneTwister(rng_seed)

    for (ee, N_ens) in enumerate(N_ens_sizes)
        # initial parameters: N_params x N_ens
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        methods = [
            Inversion(),
#            TransformInversion(),
#            GaussNewtonInversion(prior),
#            Unscented(prior),
        ]

        @info "Ensemble size: $(N_ens)"
        for (kk, method) in enumerate(methods)
            @info "Method: $(nameof(typeof(method)))"
            if isa(method, Unscented)
                ekpobj = EKP.EnsembleKalmanProcess(
                    y,
                    R,
                    deepcopy(method);
                    rng = copy(rng),
                    verbose = verbose_flag,
                    localization_method = NoLocalization(),
                    scheduler = DataMisfitController(terminate_at = terminate_at),
                )
            else
                ekpobj = EKP.EnsembleKalmanProcess(
                    initial_params,
                    y,
                    R,
                    deepcopy(method);
                    rng = copy(rng),
                    verbose = verbose_flag,
                    localization_method = NoLocalization(),
                    scheduler = DataMisfitController(terminate_at = terminate_at),
                )
            end
            Ne = get_N_ens(ekpobj)

            ens_mean_final = zeros(nu)
            G_ens_mean_final = zeros(ny)
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekpobj)

                # Calculating RMSE_e (diagnostic; not used as stopping criterion)
                ens_mean = mean(params_i, dims = 2)[:] # in constrained_space
                G_ens_mean = lorenz_forward(
                    EnsembleMemberConfig(ens_mean),
                    x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, 1),
                    lorenz_config_settings,
                    observation_config,
                )
                RMSE_e = norm(R_inv_var * (y - G_ens_mean[:])) / sqrt(size(y, 1))
                @info "RMSE (at G(u_mean)): $(RMSE_e)"

                ens_mean_final = ens_mean
                G_ens_mean_final = G_ens_mean[:]

                G_ens = hcat(
                    [
                        lorenz_forward(
                            EnsembleMemberConfig(params_i[:, j]),
                            (x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, Ne))[:, j],
                            lorenz_config_settings,
                            observation_config,
                        ) for j in 1:Ne
                    ]...,
                )
                terminated = EKP.update_ensemble!(ekpobj, G_ens)
                if !isnothing(terminated)
                    conv_alg_iters[kk, ee, rr] = i * Ne 
                    break
                end
            end
            final_parameters[kk, ee, rr, :] = ens_mean_final
            final_model_output[kk, ee, rr, :] = G_ens_mean_final
            if isnan(conv_alg_iters[kk, ee, rr]) # if didnt terminate
                conv_alg_iters[kk, ee, rr] = N_iter * Ne 
            end
            
            final_ensemble = get_ϕ_final(prior, ekpobj)

            # save ekp files
            per_method_dir = joinpath(output_dir, calib_directory(nameof(typeof(method)), cfg))
            if rr == 1 && ee == 1
                if !isdir(per_method_dir)
                    mkpath(per_method_dir)
                end
                JLD2.save(joinpath(per_method_dir, prior_filename(cfg)), "prior", prior)
            end
            if save_all_ekp
                # JLD2
                JLD2.save(
                    joinpath(per_method_dir, ekp_filename(cfg, N_ens, rr)),
                    "N_ens", N_ens,
                    "method", method,
                    "ekpobj", ekpobj,
                )
                u_stored = get_u(ekpobj, return_array = false)
                g_stored = get_g(ekpobj, return_array = false)
                JLD2.save(
                    joinpath(per_method_dir, results_filename(cfg, N_ens, rr)),
                    "y", y,
                    "R", R,
                    "inputs", u_stored,
                    "outputs", g_stored,
                    "truth_params_structure", truth_params, # EnsembleMemberConfig
                )
            end
            
        end
    end
end

# Saving data:
data_filename = joinpath(output_dir, summary_filename(cfg))
JLD2.save(
    data_filename,
    "configuration",
    configuration,
    "method_names",
    method_names,
    "conv_alg_iters",
    conv_alg_iters,
    "final_parameters",
    final_parameters,
    "final_model_output",
    final_model_output,
)
