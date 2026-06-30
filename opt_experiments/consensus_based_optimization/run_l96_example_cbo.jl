# CBO — L96 opt experiment
# Consensus-based optimization on the L96 quadratic likelihood.
# Supports three forcing cases: const-force (nu=1), vec-force (nu=40), flux-force (nu=61).
# Toggle CBO_METHOD in experiment_config.jl for CBO1 vs CBO2.
# Cost metric: count × N_ens forward-model evaluations.
#
# Local: EXPERIMENT=l96_const julia --project=. run_l96_example_cbo.jl
#        EXPERIMENT=l96_vec   julia --project=. run_l96_example_cbo.jl
#        EXPERIMENT=l96_flux  julia --project=. run_l96_example_cbo.jl
# One cell: EXPERIMENT=l96_const julia --project=. run_l96_example_cbo.jl <task_idx>

using BSON
using Dates
using Distributions
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using ConsensusOptimization
using Flux
using JLD2
using LinearAlgebra
using Random
using Statistics

const EKP = EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
include("experiment_config.jl")

########################################################################
###############  CBO structs (problem + config)  ######################
########################################################################

struct Problem{FTOrVV <: Union{AbstractFloat, AbstractVector}, SS <: AbstractString}
    cost::Function
    minimizer::FTOrVV
    name::SS
end

function WeightedQuadratic(minimizer::VV, sqrt_inv_Γ::MM) where {VV <: AbstractVector, MM <: AbstractMatrix}
    Problem(x -> norm(sqrt_inv_Γ * (x .- minimizer)), minimizer, "Quadratic-$(length(minimizer))D")
end

struct ConsensusBasedConfig{PP <: Problem, OT <: ODEType, FT <: Real}
    problem::PP
    model::OT
    weight_exponent::FT
    Δt::FT
end

########################################################################
###############  Case-specific problem setup  #########################
########################################################################

function build_l96_problem(case::String, output_dir::String)
    if case == "const-force"
        nx = 40
        phi           = ConstantEMC(8.0)
        phi_structure = nothing
        sample_range  = nothing
        nu = 1
        prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
        T = 14.0; inff = 2.0

    elseif case == "vec-force"
        nx = 40
        pl = 2.0; psig = 3.0
        sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop=nx - 1, step=1)) / nx)
        phi           = VectorEMC(sinusoid)
        phi_structure = nothing
        sample_range  = nothing
        nu = nx
        prior_cov = [psig^2 * exp(-abs(i - j) / pl) for i in 1:nx, j in 1:nx]
        prior = ParameterDistribution(
            Parameterized(MvNormal(8.0 * ones(nx), prior_cov)),
            repeat([no_constraint()], nx),
            "ml96_prior",
        )
        T = 54.0; inff = 2.0

    elseif case == "flux-force"
        nx = 100
        true_sinusoid(x) = 8 .+ 6 * sin.((4 * pi * x) / 10)
        x_train = collect(-5.0:0.01:5.0)
        y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
        true_model, _ = train_network(phi_structure, x_train, y_train)
        sample_range  = Float32.(collect(-5.0:0.1:4.9))
        phi           = FluxEMC(true_model, sample_range)

        prior_sinusoid(x) = 8.02 .+ 6.5 * sin.(1.02 * (4 * pi * x) / 10 + 0.2)
        prior_train = prior_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
        _, prior_mean = train_network(phi_structure, x_train, prior_train)
        nu = length(prior_mean)
        prior_cov = (0.1^2) * I(nu)
        prior = ParameterDistribution(
            Parameterized(MvNormal(prior_mean, prior_cov)),
            repeat([no_constraint()], nu),
            "l96_nn_prior",
        )
        T = 54.0; inff = 2.5

    else
        throw(ArgumentError("Unknown L96 case: $case"))
    end

    ny = nx * 2
    prelim_file = joinpath(output_dir, "l96_computed_preliminaries_$(case).jld2")
    if isfile(prelim_file)
        ld = load_preliminaries(prelim_file)
        @info "Loaded L96 ($case) preliminaries from $prelim_file"
    else
        rng_i     = MersenneTwister(11)
        t         = 0.01
        x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
        ld = compute_perfect_data(
            phi, nx,
            LorenzConfig(t, 1000.0), x_initial,
            LorenzConfig(t, T), ObservationConfig(4.0, T);
            R_inflation = inff,
        )
        save_preliminaries(ld, prelim_file)
        @info "Saved L96 ($case) preliminaries to $prelim_file"
    end

    return (; x0          = ld.x0,
              y           = ld.y,
              R           = ld.R,
              R_inv_var   = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config,
              nx, nu, ny, phi, phi_structure, sample_range, prior, case)
end

########################################################################
###############  Run one (N_ens, rmse_target, rng_idx) cell  ##########
########################################################################

function run_one(cfg, N_ens, rmse_target, rng_idx, prob)
    (; x0, y, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg,
       nx, nu, ny, phi, phi_structure, sample_range, prior, case) = prob

    rng = MersenneTwister(rng_idx)

    cbo_model = cfg.cbo_method == :CBO1 ?
        FirstOrder(cfg.sigma, cfg.lambda, true) :
        SecondOrder(cfg.sigma, cfg.inertia, true)

    cbo_cfg = ConsensusBasedConfig(
        WeightedQuadratic(y, R_inv_var),
        cbo_model,
        cfg.weight_exponent,
        cfg.Δt,
    )

    initial_params = construct_initial_ensemble(rng, prior, N_ens)
    # CBO2 state is [θ; momentum], so rows double to 2*nu
    state_rows  = cfg.cbo_method == :CBO2 ? 2 * ndims(prior) : ndims(prior)
    param_state = zeros(cfg.N_iter + 1, state_rows, N_ens)
    param_state[1, 1:ndims(prior), :] = initial_params

    conv_score   = NaN
    final_params = fill(NaN, nu)
    final_output = fill(NaN, ny)

    count = 0
    for i in 1:cfg.N_iter
        params_i_unconstrained = param_state[i, 1:ndims(prior), :]
        params_i = transform_unconstrained_to_constrained(prior, params_i_unconstrained)

        ens_mean   = mean(params_i, dims=2)[:]
        G_ens_mean = lorenz_forward(
            build_forcing(phi, ens_mean, phi_structure, sample_range),
            x0 .+ ic_cov_sqrt * randn(rng, nx),
            lorenz_cfg, obs_cfg,
        )
        RMSE_e = norm(R_inv_var * (y - G_ens_mean[:])) / sqrt(ny)

        if RMSE_e < rmse_target
            conv_score   = count * N_ens
            final_params = ens_mean
            final_output = G_ens_mean[:]
            break
        end

        ic_pert = x0 .+ ic_cov_sqrt * randn(rng, nx, N_ens)
        G_ens = reduce(hcat, [
            lorenz_forward(
                build_forcing(phi, params_i[:, j], phi_structure, sample_range),
                ic_pert[:, j],
                lorenz_cfg, obs_cfg,
            ) for j in 1:N_ens
        ])

        param_state[i+1, :, :] = update_ensemble(
            param_state[i, :, :],
            G_ens,
            cbo_cfg.problem.cost,
            cbo_cfg.weight_exponent,
            cbo_cfg.Δt,
            i,
            cbo_cfg.model;
            rng = rng,
        )
        count += 1
    end

    @info "case=$(case)  N_ens=$(N_ens)  rmse_target=$(rmse_target)  rng_idx=$(rng_idx)  conv=$(conv_score)"
    return (; conv_score, final_params, final_output)
end

########################################################################
###############  Main  ################################################
########################################################################

function main()
    experiment = l96_experiment()
    cfg        = experiment_config(experiment)
    tasks      = flat_tasks(cfg)
    tidx       = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prob = build_l96_problem(cfg.force_case, output_dir)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        N_ens, rmse_target, rng_idx = tasks[t]
        @info "Task $t: case=$(cfg.force_case)  N_ens=$(N_ens)  rmse_target=$(rmse_target)  rng_idx=$(rng_idx)"
        result = run_one(cfg, N_ens, rmse_target, rng_idx, prob)
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rmse_target, rng_idx))
        JLD2.save(fn,
            "conv_score",   result.conv_score,
            "final_params", result.final_params,
            "final_output", result.final_output,
            "N_ens",        N_ens,
            "rmse_target",  rmse_target,
            "rng_idx",      rng_idx,
        )
        @info "Saved: $fn"
    end
end

main()
