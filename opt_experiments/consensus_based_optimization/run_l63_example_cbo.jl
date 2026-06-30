# CBO — L63 opt experiment
# Consensus-based optimization on the L63 quadratic likelihood.
# Toggle CBO_METHOD in experiment_config.jl to switch between CBO1 (first-order)
# and CBO2 (second-order).  Cost metric: count × N_ens forward-model evaluations.
#
# Local (all cells):  julia --project=. run_l63_example_cbo.jl
# Local (one cell):   julia --project=. run_l63_example_cbo.jl <task_index>

using Dates
using Distributions
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using ConsensusOptimization
using JLD2
using LinearAlgebra
using Random
using Statistics

const EKP = EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
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
###############  Problem setup  #######################################
########################################################################

function build_l63_problem(output_dir)
    nx = 3; nu = 2; ny = 9; t = 0.01; T = 40.0
    u_truth = EnsembleMemberConfig([28.0, 8.0 / 3.0])

    prior = ParameterDistribution(
        Parameterized(MvNormal([3.3, 1.2], [0.15^2 0.0; 0.0 0.5^2])),
        repeat([no_constraint()], nu),
        "l63_prior",
    )

    prelim_file = joinpath(output_dir, "l63_computed_preliminaries.jld2")
    if isfile(prelim_file)
        ld = load_preliminaries(prelim_file)
        @info "Loaded L63 preliminaries from $prelim_file"
    else
        rng_i     = MersenneTwister(11)
        x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
        ld = compute_perfect_data(
            u_truth, nx, ny,
            LorenzConfig(t, 1000.0), x_initial,
            LorenzConfig(t, T), ObservationConfig(30.0, T),
        )
        save_preliminaries(ld, prelim_file)
        @info "Saved L63 preliminaries to $prelim_file"
    end

    return (; x0          = ld.x0,
              y           = ld.y,
              R           = ld.R,
              R_inv_var   = ld.R_inv_var,
              ic_cov_sqrt = ld.ic_cov_sqrt,
              lorenz_cfg  = ld.lorenz_config_settings,
              obs_cfg     = ld.observation_config,
              nx, nu, ny, prior)
end

########################################################################
###############  Run one (N_ens, rmse_target, rng_idx) cell  ##########
########################################################################

function run_one(cfg, N_ens, rmse_target, rng_idx, prob)
    (; x0, y, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx, nu, ny, prior) = prob

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
            EnsembleMemberConfig(exp.(ens_mean)),
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
                EnsembleMemberConfig(exp.(params_i[:, j])),
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

    @info "N_ens=$(N_ens)  rmse_target=$(rmse_target)  rng_idx=$(rng_idx)  conv=$(conv_score)"
    return (; conv_score, final_params, final_output)
end

########################################################################
###############  Main  ################################################
########################################################################

function main()
    cfg   = experiment_config(:l63)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    prob = build_l63_problem(output_dir)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        N_ens, rmse_target, rng_idx = tasks[t]
        @info "Task $t: N_ens=$(N_ens)  rmse_target=$(rmse_target)  rng_idx=$(rng_idx)"
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
