# <METHOD_NAME> — L63 opt experiment
# Runs the <METHOD_NAME> optimizer on the Lorenz-63 problem.
# Produces one result file per (N_ens, rng_idx) cell and a leaderboard netcdf.
#
# Local (all cells):    julia --project=. run_l63_<METHOD_NAME>.jl
# Local (one cell):     julia --project=. run_l63_<METHOD_NAME>.jl <task_index>
# SLURM array task:     invoked automatically via run_array.sbatch

using Distributions
using LinearAlgebra
using Random
using Statistics
using JLD2
using Dates

# ── Shared code from common/ ──────────────────────────────────────────────────
const _COMMON = joinpath(@__DIR__, "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))

# Method-specific packages (uncomment/add as needed):
# using <MethodPackage>

include("experiment_config.jl")

########################################################################
###############  Problem setup (L63)  #################################
########################################################################

function build_l63_problem(; rng_seed_init = 11)
    rng_i = MersenneTwister(rng_seed_init)

    t = 0.01
    T = 40.0
    T_long = 1000.0
    nx = 3

    u_truth = EnsembleMemberConfig([28.0, 8.0 / 3.0])
    picking_ic = LorenzConfig(t, T_long)
    x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
    x_spun_up = lorenz_solve(u_truth, x_initial, picking_ic)
    x0 = x_spun_up[:, end]

    lorenz_cfg = LorenzConfig(t, T)
    T_start, T_end = 30.0, T
    obs_cfg = ObservationConfig(T_start, T_end)
    y = lorenz_forward(u_truth, x0, lorenz_cfg, obs_cfg)

    # Observation covariance
    multiple = 36
    window = T_end - T_start
    R_cfg = LorenzConfig(t, multiple * window + T_start)
    R_run = lorenz_solve(u_truth, x_initial, R_cfg)
    R_samples = zeros(length(y), Int(ceil(multiple)))
    for ii in axes(R_samples, 2)
        obs = ObservationConfig(T_start + (ii-1)*window, T_start + ii*window)
        R_samples[:, ii] = stats(R_run, R_cfg, obs)
    end
    R = cov(R_samples, dims=2)
    R_inv_var = sqrt(inv(R))

    # Initial condition perturbation covariance
    cov_run = lorenz_solve(u_truth, x0, LorenzConfig(t, 2000.0))
    ic_cov_sqrt = sqrt(0.1 * cov(cov_run, dims=2))

    return (; x0, y, R, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx)
end

########################################################################
###############  Run one cell  ########################################
########################################################################

function run_one(cfg, N_ens, rng_idx, problem)
    (; x0, y, R, R_inv_var, ic_cov_sqrt, lorenz_cfg, obs_cfg, nx) = problem
    rng = MersenneTwister(rng_idx)

    # Prior
    nu = 2
    prior_mean = [3.3, 1.2]
    prior_cov  = diagm([0.15^2, 0.5^2])
    prior_dist = MvNormal(prior_mean, prior_cov)

    # Initial ensemble (N_params × N_ens)
    initial_params = rand(rng, prior_dist, N_ens)

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  REPLACE: initialize your method here                                   ║
    # ║  (prior, initial_params, y, R, rng are all available)                  ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    # method_state = MyMethod(initial_params, y, R; rng=rng)

    conv_score = NaN    # forward-model eval count to convergence (NaN = did not converge)
    final_params = fill(NaN, nu)
    final_output = fill(NaN, length(y))

    count = 0
    Ne = N_ens   # update if your method changes ensemble size

    for i in 1:cfg.N_iter
        # Get current parameter ensemble (nu × Ne)
        # params_i = get_current_ensemble(method_state)  # REPLACE

        # Placeholder: use initial ensemble (remove when implementing your method)
        params_i = initial_params

        # Evaluate forward map at ensemble mean
        ens_mean = mean(params_i, dims=2)[:]
        G_ens_mean = lorenz_forward(
            EnsembleMemberConfig(exp.(ens_mean)),
            x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, 1),
            lorenz_cfg, obs_cfg,
        )
        RMSE_e = norm(R_inv_var * (y - G_ens_mean[:])) / sqrt(length(y))
        @info "Iter $(i): RMSE = $(RMSE_e)"

        if RMSE_e < cfg.target_rmse
            conv_score  = count * Ne
            final_params = ens_mean
            final_output = G_ens_mean
            break
        end

        # Evaluate forward map at full ensemble
        G_ens = hcat([
            lorenz_forward(
                EnsembleMemberConfig(exp.(params_i[:, j])),
                x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, 1),
                lorenz_cfg, obs_cfg,
            ) for j in 1:Ne
        ]...)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  REPLACE: your method's ensemble update step                    ║
        # ║  Input:  params_i (nu × Ne), G_ens (ny × Ne), y, R            ║
        # ║  Output: updated parameters (via method_state or direct)        ║
        # ╚══════════════════════════════════════════════════════════════════╝
        # update!(method_state, G_ens)

        count += 1
    end

    return (; conv_score, final_params, final_output)
end

########################################################################
###############  Main entry point  ####################################
########################################################################

function main()
    experiment = :l63
    cfg = experiment_config(experiment)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    problem = build_l63_problem()

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]

    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        @info "Task $t: N_ens=$N_ens, rng_idx=$rng_idx"
        result = run_one(cfg, N_ens, rng_idx, problem)
        fn = joinpath(output_dir, result_filename(cfg, N_ens, rng_idx))
        JLD2.save(fn,
            "conv_score",    result.conv_score,
            "final_params",  result.final_params,
            "final_output",  result.final_output,
            "N_ens",         N_ens,
            "rng_idx",       rng_idx,
            "target_rmse",   cfg.target_rmse,
        )
        @info "Saved: $fn"
    end
end

main()
