# GaussNewtonKalmanInversion — calibrate stage, L63
#
# Runs GaussNewtonInversion (GNKI) to (approximate) convergence for one
# (N_ens, rng_idx) cell, storing the ensemble at every iteration. There is no
# separate emulate_sample stage: the ensemble at each iteration IS the
# UQ estimate of the posterior, and is pushed forward directly in
# pushforward_from_posterior_l63.jl.
#
# Local (all cells):  julia --project=. calibrate_l63.jl
# Local (one cell):   julia --project=. calibrate_l63.jl <task_index>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l63.jl

using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
const EKP = EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

verbose_flag = false

########################################################################

function calibrate_one(cfg, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)

    # ── Problem setup (shared across all cells of this case) ───────────
    nx = 3  # state dimension
    ny = 9  # observation dimension
    truth_params = EnsembleMemberConfig([28.0, 8.0 / 3.0])
    t = 0.01
    T = 40.0

    prelim_file = joinpath(@__DIR__, "output", prelim_filename(cfg))
    if isfile(prelim_file)
        prelim = load_preliminaries(prelim_file)
        x0                     = prelim.x0
        y                      = prelim.y
        lorenz_config_settings = prelim.lorenz_config_settings
        observation_config     = prelim.observation_config
        R                      = prelim.R
        ic_cov_sqrt            = prelim.ic_cov_sqrt
    else
        rng_i = MersenneTwister(11)
        pdc = compute_perfect_data(
            truth_params, nx, ny,
            LorenzConfig(t, 1000.0), rand(rng_i, Normal(0.0, 1.0), nx),
            LorenzConfig(t, T), ObservationConfig(30.0, T),
        )
        x0                     = pdc.x0
        y                      = pdc.y
        lorenz_config_settings = pdc.lorenz_config_settings
        observation_config     = pdc.observation_config
        R                      = pdc.R
        ic_cov_sqrt            = pdc.ic_cov_sqrt
        save_preliminaries(pdc, prelim_file)
        @info "Saved computed quantities to $(prelim_file)"
    end

    # ── Prior ────────────────────────────────────────────────────────
    prior_r = constrained_gaussian("rho", exp(3.3), 4.153, 0, Inf)
    prior_b = constrained_gaussian("beta", exp(1.2), 2.016, 0, Inf)
    prior = combine_distributions([prior_r, prior_b])

    # ── GNKI calibration ────────────────────────────────────────────────
    initial_params = construct_initial_ensemble(rng, prior, N_ens)
    process = GaussNewtonInversion(prior)

    ekpobj = EKP.EnsembleKalmanProcess(
        initial_params, y, R, deepcopy(process);
        rng = copy(rng),
        verbose = verbose_flag,
        localization_method = NoLocalization(),
        scheduler = DataMisfitController(terminate_at = cfg.terminate_at),
    )

    for i in 1:cfg.N_iter
        params_i = get_ϕ_final(prior, ekpobj)
        G_ens = hcat(
            [
                lorenz_forward(
                    EnsembleMemberConfig(params_i[:, j]),
                    x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx),
                    lorenz_config_settings,
                    observation_config,
                ) for j in 1:N_ens
            ]...,
        )
        terminated = EKP.update_ensemble!(ekpobj, G_ens)
        isnothing(terminated) || break
    end

    ϕ_stored = get_ϕ(prior, ekpobj)  # [iteration][param, ens_member]; ϕ_stored[1] = prior draw

    JLD2.save(
        joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
        "phi_stored", ϕ_stored,
        "prior", prior,
        "y", y,
        "R", R,
        "x0", x0,
        "ic_cov_sqrt", ic_cov_sqrt,
        "lorenz_config_settings", lorenz_config_settings,
        "observation_config", observation_config,
        "truth_params_constrained", truth_params.u,
    )
    @info "Calibrate done: N_ens=$N_ens, rng_idx=$rng_idx ($(length(ϕ_stored) - 1) EKI iterations stored)"
end

function main()
    cfg = experiment_config(:l63)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))
    mkpath(output_dir)

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        calibrate_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
