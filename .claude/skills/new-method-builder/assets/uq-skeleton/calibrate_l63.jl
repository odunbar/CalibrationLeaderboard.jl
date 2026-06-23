# <METHOD_NAME> — calibrate stage, L63
# Runs the calibration (EKP) loop for one (N_ens, rng_idx) cell or all cells.
#
# Local (all cells):  julia --project=. calibrate_l63.jl
# Local (one cell):   julia --project=. calibrate_l63.jl <task_index>
# SLURM:              invoked via calibrate_array.sbatch with SCRIPT=calibrate_l63.jl

using Distributions
using LinearAlgebra
using Random
using JLD2

using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses

const _COMMON = joinpath(@__DIR__, "..", "common")
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
include("experiment_config.jl")

########################################################################

function calibrate_one(cfg, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(rng_idx)

    # ── Problem setup ──────────────────────────────────────────────────
    t = 0.01; T = 40.0; nx = 3
    u_truth = EnsembleMemberConfig([28.0, 8.0 / 3.0])
    lorenz_cfg = LorenzConfig(t, T)
    obs_cfg = ObservationConfig(30.0, T)

    x_initial = rand(MersenneTwister(11), Normal(0.0, 1.0), nx)
    x_spun_up = lorenz_solve(u_truth, x_initial, LorenzConfig(t, 1000.0))
    x0 = x_spun_up[:, end]

    y = lorenz_forward(u_truth, x0, lorenz_cfg, obs_cfg)

    multiple = 36; window = T - 30.0
    R_run = lorenz_solve(u_truth, x_initial, LorenzConfig(t, multiple*window + 30.0))
    R_samples = hcat([stats(R_run, LorenzConfig(t, multiple*window+30.0),
                            ObservationConfig(30.0+(ii-1)*window, 30.0+ii*window))
                      for ii in 1:Int(ceil(multiple))]...)
    R = cov(R_samples, dims=2)

    cov_run = lorenz_solve(u_truth, x0, LorenzConfig(t, 2000.0))
    ic_cov_sqrt = sqrt(0.1 * cov(cov_run, dims=2))

    # ── Prior ──────────────────────────────────────────────────────────
    prior = ParameterDistribution(
        Parameterized(MvNormal([3.3, 1.2], diagm([0.15^2, 0.5^2]))),
        repeat([no_constraint()], 2),
        "l63_prior",
    )

    # ── EKP calibration ────────────────────────────────────────────────
    initial_params = construct_initial_ensemble(rng, prior, N_ens)

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  REPLACE: choose your EKP process variant                       ║
    # ║  Options: Inversion(prior), TransformInversion(prior),          ║
    # ║           GaussNewtonInversion(prior), Unscented(prior)         ║
    # ╚══════════════════════════════════════════════════════════════════╝
    process = Inversion(prior)

    ekpobj = EKP.EnsembleKalmanProcess(initial_params, y, R, deepcopy(process);
                                       rng=copy(rng))

    G_ens_store = []
    for i in 1:cfg.N_iter
        params_i = get_ϕ_final(prior, ekpobj)
        G_ens = hcat([lorenz_forward(
            EnsembleMemberConfig(exp.(params_i[:, j])),
            x0 .+ ic_cov_sqrt * rand(rng, Normal(), nx, 1),
            lorenz_cfg, obs_cfg,
        ) for j in 1:N_ens]...)
        EKP.update_ensemble!(ekpobj, G_ens)
        push!(G_ens_store, G_ens)
        if i >= cfg.max_iter; break; end
    end

    JLD2.save(joinpath(output_dir, results_filename(cfg, N_ens, rng_idx)),
        "ekpobj", ekpobj, "prior", prior, "G_ens_store", G_ens_store,
        "y", y, "R", R, "x0", x0, "ic_cov_sqrt", ic_cov_sqrt,
        "lorenz_cfg", lorenz_cfg, "obs_cfg", obs_cfg)
    @info "Calibrate done: N_ens=$N_ens, rng_idx=$rng_idx"
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
