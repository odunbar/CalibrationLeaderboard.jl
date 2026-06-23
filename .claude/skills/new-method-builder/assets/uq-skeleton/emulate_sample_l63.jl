# <METHOD_NAME> — emulate + sample stage, L63
# For each calibration cell: build a surrogate emulator and draw posterior samples.
#
# Local (all cells):  julia --project=. emulate_sample_l63.jl
# Local (one cell):   julia --project=. emulate_sample_l63.jl <task_index>
# SLURM:              invoked via emulate_sample_array.sbatch

ENV["GKSwstype"] = "100"  # headless GR for cluster runs

using Distributions
using LinearAlgebra
using Random
using JLD2

using CalibrateEmulateSample
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo

include("experiment_config.jl")

function emulate_sample_one(cfg, N_ens, rng_idx, output_dir)
    fn = joinpath(output_dir, results_filename(cfg, N_ens, rng_idx))
    d  = JLD2.load(fn)
    ekpobj = d["ekpobj"]; prior = d["prior"]
    y = d["y"]; R = d["R"]

    posteriors_by_k = []
    K = cfg.max_iter

    for k in 1:K
        # Build emulator from calibration data at iteration k
        input_output_pairs = DataContainers.PairedDataContainer(
            EnsembleKalmanProcesses.get_u(ekpobj, k; return_array=true),
            EnsembleKalmanProcesses.get_g(ekpobj, k; return_array=true),
        )

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  REPLACE: choose your emulator and MCMC settings            ║
        # ╚══════════════════════════════════════════════════════════════╝
        emulator = Emulators.Emulator(
            Emulators.ScalarRandomFeatureInterface(cfg.n_features, size(y,1);
                optimizer_options=Emulators.SKLJL.Hyperparameter_Optimizer(n_features_opt=cfg.n_features_opt)),
            input_output_pairs; obs_noise_cov=R,
        )
        Emulators.optimize_hyperparameters!(emulator)

        mcmc = MCMCWrapper(RWMHSampling(), y, prior, emulator; init_params=get_ϕ_mean_final(prior, ekpobj))
        new_step = optimize_stepsize(mcmc)
        chain = MarkovChainMonteCarlo.sample(MCMCWrapper(RWMHSampling(), y, prior, emulator;
                                             init_params=get_ϕ_mean_final(prior, ekpobj),
                                             mcmc_alg=RWMHSampling(step_size=new_step)),
                                             5_000; discard_initial=1_000)
        push!(posteriors_by_k, chain)
    end

    JLD2.save(joinpath(output_dir, posterior_filename(cfg, N_ens, rng_idx)),
        "posteriors_by_k", posteriors_by_k, "prior", prior,
        "y", d["y"], "R", d["R"],
        "x0", d["x0"], "ic_cov_sqrt", d["ic_cov_sqrt"],
        "lorenz_cfg", d["lorenz_cfg"], "obs_cfg", d["obs_cfg"])
    @info "Emulate+sample done: N_ens=$N_ens, rng_idx=$rng_idx"
end

function main()
    cfg = experiment_config(:l63)
    tasks = flat_tasks(cfg)
    tidx  = task_index_from_args()
    output_dir = joinpath(@__DIR__, "output", calib_directory(cfg))

    run_cells = tidx === nothing ? eachindex(tasks) : [tidx]
    for t in run_cells
        (N_ens, rng_idx) = tasks[t]
        emulate_sample_one(cfg, N_ens, rng_idx, output_dir)
    end
end

main()
