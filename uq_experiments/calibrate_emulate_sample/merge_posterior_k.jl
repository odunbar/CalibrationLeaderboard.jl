# Merge the per-k emulate_sample outputs (output/<calib_dir>/emu_k/*_k<k>.jld2, written
# when emulate_sample_l63.jl / emulate_sample_l96.jl run with K_LO / K_HI set) into the
# combined posterior file posterior_filename(cfg, N_ens, rng_idx), with exactly the keys
# the serial emulate_sample writes. pushforward_from_posterior, the diagnostic plots and
# exp_to_leaderboard then run unchanged.
#
# One task per (N_ens, rng_idx) cell (SLURM_ARRAY_TASK_ID, or first CLI arg; with no index
# every cell is merged serially). Model/case: EXPERIMENT env var (l63, l96_const,
# l96_vec, l96_flux) or second CLI arg.
#
# A cell is merged only if its per-k files cover k = 1:K_total contiguously (K_total is
# stored in every per-k file); otherwise it is skipped with a warning and the per-k files
# are left in place so the missing k can be rerun (resubmit the emulate band for that k;
# finished (N_ens, rng_idx, k) are skipped), then rerun this script.
#
# Idempotent: a cell whose combined file already exists is skipped (re-merging would
# overwrite any pushforward already appended to it). Per-k files are deleted after a
# successful merge unless KEEP_K_FILES=1.

using JLD2
using Distributions
using LinearAlgebra
using Random
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

include("experiment_config.jl")

const merged_by_k_keys = ("posteriors_by_k", "input_output_pairs_by_k", "encoder_schedules_by_k")
const cell_constant_keys = ("priors", "truth_params_constrained", "final_params_constrained", "truth_params")

function merge_one(cfg, N_ens, rng_idx; method = method_cases[1])
    suffix = case_suffix(cfg, N_ens, rng_idx)
    data_save_directory = joinpath(@__DIR__, "output", calib_directory(method, cfg))
    kdir   = emu_k_directory(data_save_directory)
    out_fn = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))

    if isfile(out_fn)
        @info "Combined posterior already present for $(suffix); skipping merge."
        return
    end

    kfile(k) = joinpath(kdir, posterior_k_filename(cfg, N_ens, rng_idx, k))
    present = [k for k in 1:cfg.max_iter if isfile(kfile(k))]
    if isempty(present)
        @warn "No per-k posterior files for $(suffix) (calibrate failed, or fewer than 2 EKP iterations?); skipping."
        return
    end

    K_total = JLD2.load(kfile(first(present)), "K_total")
    missing_k = setdiff(1:K_total, present)
    if !isempty(missing_k)
        @warn "Cell $(suffix): per-k files missing for k=$(missing_k) of 1:$(K_total); NOT merging. Rerun those k, then rerun merge."
        return
    end

    merged = Dict(key => Dict{Int, Any}() for key in merged_by_k_keys)
    constants = Dict{String, Any}()
    for k in 1:K_total
        d = JLD2.load(kfile(k))
        for key in merged_by_k_keys
            merge!(merged[key], d[key])
        end
        if k == 1
            for key in cell_constant_keys
                constants[key] = d[key]
            end
        end
    end

    tmp_fn = joinpath(dirname(out_fn), ".tmp_" * basename(out_fn))   # keep the .jld2 extension (FileIO infers the format from it)
    save(
        tmp_fn,
        "posteriors_by_k",          merged["posteriors_by_k"],
        "input_output_pairs_by_k",  merged["input_output_pairs_by_k"],
        "encoder_schedules_by_k",   merged["encoder_schedules_by_k"],
        "priors",                   constants["priors"],
        "truth_params_constrained", constants["truth_params_constrained"],
        "final_params_constrained", constants["final_params_constrained"],
        "truth_params",             constants["truth_params"],
        "k_values",                 collect(1:K_total),
    )
    mv(tmp_fn, out_fn; force = true)   # atomic: downstream never sees a partial file
    @info "Merged $(K_total) k values for $(suffix) -> $(out_fn)"

    if get(ENV, "KEEP_K_FILES", "0") != "1"
        foreach(k -> rm(kfile(k); force = true), 1:K_total)
    end
end

function main()
    exp   = l96_experiment()   # EXPERIMENT env var, else 2nd CLI arg, else the toggle in experiment_config.jl
    @assert(exp in (:l63, :l96_const, :l96_vec, :l96_flux), "Unknown EXPERIMENT $(exp)")
    cfg   = experiment_config(exp)
    tasks = flat_tasks(cfg)
    idx   = task_index_from_args()

    if isnothing(idx)
        for (N_ens, rng_idx) in tasks
            merge_one(cfg, N_ens, rng_idx)
        end
    else
        if idx < 1 || idx > length(tasks)
            error("SLURM_ARRAY_TASK_ID $(idx) out of range 1:$(length(tasks))")
        end
        N_ens, rng_idx = tasks[idx]
        merge_one(cfg, N_ens, rng_idx)
    end
end

main()
