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
# A cell is merged from whichever per-k files exist; the k need not be contiguous or start
# at 1. K_total (stored in every per-k file) is the number of k the cell should have; any k
# in 1:K_total without a file is recorded in the combined file as "missing_k" (empty when
# complete), and warned about. Downstream indexes by the k value, so missing k are NaN in the
# leaderboard arrays. NB: budget_to_target takes the first k that is present and within
# tolerance, so a budget that follows a gap is an upper bound. A cell is skipped only if it has
# no per-k files at all.
#
# To fill gaps later: rerun the missing k (resubmit the emulate band for that k; finished
# (N_ens, rng_idx, k) are skipped), delete the combined file, then rerun this script.
#
# Idempotent: a cell whose combined file already exists is skipped (re-merging would
# overwrite any pushforward already appended to it). Per-k files are deleted after a
# successful, complete merge unless KEEP_K_FILES=1; those of an incomplete cell are kept.

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
    # Merge whatever k exist (gaps allowed): downstream indexes by the k value, and an unmerged
    # k simply stays NaN in the leaderboard arrays. The gaps are recorded as "missing_k".
    k_present = [k for k in present if k <= K_total]
    missing_k = setdiff(1:K_total, k_present)
    if !isempty(missing_k)
        @warn "Cell $(suffix): per-k files missing for k=$(missing_k) of 1:$(K_total); merging the $(length(k_present)) available k=$(k_present) and recording missing_k. " *
              "Those k will be NaN downstream (a budget_to_target after a gap is an upper bound). " *
              "Per-k files are kept; to fill the gaps later, rerun the missing k, delete $(basename(out_fn)), then rerun this merge."
    end

    merged = Dict(key => Dict{Int, Any}() for key in merged_by_k_keys)
    constants = Dict{String, Any}()
    for (n, k) in enumerate(k_present)
        d = JLD2.load(kfile(k))
        for key in merged_by_k_keys
            merge!(merged[key], d[key])
        end
        if n == 1   # constants are identical in every per-k file; take them from the first one present
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
        "k_values",                 k_present,
        "K_total",                  K_total,
        "missing_k",                missing_k,   # empty when the cell is complete
    )
    mv(tmp_fn, out_fn; force = true)   # atomic: downstream never sees a partial file
    @info "Merged $(length(k_present)) of $(K_total) k values for $(suffix) -> $(out_fn)"

    # Keep the per-k files of an incomplete cell so the gaps can be filled and re-merged.
    if isempty(missing_k) && get(ENV, "KEEP_K_FILES", "0") != "1"
        foreach(k -> rm(kfile(k); force = true), k_present)
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
