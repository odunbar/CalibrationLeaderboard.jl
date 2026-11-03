# Post-hoc removal of nullspace noise from existing :PCA posterior files.
#
# Runs made before the noise injector was tied to DIM_REDUCTION (see experiment_config.jl,
# `noise_injector_threshold_case`) decoded their MCMC samples with CES's default threshold
# (0.001). Where the PCA input encoder dropped more than that fraction of the prior variance
# (l96_flux and l96_vec at 90% input retention), prior-conditional noise was injected into the
# dropped directions, which inflates the posterior (see README, "Nullspace noise injector").
#
# The saved samples can be corrected exactly, without rerunning emulate_sample or MCMC: for the
# injected decode, encode(sample) == the MCMC chain's encoded sample z (the noise lives in the
# encoder's null space and is annihilated by encoding), so decode_data(encode_data(sample)) is
# precisely what a run with the injector off produces for the SAME chain. The kept coordinates
# are unchanged; the dropped ones are pinned at the encoder mean.
#
# For every cell this script, per k, decides whether a correction applies
#   (injector was on at the original threshold) && (injector is off at the current threshold),
# replaces posteriors_by_k[k], and DROPS the stored pushforward arrays (they were computed from
# the injected samples) so that pushforward_from_posterior_l*.jl recomputes them. Rerun, in order:
#   pushforward_from_posterior_l*.jl -> posterior_diagnostic_plots_l*.jl -> exp_to_leaderboard.jl
# (hpc-variant/submit_correct_nullspace.sh chains all of this).
#
# Safe by construction:
#   * idempotent: corrected files carry "nullspace_noise_removed" = true and are skipped;
#   * cells/k where no correction applies are untouched (the file is not rewritten at all);
#   * the correction is verified (re-encoding the new samples must reproduce the chain) before
#     anything is written; the new file is written to a temp name first;
#   * the original file is moved to <calib_dir>/injected_orig/ (KEEP_ORIG=0 deletes it instead).
#
# Run with the same DIM_REDUCTION (must be PCA), EXPERIMENT and CALIBRATE_DATE as the run being
# corrected. One task per (N_ens, rng_idx) cell (SLURM_ARRAY_TASK_ID or first CLI arg; with no
# index every cell is processed serially). EXPERIMENT is required (including l63).
# OUTPUT_ROOT overrides the output directory (default: <this dir>/output), e.g. for testing.

using JLD2
using LinearAlgebra
using Statistics
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions

include("experiment_config.jl")

# CES default; what the runs being corrected used.
const ORIGINAL_THRESHOLD = 0.001
const PUSHFORWARD_KEYS = (
    "pushforward_forcing_samples",
    "pushforward_output_samples",
    "pushforward_k_values",
    "pushforward_n_samples",
    "truth_forcing",
)

injector_on(schedule, prior, threshold) = begin
    inj = Utilities.create_noise_injector(schedule, prior, threshold, 1.0)
    !isnothing(inj) && inj.use_noise
end

# Same construction as CES's get_posterior (per-name atoms combined), from a (n_params x n_samples) array.
function rebuild_posterior(post, samples::AbstractMatrix)
    names = get_name(post)
    slices = batch(post)
    flat_constraints = get_all_constraints(post)
    return combine_distributions([
        ParameterDistribution(Samples(samples[sl, :]; params_are_columns = true), flat_constraints[sl], nm) for
        (nm, sl) in zip(names, slices)
    ])
end

function correct_one(cfg, N_ens, rng_idx; method = method_cases[1])
    root = get(ENV, "OUTPUT_ROOT", joinpath(@__DIR__, "output"))
    dir = joinpath(root, calib_directory(method, cfg))
    fn = joinpath(dir, posterior_filename(cfg, N_ens, rng_idx))
    suffix = case_suffix(cfg, N_ens, rng_idx)

    if !isfile(fn)
        @warn "No combined posterior file for $(suffix) ($(fn)); skipping."
        return
    end
    loaded = JLD2.load(fn)
    if get(loaded, "nullspace_noise_removed", false)
        @info "Already corrected: $(suffix); skipping."
        return
    end

    prior = loaded["priors"]
    schedules = loaded["encoder_schedules_by_k"]
    posteriors = loaded["posteriors_by_k"]

    new_posteriors = Dict{Int, Any}()
    fixed_ks = Int[]
    for k in sort(collect(keys(posteriors)))
        sched = schedules[k]
        if !(injector_on(sched, prior, ORIGINAL_THRESHOLD) && !injector_on(sched, prior, noise_injector_threshold_case))
            new_posteriors[k] = posteriors[k]
            continue
        end
        post = posteriors[k]
        P = reduce(vcat, [get_distribution(post)[nm] for nm in get_name(post)])
        Z = Utilities.encode_data(sched, P, "in")
        Pp = Utilities.decode_data(sched, Z, "in")
        # verification: the corrected samples must encode back to the same chain
        err = maximum(abs.(Utilities.encode_data(sched, Pp, "in") .- Z))
        err <= 1e-8 * (1 + maximum(abs.(Z))) ||
            error("Correction check failed for $(suffix), k=$(k): re-encoded samples differ from the chain by $(err)")
        new_posteriors[k] = rebuild_posterior(post, Pp)
        push!(fixed_ks, k)
    end

    if isempty(fixed_ks)
        @info "No correction applies to $(suffix) (injector was off, or is still on at threshold $(noise_injector_threshold_case)); file untouched."
        return
    end

    out = Dict{String, Any}(key => val for (key, val) in loaded if !(key in PUSHFORWARD_KEYS))
    out["posteriors_by_k"] = new_posteriors
    out["nullspace_noise_removed"] = true
    out["nullspace_noise_threshold"] = noise_injector_threshold_case
    out["nullspace_noise_fixed_ks"] = fixed_ks

    tmp = joinpath(dir, ".tmp_" * basename(fn))   # keep the .jld2 extension (FileIO infers the format from it)
    JLD2.save(tmp, out)
    if get(ENV, "KEEP_ORIG", "1") == "1"
        backup_dir = joinpath(dir, "injected_orig")
        mkpath(backup_dir)
        mv(fn, joinpath(backup_dir, basename(fn)); force = true)
    else
        rm(fn)
    end
    mv(tmp, fn; force = true)
    @info "Corrected $(suffix): k = $(fixed_ks); dropped pushforward arrays (rerun pushforward_from_posterior)."
end

function main()
    dim_reduction_case == :PCA || error("correct_nullspace_noise.jl is for DIM_REDUCTION=PCA (got $(dim_reduction_case)).")
    isinf(noise_injector_threshold_case) ||
        @warn "noise_injector_threshold_case = $(noise_injector_threshold_case) (not Inf): only cells whose input-encoder loss is below it are corrected."
    exp = l96_experiment()   # EXPERIMENT env var, else 2nd CLI arg, else the toggle in experiment_config.jl
    @assert(exp in (:l63, :l96_const, :l96_vec, :l96_flux), "Unknown EXPERIMENT $(exp)")
    cfg = experiment_config(exp)
    tasks = flat_tasks(cfg)
    idx = task_index_from_args()

    if isnothing(idx)
        for (N_ens, rng_idx) in tasks
            correct_one(cfg, N_ens, rng_idx)
        end
    else
        (idx < 1 || idx > length(tasks)) && error("SLURM_ARRAY_TASK_ID $(idx) out of range 1:$(length(tasks))")
        N_ens, rng_idx = tasks[idx]
        correct_one(cfg, N_ens, rng_idx)
    end
end

main()
