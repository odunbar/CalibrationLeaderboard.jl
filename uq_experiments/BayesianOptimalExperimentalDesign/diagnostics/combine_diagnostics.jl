# combine_diagnostics.jl — BayesianOptimalExperimentalDesign (uq) — per-cell diagnostics → ONE file per experiment.
#
# Lives in BayesianOptimalExperimentalDesign/diagnostics/. The file layout, the save and the summary are
# common/diagnostics/combine_core.jl (`write_combined`); file names come from DiagFiles.
#
# One combined file per (variant, case): the tag is the variant's leaderboard key (`method_key`: boed-tmcmc-eig, boed-tmcmc,
# boed-iekf-eig, boed-iekf), so the four variants never overwrite each other and `DiagFiles.find_combined_by_tag` finds them.
#
# Local (from the method directory):
#   BOED_VARIANT=iekf julia --project=. diagnostics/combine_diagnostics.jl                       # every case with cells, CALIBRATE_DATE or today
#   BOED_VARIANT=iekf EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-09 julia --project=. diagnostics/combine_diagnostics.jl
#   DIAG_SOURCE=<dir with cell JLD2s>   DIAG_NC=<leaderboard .nc for status>   (optional overrides; from-hpc nc files work)
#
# Env-only CLI on purpose: experiment_config.jl's l96_experiment() reads ARGS[2].
#
# Every per-iteration scalar the calibrate stage recorded (diagnostics/boed_diagnostics.jl, keys described in
# describe!) is read from the cell's "diagnostics" key; the objects saved next to them (Z, Yw, gp_hyper_by_k, ...) are NOT
# combined: the offline test scripts in this directory read them from the cell files.
#
# Output: diagnostics/output/diagnostics_<tag>_<model>[_<fc>]_<date>.jld2 (layout: combine_core.jl) with
#   "<key>"           Float32 [record, seed, slot] (record = GBOED iteration k), NaN-padded
#   "iter"            Float32 [record, seed, slot]
#   "n_records"       Int [seed, slot]
#   "status"          String [seed, slot]: "reached" | "partial" | "never" | "missing"   (budget_to_target, scaling 1)
#   "frac_q_reached"  Float64 [seed, slot]: fraction of coverage quantiles that reached the target
#   "metric"          Float64 [seed, slot]: median budget_to_target over reached quantiles (NaN if none)
#   "seeds", "slot_name", "slot_values" (= N_ens), "selected_seeds" (Int [3, slot]; 0 = none), "selected_names", "meta"

using JLD2
using Dates
using LinearAlgebra
using NCDatasets
using Statistics

include(joinpath(@__DIR__, "..", "experiment_config.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "combine_core.jl"))
using .DiagFiles, .DiagCombine

const OUT_DIR = joinpath(@__DIR__, "output")

# ╔══ METHOD-SPECIFIC ═══════════════════════════════════════════════════════╗
const METHOD_TAG = method_key        # "boed-tmcmc-eig" | "boed-tmcmc" | "boed-iekf-eig" | "boed-iekf" (set by BOED_VARIANT)
# Packages needed to deserialise saved objects, e.g.:
# using EnsembleKalmanProcesses, EnsembleKalmanProcesses.ParameterDistributions

slot_name(cfg) = "ensemble_size"
slot_values(cfg) = cfg.N_ens_sizes
cell_dir(cfg) = get(ENV, "DIAG_SOURCE", joinpath(@__DIR__, "..", "output", calib_directory(cfg)))
cell_file(cfg, N_ens, rng_idx) = joinpath(cell_dir(cfg), results_filename(cfg, N_ens, rng_idx))
nc_path(cfg) = get(ENV, "DIAG_NC", joinpath(@__DIR__, "..", "output", nc_filename(cfg)))
run_date(cfg) = cfg.calibrate_date

"""
    cell_diagnostics(f, cfg) -> Dict{String, Vector{Float64}}

`f` is an open jldopen handle on one cell. Returns "iter" plus every per-iteration scalar the run recorded, and three derived keys.
"""
function cell_diagnostics(f, cfg)
    haskey(f, "diagnostics") || error("cell has no \"diagnostics\" key (run predates the instrumentation, or DIAG=0)")
    d = f["diagnostics"]
    get(d, "enabled", true) || error("cell was run with DIAG=0")
    it = Int.(d["iter"])
    out = Dict{String, Vector{Float64}}("iter" => Float64.(it))
    for (k, v) in d
        (k == "iter" || !(v isa AbstractVector{Float32}) || length(v) != length(it)) && continue
        out[k] = Float64.(v)
    end
    # derived: how much the EIG optimiser improved the batch over plain posterior draws / over its own starting batch (≈ 0 for non-EIG variants)
    haskey(out, "eig_batch") && (out["eig_gain_over_postdraw"] = out["eig_batch"] .- out["eig_postdraw"])
    haskey(out, "eig_batch") && (out["eig_gain_over_init"] = out["eig_batch"] .- out["eig_init"])
    # derived: surrogate log-likelihood at the truth relative to the posterior median (≤ 0 when the posterior sits on a higher-likelihood region than the truth)
    haskey(out, "ll_truth") && (out["ll_truth_minus_post"] = out["ll_truth"] .- out["ll_post_med"])
    return out
end
# ╚══════════════════════════════════════════════════════════════════════════╝

# Graded status from the leaderboard nc (dims: random_seed, ensemble_size, target_scaling, coverage_quantile).
# Also returns per-k coverage error mean_q |S_k(q) - q| keyed by (seed, N_ens) for alignment with records.
function leaderboard_status(cfg, seeds, slots)
    n_seed, n_slot = length(seeds), length(slots)
    frac = fill(NaN, n_seed, n_slot)
    metric = fill(NaN, n_seed, n_slot)
    cov_err = Dict{Tuple{Int, Int}, Tuple{Vector{Float64}, Vector{Float64}}}()
    path = nc_path(cfg)
    if !isfile(path)
        @warn "No leaderboard nc at $path — status will be \"missing\" (set DIAG_NC to a from-hpc file)"
        return frac, metric, cov_err
    end
    NCDataset(path) do ds
        nc_seeds = Int.(ds["random_seed"][:])
        nc_ens = Int.(ds["ensemble_size"][:])
        budget = Array(ds["output_budget_to_target"][:, :, :, :])
        has_cov = haskey(ds, "output_coverage") && haskey(ds, "k_iter")
        cov = has_cov ? Array(ds["output_coverage"][:, :, :, :]) : nothing
        ks = has_cov ? Float64.(ds["k_iter"][:]) : Float64[]
        qs = has_cov ? Float64.(ds["coverage_quantile"][:]) : Float64[]
        for (j, ne) in enumerate(slots), (i, s) in enumerate(seeds)
            si = findfirst(==(s), nc_seeds)
            ei = findfirst(==(ne), nc_ens)
            (si === nothing || ei === nothing) && continue
            b = coalesce.(budget[si, ei, 1, :], NaN)
            frac[i, j] = mean(isfinite, b)
            fb = filter(isfinite, b)
            metric[i, j] = isempty(fb) ? NaN : median(fb)
            if has_cov
                c = coalesce.(cov[si, ei, :, :], NaN)
                cov_err[(s, ne)] = (ks, [mean(abs.(c[k, :] .- qs)) for k in axes(c, 1)])
            end
        end
    end
    return frac, metric, cov_err
end

uq_label(f) = isnan(f) ? "missing" : f == 1 ? "reached" : f == 0 ? "never" : "partial"

# best = highest frac then lowest budget; failures = frac < 1 (typical = median frac, closest = highest frac).
# If nothing failed, "typical_failure" is the seed with the largest budget.
function pick_seeds(frac, metric)
    valid = findall(isfinite, frac)
    isempty(valid) && return [0, 0, 0]
    best = sort(valid; by = i -> (-frac[i], isfinite(metric[i]) ? metric[i] : Inf))[1]
    fail = sort(filter(i -> frac[i] < 1, valid); by = i -> (frac[i], i))
    if isempty(fail)
        worst = sort(valid; by = i -> (isfinite(metric[i]) ? -metric[i] : -Inf))[1]
        return [best, worst, 0]
    end
    return [best, fail[cld(length(fail), 2)], fail[end]]
end

function combine_case(case)
    cfg = experiment_config(case)
    seeds = collect(1:cfg.n_repeats)
    slots = slot_values(cfg)
    n_seed, n_slot = length(seeds), length(slots)
    frac, metric, cov_err = leaderboard_status(cfg, seeds, slots)

    cells = Array{Any}(nothing, n_seed, n_slot)
    for (j, ne) in enumerate(slots), (i, s) in enumerate(seeds)
        fn = cell_file(cfg, ne, s)
        isfile(fn) || continue
        cdg = try
            jldopen(f -> cell_diagnostics(f, cfg), fn, "r")
        catch err
            @warn "Could not read diagnostics from $fn" err
            continue
        end
        if haskey(cov_err, (s, ne))           # align per-k coverage error to records by iteration value
            ks, ce = cov_err[(s, ne)]
            cdg["cov_err"] = [(m = findfirst(==(it), ks)) === nothing ? NaN : ce[m] for it in cdg["iter"]]
        end
        cells[i, j] = cdg
    end
    # keys this variant never populates (tmcmc_* for the IEKF variants, iekf_* and eig_opt_* for the others, eig_* for the no-EIG variants) would be all-NaN panels
    live = Set(k for c in cells if c !== nothing for (k, v) in c if any(isfinite, v))
    for c in cells
        c === nothing && continue
        for k in collect(keys(c))
            (k == "iter" || k in live) || delete!(c, k)
        end
    end
    n_found = count(!isnothing, cells)
    if n_found == 0
        @info "No cells for $case in $(cell_dir(cfg)) — skipping"
        return nothing
    end

    status = uq_label.(frac)
    status[map(isnothing, cells)] .= "missing"   # a leaderboard entry without a (diagnosed) cell is not a diagnosed run; also hides it from pick_seeds
    frac[map(isnothing, cells)] .= NaN
    selected = reduce(hcat, [pick_seeds(frac[:, j], metric[:, j]) for j in 1:n_slot])
    meta = Dict{String, Any}(
        "method" => METHOD_TAG, "case" => String(case), "date" => string(run_date(cfg)),
        "git_hash" => git_hash(@__DIR__), "cell_dir" => abspath(cell_dir(cfg)), "nc" => abspath(nc_path(cfg)),
        "status_rule" => "budget_to_target at target_scaling index 1: reached = all quantiles, never = none",
    )
    path = joinpath(OUT_DIR, diag_filename(nc_filename(cfg)))
    return write_combined(path, cells, status, metric, selected; seeds, slot_name = slot_name(cfg), slots, meta,
                          extras = Dict{String, Any}("frac_q_reached" => frac),
                          count_labels = ("reached", "partial", "never", "missing"), summary_labels = ("reached", "partial", "never"),
                          tag = METHOD_TAG, case, date = string(run_date(cfg)))
end

for case in env_cases(experiments)
    combine_case(case)
end
