# combine_diagnostics.jl — <METHOD> (uq) — per-cell diagnostics → ONE file per experiment.
#
# Template: .claude/skills/algorithm-diagnoser/assets/combine_diagnostics_uq.jl
# Lives in <method>/diagnostics/. Fill in the METHOD-SPECIFIC blocks only. The file layout, the save and the
# summary are common/diagnostics/combine_core.jl (`write_combined`); file names come from DiagFiles.
# (Not yet run on a uq method: check the first combined file against the layout below.)
#
# Local (from the method directory):
#   julia --project=. diagnostics/combine_diagnostics.jl                       # every case with cells, CALIBRATE_DATE or today
#   EXPERIMENT=l96_flux CALIBRATE_DATE=2026-07-06 julia --project=. diagnostics/combine_diagnostics.jl
#   DIAG_SOURCE=<dir with cell JLD2s>   DIAG_NC=<leaderboard .nc for status>   (optional overrides)
#
# Env-only CLI on purpose: experiment_config.jl's l96_experiment() reads ARGS[2].
#
# Diagnostics are computed post-hoc from what each cell saved (ensemble history,
# ekpobj, GP waves, ...) plus any instrumented "diagnostics" key. No forward runs.
#
# Output: diagnostics/output/diagnostics_<tag>_<model>[_<fc>]_<date>.jld2 (layout: combine_core.jl) with
#   "<key>"           Float32 [record, seed, slot] (record = iteration / wave / k), NaN-padded
#   "iter"            Float32 [record, seed, slot]
#   "n_records"       Int [seed, slot]
#   "status"          String [seed, slot]: "reached" | "partial" | "never" | "missing"   (budget_to_target, scaling 1)
#   "frac_q_reached"  Float64 [seed, slot]: fraction of coverage quantiles that reached the target
#   "metric"          Float64 [seed, slot]: median budget_to_target over reached quantiles (NaN if none)
#   "seeds", "slot_name", "slot_values", "selected_seeds" (Int [3, slot]; 0 = none), "selected_names", "meta"

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
const METHOD_TAG = "<tag>"            # e.g. "gnki", "history-matching"
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

`f` is an open jldopen handle on one cell. Return "iter" plus one vector per
diagnostic, all the same length (one entry per iteration/wave/k). Read only the
keys you need (`haskey(f, k) && f[k]`). Example below: ensemble-history
diagnostics from `phi_stored` (GNKI/ABC-style); see references/worked-examples.md
for the ekpobj (GNKI) and GP-wave (HM) versions.
"""
function cell_diagnostics(f, cfg)
    ens = f["phi_stored"]                      # Vector over k of (param × member) matrices
    K = length(ens)
    out = Dict{String, Vector{Float64}}("iter" => collect(0:(K - 1)))
    out["spread"] = [sqrt(sum(var(E; dims = 2)) / size(E, 1)) for E in ens]
    out["n_nonfinite"] = [count(!isfinite, E) for E in ens]
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
    n_found = count(!isnothing, cells)
    if n_found == 0
        @info "No cells for $case in $(cell_dir(cfg)) — skipping"
        return nothing
    end

    status = uq_label.(frac)
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
