# diag_files.jl — file naming, key vocabulary and loaders shared by every diagnostics script.
#
# Used by combine_core.jl, diagnosis_plot_core.jl and the cross-method comparison scripts in
# <opt|uq>_experiments/diagnostics/<topic>/, so that a filename or a canonical key is defined once.
#
# File grammar (date is always last; the N_ens token is present only when the method has an N_ens axis):
#   diagnostics_<tag>_<model>[_<fc>][_N<n>]_<date>.jld2     combined file (combine_core.jl)
#   diagnosis_<tag>_<model>[_<fc>][_N<n>]_<date>.{png,pdf}  figure (diagnosis_plot_core.jl), next to it
# <tag> carries the method and any variant (e.g. sblm_kgmm_parallel), <model>[_<fc>] is the case stem
# (l63, l96_const-force, ...).
#
# Canonical keys (KEY_ALIASES): the same quantity is recorded under different names (a point value `cos_truth` by LM,
# a window mean `cos_truth_wmean` by Adam) and sometimes different units. `resolve_series(c, "cos_truth")` returns
# whichever the combined file holds, so one PANELS list or comparison table works for every method.

module DiagFiles

using JLD2

export env_cases, CASE_STEM, case_stem, diag_filename, figure_stem, find_combined, find_combined_by_tag, load_combined,
       KEY_ALIASES, resolve_series, git_hash, default_closeness, opt_theta_true_note, pick_seeds_opt

const CASE_STEM = Dict("l63" => "l63", "l96_const" => "l96_const-force", "l96_vec" => "l96_vec-force",
                       "l96_flux" => "l96_flux-force")
case_stem(case) = get(CASE_STEM, String(case), String(case))

# cases to process: EXPERIMENT from the environment (the scripts are env-only because experiment_config.jl reads ARGS[2]), else all
env_cases(all) = haskey(ENV, "EXPERIMENT") ? [Symbol(ENV["EXPERIMENT"])] : all

"""
    diag_filename(nc_name; n_ens = nothing) -> String

Combined-file name from the leaderboard nc name of the same run (the nc name already encodes tag, case and date).
`n_ens` inserts the `_N<n>` token before the date.
"""
function diag_filename(nc_name::AbstractString; n_ens = nothing)
    n = startswith(nc_name, "leaderboard_") ? nc_name[(length("leaderboard_") + 1):end] : nc_name
    n = replace(n, r"\.nc$" => "")
    m = match(r"^(.*)_(\d{4}-\d{2}-\d{2})$", n)
    m === nothing && return "diagnostics_$(n).jld2"
    stem = n_ens === nothing ? m.captures[1] : "$(m.captures[1])_N$(n_ens)"
    return "diagnostics_$(stem)_$(m.captures[2]).jld2"
end

figure_stem(path::AbstractString) = replace(basename(path), r"^diagnostics_" => "diagnosis_", r"\.jld2$" => "")

"""
    find_combined(dir, nc_name; n_ens = nothing, exact = false) -> path or nothing

The combined file for this run: the exact date if `exact` (a date was pinned, e.g. RUN_DATE) or if that file
exists, otherwise the latest date available for the same tag/case/N_ens.
"""
function find_combined(dir, nc_name; n_ens = nothing, exact::Bool = false)
    name = diag_filename(nc_name; n_ens)
    path = joinpath(dir, name)
    (exact || isfile(path)) && return isfile(path) ? path : nothing
    isdir(dir) || return nothing
    stem = replace(name, r"\d{4}-\d{2}-\d{2}\.jld2$" => "")
    hits = sort(filter(f -> startswith(f, stem) && occursin(r"^\d{4}-\d{2}-\d{2}\.jld2$", f[(length(stem) + 1):end]), readdir(dir)))
    return isempty(hits) ? nothing : joinpath(dir, hits[end])
end

"""
    find_combined_by_tag(dir, tag, case; n_ens = nothing) -> (path, n, date) or nothing

Cross-method lookup that needs no config: matches `diagnostics_<tag>[_variant]_<case stem>[_N<n>]_<date>.jld2` in
`dir` (a method's `diagnostics/output`). Without `n_ens`, prefers the smallest N_ens (the like-for-like cell with
single-member methods); the latest date wins.
"""
function find_combined_by_tag(dir, tag, case; n_ens = nothing)
    isdir(dir) || return nothing
    re = Regex("^diagnostics_$(tag)(?:_[a-z]+)*_$(case_stem(case))(?:_N(\\d+))?_(\\d{4}-\\d{2}-\\d{2})\\.jld2\$")
    hits = []
    for f in readdir(dir)
        mt = match(re, f)
        mt === nothing && continue
        n = mt.captures[1] === nothing ? nothing : parse(Int, mt.captures[1])
        push!(hits, (path = joinpath(dir, f), n = n, date = mt.captures[2]))
    end
    isempty(hits) && return nothing
    if n_ens !== nothing
        sel = filter(h -> h.n === nothing || h.n == n_ens, hits)
        isempty(sel) || (hits = sel)
    end
    sort!(hits; by = h -> (h.n === nothing ? 0 : h.n, h.date))
    best_n = hits[1].n
    return last(filter(h -> h.n == best_n, hits))
end

load_combined(path) = JLD2.load(path)

# canonical key => [(stored key, transform to the canonical unit), ...] in order of preference.
const KEY_ALIASES = Dict{String, Vector{Tuple{String, Function}}}(
    "rmse"       => [("rmse", identity)],
    "log10_J"    => [("log10_J", identity)],
    "jac_plaus"  => [("jac_plaus", identity)],
    "log10_g"    => [("log10_g", identity)],
    "cos_truth"  => [("cos_truth", identity), ("cos_truth_wmean", identity)],
    "cos_prev"   => [("cos_prev", identity), ("cos_prev_wmean", identity)],
    "dist_truth" => [("dist_truth", identity)],
    # log10 of the prior-whitened step norm ‖Lp⁻¹Δθ‖ (exact zeros floored at 1e-16)
    "log10_step" => [("log10_step_wmean", identity), ("log10_step", identity),
                     ("step_w_wmean", x -> log10(max(x, 1e-16))), ("step_w", x -> log10(max(x, 1e-16)))],
)

"""
    resolve_series(c, name) -> array or nothing

The `[record, seed, slot]` array for `name` from a loaded combined file `c`: an exact stored key wins, otherwise the
first present alias of a canonical key (see KEY_ALIASES), converted to the canonical unit.
"""
function resolve_series(c, name::AbstractString)
    haskey(c, name) && return c[name]
    for (k, f) in get(KEY_ALIASES, name, Tuple{String, Function}[])
        haskey(c, k) && return f === identity ? c[k] : f.(c[k])
    end
    return nothing
end

git_hash(dir = @__DIR__) = try
    readchomp(`git -C $(dir) rev-parse --short HEAD`)
catch
    "unknown"
end

# Per-seed "closeness" for picking the closest failure: smaller = closer to target. NaN if no usable key.
default_closeness(diag) = haskey(diag, "rmse_wmin") ? minimum(filter(isfinite, diag["rmse_wmin"]); init = Inf) :
                          haskey(diag, "rmse") ? minimum(filter(isfinite, diag["rmse"]); init = Inf) : NaN

# θ_true used by the opt runs' diagnostics (copy of the *_preliminaries.jl values), stored in meta for provenance.
opt_theta_true_note(case) =
    case == :l63       ? "log.([28, 8/3])" :
    case == :l96_const ? "[log(8.0)]" :
    case == :l96_vec   ? "8 .+ 6 sin.(4π (0:39) / 40)" : "none (flux NN weights non-identifiable)"

# best success = converged with lowest metric; typical failure = failed with median closeness;
# closest failure = failed with lowest closeness. Ties → lowest seed. 0 = none.
function pick_seeds_opt(status, metric, closev)
    succ = findall(==("converged"), status)
    fail = findall(==("failed"), status)
    best = isempty(succ) ? 0 : succ[argmin(metric[succ])]
    fc = [(isfinite(closev[i]) ? closev[i] : Inf, i) for i in fail]
    sort!(fc)
    closest = isempty(fc) ? 0 : fc[1][2]
    typical = isempty(fc) ? 0 : fc[cld(length(fc), 2)][2]
    return [best, typical, closest]
end

end # module DiagFiles
