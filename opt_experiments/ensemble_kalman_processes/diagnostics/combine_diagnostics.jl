# combine_diagnostics.jl — ensemble_kalman_processes (opt) — the "diagnostics" key of an example output file → the
# standard combined layout (common/diagnostics/combine_core.jl): one combined file per (case, N_ens), one slot per
# EKP variant (TEKI, ETKI, GNKI, UKI), [record, seed, slot].
#
# The example scripts run every variant × N_ens × seed in one process and save one output file, so there are no
# per-seed result files to scan; this script only reshapes that file (arrays [iteration, variant, N_ens, seed]).
# status: "converged" when conv_alg_iters > 0, metric = conv_alg_iters (cost, as the example reports it).
#
# Local (from the method directory; it has no Manifest, e.g. use the adam environment):
#   julia --project=../adam diagnostics/combine_diagnostics.jl
#   EXPERIMENT=l96_const julia --project=../adam diagnostics/combine_diagnostics.jl
# Options (env):
#   EXPERIMENT=l63|l96_const|l96_vec|l96_flux   which output file to look for (default l63)
#   EKP_FILE=<path>        combine exactly this output file
#   DIAG_N_ENS=<n>         only this N_ens (default: every N_ens in the file)
#
# Output: diagnostics/output/diagnostics_ekp_<model>[_<fc>]_N<N_ens>_<date>.jld2

using JLD2
include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "combine_core.jl"))
using .DiagFiles, .DiagCombine

const IN_DIR = joinpath(@__DIR__, "..", "output")
const OUT_DIR = joinpath(@__DIR__, "output")

function find_output_file(case)
    haskey(ENV, "EKP_FILE") && return ENV["EKP_FILE"]
    pat = case == "l63" ? r"^l63_output_(\d{4}-\d{2}-\d{2})\.jld2$" :
          Regex("^l96_output_$(replace(case, "l96_" => ""))-force_(\\d{4}-\\d{2}-\\d{2})\\.jld2\$")
    hits = sort(filter(f -> occursin(pat, f), isdir(IN_DIR) ? readdir(IN_DIR) : String[]))
    isempty(hits) && error("No output file for $case in $IN_DIR (run the example script first, or set EKP_FILE)")
    return joinpath(IN_DIR, hits[end])
end

const SKIP = ("schema_version", "enabled", "descriptions", "dims")

function combine_file(file)
    d = JLD2.load(file)
    haskey(d, "diagnostics") || error("$file has no \"diagnostics\" key — rerun the example script")
    dg = d["diagnostics"]
    cfg = d["configuration"]
    raw = get(cfg, "case", "l63")   # "const-force" etc. in the l96 file; "l63" has no entry
    case = raw == "l63" ? "l63" : "l96_" * replace(raw, "-force" => "")
    date = match(r"(\d{4}-\d{2}-\d{2})\.jld2$", file).captures[1]
    variants = [m[2] for m in d["method_names"]]
    keyset = sort([k for (k, v) in dg if !(k in SKIP) && v isa AbstractArray])
    n_seed = length(cfg["rng_seeds"])
    paths = String[]
    for (ee, N_ens) in enumerate(cfg["N_ens_sizes"])
        haskey(ENV, "DIAG_N_ENS") && N_ens != parse(Int, ENV["DIAG_N_ENS"]) && continue
        cells = Array{Any}(nothing, n_seed, length(variants))
        status = fill("missing", n_seed, length(variants))
        metric = fill(NaN, n_seed, length(variants))
        closev = fill(NaN, n_seed, length(variants))
        for kk in eachindex(variants), rr in 1:n_seed
            n = something(findlast(isfinite, Float64.(dg["rmse"][:, kk, ee, rr])), 0)
            n == 0 && continue
            cell = Dict{String, Any}("iter" => collect(1.0:n))
            for k in keyset
                cell[k] = Float64.(dg[k][1:n, kk, ee, rr])
            end
            cells[rr, kk] = cell
            conv = d["conv_alg_iters"][kk, ee, rr]
            status[rr, kk] = conv > 0 ? "converged" : "failed"
            metric[rr, kk] = conv > 0 ? conv : NaN
            closev[rr, kk] = minimum(filter(isfinite, cell["rmse"]); init = Inf)
        end
        selected = reduce(hcat, [pick_seeds_opt(status[:, j], metric[:, j], closev[:, j]) for j in eachindex(variants)])
        meta = Dict{String, Any}("method" => "ekp", "case" => case, "date" => date, "git_hash" => git_hash(@__DIR__),
                                 "theta_true" => opt_theta_true_note(Symbol(case)),
                                 "descriptions" => get(dg, "descriptions", Dict{String, String}()),
                                 "result_dir" => abspath(file), "N_ens" => N_ens, "target_rmse" => cfg["target_rmse"],
                                 "rng_seeds" => cfg["rng_seeds"])
        path = joinpath(OUT_DIR, diag_filename("ekp_$(case_stem(case))_$(date).nc"; n_ens = N_ens))
        push!(paths, write_combined(path, cells, status, metric, selected; seeds = collect(1:n_seed), slot_name = "variant",
                                    slots = variants, meta, tag = "ekp", case, date))
    end
    return paths
end

combine_file(find_output_file(get(ENV, "EXPERIMENT", "l63")))
