# compare_common.jl — shared loader for the cross-variant tests (test3_eig_placement.jl, test4_eig_helpfulness.jl).
#
# Reads the combined files written by combine_diagnostics.jl (one per variant × case) from diagnostics/output/. No algorithm code is loaded.
#
# Environment: EXPERIMENT=l63|l96_const|l96_vec|l96_flux (default l63)
#              CALIBRATE_DATE=<date> pin the combined files' date (default: the latest date found per variant; set it when the variants were run on different dates)
#              DIAG_N_ENS=<N_ens> choose the slot (default: the middle N_ens)    DIAG_VARIANTS=tmcmc_eig,iekf_eig,... (default all four)

using JLD2
using Plots
using Statistics

const COMPARE_OUT = joinpath(@__DIR__, "output")
const VARIANT_TAGS = Dict("tmcmc_eig" => "boed-tmcmc-eig", "tmcmc" => "boed-tmcmc", "iekf_eig" => "boed-iekf-eig", "iekf" => "boed-iekf")
const VARIANT_COLORS = Dict("tmcmc_eig" => "#2a78d6", "tmcmc" => "#8ab6ee", "iekf_eig" => "#eb6834", "iekf" => "#f4b093")
const VARIANT_STYLES = Dict("tmcmc_eig" => :solid, "tmcmc" => :dash, "iekf_eig" => :solid, "iekf" => :dash)
const CASE_STEMS = Dict("l63" => "l63", "l96_const" => "l96_const-force", "l96_vec" => "l96_vec-force", "l96_flux" => "l96_flux-force")

compare_case() = get(ENV, "EXPERIMENT", "l63")
compare_variants() = haskey(ENV, "DIAG_VARIANTS") ? String.(split(ENV["DIAG_VARIANTS"], ",")) : ["tmcmc_eig", "tmcmc", "iekf_eig", "iekf"]

# Latest (or pinned-date) combined file for a variant: diagnostics_<tag>_<case stem>[_ensemble_results]_<date>.jld2
function find_variant_file(variant::AbstractString, case::AbstractString)
    tag, stem = VARIANT_TAGS[variant], CASE_STEMS[case]
    re = Regex("^diagnostics_$(tag)_$(stem)(?:_ensemble_results)?_(\\d{4}-\\d{2}-\\d{2})\\.jld2\$")
    hits = [(m.captures[1], f) for f in (isdir(COMPARE_OUT) ? readdir(COMPARE_OUT) : String[]) for m in [match(re, f)] if m !== nothing]
    haskey(ENV, "CALIBRATE_DATE") && filter!(h -> h[1] == ENV["CALIBRATE_DATE"], hits)
    isempty(hits) && return nothing
    return joinpath(COMPARE_OUT, sort(hits)[end][2])
end

"""
    load_variants(case) -> Dict{String, Dict}  (variant => combined-file contents; only variants that have a file)
"""
function load_variants(case::AbstractString)
    out = Dict{String, Dict{String, Any}}()
    for v in compare_variants()
        path = find_variant_file(v, case)
        if path === nothing
            @info "No combined diagnostics file for variant $v / $case in $COMPARE_OUT — run combine_diagnostics.jl for it (BOED_VARIANT=$v)"
        else
            out[v] = JLD2.load(path)
            out[v]["__path"] = path
        end
    end
    isempty(out) && error("No combined files found for $case in $COMPARE_OUT")
    return out
end

# The slot index (N_ens) shared by the variants: DIAG_N_ENS, else the middle slot of the first variant.
function chosen_slot(vars)
    c = first(values(vars))
    if haskey(ENV, "DIAG_N_ENS")
        j = findfirst(==(parse(Int, ENV["DIAG_N_ENS"])), c["slot_values"])
        j === nothing && error("DIAG_N_ENS=$(ENV["DIAG_N_ENS"]) is not one of the file's N_ens values $(c["slot_values"])")
        return j
    end
    return cld(length(c["slot_values"]), 2)
end

# [record, seed] matrix of `key` for slot j, only for seeds that have records (NaN-padded), or nothing if the key was never recorded for this variant.
function seed_matrix(c, key, j)
    haskey(c, key) || return nothing
    keep = findall(>(0), c["n_records"][:, j])
    isempty(keep) && return nothing
    return Float64.(c[key][:, keep, j])
end

finite_med(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
finite_q(x, q) = (v = filter(isfinite, x); isempty(v) ? NaN : quantile(v, q))

# median and IQR across seeds at every record
function band_stats(M::AbstractMatrix)
    med = [finite_med(M[r, :]) for r in axes(M, 1)]
    lo = [finite_q(M[r, :], 0.25) for r in axes(M, 1)]
    hi = [finite_q(M[r, :], 0.75) for r in axes(M, 1)]
    return med, lo, hi
end

# Median ± IQR band of `key` over seeds for each variant that recorded it (dotted line style for the no-EIG variants).
function variant_band_panel(vars, key, j; title, yscale = :identity, ref = nothing, baseline_key = nothing, legend = false)
    p = plot(; title, titlefontsize = 9, xlabel = "GBOED iteration k", yscale, legend)
    for (v, c) in sort(collect(vars); by = first)
        for (kk, ls, alpha, lab) in ((key, VARIANT_STYLES[v], 0.15, v), (baseline_key, :dot, 0.0, nothing))
            kk === nothing && continue
            M = seed_matrix(c, kk, j)
            M === nothing && continue
            med, lo, hi = band_stats(M)
            ok = findall(isfinite, med)
            isempty(ok) && continue
            yscale === :log10 && (med = max.(med, 1e-12); lo = max.(lo, 1e-12); hi = max.(hi, 1e-12))
            plot!(p, ok, med[ok]; ribbon = (med[ok] .- lo[ok], hi[ok] .- med[ok]), fillalpha = alpha, color = VARIANT_COLORS[v], ls, lw = 2, label = lab === nothing ? false : lab)
        end
    end
    ref === nothing || hline!(p, [ref]; color = :gray, ls = :dash, label = false)
    return p
end
