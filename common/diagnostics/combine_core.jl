# combine_core.jl — merge per-run diagnostics into ONE combined file per experiment (opt and uq).
#
# A method's <method>/diagnostics/combine_diagnostics.jl only states what is specific to the method (tag, slots,
# where its result files are) and calls `DiagCombine.combine_opt(...)`; uq combine scripts read their cells
# themselves and hand the assembled cells to `DiagCombine.write_combined(...)`.
#
# Output layout (one file per experiment; name from DiagFiles.diag_filename):
#   "<key>"          Float32 [record, seed, slot], NaN-padded (records past a run's end, missing seeds)
#   "iter"           Float32 [record, seed, slot] iteration number of each record
#   "n_records"      Int [seed, slot]
#   "status"         String [seed, slot]: opt "converged" | "failed" | "no_diag" | "missing"; uq "reached" | "partial" | "never" | "missing"
#   "metric"         Float64 [seed, slot]: opt conv_score (NaN = failed), as on the leaderboard
#   "seeds", "slot_name", "slot_values", "selected_seeds" (Int [3, slot]; 0 = none), "selected_names", "meta"
#   "meta"           method, case, date, git_hash, descriptions, ... (+ any method-specific entries)

isdefined(@__MODULE__, :DiagFiles) || include(joinpath(@__DIR__, "diag_files.jl"))

module DiagCombine

using JLD2
using Statistics
using ..DiagFiles

export combine_opt, write_combined

const META_KEYS = ("schema_version", "enabled", "iter", "max_records", "descriptions")

"""
    write_combined(path, cells, status, metric, selected; seeds, slot_name, slots, meta, extras, count_labels,
                   summary_labels, skip_keys, tag, case, date)

Assemble `cells[seed, slot]` (a `Dict` of equal-length vectors including "iter", or `nothing`) into the combined
layout above, save it at `path`, and print per-status medians of each key at the final record. `extras` are further
arrays saved as-is (e.g. uq `frac_q_reached`). Returns `path`.
"""
function write_combined(path, cells, status, metric, selected; seeds, slot_name, slots, meta, extras = Dict{String, Any}(),
                        count_labels = ("converged", "failed", "no_diag", "missing"),
                        summary_labels = ("converged", "failed"), skip_keys = ("iter",), tag = "", case = "", date = "")
    n_seed, n_slot = length(seeds), length(slots)
    max_records = maximum([length(c["iter"]) for c in cells if c !== nothing]; init = 0)
    keyset = sort(unique(k for c in cells if c !== nothing for k in keys(c) if !(k in skip_keys)))
    out = Dict{String, Any}()
    out["iter"] = fill(NaN32, max_records, n_seed, n_slot)
    for k in keyset
        out[k] = fill(NaN32, max_records, n_seed, n_slot)
    end
    n_records = zeros(Int, n_seed, n_slot)
    for j in 1:n_slot, i in 1:n_seed
        c = cells[i, j]
        c === nothing && continue
        n = length(c["iter"])
        n_records[i, j] = n
        out["iter"][1:n, i, j] .= Float32.(c["iter"])
        for k in keyset
            haskey(c, k) && (out[k][1:n, i, j] .= Float32.(c[k]))
        end
    end
    out["n_records"] = n_records
    out["status"] = status
    out["metric"] = metric
    out["seeds"] = seeds
    out["slot_name"] = slot_name
    out["slot_values"] = collect(slots)
    out["selected_seeds"] = selected
    out["selected_names"] = ["best_success", "typical_failure", "closest_failure"]
    out["meta"] = meta
    for (k, v) in extras
        out[k] = v
    end

    mkpath(dirname(path))
    JLD2.save(path, out)

    n_found = count(!=("missing"), status)
    println("\n== $(tag) $(case) $(date): $(n_found)/$(n_seed * n_slot) cells found → $path")
    for (j, slot) in enumerate(slots)
        st = status[:, j]
        println("  $(slot_name)=$(slot): ", join(["$(l)=$(count(==(l), st))" for l in count_labels], " "),
                "  selected=$(selected[:, j])")
        for k in keyset
            parts = String[]
            for lab in summary_labels
                idx = findall(==(lab), st)
                vals = filter(isfinite, Float64[out[k][n_records[i, j], i, j] for i in idx if n_records[i, j] > 0])
                push!(parts, isempty(vals) ? "$(lab)=—" : "$(lab)=$(round(median(vals); sigdigits = 3))")
            end
            println("    final $(rpad(k, 22)) ", join(parts, "  "))
        end
    end
    return path
end

"""
    combine_opt(; tag, cases, config, nc_name, result_dir, out_dir, slot_name, slot_values, cell_file,
                closeness = default_closeness, theta_true_note = opt_theta_true_note, n_ens = nothing,
                extra_meta = cfg -> Dict(), git_dir = @__DIR__)

Combine the per-seed result JLD2s (`conv_score` + `"diagnostics"`) of an opt method, one combined file per case.
`config(case)` → cfg, `nc_name(cfg)` → the leaderboard nc filename (gives the file stem and date),
`slot_values(cfg)`/`slot_name(cfg)`/`cell_file(cfg, slot, rng_idx)` define the slot axis (everything in a task
except rng_idx; for methods with an N_ens axis fix N_ens in `cell_file` and pass `n_ens`).
"""
function combine_opt(; tag, cases, config, nc_name, result_dir, out_dir, slot_name, slot_values, cell_file,
                     closeness = default_closeness, theta_true_note = opt_theta_true_note, n_ens = nothing,
                     extra_meta = cfg -> Dict{String, Any}(), git_dir = @__DIR__)
    paths = String[]
    for case in cases
        p = combine_opt_case(case; tag, config, nc_name, result_dir, out_dir, slot_name, slot_values, cell_file,
                             closeness, theta_true_note, n_ens, extra_meta, git_dir)
        p === nothing || push!(paths, p)
    end
    return paths
end

function combine_opt_case(case; tag, config, nc_name, result_dir, out_dir, slot_name, slot_values, cell_file,
                          closeness, theta_true_note, n_ens, extra_meta, git_dir)
    cfg = config(case)
    seeds = collect(1:cfg.n_repeats)
    slots = slot_values(cfg)
    n_seed, n_slot = length(seeds), length(slots)

    cells = Array{Any}(nothing, n_seed, n_slot)
    status = fill("missing", n_seed, n_slot)
    metric = fill(NaN, n_seed, n_slot)
    closev = fill(NaN, n_seed, n_slot)
    descriptions = Dict{String, String}()

    for (j, slot) in enumerate(slots), (i, s) in enumerate(seeds)
        fn = cell_file(cfg, slot, s)
        isfile(fn) || continue
        d = try
            JLD2.load(fn)
        catch err
            @warn "Unreadable: $fn" err
            continue
        end
        metric[i, j] = d["conv_score"]
        dg = get(d, "diagnostics", nothing)
        if dg === nothing || !get(dg, "enabled", false)
            status[i, j] = "no_diag"
            continue
        end
        status[i, j] = isfinite(metric[i, j]) ? "converged" : "failed"
        cells[i, j] = dg
        closev[i, j] = closeness(dg)
        merge!(descriptions, get(dg, "descriptions", Dict{String, String}()))
    end

    if count(!=("missing"), status) == 0
        @info "No result files for $case in $result_dir — skipping"
        return nothing
    end

    selected = reduce(hcat, [pick_seeds_opt(status[:, j], metric[:, j], closev[:, j]) for j in 1:n_slot])
    date = string(cfg.run_date)
    meta = Dict{String, Any}(
        "method" => tag, "case" => String(case), "date" => date,
        "git_hash" => git_hash(git_dir), "theta_true" => theta_true_note(case),
        "descriptions" => descriptions, "result_dir" => abspath(result_dir),
    )
    merge!(meta, extra_meta(cfg))
    path = joinpath(out_dir, diag_filename(nc_name(cfg); n_ens))
    return write_combined(path, cells, status, metric, selected; seeds, slot_name = slot_name(cfg), slots, meta,
                          skip_keys = META_KEYS, tag, case, date)
end

end # module DiagCombine
