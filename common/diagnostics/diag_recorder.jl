# diag_recorder.jl — thinning per-iteration diagnostics recorder.
#
# The single copy, shared by every method: included by <method>/diagnostics/*_diagnostics.jl as
#   include(joinpath(@__DIR__, "..", "..", "..", "common", "diagnostics", "diag_recorder.jl"))
#
# Base only (no package deps), never touches an RNG, never mutates its inputs.
#
#   rec = DiagRecorder(N_iter)                     # DIAG=0 disables; DIAG_MAX_RECORDS caps rows
#   want(rec, it)                                  # is `it` a scheduled record?
#   record!(rec, it; rmse = r, log10_J = j)        # one row (only if want(rec, it))
#   accumulate!(rec, "cos_prev", c)                # every iteration; flushed as cos_prev_wmean
#   accumulate!(rec, "rmse", r; mode = :min)       # flushed as rmse_wmin (also :max)
#   finalize!(rec, it; rmse = r)                   # forces a row for the last iteration
#   describe!(rec, "rmse" => "whitened RMSE before the step")
#   diagnostics_dict(rec)                          # Dict{String,Any} to JLD2.save
#
# Vector values with length <= 10 are expanded into key_1, key_2, ...; longer
# vectors are rejected (storage cap: store scalars and small nu-vectors only).

const DIAG_SCHEMA_VERSION = 1

"""
    diag_schedule(N_iter; max_records = 500, n_dense = 50) -> BitVector

Dense for the first `n_dense` iterations, log-spaced after, last iteration always included.
"""
function diag_schedule(N_iter::Integer; max_records::Integer = 500, n_dense::Integer = 50)
    sched = falses(N_iter)
    N_iter == 0 && return sched
    if N_iter <= max_records
        sched .= true
        return sched
    end
    n_dense = min(n_dense, max_records - 1)
    sched[1:n_dense] .= true
    n_log = max_records - n_dense
    lo, hi = log(n_dense + 1), log(N_iter)
    for k in 0:(n_log - 1)
        it = round(Int, exp(lo + (hi - lo) * k / max(n_log - 1, 1)))
        sched[clamp(it, 1, N_iter)] = true
    end
    sched[N_iter] = true
    return sched
end

mutable struct DiagRecorder
    enabled::Bool
    schedule::BitVector
    max_records::Int
    iters::Vector{Int}
    data::Dict{String, Vector{Float32}}
    acc::Dict{String, Tuple{Symbol, Float64, Int}}   # key => (mode, value, count)
    descriptions::Dict{String, String}
    finalized::Bool
end

function DiagRecorder(
    N_iter::Integer;
    max_records::Integer = parse(Int, get(ENV, "DIAG_MAX_RECORDS", "500")),
    n_dense::Integer = 50,
    enabled::Bool = get(ENV, "DIAG", "1") != "0",
)
    max_records = clamp(max_records, 2, 1000)
    return DiagRecorder(
        enabled,
        diag_schedule(N_iter; max_records, n_dense),
        max_records,
        Int[],
        Dict{String, Vector{Float32}}(),
        Dict{String, Tuple{Symbol, Float64, Int}}(),
        Dict{String, String}(),
        false,
    )
end

want(rec::DiagRecorder, it::Integer) =
    rec.enabled && !rec.finalized && 1 <= it <= length(rec.schedule) && rec.schedule[it]

function describe!(rec::DiagRecorder, descs::Pair...)
    for (k, v) in descs
        rec.descriptions[String(k)] = String(v)
    end
    return rec
end

# cosine of two vectors; NaN if either is zero or non-finite
function diag_cos(a, b)
    na = sqrt(sum(abs2, a))
    nb = sqrt(sum(abs2, b))
    (na > 0 && nb > 0 && isfinite(na) && isfinite(nb)) || return NaN
    return sum(a .* b) / (na * nb)
end

_diag_f32(x::Bool) = Float32(x)
_diag_f32(x::Real) = Float32(x)   # values beyond Float32 range become ±Inf32 (store log10 of huge norms)
_diag_f32(::Nothing) = NaN32

function accumulate!(rec::DiagRecorder, key::AbstractString, x::Real; mode::Symbol = :mean)
    rec.enabled || return rec
    isfinite(x) || return rec          # window statistics skip NaN/Inf; count those separately if needed
    mode in (:mean, :min, :max) || throw(ArgumentError("mode must be :mean, :min or :max"))
    m, v, c = get(rec.acc, key, (mode, mode === :min ? Inf : mode === :max ? -Inf : 0.0, 0))
    v = mode === :mean ? v + x : mode === :min ? min(v, x) : max(v, x)
    rec.acc[key] = (mode, v, c + 1)
    return rec
end

function _diag_setcol!(rec::DiagRecorder, key::String, n_before::Int, val::Float32)
    col = get!(() -> fill(NaN32, n_before), rec.data, key)
    length(col) == n_before && push!(col, NaN32)
    col[end] = val
    return nothing
end

function _diag_push_row!(rec::DiagRecorder, it::Integer, vals)
    n = length(rec.iters)
    push!(rec.iters, it)
    for col in values(rec.data)
        push!(col, NaN32)
    end
    for (k, v) in pairs(vals)
        ks = String(k)
        if v isa AbstractVector
            length(v) <= 10 || throw(ArgumentError("diagnostic `$ks` has length $(length(v)) > 10; store a summary scalar instead"))
            for (i, vi) in enumerate(v)
                _diag_setcol!(rec, "$(ks)_$(i)", n, _diag_f32(vi))
            end
        else
            _diag_setcol!(rec, ks, n, _diag_f32(v))
        end
    end
    for (k, (mode, v, c)) in rec.acc
        suffix = mode === :mean ? "_wmean" : mode === :min ? "_wmin" : "_wmax"
        val = c == 0 ? NaN32 : _diag_f32(mode === :mean ? v / c : v)
        _diag_setcol!(rec, k * suffix, n, val)
        _diag_setcol!(rec, k * "_wcount", n, Float32(c))
        rec.acc[k] = (mode, mode === :min ? Inf : mode === :max ? -Inf : 0.0, 0)
    end
    return rec
end

function record!(rec::DiagRecorder, it::Integer; kwargs...)
    want(rec, it) || return rec
    return _diag_push_row!(rec, it, kwargs)
end

"""
    finalize!(rec, it; kwargs...)

Force a row for the final iteration `it` (convergence break or loop end). If `it` was
already recorded, merges `kwargs` into that row. Later calls are no-ops.
"""
function finalize!(rec::DiagRecorder, it::Integer; kwargs...)
    (rec.enabled && !rec.finalized) || return rec
    if !isempty(rec.iters) && rec.iters[end] == it
        n = length(rec.iters) - 1
        for (k, v) in pairs(kwargs)
            ks = String(k)
            if v isa AbstractVector
                for (i, vi) in enumerate(v)
                    _diag_setcol!(rec, "$(ks)_$(i)", n, _diag_f32(vi))
                end
            else
                _diag_setcol!(rec, ks, n, _diag_f32(v))
            end
        end
    else
        _diag_push_row!(rec, it, kwargs)
    end
    rec.finalized = true
    return rec
end

function diagnostics_dict(rec::DiagRecorder)
    d = Dict{String, Any}("schema_version" => DIAG_SCHEMA_VERSION, "enabled" => rec.enabled)
    rec.enabled || return d
    d["iter"] = copy(rec.iters)
    d["max_records"] = rec.max_records
    d["descriptions"] = copy(rec.descriptions)
    for (k, col) in rec.data
        d[k] = copy(col)
    end
    return d
end
