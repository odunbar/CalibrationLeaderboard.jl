# checkpoint.jl — mid-run checkpoint of a run script's recorded diagnostics (score_based_lm and score_based_adam, all cases).
#
# Result JLD2s are written once, at the end of a cell, so a task killed by the wall-clock limit (HPC) leaves nothing, and a missing file
# is indistinguishable from "never ran". save_checkpoint writes the diagnostics recorded so far plus the fields passed to it (current θ,
# iterations done, ...) to `path` every `cfg.checkpoint_every` iterations (config key; env CKPT_EVERY; 0 disables).
#
#   * `path` is partial_path(<result filename>) = <result>_partial.jld2, NOT the result filename: run_to_leaderboard.jl looks cells up by
#     exact result filename, so a time-out is never scored as a finished or a failed run. The run script deletes the partial file when the
#     cell finishes normally.
#   * Atomic: written to <path>.tmp.jld2 and renamed, so a kill mid-write cannot leave a truncated file.
#   * A pure write: no forward evaluation, no RNG draw, no change to the loop state; a failed write only warns and the run continues.
#   * Needs `diagnostics_dict` (diag_recorder.jl, already included by lm_diagnostics.jl / sbadam_diagnostics.jl) and JLD2, which every
#     run script already loads. Include this file after the method's diagnostics file.
#
# Usage in a run loop (after the iteration's update):
#   (cfg.checkpoint_every > 0 && outer_iter % cfg.checkpoint_every == 0) &&
#       save_checkpoint(ckpt, diag.rec; iter_done = outer_iter, conv_score = NaN, final_params = copy(theta), ...)

using JLD2

partial_path(fn::AbstractString) = replace(fn, r"\.jld2$" => "_partial.jld2")

function save_checkpoint(path, rec; fields...)
    path === nothing && return nothing
    try
        tmp = replace(path, r"\.jld2$" => ".tmp.jld2")
        jldopen(tmp, "w") do f
            for (k, v) in pairs(fields)
                f[String(k)] = v
            end
            f["partial"] = true
            f["diagnostics"] = diagnostics_dict(rec)
        end
        mv(tmp, path; force = true)
    catch err
        @warn "checkpoint write failed (the run continues)" exception = err
    end
    return nothing
end
