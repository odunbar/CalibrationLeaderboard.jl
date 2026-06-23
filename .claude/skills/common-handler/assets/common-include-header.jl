# ──────────────────────────────────────────────────────────────────────────────
# Shared code from common/
# All paths are @__DIR__-relative so this file works regardless of the working
# directory when Julia is invoked (local run, SLURM array task, CI, etc.).
# ──────────────────────────────────────────────────────────────────────────────

const _COMMON = joinpath(@__DIR__, "..", "common")

# Forward map (pick the one(s) this experiment uses)
include(joinpath(_COMMON, "forward_maps", "Lorenz63.jl"))
# include(joinpath(_COMMON, "forward_maps", "Lorenz96.jl"))

# Leaderboard metrics (pick opt OR uq, not both)
# include(joinpath(_COMMON, "opt_metrics", "write_results_nc.jl"))
# include(joinpath(_COMMON, "uq_metrics", "coverage_metrics.jl"))
# include(joinpath(_COMMON, "uq_metrics", "write_uq_nc.jl"))
