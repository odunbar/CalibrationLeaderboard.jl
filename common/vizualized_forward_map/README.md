# Visualised forward maps

Plots of the Lorenz 63 / Lorenz 96 forward maps in `common/forward_maps/`. They only run the
simulator: **no optimisation / calibration / UQ algorithm is involved.**

```
julia --project=common/vizualized_forward_map -e 'using Pkg; Pkg.instantiate()'   # once

# from the repo root
julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l63_forward_map.jl
julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l63_prior_pushforward.jl
julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l96_forward_map.jl [case ...]
julia --project=common/vizualized_forward_map common/vizualized_forward_map/plot_l96_prior_pushforward.jl [case ...]
```

`case` is any of `const-force`, `vec-force`, `flux-force` (default: all three).
`N_PRIOR=200` sets the number of simulated prior samples in the prior-pushforward plots (default 100).
PNGs are written to `output/`.

Style knobs live at the top of `viz_common.jl`: `FONT_SCALE` (fonts), `SIZE_SCALE` (canvas, so the large fonts fit),
`LW_SCALE` (line widths) and `dpi` (60: layout and font proportions do not depend on it, only the pixel count and so the file size;
the PNGs are roughly 0.5-1 MB each).

## Plots

| Script | Figure | What it shows |
|---|---|---|
| `plot_l63_forward_map.jl` | `l63_forward_map.png` | θ=(ρ,β) → the 9 window statistics G(θ) against the observation y ± σ → x–z phase portrait (bold = statistics window) |
| `plot_l96_forward_map.jl` | `l96_<case>_forward_map.png` | forcing F(i) → window mean/std per site G(θ) with ±σ → Hovmöller diagram per run with the window start marked |
| `plot_l63_prior_pushforward.jl` | `l63_prior_pushforward.png` | shaded prior densities of ρ, β; x–z trajectories of a few prior samples; histograms of the prior pushforward on each of the 9 statistics vs y ± σ |
| `plot_l96_prior_pushforward.jl` | `l96_<case>_prior_pushforward.png` | shaded prior density (const) or quantile bands (vec/flux) of the forcing; pushforward bands of window mean/std vs y ± σ |

The forward-map figures overlay three runs, in the same colours in every panel:

- **black**: truth parameter, spun-up initial condition `x0` (its statistics are exactly the observation `y`)
- **orange**: a different parameter (one seeded prior draw; for L63 the first draw with β < the true β, via `run_scenarios(...; alt_accept = ...)`), same `x0`
- **green**: truth parameter, a different initial condition (`x0 + ic_cov_sqrt*randn`, as used by the leaderboard methods)

In the L96 output panels, grey curves are statistics of more initial-condition-perturbed runs at the true parameter. Together with the
`±σ = sqrt(diag R)` band they show how much of G is chaos noise rather than signal (the L63 figure omits this cloud: `n_ic_cloud = 0`).

## Files

- `viz_common.jl`: model-agnostic helpers (palette, window shading, quantile bands, `run_scenarios`, `prior_pushforward`)
- `l63_problem.jl`, `l96_problem.jl`: truth, prior, observation window of each problem. They mirror `uq_experiments/*/calibrate_l*.jl` and `l*_preliminaries.jl`
  (priors and `force_case_setup` are experiment-specific, so they are duplicated here rather than moved to `common/`).
  Spun-up `x0`, `R` and `ic_cov_sqrt` come from `compute_perfect_data` and are cached in `output/*.jld2` (git-ignored). The L96 `R` estimate takes a few minutes the first time.
- The two forward maps define clashing names (`LorenzConfig`, `stats`, ...), so L63 and L96 scripts run in separate Julia processes.

## Notes

- The shaded window is `stats_window_indices(config, obs)` converted to times. Column *j* holds time `(j-1)*dt`, so it reads e.g. `[29.99, 39.99]`
  rather than `[30, 40]`. This one-step offset is deliberate in the forward map (stored `y` files depend on it).
- A prior draw can land in a non-chaotic regime (e.g. L63 with small ρ relaxes to a fixed point); this is real behaviour of the forward map, not a bug.
  Change `seed` in `run_scenarios`, or pass an `alt_accept` predicate, to draw a different "different parameter".
