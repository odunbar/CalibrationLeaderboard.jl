# test2_sampler_convergence.jl — TEST 2: have the posterior samplers (ST-MCMC, IEKF) converged?
#
# For ONE instrumented cell, rebuild GP_k at a few iterations and run the samplers on that FROZEN GP (no forward model, no new design points), so any difference
# between runs is pure sampler error: the target density is identical.
#   tmcmc_prod    production settings (cfg.tmcmc_burnin / tmcmc_thin), REPEATS independent runs
#   tmcmc_long    TransitionalMCMC's own defaults (burn-in 20, thin 3), REPEATS runs; pooled into the REFERENCE posterior
#   tmcmc_novar   production settings, likelihood WITHOUT the GP predictive variance (isolates what the IEKF ignores)
#   iekf_prod     production IEKF (cfg.iekf_iters × cfg.iekf_step; currently 30 × 0.1, T = 3)
#   iekf_T1       10 × 0.1 and 40 × 0.025 (T = 1, the earlier finite-time setting; two step sizes show discretisation stability)
#   iekf_T3fine   120 × 0.025 (T = 3 with a finer step: discretisation stability of the production setting)
# Each sampler's sample is compared with the reference (the pooled tmcmc_long draws):
#   dmean   ‖mean − mean_ref‖ / (√k · spread_ref)         mean shift in reference posterior widths
#   sratio  spread / spread_ref                           < 1 over-confident, > 1 over-dispersed (spread = sqrt(mean variance), whitened coordinates)
#   energy  energy distance to the reference sample
# The MC noise floor is the energy distance between two independent tmcmc_long runs ("floor"). Anything within ≈ the floor of the reference is converged.
# The saved production posterior of the run (a fixed 100-point subsample) is scored the same way ("saved").
#
# Local (from the method directory):
#   EXPERIMENT=l63 BOED_VARIANT=tmcmc_eig CALIBRATE_DATE=2099-01-01 DIAG_N_ENS=10 julia --project=. -t 8 diagnostics/test2_sampler_convergence.jl
# Knobs: DIAG_ITERS=1,5,10 (iterations to test, default first/middle/last), DIAG_REPEATS=3, DIAG_NPART=<particles> (default cfg.n_posterior_samples).
# Output: diagnostics/output/test2_<variant key>_<case>_N<N>_seed<s>_<date>.jld2 and .png

using Plots
include(joinpath(@__DIR__, "offline_common.jl"))

S = load_offline_cell()
D, cfg = S.D, S.cfg
prob = proxy_problem(D)
K = length(D["iter"])
k_in = prob.prior_basis.k_R
ITERS = haskey(ENV, "DIAG_ITERS") ? parse.(Int, split(ENV["DIAG_ITERS"], ",")) : unique([1, cld(K, 2), K])
REPEATS = parse(Int, get(ENV, "DIAG_REPEATS", "3"))
NPART = parse(Int, get(ENV, "DIAG_NPART", string(cfg.n_posterior_samples)))
println("test2: $(method_key) $(OFFLINE_CASE) N_ens=$(S.N_ens) seed=$(S.seed); iterations $ITERS; $REPEATS repeats × $NPART particles; k_in=$k_in")

# ST-MCMC without the GP predictive variance in the likelihood (identical structure to run_tmcmc / boed_loglik otherwise).
function run_tmcmc_novar(prob, gps, n, rng; burnin, thin)
    k = prob.prior_basis.k_R
    function ll(z)
        mu = predict_boed(gps, z isa Real ? [z] : z)[1]
        return sum(-0.5 .* (mu .- prob.y_whitened) .^ 2 .- 0.5 * log(2π))
    end
    logprior(z) = sum(logpdf.(Normal(), z))
    priorRnd(n) = Matrix(lhs_standard_normal_sample(k, n, rng)')
    samps, _ = TransitionalMCMC.tmcmc(ll, logprior, priorRnd, n, burnin, thin)
    return Matrix(reshape(samps, n, k)')
end

spreadof(X) = sqrt(mean(var(X; dims = 2)))
iekf_runs = [("iekf_prod", cfg.iekf_iters, cfg.iekf_step), ("iekf_T1_10x0.1", 10, 0.1), ("iekf_T1_40x0.025", 40, 0.025), ("iekf_T3fine_120x0.025", 120, 0.025)]
sampler_names = ["saved", "tmcmc_prod", "tmcmc_long", "tmcmc_novar", first.(iekf_runs)...]
res = Dict(m => fill(NaN, length(sampler_names), length(ITERS)) for m in ("dmean", "sratio", "energy"))
floor_energy = fill(NaN, length(ITERS))
logev = Dict(name => fill(NaN, REPEATS, length(ITERS)) for name in ("tmcmc_prod", "tmcmc_long"))
secs = Dict(name => fill(NaN, length(ITERS)) for name in sampler_names)

for (ik, k) in enumerate(ITERS)
    gps = rebuild_gps(D, k)
    c = selfcheck_gps(gps, D, k)
    println("\n--- iteration $k (n_train=$(Int(D["n_train"][k])); GP self-check Δ log target = $(round(c.rebuilt - c.recorded; sigdigits = 3))) ---")

    longs, prods = Matrix{Float64}[], Matrix{Float64}[]
    for r in 1:REPEATS
        for (name, burn, thin, store) in (("tmcmc_long", 20, 3, longs), ("tmcmc_prod", cfg.tmcmc_burnin, cfg.tmcmc_thin, prods))
            Random.seed!(10_000 * k + 100 * r + (name == "tmcmc_long" ? 1 : 2))   # ST-MCMC's MH moves use the global RNG
            dg = Dict{String, Any}()
            t0 = time()
            push!(store, run_tmcmc(prob, gps, NPART, MersenneTwister(1000 * k + r + (name == "tmcmc_long" ? 0 : 500)); burnin = burn, thin = thin, diag = dg))
            secs[name][ik] = (isnan(secs[name][ik]) ? 0.0 : secs[name][ik]) + (time() - t0) / REPEATS
            logev[name][r, ik] = dg["tmcmc_log_ev"]
        end
    end
    Xref = reduce(hcat, longs)
    μref, sref = vec(mean(Xref; dims = 2)), spreadof(Xref)
    floor_energy[ik] = REPEATS >= 2 ? energy_distance(longs[1], longs[2]) : NaN

    Random.seed!(10_000 * k + 7)
    t0 = time()
    novar = run_tmcmc_novar(prob, gps, NPART, MersenneTwister(1000 * k + 700); burnin = cfg.tmcmc_burnin, thin = cfg.tmcmc_thin)
    secs["tmcmc_novar"][ik] = time() - t0
    iekfs = Dict{String, Matrix{Float64}}()
    for (name, iters, step) in iekf_runs
        t0 = time()
        iekfs[name] = run_iekf(prob, gps, NPART, MersenneTwister(1000 * k + 900); step = step, iters = iters)
        secs[name][ik] = time() - t0
    end
    samples = Dict("saved" => saved_posterior(D, k), "tmcmc_prod" => reduce(hcat, prods), "tmcmc_long" => Xref, "tmcmc_novar" => novar, iekfs...)
    for (is, name) in enumerate(sampler_names)
        X = samples[name]
        res["dmean"][is, ik] = norm(vec(mean(X; dims = 2)) .- μref) / (sqrt(k_in) * sref)
        res["sratio"][is, ik] = spreadof(X) / sref
        # tmcmc_long is scored as run 1 against the pool of the OTHER runs (against a pool containing itself it would be biased low)
        res["energy"][is, ik] = name == "tmcmc_long" && REPEATS >= 2 ? energy_distance(longs[1], reduce(hcat, longs[2:end])) : energy_distance(X, Xref)
    end
    println(rpad("  sampler", 26), rpad("dmean", 10), rpad("sratio", 10), rpad("energy", 10), "secs")
    for (is, name) in enumerate(sampler_names)
        println("  ", rpad(name, 24), rpad(round(res["dmean"][is, ik]; sigdigits = 3), 10), rpad(round(res["sratio"][is, ik]; sigdigits = 3), 10),
                rpad(round(res["energy"][is, ik]; sigdigits = 3), 10), round(get(secs, name, [NaN])[ik]; sigdigits = 3))
    end
    println("  MC noise floor (energy distance between two independent tmcmc_long runs): ", round(floor_energy[ik]; sigdigits = 3))
    println("  log-evidence  tmcmc_prod: ", round.(logev["tmcmc_prod"][:, ik]; sigdigits = 5), "   tmcmc_long: ", round.(logev["tmcmc_long"][:, ik]; sigdigits = 5))
end

tag = "$(method_key)_$(case_stem(OFFLINE_CASE))_N$(S.N_ens)_seed$(S.seed)_$(cfg.calibrate_date)"
JLD2.save(joinpath(OFFLINE_OUT, "test2_$(tag).jld2"), "results", res, "samplers", sampler_names, "iters", ITERS, "floor_energy", floor_energy, "logev", logev, "secs", secs,
          "npart", NPART, "repeats", REPEATS, "N_ens", S.N_ens, "seed", S.seed, "variant", String(cfg.variant), "case", String(OFFLINE_CASE))
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a63d2"]
panels = []
for (m, ttl, ysc, ref) in (("energy", "energy distance to reference (dotted: MC floor)", :log10, nothing), ("dmean", "mean shift / reference spread", :log10, nothing), ("sratio", "spread / reference spread", :identity, 1.0))
    p = plot(; title = ttl, titlefontsize = 9, yscale = ysc, xticks = (1:length(sampler_names), sampler_names), xrotation = 30, legend = m == "energy" ? :topleft : false, bottom_margin = 8Plots.mm)
    for (ik, k) in enumerate(ITERS)
        scatter!(p, 1:length(sampler_names), max.(res[m][:, ik], 1e-6); label = "k=$k", color = cols[mod1(ik, length(cols))], ms = 6)
        m == "energy" && hline!(p, [floor_energy[ik]]; color = cols[mod1(ik, length(cols))], ls = :dot, label = false)
    end
    ref === nothing || hline!(p, [ref]; color = :gray, ls = :dash, label = false)
    push!(panels, p)
end
fig = plot(panels...; layout = (1, 3), size = (1500, 480), plot_title = "test2: sampler convergence on a frozen GP — $(method_key) $(OFFLINE_CASE) N_ens=$(S.N_ens) seed=$(S.seed)", plot_titlefontsize = 10, dpi = 130)
savefig(fig, joinpath(OFFLINE_OUT, "test2_$(tag).png"))
println("\nSaved ", joinpath(OFFLINE_OUT, "test2_$(tag).{jld2,png}"))
