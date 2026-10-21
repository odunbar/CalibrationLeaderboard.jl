# validate_offline_tmcmc.jl — check that the threaded and batched ST-MCMC of offline_tmcmc.jl agree with TransitionalMCMC.tmcmc on a FROZEN GP.
#
# For one cell and iteration k, runs each implementation REPEATS times (same GP, same target density) with the production (burnin, thin) and, with
# DIAG_VAL_LONG=1, the reference (20, 3) settings, and compares:
#   log-evidence   mean ± sd over repeats: the implementations should agree within the package's own run-to-run sd
#   dmean, sratio  of each implementation's pooled sample against the package's pooled sample (mean shift / package spread; spread ratio)
#   energy         energy distance of the pooled sample to the package pool, next to the MC floor = energy distance between the two halves of the package's runs
#   secs           mean wall seconds per run
# Agreement: log-ev differences ≲ sd, sratio ≈ 1, dmean ≈ 0, energy ≈ the floor.
#
# Local (from the method directory), e.g. the cheap l63 case, or vec with DIAG_VAL_PKG=0 to skip the slow package runs:
#   EXPERIMENT=l63 BOED_VARIANT=iekf CALIBRATE_DATE=2026-10-16 DIAG_N_ENS=10 DIAG_SOURCE=<dir of cells> julia --project=. -t 16 diagnostics/validate_offline_tmcmc.jl
# Knobs: DIAG_VAL_ITER (default last), DIAG_VAL_REPEATS (default 6, even), DIAG_VAL_PKG=0|1 (default 1), DIAG_VAL_LONG=0|1, DIAG_NPART.

include(joinpath(@__DIR__, "offline_common.jl"))

S = load_offline_cell()
D, cfg = S.D, S.cfg
prob = proxy_problem(D)
K = length(D["iter"])
k = parse(Int, get(ENV, "DIAG_VAL_ITER", string(K)))
R = parse(Int, get(ENV, "DIAG_VAL_REPEATS", "6"))
iseven(R) || error("DIAG_VAL_REPEATS must be even (the MC floor splits the runs in halves)")
NPART = parse(Int, get(ENV, "DIAG_NPART", string(cfg.n_posterior_samples)))
do_pkg = get(ENV, "DIAG_VAL_PKG", "1") == "1"
settings = Pair{String, Tuple{Int, Int}}["prod" => (cfg.tmcmc_burnin, cfg.tmcmc_thin)]
get(ENV, "DIAG_VAL_LONG", "0") == "1" && push!(settings, "long" => (20, 3))
impls = do_pkg ? [:package, :threaded, :batched] : [:threaded, :batched]
println("validate: $(method_key) $(OFFLINE_CASE) N_ens=$(S.N_ens) seed=$(S.seed) iteration $k, k_in=$(prob.prior_basis.k_R), k_out=$(prob.output_basis.k_R), $R repeats × $NPART particles, $(Threads.nthreads()) threads")

gps = rebuild_gps(D, k)
spreadof(X) = sqrt(mean(var(X; dims = 2)))
# warm-up (compile) so the timings below are not dominated by it
for impl in impls
    offline_tmcmc(prob, gps, 20, MersenneTwister(1); burnin = 1, thin = 1, impl = impl)
end

for (sname, (burn, thin)) in settings
    println("\n=== settings '$sname' (burnin $burn, thin $thin) ===")
    runs, logev, secs = Dict{Symbol, Vector{Matrix{Float64}}}(), Dict{Symbol, Vector{Float64}}(), Dict{Symbol, Float64}()
    for impl in impls
        runs[impl], logev[impl] = Matrix{Float64}[], Float64[]
        t0 = time()
        for r in 1:R
            Random.seed!(500 * r + 1)   # the package draws its moves from the global RNG
            dg = Dict{String, Any}()
            push!(runs[impl], offline_tmcmc(prob, gps, NPART, MersenneTwister(1000 * r + 7); burnin = burn, thin = thin, impl = impl, diag = dg))
            push!(logev[impl], dg["tmcmc_log_ev"])
        end
        secs[impl] = (time() - t0) / R
    end
    ref = do_pkg ? :package : :threaded
    Xref = reduce(hcat, runs[ref])
    μref, sref = vec(mean(Xref; dims = 2)), spreadof(Xref)
    half(impl) = (reduce(hcat, runs[impl][1:(R ÷ 2)]), reduce(hcat, runs[impl][(R ÷ 2 + 1):end]))
    println(rpad("  impl", 12), rpad("log-ev mean ± sd", 24), rpad("dmean", 10), rpad("sratio", 10), rpad("energy→ref", 13), rpad("floor (halves)", 16), "secs/run")
    for impl in impls
        X = reduce(hcat, runs[impl])
        h1, h2 = half(impl)
        println("  ", rpad(String(impl), 10), rpad("$(round(mean(logev[impl]); digits = 3)) ± $(round(std(logev[impl]); digits = 3))", 24),
                rpad(round(norm(vec(mean(X; dims = 2)) .- μref) / (sqrt(prob.prior_basis.k_R) * sref); sigdigits = 3), 10), rpad(round(spreadof(X) / sref; sigdigits = 3), 10),
                rpad(impl === ref ? "—" : round(energy_distance(X, Xref); sigdigits = 3), 13), rpad(round(energy_distance(h1, h2); sigdigits = 3), 16), round(secs[impl]; sigdigits = 3))
    end
    if do_pkg
        println("  speedup vs package: ", join(["$(impl) $(round(secs[:package] / secs[impl]; digits = 1))×" for impl in impls if impl !== :package], ",  "))
    end
end
