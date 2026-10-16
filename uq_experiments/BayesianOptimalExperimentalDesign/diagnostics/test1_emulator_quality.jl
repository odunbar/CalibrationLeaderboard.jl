# test1_emulator_quality.jl — TEST 1: does the emulator improve, over the iterations, where it matters?
#
# For ONE instrumented cell, rebuild GP_k for every iteration k (no optimisation, from the saved hyperparameters) and score it against the TRUE forward
# model (replicate-averaged, common random numbers) on three held-out reference sets:
#   truth   the truth θ* projected into the truncated whitened input space (skipped for flux: no identifiable truth)
#   post    held-out points drawn from the FINAL iteration's posterior (the region the method ends up exploring; never in the training set)
#   prior   fresh prior-LHS points (global control: does the emulator get better everywhere, or only where the posterior is?)
# Scores per (k, set):
#   rmse      sqrt(mean over points and modes of (GP mean − true replicate-mean output)²), whitened units (noise floor = sqrt(mean replicate var / nrep))
#   msse      mean of (GP mean − truth)² / (GP var + replicate var / nrep): ≈ 1 calibrated, ≫ 1 overconfident
#   cov95     fraction of modes/points whose true mean lies inside the GP's 95% interval
#   ll_gap    mean |surrogate loglik − true loglik| (the quantity the posterior actually uses)
#   spearman  rank correlation of surrogate vs true loglik across the post ∪ prior points (does the emulator rank candidates correctly?)
#
# This is the 'plain truth' reference. A reference from an IEKF on the TRUE model (GNKI) would be a further set: add its ensemble as `post_ref`.
#
# Local (from the method directory):
#   EXPERIMENT=l63 BOED_VARIANT=iekf CALIBRATE_DATE=2099-01-01 DIAG_N_ENS=10 julia --project=. -t 8 diagnostics/test1_emulator_quality.jl
# Knobs: DIAG_NPTS (points per held-out set, default 30), DIAG_NREP (forward replicates per point, default 8).
# Output: diagnostics/output/test1_<variant key>_<case>_N<N>_seed<s>_<date>.jld2 and .png

using Plots
include(joinpath(@__DIR__, "offline_common.jl"))

const NPTS = parse(Int, get(ENV, "DIAG_NPTS", "30"))
const NREP = parse(Int, get(ENV, "DIAG_NREP", "8"))

S = load_offline_cell()
D, cell, cfg = S.D, S.cell, S.cfg
prob = proxy_problem(D)
K = length(D["iter"])
k_in = prob.prior_basis.k_R
println("test1: $(method_key) $(OFFLINE_CASE) N_ens=$(S.N_ens) seed=$(S.seed): $K iterations, k_in=$k_in, k_out=$(prob.output_basis.k_R)")

# ---- held-out reference sets (fixed seed, so every iteration is scored on the same points) ----
rng = MersenneTwister(77)
Zpost_all = saved_posterior(D, K)
Zpost = Zpost_all[:, unique(round.(Int, range(1, size(Zpost_all, 2); length = min(NPTS, size(Zpost_all, 2)))))]
Zprior = lhs_standard_normal_sample(k_in, NPTS, rng)
ztruth = saved_truth_z(D)
sets = Pair{String, Matrix{Float64}}["post" => Zpost, "prior" => Zprior]
ztruth === nothing || pushfirst!(sets, "truth" => reshape(ztruth, :, 1))

truthfwd = Dict{String, Tuple{Matrix{Float64}, Matrix{Float64}}}()
for (name, Zp) in sets
    @info "test1: true forward model at the '$name' set ($(size(Zp, 2)) points × $NREP replicates)"
    truthfwd[name] = true_forward_whitened(prob, cell, Zp; nrep = NREP)
end
ll_true = Dict(name => true_loglik(prob, truthfwd[name]...) for (name, _) in sets)
floor_rmse = Dict(name => sqrt(mean(truthfwd[name][2]) / NREP) for (name, _) in sets)

# ---- score GP_k at every iteration ----
names = first.(sets)
res = Dict(name => Dict(m => fill(NaN, K) for m in ("rmse", "msse", "cov95", "ll_gap")) for name in names)
spear = fill(NaN, K)
selfchk = fill(NaN, K)
for k in 1:K
    gps = rebuild_gps(D, k)
    c = selfcheck_gps(gps, D, k)
    selfchk[k] = c.rebuilt - c.recorded
    sur_ll = Dict{String, Vector{Float64}}()
    for (name, Zp) in sets
        mu, v = zeros(size(truthfwd[name][1])), zeros(size(truthfwd[name][1]))
        for j in 1:length(gps.gps)
            m, s = GaussianProcesses.predict_y(gps.gps[j], Zp)
            mu[j, :], v[j, :] = m, s
        end
        TM, TV = truthfwd[name]
        err = mu .- TM
        tot = v .+ TV ./ NREP
        res[name]["rmse"][k] = sqrt(mean(err .^ 2))
        res[name]["msse"][k] = mean(err .^ 2 ./ tot)
        res[name]["cov95"][k] = mean(abs.(err) .< 1.959964 .* sqrt.(tot))
        sur_ll[name] = surrogate_loglik(prob, gps, Zp)
        res[name]["ll_gap"][k] = mean(abs.(sur_ll[name] .- ll_true[name]))
    end
    if haskey(sur_ll, "post") && haskey(sur_ll, "prior")
        spear[k] = spearman(vcat(sur_ll["post"], sur_ll["prior"]), vcat(ll_true["post"], ll_true["prior"]))
    end
end

# ---- report ----
println("\nGP self-check (rebuilt log target − recorded): max |Δ| = ", round(maximum(abs.(filter(isfinite, selfchk))); sigdigits = 3), " (≈0 means the rebuilt GPs match the run's)")
println("forward-model noise floor of rmse (sqrt(mean replicate var/nrep)): ", join(["$n=$(round(floor_rmse[n]; sigdigits = 3))" for n in names], "  "))
for name in names
    println("\n[$name]  n_points=$(size(sets[findfirst(p -> p.first == name, sets)].second, 2))")
    println(rpad("  iter", 8), rpad("n_train", 9), rpad("rmse", 10), rpad("msse", 10), rpad("cov95", 9), rpad("ll_gap", 10), name == "post" ? "spearman(post∪prior)" : "")
    for k in 1:K
        println("  ", rpad(k, 6), rpad(Int(D["n_train"][k]), 9), rpad(round(res[name]["rmse"][k]; sigdigits = 3), 10), rpad(round(res[name]["msse"][k]; sigdigits = 3), 10),
                rpad(round(res[name]["cov95"][k]; sigdigits = 3), 9), rpad(round(res[name]["ll_gap"][k]; sigdigits = 3), 10), name == "post" ? round(spear[k]; sigdigits = 3) : "")
    end
end
println("\nHypothesis check: emulator improves where it matters if rmse/ll_gap in 'post' fall with k; 'prior' falling too means global improvement.")

# ---- save + figure ----
tag = "$(method_key)_$(case_stem(OFFLINE_CASE))_N$(S.N_ens)_seed$(S.seed)_$(cfg.calibrate_date)"
JLD2.save(joinpath(OFFLINE_OUT, "test1_$(tag).jld2"), "results", res, "spearman", spear, "noise_floor_rmse", floor_rmse, "iter", Int.(D["iter"]),
          "n_train", Float64.(D["n_train"]), "N_ens", S.N_ens, "seed", S.seed, "variant", String(cfg.variant), "case", String(OFFLINE_CASE))
colors = Dict("truth" => "#eda100", "post" => "#2a78d6", "prior" => "#eb6834")
panels = []
for (m, ttl, ysc, refs) in (("rmse", "emulator rmse vs true model (whitened)", :log10, nothing), ("msse", "msse (≈1 calibrated)", :log10, 1.0),
                            ("cov95", "95% coverage of the true mean", :identity, 0.95), ("ll_gap", "mean |surrogate − true loglik|", :log10, nothing))
    p = plot(; title = ttl, xlabel = "GBOED iteration k", yscale = ysc, legend = m == "rmse" ? :topright : false, titlefontsize = 9)
    for name in names
        plot!(p, 1:K, max.(res[name][m], 1e-12); label = name, color = colors[name], lw = 2, marker = :circle, ms = 3)
        m == "rmse" && hline!(p, [floor_rmse[name]]; color = colors[name], ls = :dot, label = false)
    end
    refs === nothing || hline!(p, [refs]; color = :gray, ls = :dash, label = false)
    push!(panels, p)
end
p5 = plot(1:K, spear; title = "Spearman(surrogate, true loglik), post∪prior", xlabel = "GBOED iteration k", color = "#2a78d6", lw = 2, marker = :circle, ms = 3, legend = false, titlefontsize = 9)
push!(panels, p5)
fig = plot(panels...; layout = (2, 3), size = (1300, 820), left_margin = 4Plots.mm, bottom_margin = 7Plots.mm, plot_title = "test1: emulator quality — $(method_key) $(OFFLINE_CASE) N_ens=$(S.N_ens) seed=$(S.seed)", plot_titlefontsize = 10, dpi = 130)
savefig(fig, joinpath(OFFLINE_OUT, "test1_$(tag).png"))
println("Saved ", joinpath(OFFLINE_OUT, "test1_$(tag).{jld2,png}"))
