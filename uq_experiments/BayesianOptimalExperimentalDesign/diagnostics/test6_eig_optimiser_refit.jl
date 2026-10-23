# test6_eig_optimiser_refit.jl — TEST 6: re-run the EIG optimiser of every (seed, iteration) cell offline, with a much larger evaluation budget, and ask
#   (1) did it converge (more stages converged than the run's 2000-call limit allowed)?
#   (2) was it successful by ITS OWN criteria (gradient / relative-change tolerance, stagnation of the best objective)?
#   (3) do those criteria lead to overfitting: is the EIG gained on the optimiser's own posterior points also gained on NEWLY DRAWN points?
#
# No GP is refit and no forward model is run. For stage k (k ≥ 2) the saved cell gives everything the optimiser used:
#   hyperparameters of GP_{k-1}       D["gp_hyper_by_k"][k-1]                 (eig_objective only needs [iℓ2…, σ2] per output mode)
#   the k-1 posterior                 cell["posteriors_by_k"][k-1] (raw θ, 1000 draws) → whitened with the saved prior basis; checked against the run's own
#                                     100 saved draws D["X_post_sub_by_k"][k-1]; if that check fails the 100 saved draws are the pool instead
#   the optimiser's starting batch    D["X_cand0_by_k"][k]  (exact)
#   the batch the run actually chose  D["Z"][:, (k-1)N+1 : kN]
# What cannot be recovered is the run's own random 50-point subsample X_eig; here `X_opt` is a fresh random n_eig-subsample (DIAG_REPS of them per stage), so the
# test reproduces the DISTRIBUTION of what the run did, not each trajectory. The optimiser is `run_fminbox_with_call_limit` itself (boed_core.jl, unchanged), on the
# same :ball reparameterisation as `optimize_batch`, with the run's tolerances and a raised call / iteration budget.
#
# Per stage the table records (all EIG values = eig_objective, GP_{k-1} hyperparameters):
#   EIG of the raw start batch X_cand0, of the optimiser's actual start (ball_squash∘ball_unsquash: draws beyond 0.95R are pulled in), of the optimised batch and of the
#   batch the run chose — each on X_opt (in-sample) and on `n_fresh` disjoint newly drawn sets of the same size (held-out; mean and sd over the sets);
#   stopping: n_calls, converged / hit_limit flags, |grad|∞ and |grad|2 at the returned point and at the start (against g_tol), relative change of the best objective over
#   the last 100 / 500 calls (against f_reltol), calls needed for 90% / 99% of the in-sample gain, fraction of the gain made after the run's old 2000-call limit;
#   checkpoints (CKPTS calls): in-sample best and held-out EIG of the best batch so far (does more optimisation raise the held-out EIG or only the in-sample one?);
#   selfchecks: the recorded eig_batch / eig_init recomputed on the run's own diagnostic subsample (should be ~1e-4), and the first objective value vs EIG(start, X_opt).
# "n_calls" counts every forward pass of the objective, including the one inside each gradient evaluation.
#
# Environment (offline_common.jl documents EXPERIMENT, BOED_VARIANT, CALIBRATE_DATE, DIAG_SOURCE):
#   DIAG_SOURCE   directory of the cell files (default from-hpc/<CALIBRATE_DATE>/<variant with dashes>, else output/<variant key>_<date>)
#   DIAG_N_ENS    comma list of N_ens (default every N_ens whose cells exist)        DIAG_SEEDS  e.g. 1,2,3 or 1:20 (default 1:n_repeats)     DIAG_K  stage list (default all)
#   DIAG_CALL_LIMIT (20000)   DIAG_OPTIM_ITERS (5000; the run used 500)   DIAG_G_TOL / DIAG_F_RELTOL (the run's)   DIAG_N_EIG (the run's 50)
#   DIAG_N_FRESH (20 held-out sets per stage)   DIAG_REPS (1 X_opt draw per stage)   DIAG_TAG (suffix of the output name)
#
# Local (from the method directory):
#   BOED_VARIANT=iekf_eig EXPERIMENT=l96_vec CALIBRATE_DATE=2026-10-19 julia --project=. -t 16 diagnostics/test6_eig_optimiser_refit.jl
# Output: diagnostics/output/test6_eig_refit_<case>_<variant>_<date>[_<tag>].jld2 and a printed table. Plot: test6_plot.jl.

include(joinpath(@__DIR__, "offline_common.jl"))
using Logging
using Printf

const CFG = experiment_config(OFFLINE_CASE)
CFG.variant in (:tmcmc_eig, :iekf_eig) || error("test6 refits the EIG optimiser: BOED_VARIANT must be tmcmc_eig or iekf_eig (got $(CFG.variant))")

_env_int(k, d) = parse(Int, get(ENV, k, string(d)))
_env_flt(k, d) = parse(Float64, get(ENV, k, string(d)))
function _env_list(k, default)
    haskey(ENV, k) || return default
    out = Int[]
    for tok in split(ENV[k], ",")
        if occursin(":", tok)
            a, b = parse.(Int, split(tok, ":"))
            append!(out, a:b)
        else
            push!(out, parse(Int, tok))
        end
    end
    return out
end

const CALL_LIMIT = _env_int("DIAG_CALL_LIMIT", 20_000)
const OPTIM_ITERS = _env_int("DIAG_OPTIM_ITERS", 5_000)
const G_TOL = _env_flt("DIAG_G_TOL", CFG.eig_g_tol)
const F_RELTOL = _env_flt("DIAG_F_RELTOL", CFG.eig_f_reltol)
const N_EIG = _env_int("DIAG_N_EIG", CFG.n_eig_posterior_samples)
const N_FRESH = _env_int("DIAG_N_FRESH", 20)
const N_REP = _env_int("DIAG_REPS", 1)
const CKPTS = sort(unique(vcat(filter(<=(CALL_LIMIT), [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]), CALL_LIMIT)))
const TRACE_GRID = unique(round.(Int, exp.(range(log(1.0), log(Float64(CALL_LIMIT)); length = 40))))

########################################################################
###############  objective trace (offline only, outside the run)  ######
########################################################################
mutable struct TraceState
    ncalls::Int
    first::Float64                     # first objective value (= -EIG at the start, on X_opt)
    best::Float64                      # best (lowest) objective so far
    best_v::Vector{Float64}
    trace::Vector{Float32}             # best-so-far objective after each call
    nck::Int
    ck_best::Vector{Float64}
    ck_v::Vector{Vector{Float64}}
end
TraceState(n) = TraceState(0, NaN, Inf, Float64[], Float32[], 0, fill(NaN, length(CKPTS)), [Float64[] for _ in CKPTS])

# Called from inside the objective, which Zygote differentiates: mutation must be invisible to AD.
function _tr_record!(st::TraceState, v, val)
    st.ncalls += 1
    st.ncalls == 1 && (st.first = val)
    if val < st.best
        st.best = val
        st.best_v = collect(Float64, v)
    end
    push!(st.trace, Float32(st.best))
    while st.nck < length(CKPTS) && st.ncalls >= CKPTS[st.nck + 1]
        st.nck += 1
        st.ck_best[st.nck] = st.best
        st.ck_v[st.nck] = st.best_v        # best_v is replaced, never mutated, so sharing is safe
    end
    return nothing
end
Zygote.@nograd _tr_record!

########################################################################
###############  load the cells  #######################################
########################################################################
function source_dir()
    haskey(ENV, "DIAG_SOURCE") && return ENV["DIAG_SOURCE"]
    d = joinpath(BOED_DIR, "from-hpc", string(CFG.calibrate_date), replace(string(CFG.variant), "_" => "-"))
    return isdir(d) ? d : joinpath(BOED_DIR, "output", calib_directory(CFG))
end

struct Cell
    N::Int
    seed::Int
    D::Dict{String, Any}
    prob::Any
    pools::Dict{Int, Matrix{Float64}}     # stage k => whitened k-1 posterior (k_in × M)
    pool_full::Dict{Int, Bool}            # true: the full posterior was reconstructed and matched the run's saved subsample
    pool_err::Dict{Int, Float64}
end

function load_cell(path, N, seed, kfilter)
    cell = JLD2.load(path)
    haskey(cell, "diagnostics") && haskey(cell["diagnostics"], "Z") || (@warn "no instrumented diagnostics in $path"; return nothing)
    D = cell["diagnostics"]
    prob = proxy_problem(D)
    K = Int(cell["n_iters_completed"])
    pools, full, err = Dict{Int, Matrix{Float64}}(), Dict{Int, Bool}(), Dict{Int, Float64}()
    for k in 2:K
        (haskey(D["X_cand0_by_k"], k) && haskey(D["X_post_sub_by_k"], k - 1) && haskey(D["gp_hyper_by_k"], k - 1)) || continue
        (kfilter === nothing || k in kfilter) || continue
        saved = Float64.(D["X_post_sub_by_k"][k - 1])
        pool = saved
        isfull = false
        e = NaN
        if haskey(cell, "posteriors_by_k") && haskey(cell["posteriors_by_k"], k - 1)
            P = Matrix{Float64}(to_prior_whitened(prob, cell["posteriors_by_k"][k - 1]))
            if size(P, 1) == size(saved, 1)
                e = maximum(abs, P[:, _diag_idx(size(P, 2), size(saved, 2))] .- saved)
                if e < 1e-3 * max(1.0, maximum(abs, saved))
                    pool, isfull = P, true
                end
            end
        end
        pools[k], full[k], err[k] = pool, isfull, e
    end
    return Cell(N, seed, D, prob, pools, full, err)
end

########################################################################
###############  one stage  ############################################
########################################################################
function refit_stage(c::Cell, k::Int, rep::Int)
    D, N = c.D, c.N
    kin, B = size(D["Z"], 1), N
    pool = c.pools[k]
    M = size(pool, 2)
    Hk = D["gp_hyper_by_k"][k - 1]
    hyp = [(Hk[j, 1:kin], Hk[j, kin + 1]) for j in 1:size(Hk, 1)]
    X0 = Float64.(D["X_cand0_by_k"][k])
    Zrun = Float64.(D["Z"][:, ((k - 1) * B + 1):(k * B)])
    R = ball_radius(kin, CFG.eig_ball_quantile)
    jit = CFG.eig_jitter

    rng = MersenneTwister(hash((c.seed, N, k, rep)))
    perm = randperm(rng, M)
    nopt = min(N_EIG, M)
    Xopt = pool[:, perm[1:nopt]]
    nf = min(N_FRESH, (M - nopt) ÷ nopt)
    fresh = [pool[:, perm[(nopt + (i - 1) * nopt + 1):(nopt + i * nopt)]] for i in 1:nf]

    eig(Zb, Xs) = eig_objective(vec(Matrix(Zb)), Xs, hyp, kin, B; jitter = jit)
    held(Zb) = (v = [eig(Zb, F) for F in fresh]; isempty(v) ? (NaN, NaN) : (mean(v), length(v) > 1 ? std(v) : NaN))
    core(v) = -eig_objective(vec(ball_squash(reshape(v, kin, B), R)), Xopt, hyp, kin, B; jitter = jit)

    v0 = vec(ball_unsquash(X0, R))
    z0 = ball_squash(reshape(v0, kin, B), R)             # the optimiser's actual starting batch
    st = TraceState(length(CKPTS))
    f(v) = (val = core(v); _tr_record!(st, v, val); val)
    dd = Dict{String, Any}()
    t0 = time()
    with_logger(NullLogger()) do
        run_fminbox_with_call_limit(f, copy(v0), nothing, nothing; iters = OPTIM_ITERS, outer_iters = CFG.eig_outer_iters,
            g_tol = G_TOL, f_reltol = F_RELTOL, call_limit = CALL_LIMIT, label = "refit", diag = dd)
    end
    secs = time() - t0
    zf = ball_squash(reshape(st.best_v, kin, B), R)
    for i in (st.nck + 1):length(CKPTS)                  # converged before this checkpoint: the checkpoint batch is the final one
        st.ck_best[i], st.ck_v[i] = st.best, st.best_v
    end

    g_best = only(Zygote.gradient(core, st.best_v))
    g_start = only(Zygote.gradient(core, v0))
    tr = -Float64.(st.trace)                             # best-so-far EIG on X_opt after each call
    ncalls = length(tr)
    total = tr[end] - tr[1]
    n_to(fr) = total > 0 ? Float64(findfirst(i -> tr[i] - tr[1] >= fr * total, eachindex(tr))) : NaN
    late = ncalls > 2000 && total > 0 ? (tr[end] - tr[2000]) / total : 0.0
    relchg(m) = ncalls > m ? (tr[end] - tr[end - m]) / abs(tr[end]) : NaN

    # selfchecks against what the run recorded (valid only when the full posterior was reconstructed: same evenly spaced 50 as the run's diagnostic)
    sc_batch = sc_init = NaN
    r = D["iter"] isa AbstractVector ? findfirst(==(k), Int.(round.(D["iter"]))) : nothing
    if c.pool_full[k] && r !== nothing
        sub50 = pool[:, _diag_idx(M, BOED_DIAG_N_SUB)]
        sc_batch = eig(Zrun, sub50) - Float64(D["eig_batch"][r])
        sc_init = eig(X0, sub50) - Float64(D["eig_init"][r])
    end

    ho_start, ho_final, ho_run, ho_raw = held(z0), held(zf), held(Zrun), held(X0)
    ck_ins = -st.ck_best
    ck_held = [held(ball_squash(reshape(st.ck_v[i], kin, B), R))[1] for i in eachindex(CKPTS)]
    grid_ins = [tr[min(g, ncalls)] for g in TRACE_GRID]

    return Dict{String, Any}(
        "N_ens" => N, "seed" => c.seed, "k" => k, "rep" => rep, "pool_size" => M, "pool_full" => c.pool_full[k], "n_fresh" => nf,
        "eig_raw_opt" => eig(X0, Xopt), "eig_raw_fresh" => ho_raw[1],
        "eig_start_opt" => -st.first, "eig_start_fresh" => ho_start[1], "eig_start_fresh_sd" => ho_start[2],
        "eig_final_opt" => -st.best, "eig_final_fresh" => ho_final[1], "eig_final_fresh_sd" => ho_final[2],
        "eig_run_opt" => eig(Zrun, Xopt), "eig_run_fresh" => ho_run[1],
        "n_calls" => Float64(ncalls), "secs" => secs, "converged" => Float64(get(dd, "eig_converged", NaN)), "hit_limit" => Float64(get(dd, "eig_hit_limit", NaN)),
        "grad_inf_final" => maximum(abs, g_best), "grad_2_final" => norm(g_best), "grad_inf_start" => maximum(abs, g_start), "grad_2_start" => norm(g_start),
        "relchg_last100" => relchg(100), "relchg_last500" => relchg(500), "calls_to_90" => n_to(0.9), "calls_to_99" => n_to(0.99), "late_gain_frac_2000" => late,
        "selfcheck_batch" => sc_batch, "selfcheck_init" => sc_init, "selfcheck_first" => abs(-st.first - eig(z0, Xopt)),
        "ck_insample" => ck_ins, "ck_heldout" => ck_held, "trace_grid_insample" => grid_ins,
    )
end

########################################################################
###############  run  ##################################################
########################################################################
dir = source_dir()
ns = _env_list("DIAG_N_ENS", CFG.N_ens_sizes)
seeds = _env_list("DIAG_SEEDS", collect(1:CFG.n_repeats))
kfilter = haskey(ENV, "DIAG_K") ? Set(_env_list("DIAG_K", Int[])) : nothing
println("test6: $(OFFLINE_CASE) $(CFG.variant); cells from $dir\n  budget: call_limit=$CALL_LIMIT optim_iters=$OPTIM_ITERS g_tol=$G_TOL f_reltol=$F_RELTOL n_eig=$N_EIG n_fresh=$N_FRESH reps=$N_REP; threads=$(Threads.nthreads())")
Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)

cells = Cell[]
for N in ns, seed in seeds
    path = joinpath(dir, results_filename(CFG, N, seed))
    isfile(path) || continue
    c = load_cell(path, N, seed, kfilter)
    c === nothing || isempty(c.pools) || push!(cells, c)
end
isempty(cells) && error("no cells found in $dir for N_ens ∈ $ns, seeds ∈ $seeds")
tasks = [(c, k, rep) for c in cells for k in sort(collect(keys(c.pools))) for rep in 1:N_REP]
println("  $(length(cells)) cells, $(length(tasks)) stages; full posterior reconstructed for ",
        sum(count(values(c.pool_full)) for c in cells), "/", sum(length(c.pools) for c in cells), " (cell, stage) pools; max |whitened − saved| = ",
        round(let v = filter(isfinite, [e for c in cells for e in values(c.pool_err)]); isempty(v) ? NaN : maximum(v) end; sigdigits = 2))

results = Vector{Any}(undef, length(tasks))
done = Threads.Atomic{Int}(0)
t_start = time()
Threads.@threads :dynamic for i in eachindex(tasks)
    c, k, rep = tasks[i]
    results[i] = try
        refit_stage(c, k, rep)
    catch e
        @warn "stage failed" N = c.N seed = c.seed k exception = (e, catch_backtrace())
        nothing
    end
    n = Threads.atomic_add!(done, 1) + 1
    n % max(1, length(tasks) ÷ 20) == 0 && println("  $n/$(length(tasks)) stages, $(round(Int, time() - t_start)) s")
end
rows = filter(!isnothing, results)
println("  finished $(length(rows))/$(length(tasks)) stages in $(round(Int, time() - t_start)) s")

########################################################################
###############  save + table  #########################################
########################################################################
scalar_keys = [k for (k, v) in rows[1] if v isa Number || v isa Bool]
out = Dict{String, Any}(k => [Float64(r[k]) for r in rows] for k in sort(scalar_keys))
out["ck_insample"] = reduce(hcat, [r["ck_insample"] for r in rows])'
out["ck_heldout"] = reduce(hcat, [r["ck_heldout"] for r in rows])'
out["trace_grid_insample"] = reduce(hcat, [r["trace_grid_insample"] for r in rows])'
out["ckpts"], out["trace_grid"] = CKPTS, TRACE_GRID
out["settings"] = Dict("call_limit" => CALL_LIMIT, "optim_iters" => OPTIM_ITERS, "g_tol" => G_TOL, "f_reltol" => F_RELTOL, "n_eig" => N_EIG, "n_fresh" => N_FRESH, "reps" => N_REP,
                       "variant" => String(CFG.variant), "case" => String(OFFLINE_CASE), "date" => string(CFG.calibrate_date), "source" => dir, "run_call_limit" => CFG.eig_call_limit)
tag = get(ENV, "DIAG_TAG", "")
path = joinpath(OFFLINE_OUT, "test6_eig_refit_$(DiagFiles.case_stem(String(OFFLINE_CASE)))_$(CFG.variant)_$(CFG.calibrate_date)$(isempty(tag) ? "" : "_" * tag).jld2")
JLD2.save(path, out)

col(k) = out[k]
fmed(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
rel(a, base) = a ./ abs.(base)
ins_gain = rel(col("eig_final_opt") .- col("eig_start_opt"), col("eig_start_opt"))
held_gain = rel(col("eig_final_fresh") .- col("eig_start_fresh"), col("eig_start_fresh"))
optimism = rel((col("eig_final_opt") .- col("eig_final_fresh")) .- (col("eig_start_opt") .- col("eig_start_fresh")), col("eig_start_fresh"))
base_gap = rel(col("eig_start_opt") .- col("eig_start_fresh"), col("eig_start_fresh"))
conv, hit = col("converged") .> 0.5, col("hit_limit") .> 0.5
other = .!conv .& .!hit
ck2000 = findfirst(==(2000), CKPTS)
held_gain_2000 = ck2000 === nothing ? fill(NaN, length(rows)) : rel(out["ck_heldout"][:, ck2000] .- col("eig_start_fresh"), col("eig_start_fresh"))
println("\nsummary (medians over stages; relative = / |EIG of the optimiser's start|, in-sample = on X_opt, held-out = mean over $(N_FRESH) fresh sets):")
@printf("  %-22s %6s %9s %9s %10s %11s %9s %10s %9s %10s\n", "stage set", "n", "med calls", "in-samp", "held-out", "frac held>0", "optimism", "held@2000", "late>2000", "calls→99%")
for (lab, sel) in (("all", trues(length(rows))), ("converged (own crit.)", conv), ("hit call limit", hit), ("stopped otherwise", other))
    any(sel) || continue
    @printf("  %-22s %6d %9.0f %9.3f %10.3f %11.2f %9.3f %10.3f %9.2f %10.0f\n", lab, count(sel), fmed(col("n_calls")[sel]), fmed(ins_gain[sel]), fmed(held_gain[sel]),
        mean(held_gain[sel][isfinite.(held_gain[sel])] .> 0), fmed(optimism[sel]), fmed(held_gain_2000[sel]), fmed(col("late_gain_frac_2000")[sel]), fmed(col("calls_to_99")[sel]))
end
@printf("\n  baseline subsample gap (start_opt − start_fresh)/|start| (no optimisation yet): median %.3f;  projection effect (start − raw start, held-out)/|raw|: median %.4f\n",
        fmed(base_gap), fmed(rel(col("eig_start_fresh") .- col("eig_raw_fresh"), col("eig_raw_fresh"))))
@printf("  at the optimum: |grad|∞ median %.2e (g_tol %.0e), relative change over last 100 calls median %.2e (f_reltol %.0e)\n", fmed(col("grad_inf_final")), G_TOL, fmed(col("relchg_last100")), F_RELTOL)
@printf("  selfchecks (should be ≈0): recorded eig_batch %.1e, eig_init %.1e (median |diff|); first objective vs EIG(start,X_opt) %.1e\n",
        fmed(abs.(col("selfcheck_batch"))), fmed(abs.(col("selfcheck_init"))), fmed(col("selfcheck_first")))
println("\nper N_ens: ", join(["N=$N: n=$(count(==(N), col("N_ens"))) conv=$(round(mean(conv[col("N_ens") .== N]); digits = 2)) held>0=$(round(mean(held_gain[col("N_ens") .== N][isfinite.(held_gain[col("N_ens") .== N])] .> 0); digits = 2))" for N in sort(unique(Int.(col("N_ens"))))], "  "))
println("Saved $path")
