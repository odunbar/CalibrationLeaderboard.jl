# offline_common.jl — shared setup for the offline GBOED tests (test1_emulator_quality.jl, test2_sampler_convergence.jl).
#
# Everything is rebuilt from ONE instrumented calibrate cell (its "diagnostics" key, written by boed_diagnostics.jl): the cumulative
# design Z and whitened outputs Yw, the fitted GP hyperparameters per iteration, and the whitening bases. No re-run of the algorithm.
# These scripts MAY evaluate the true forward model (the "don't change the run" rule applies to the calibrate stage, not to offline checks).
#
# Environment (all the offline tests):
#   EXPERIMENT=l63|l96_const|l96_vec|l96_flux   (default l63)      BOED_VARIANT=tmcmc_eig|tmcmc|iekf_eig|iekf (default: experiment_config.jl)
#   CALIBRATE_DATE=<date of the cell directory>  (default today)    DIAG_SOURCE=<dir holding the cell JLD2s> (default output/<variant>_<date>)
#   DIAG_N_ENS=<N_ens>  (default: the middle N_ens of the experiment)   DIAG_SEED=<rng_idx> (default 1)

using Distributions
using GaussianProcesses
using JLD2
using LinearAlgebra
using Random
using Statistics

const BOED_DIR = dirname(@__DIR__)
const _COMMON = joinpath(BOED_DIR, "..", "..", "common")
const OFFLINE_CASE = Symbol(get(ENV, "EXPERIMENT", "l63"))
OFFLINE_CASE in (:l63, :l96_const, :l96_vec, :l96_flux) || error("EXPERIMENT must be l63, l96_const, l96_vec or l96_flux")
OFFLINE_CASE === :l63 || Core.eval(Main, :(using Flux))   # the l96 cells hold Flux objects (flux-force: the truth network)
include(joinpath(_COMMON, "forward_maps", OFFLINE_CASE === :l63 ? "Lorenz63.jl" : "Lorenz96.jl"))   # types needed to load the cell
include(joinpath(_COMMON, "diagnostics", "diag_files.jl"))   # file-name grammar (case_stem)
using .DiagFiles
include(joinpath(BOED_DIR, "experiment_config.jl"))
include(joinpath(BOED_DIR, "boed_core.jl"))

const OFFLINE_OUT = joinpath(@__DIR__, "output")
mkpath(OFFLINE_OUT)

"""
    load_offline_cell() -> (; cfg, N_ens, seed, path, cell, D)

`cell` is the whole result file, `D = cell["diagnostics"]`. Errors with the command to run if the cell was not instrumented.
"""
function load_offline_cell()
    cfg = experiment_config(OFFLINE_CASE)
    N_ens = parse(Int, get(ENV, "DIAG_N_ENS", string(cfg.N_ens_sizes[cld(length(cfg.N_ens_sizes), 2)])))
    seed = parse(Int, get(ENV, "DIAG_SEED", "1"))
    dir = get(ENV, "DIAG_SOURCE", joinpath(BOED_DIR, "output", calib_directory(cfg)))
    path = joinpath(dir, results_filename(cfg, N_ens, seed))
    isfile(path) || error("No cell at $path. Run it first, e.g.\n  BOED_VARIANT=$(cfg.variant) EXPERIMENT=$(OFFLINE_CASE) CALIBRATE_DATE=$(cfg.calibrate_date) julia --project=. -t 8 calibrate_$(OFFLINE_CASE === :l63 ? "l63" : "l96").jl <task_index>")
    cell = JLD2.load(path)
    haskey(cell, "diagnostics") && get(cell["diagnostics"], "enabled", false) && haskey(cell["diagnostics"], "Z") ||
        error("Cell $path has no instrumented diagnostics (run predates boed_diagnostics.jl, or DIAG=0)")
    return (; cfg, N_ens, seed, path, cell, D = cell["diagnostics"])
end

# BOEDProblem rebuilt from the saved bases (only the fields the samplers/likelihood use are meaningful; the cum_var field is a dummy).
function proxy_problem(D)
    ob = WhitenedPCABasis(D["output_V"], D["output_lambda"], length(D["output_lambda"]), Float64[])
    pb = WhitenedPCABasis(D["prior_V"], D["prior_lambda"], length(D["prior_lambda"]), Float64[])
    log_cases = OFFLINE_CASE in (:l63, :l96_const)   # log-normal priors; the others are Gaussian in raw space
    return BOEDProblem(ob, D["y_whitened"], D["prior_mean"], pb, log_cases ? log : identity, log_cases ? exp : identity)
end

"""
    rebuild_gps(D, k) -> BOEDGPs

GP_k from the saved cumulative design, whitened outputs and fitted hyperparameters (no optimisation). Differs from the run's GP only by the
Float32 rounding of Z and Yw; `selfcheck_gps` compares its log target with the one recorded in the run.
"""
function rebuild_gps(D, k::Int)
    n = Int(D["n_train"][k])
    Z = Float64.(D["Z"][:, 1:n])
    Y = Float64.(D["Yw"][1:n, :])
    H = D["gp_hyper_by_k"][k]       # k_out x (k_in + 2): [iℓ2..., σ2, noise variance]
    k_in = size(Z, 1)
    gps = Vector{GaussianProcesses.GPE}(undef, size(Y, 2))
    for j in 1:size(Y, 2)
        ll = -0.5 .* log.(H[j, 1:k_in])
        kern = GaussianProcesses.SEArd(ll, 0.5 * log(H[j, k_in + 1]))
        gps[j] = GaussianProcesses.GPE(Z, Y[:, j], GaussianProcesses.MeanZero(), kern, 0.5 * log(H[j, k_in + 2]))
    end
    return BOEDGPs(gps)
end

selfcheck_gps(gps::BOEDGPs, D, k::Int) = (rebuilt = sum(gp.target for gp in gps.gps), recorded = Float64(D["gp_mll_sum"][k]))

# Posterior samples saved by the run, whitened input coordinates k_in x n (a fixed evenly spaced subsample of the iteration-k posterior).
saved_posterior(D, k::Int) = Float64.(D["X_post_sub_by_k"][k])

# θ* projected into whitened coordinates (nothing for flux), recorded by the run.
saved_truth_z(D) = isempty(D["z_true"]) ? nothing : Float64.(D["z_true"])

# The cell's N_ens (initial LHS size and acquisition batch size).
cell_N_ens(D) = Int(D["n_train"][1])

"""
    true_forward_whitened(prob, cell, Zpts; nrep, ic_seed) -> (mean, var)  (both k_out x M)

TRUE forward model at the whitened-input points `Zpts` (k_in x M): decode to raw θ, run `nrep` initial-condition replicates with common random
numbers across points, whiten the outputs with the run's output basis. The replicate variance is the forward model's own noise floor.
"""
function true_forward_whitened(prob, cell, Zpts::AbstractMatrix; nrep::Int = 8, ic_seed::Int = 2024)
    M = size(Zpts, 2)
    theta = from_prior_whitened(prob, Zpts)
    x0 = cell["x0"]
    ics_rng = MersenneTwister(ic_seed)
    ics = [x0 .+ cell["ic_cov_sqrt"] * randn(ics_rng, length(x0)) for _ in 1:nrep]
    n_out = length(cell["y"])
    raw = zeros(nrep, M, n_out)
    Threads.@threads for idx in 1:(M * nrep)
        j, r = fldmod1(idx, nrep)
        raw[r, j, :] = if OFFLINE_CASE === :l63
            lorenz_forward(EnsembleMemberConfig(theta[:, j]), ics[r], cell["lorenz_cfg"], cell["obs_cfg"])
        else
            forcing = build_forcing(cell["truth_phi"], theta[:, j], cell["phi_structure"], cell["sample_range"])
            lorenz_forward(forcing, ics[r], cell["lorenz_cfg"], cell["obs_cfg"])
        end
    end
    TM = zeros(length(prob.y_whitened), M)
    TV = zeros(length(prob.y_whitened), M)
    for j in 1:M
        W = whiten_samples(prob.output_basis, raw[:, j, :])   # nrep x k_out
        TM[:, j] = vec(mean(W; dims = 1))
        TV[:, j] = vec(var(W; dims = 1))
    end
    return TM, TV
end

# Surrogate log-likelihood of each column of Zpts under `gps` (the same form as boed_loglik).
surrogate_loglik(prob, gps, Zpts) = [boed_loglik(prob, gps, Zpts[:, i]) for i in 1:size(Zpts, 2)]

# The true log-likelihood in the same functional form, with the replicate-mean forward output (var = replicate variance / nrep, or 0).
true_loglik(prob, TM, TV) = [sum(-0.5 .* (TM[:, i] .- prob.y_whitened) .^ 2 ./ (1 .+ TV[:, i]) .- 0.5 .* log.(2π .* (1 .+ TV[:, i]))) for i in 1:size(TM, 2)]

ranks(x) = (r = zeros(length(x)); r[sortperm(x)] = 1:length(x); r)
spearman(a, b) = length(a) < 3 ? NaN : cor(ranks(a), ranks(b))

# Energy distance between two samples (columns), on at most `n` evenly spaced columns each.
function energy_distance(A::AbstractMatrix, B::AbstractMatrix; n::Int = 300)
    pick(X) = X[:, unique(round.(Int, range(1, size(X, 2); length = min(n, size(X, 2)))))]
    A, B = pick(A), pick(B)
    md(X, Y) = mean(norm(X[:, i] .- Y[:, j]) for i in 1:size(X, 2), j in 1:size(Y, 2))
    return 2 * md(A, B) - md(A, A) - md(B, B)
end
