# ReinforcementLearning — DDPG actor-critic as an optimizer for the inverse problem

Reinforcement learning (RL) applied to the benchmark by identifying the unknown
forcing parameterization with the **policy** and the finite-time statistics misfit
with the **reward**. The actor-critic method used is **DDPG** (with the TD3
robustness heuristics available as config switches, off by default).

Unlike `adam` / `levenberg_marquardt` (ForwardDiff Jacobians) and
`score_based_{adam,lm}` (GFDT/score Jacobians), no Jacobian of the simulator is ever
formed. The parameter gradient comes from **differentiating a learned critic**.

## 1. RL formulation

Notation follows the write-up this experiment implements.

| RL object | Inverse-problem meaning |
|---|---|
| policy `π_θ(x) = F_θ(x)` | the parameterized unknown forcing (L63: `(log ρ, log β)`; L96: `log F`, per-site `F`, or NN weights) |
| state `x_k` | Lorenz state at the start of RL step `k` |
| action `F_k` | forcing applied during step `k` (policy output + exploration noise) |
| `Φ_k = ‖y − G(x_{T_k})‖²_Γ` | whitened misfit of the statistics window of step `k` (= `ny·RMSE²`) |
| reward `r_k = Φ_{k−1} − Φ_k` | decrease in misfit |
| return / critic `Q(x_k,F_k) = E[Σ_j γʲ r_{k+j}]` | with `γ=1`: `Φ_{k−1} − E[Φ_∞ ∣ x_k,F_k]` |
| `L(π_θ) = E[Q(x_0,F_θ(x_0))]` | `const − J(θ)`, so maximizing `L` minimizes the inverse-problem objective `J` |

The actor update uses `∇_θ J = −E_x[ ∇_θF_θ(x) ∇_F Q(x,F)|_{F=F_θ} ]`, with `Q` replaced
by a learned critic `Q_ϑ` trained on Bellman targets
`z_k = r_k + γ Q_ϑ(x_{k+1}, π_θ(x_{k+1}))`.

In this benchmark the policy is **state-independent** (the parameter vector *is* the
policy), so `∇_θF_θ(x)` is the identity in the coordinates used here and the actor
gradient is simply `E_x[∇_F Q_ϑ(x,F)]` evaluated at `F = F_θ`.

### Implementation choices that differ from, or sharpen, the write-up

1. **`γ < 1` (default 0.5), not `γ = 1`.**
   For a state-independent policy at `γ=1` the reward telescopes and the Bellman fixed
   point of `Q` has **no `F`-dependence**: a single window run under `F ≠ π(x)` followed
   by `π` forever has the same long-time statistics, so `∇_F Q = 0`. The telescoped
   identity `Q = Φ_{k−1} − E[Φ_∞]` holds for the objective `J` (where `F` is held fixed
   forever), but DDPG bootstraps with `π`, not with `F`. With `γ<1` the reward keeps an
   immediate `−(1−γ)Φ_k(F)` term, so `∇_F Q ≈ −(1−γ)∇_F E[Φ_k]` plus a discounted
   state-transition term. This is confirmed empirically (see §6): `γ=1` reproduces the
   no-learning control exactly. **Treat `γ` as a method hyperparameter, not a free
   choice of the reduction.**
2. **Critic is learned on `q`, with the `Φ_{k−1}` offset removed analytically.**
   `Q(x_k,F_k) = Φ_{k−1} + q(x_k,F_k)`; `Φ_{k−1}` is observed at `x_k` and cannot be
   influenced by `F_k`. Substituting in the Bellman target, `Φ_{k−1}` cancels
   *exactly*:
   `q-target_k = −(1−γ)Φ_k + γ q(x_{k+1}, π(x_{k+1}))`.
   So the buffer stores `(x_k, F_k, Φ_k, x_{k+1})`, never `Φ_{k−1}`, and
   `∇_F Q = ∇_F q`. This is mathematically identical to the stated reward, minus a
   history-dependent offset the network could not otherwise see.
3. **Actor targets.** `π_θ(x_{k+1})` in the target uses the *current* actor (no target
   actor). A Polyak-averaged **target critic** is used (`polyak = 0.05`; `polyak = 1`
   recovers the un-targeted recursion written in the description).
4. **Actor coordinates.** The actor is stored as `θn` in prior-std units
   (`F = prior_mean + prior_std .* θn`) and updated with Adam (`Optimisers.jl`);
   `actor_lr` is therefore in units of prior standard deviations.
5. **Critic scaling.** Inputs are standardized (state mean/std and prior-scaled
   action), and the target is divided by `φ_scale` (median warm-up `Φ`). These are
   frozen at the first update.
6. **Algorithm: DDPG vs TD3** (`ALGORITHM` toggle / env var, `:ddpg` or `:td3`). Both use
   Gaussian exploration noise on the applied action. `rl-ddpg` is plain DDPG
   (`n_critics, policy_delay, target_noise = 1, 1, 0`); `rl-td3` switches on the three
   TD3 additions (`2, 2, 0.2` with clip `0.5`): twin critics with a min-target, actor and
   target-critic updates once per `policy_delay` critic updates, and clipped Gaussian
   smoothing noise on the target action (added to `θn` in prior-std units, since the
   actor is a single vector). The actor ascends critic 1 only. Neither uses a target actor
   (the Bellman target uses the current actor).

## 2. The two trajectory cases

The two cases differ in the **length of one RL step** relative to the statistics window.
A single trajectory is followed continuously: the state at the end of step `k`
starts step `k+1`. Initial condition is `x0 + ic_cov_sqrt·randn`, as everywhere else.
Let `W = T_end − T_start` be the statistics-window length of the other experiments and
`T` the full forward-evaluation length (spin-up + window).

| | **long** (`UPDATE_FREQUENCY=long`) | **short** (`UPDATE_FREQUENCY=short`) |
|---|---|---|
| RL step length | the whole statistics window: one standard forward map (`T`, statistics over `[T_start,T_end]`) | a fraction `1/steps_per_window` of `W` (stride `W/steps_per_window`) |
| `steps_per_window` | 1 | 10 (set 100 for a 100x shorter step) |
| RL updates per simulation | **1 per forward evaluation** | **many**: 10 per `W` of simulated time |
| statistic `G_k` | `lorenz_forward_with_states` (exactly the other experiments' `G`) | `stats` over the trailing window of length `W` (identical statistic and sample count; consecutive windows overlap) |
| cost per step | 1 unit | `stride/T` units — a fraction (plus one initial full evaluation of 1 unit) |
| spin-up | paid every step (as in every other method) | paid **once** |

Both use one critic update per environment step (`updates_per_step = 1`). In the short
case the statistic is computed over a trailing window of the full length `W` rather than
over the (much shorter) step itself, so the statistics are not biased by a shortened
estimation window (the variance/covariance statistics of a window `≪ W` have a different
expectation than the observation `y`, which would move the optimum).

> **Naming.** *long* = RL step equal to the window (one RL update per simulation
> evaluation); *short* = RL step far shorter than the window (many RL updates per window,
> each at fractional cost). `steps_per_window` is the single knob between them.

## 3. Cost metric

One cost unit = one forward-model evaluation of the standard length `T`.

```
long  :  conv_score = number of RL steps until RMSE_k < rmse_target
short :  conv_score = 1 + (steps · stride·dt) / T   until RMSE_k < rmse_target
```

There is **no `(nu+1)` Jacobian factor** because there is no Jacobian. As in
`score_based_*`, only simulation cost is counted; the compute spent training the critic
is real but is not part of the metric. The per-cell budget is `max_evals = 10000`
(twice `score_based_adam`'s `budget_total = 5000`, so budgets are not matched to that
experiment); NaN if not reached.

`RMSE_k = sqrt(Φ_k/ny)` is evaluated on the window rolled out under the **applied**
(exploratory) action `F_k = π_θ + noise`, so it is conservative relative to the
noise-free policy. `final_params` is the applied `F_k`; the noise-free policy is saved
as `policy_params`.

Caveats on comparability (also relevant when reading the leaderboard):

* The **short** variant does not re-pay the spin-up every step, which is a large part of
  its cost advantage (for L63, `T_start/T = 0.75`). This is a property of the continuing
  RL trajectory, not a tuning gain.
* In the short variant the convergence test is applied at every stride on overlapping
  windows (correlated checks), and the window contains data generated under earlier
  actions, so it lags the policy.
* As for every method here, RMSE is checked from one noisy realization per step, so
  targets near the noise floor (1.0–1.2) can be hit by chance. The `actor_lr = 0`
  control in §6 quantifies that luck.

## 4. Layout

| File | Role |
|---|---|
| `experiment_config.jl` | per-case config, filename builders (update-frequency-aware), array/algorithm/update-frequency dispatch (`rl_algorithm()`, `rl_update_frequency()`; toggles `ALGORITHM`, `UPDATE_FREQUENCY`) |
| `l63_preliminaries.jl`, `l96_preliminaries.jl` | shared truth data — identical to `adam`'s (so comparable) |
| `ddpg_core.jl` | model-agnostic: replay buffer, critic MLP(s), actor (`θn` + Adam), `ddpg_update!` |
| `rl_env.jl` | the Lorenz forward map as an environment; `:long` (full-window) and `:short` (fractional-window) stepping |
| `rl_run.jl` | `run_one` training loop for one `(rmse_target, rng_idx)` cell, `rl_main` |
| `l96_problem.jl` | L96 forcing parameterizations, priors, `make_forcing` |
| `run_l63_ddpg.jl`, `run_l96_ddpg.jl` | drivers (problem setup + `rl_main`); update frequency (`:short` / `:long`) set by `UPDATE_FREQUENCY` |
| `run_to_leaderboard.jl` | per-cell JLD2 → leaderboard netcdf (`algorithm_type = rl-{ddpg,td3}-{short,long}`) |

`Lorenz63.jl` and `Lorenz96.jl` define clashing types, so the L63 and L96 drivers are
separate; everything in `ddpg_core.jl`/`rl_env.jl`/`rl_run.jl` is shared.

## 5. Hyperparameters (`experiment_config.jl`)

| Group | Parameter | Default |
|---|---|---|
| Bellman | `gamma` | 0.5 |
| | `polyak` | 0.05 |
| TD3 switches (`rl-ddpg` / `rl-td3`) | `n_critics` / `policy_delay` / `target_noise` (clip 0.5) | 1 / 1 / 0.0  vs  2 / 2 / 0.2 |
| Optimization | `critic_lr` / `actor_lr` (prior-std units) | 1e-3 / 0.02 |
| | `batch_size` / `updates_per_step` | 64 / 1 |
| Exploration | `explore_frac` / `warmup_frac` (prior-std units) | 0.1 / 0.5 |
| Critic | hidden width (2 hidden layers, ReLU) | 64 (L63, const), 128 (vec, flux) |
| long | `steps_per_window` / `n_warmup` / `buffer_size` | 1 / 20 / 2000 |
| short | `steps_per_window` / `n_warmup` / `buffer_size` | 10 / 50 / 5000 |
| Protocol | `n_repeats` / `rmse_targets` / `max_evals` | 30 / [1.0, 1.1, 1.2] / 10000 |

The critic is deliberately small (as in `score_based_adam/score_nets.jl`): each window
provides one sample of a noisy misfit, so a large network cannot be supported by the
data. Hyperparameters were set once on L63 and **not tuned per case**.

## 6. Smoke-test results (not leaderboard runs)

Quick local checks during development (target RMSE 1.2 unless stated; the 10000-unit
config budget was *not* used — see the Budget column). `control` = same code with `actor_lr = 0` (policy never
moves, only exploration noise + prior draw).

| Case | Variant | Budget | Seeds | Converged (learned) | Converged (control) |
|---|---|---|---|---|---|
| L63 | long  | 1000 | 10 | 10/10 (median 112 evals) | 3/10 |
| L63 | short | 1000 | 10 | 10/10 (median ≈ 6 units) | 3/10 |
| L96 const | long  | 500 | 6 | 6/6 | 5/6 |
| L96 const | short | 500 | 6 | 6/6 | 3/6 |
| L96 vec  | short, long | 300 | 2 | 0/2 | — |
| L96 vec  | long  | 5000 | 5 | 0/5 — mean policy RMSE 6.50 → 3.99 | 0/5 — RMSE unchanged |
| L96 vec  | short | 5000 | 5 | 0/5 — mean policy RMSE 6.50 → 2.11 | 0/5 — RMSE unchanged |
| L96 flux | short, long | 200 | 2 | 0/2 | — |

L63 `gamma` sweep (long, 10 seeds, budget 1000): `γ=0` 10/10 (median 214), `0.5` 10/10
(112), `0.9` 8/10 (125), `1.0` **3/10 — identical to the control**, as predicted in §1.1.
L63 at target 1.0 / 1.1 (long): 9/10 each.

**L96 vec detail** (5000-unit budget, i.e. half the current 10000 config budget; target
1.2; seeds 1–5; noise-free policy RMSE averaged over 5 fixed ICs, initial → final):

| Seed | initial | long | short |
|---|---|---|---|
| 1 | 7.64 | 5.01 | 1.99 |
| 2 | 4.56 | 3.06 | 2.40 |
| 3 | 6.22 | 3.93 | 1.90 |
| 4 | 6.90 | 3.28 | 2.19 |
| 5 | 7.19 | 4.66 | 2.06 |

Neither variant reached 1.2 on any seed, so no vec cell has a finite score yet. The
learning is real (all ten `actor_lr = 0` controls leave the RMSE within ~0.2 of its
initial value), and the short variant gets clearly closer than the long one at equal
cost (about 2.1 vs 4.0 on average), consistent with it making ~10x more RL updates per
unit of cost. Whether vec reaches the targets within the 10000-unit config budget is
untested (these runs started before the budget was raised); flux has not been run
beyond 200 units. **Read this table with care:** seeds are few and the smaller cases use
budgets below the leaderboard budget. These are the hard cases for a learned critic and
their leaderboard behaviour is **not yet established**. L96 const converges often by luck alone (the prior is already
close to the truth for many seeds), so the control matters for interpreting it.

## 7. Running

```bash
# one-time setup
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. l63_preliminaries.jl
for c in l96_const l96_vec l96_flux; do EXPERIMENT=$c julia --project=. l96_preliminaries.jl; done

# The algorithm and update frequency default to the ALGORITHM / UPDATE_FREQUENCY toggles in
# experiment_config.jl; override with the env vars of the same name.
# Leaderboard names: rl-{ddpg,td3}-{short,long}.
# L63 — all cells / one cell
ALGORITHM=ddpg UPDATE_FREQUENCY=short julia --project=. run_l63_ddpg.jl
ALGORITHM=td3  UPDATE_FREQUENCY=long  julia --project=. run_l63_ddpg.jl 1

# L96 — all cells (one case at a time) / one cell
ALGORITHM=ddpg UPDATE_FREQUENCY=short EXPERIMENT=l96_const julia --project=. run_l96_ddpg.jl
ALGORITHM=td3  UPDATE_FREQUENCY=long  EXPERIMENT=l96_vec   julia --project=. run_l96_ddpg.jl 1
ALGORITHM=td3  UPDATE_FREQUENCY=short EXPERIMENT=l96_flux  julia --project=. run_l96_ddpg.jl

# Leaderboard netcdf (per algorithm, update frequency and case)
ALGORITHM=td3 UPDATE_FREQUENCY=short julia --project=. run_to_leaderboard.jl
ALGORITHM=ddpg UPDATE_FREQUENCY=long EXPERIMENT=l96_const julia --project=. run_to_leaderboard.jl
```

Tasks are indexed over `rmse_targets × n_repeats` (`3 × 30 = 90` per case, algorithm and update frequency).
Set `RUN_DATE` (or pin `run_date` in `experiment_config.jl`) so all cells of a batch
share filenames.

### HPC (SLURM, Caltech Resnick)

`hpc-variant/` holds only sbatch + submit scripts; the `.jl` files, `experiment_config.jl`
and `Project.toml` live once, in this directory (the sbatch files run `julia --project=.. ../${SCRIPT}`).

```bash
cd hpc-variant
bash submit_precompile.sh                       # once, and after package updates

# Toggles (env vars): ALGORITHM = ddpg | td3 | all, UPDATE_FREQUENCY = short | long | all
bash submit_l63.sh                                           # ddpg, short (defaults)
ALGORITHM=td3 UPDATE_FREQUENCY=long bash submit_l96_const.sh # one arm
ALGORITHM=all UPDATE_FREQUENCY=all  bash submit_l96_vec.sh   # all four arms
bash submit_l96_flux.sh [EXP_ID]                             # EXP_ID labels the job names
```

Each `(ALGORITHM, UPDATE_FREQUENCY)` arm is a separate array job (`--array=1-90`) with its
own leaderboard job, and writes its own netcdf (`algorithm_type = rl-<alg>-<freq>`). One
`preliminaries` job per submission is shared by all of its arms (the prelim file does not
depend on the arm, so submitting arms through one `ALGORITHM=all` call avoids concurrent
writes of that file). `RUN_DATE` is pinned once per submission.
Run `run_array` smoke tests with `--array=1-1` (see the reference block at the end of each `submit_*.sh`).
The array wall time is `12:00:00`; the short arm and non-converging cells (L96 vec / flux)
are the slow ones, so raise `--time` in `run_array.sbatch` if cells hit the limit.

## 8. Pipeline

```
l63/l96_preliminaries →(afterok)→ run_{l63,l96}_ddpg (array over cells) →(afterany)→ run_to_leaderboard
```
(on HPC the array + leaderboard stages are repeated per `(ALGORITHM, UPDATE_FREQUENCY)` arm)
