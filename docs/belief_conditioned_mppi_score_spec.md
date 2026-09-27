# Belief-Conditioned MPPI Score Learning — Implementation Spec

**One-line goal.** Learn per-hypothesis MPPI update directions offline so that, at runtime, robust control under a *changing* belief over dynamics parameters costs one network forward pass instead of `M` full rollout sets.

**Target task.** 7-DOF arm transporting an unknown payload (unknown mass, CoM offset, grasp friction).

---

## 0. TL;DR of the method

```
Offline:  learn  K1(U, θ)   = MPPI update direction if hypothesis θ were true
          learn  K2(U, θ,θ')= interaction (disagreement) between hypotheses
          learn  cbar(U, θ) = smoothed cost value for hypothesis θ   [aux head]

Runtime:  belief w = {(θ_m, w_m)}_{m=1..M}   ← from particle filter, changes every step
          g = Σ_m w_m K1(U,θ_m) + Σ_r v_r (Σ_m w_m a_r(θ_m))²
          U ← U + η Σ g                      ← score ascent
```

Key property: `K1`, `K2`, `cbar` **do not depend on `w`**. Belief updates are free.

---

## 1. Notation

| Symbol | Shape | Meaning |
|---|---|---|
| `θ` | `(5,)` | unknown dynamics params: `(m, c_x, c_y, c_z, μ)` |
| `θ_m`, `w_m` | — | particle `m` and its belief weight; `Σ_m w_m = 1`, `M` particles |
| `U` | `(H, 7)` flattened `(d,)`, `d = H*7` | control sequence (joint velocity cmds) |
| `ε` | `(K, d)` | exploration noise, `ε ~ N(0, Σ)`, `Σ = σ² I` |
| `c(U, θ)` | scalar | rollout cost of `U` under hypothesis `θ` |
| `λ` | scalar | MPPI temperature |
| `C_w(U)` | scalar | `Σ_m w_m c(U, θ_m)` |

---

## 2. Math (what we learn and why)

### 2.1 Target quantity

MPPI's update is the score of the Gaussian-smoothed Gibbs measure:

```
J_w(U) = E_ε[ exp(-C_w(U+ε)/λ) ]
G[w](U) = ∇_U log J_w(U)
ΔU = Σ G[w](U)
```

### 2.2 Why naive linear mixing is wrong

`C_w` is linear in `w`, but `log J_w` is **not** (log and expectation do not commute).
Expanding the cumulant generating function gives a Volterra series in `w`:

```
G[w](U) =  Σ_m w_m · K1(U, θ_m)                     # 1st order  (= naive mixing)
         + Σ_{m,m'} w_m w_m' · K2(U, θ_m, θ_m')      # 2nd order  (missing term)
         + O(w³)
```

with

```
K1(U, θ)     = -(1/λ)   ∇_U E_ε[ c(U+ε, θ) ]
K2(U, θ, θ') = (1/2λ²)  ∇_U Cov_ε( c(U+ε,θ), c(U+ε,θ') )
```

**Interpretation.** `K1` = "what each hypothesis advises". `K2` = "how much hypotheses agree
about what is risky". Averaging advice (1st order) cannot hedge; the covariance term can.
Example: CoM-left and CoM-right hypotheses give `+3°` and `-3°` wrist tilt advice; their
average is `0°` (do nothing), which hedges nothing.

### 2.3 Low-rank factorization of K2

`K2` is a function of two parameter vectors → factorize with rank `R`:

```
K2(U, θ, θ') = Σ_{r=1..R} v_r(U) · a_r(U,θ) · a_r(U,θ')
```

so the double sum collapses:

```
Σ_{m,m'} w_m w_m' K2  =  Σ_r v_r(U) · ( Σ_m w_m a_r(U,θ_m) )²
```

Cost `O(M²) → O(RM)`. Symmetry is automatic. Use `R = 8..16`.

### 2.4 Validity criterion (when truncation is enough)

Series convergence is governed by `Var(C_w)/λ²`, which is directly observable as the
effective sample size of the MPPI weights:

```
ω_k = softmax(-C_w(U+ε_k)/λ)
ESS = 1 / Σ_k ω_k²            # report ESS/K
```

- `ESS/K` high (> ~0.2) → 2nd-order truncation valid
- `ESS/K` low → apply **penalty continuation**: scale `w ← γ·w` for `γ ∈ [0.1, 0.3, 1.0, 3.0]`,
  re-assembling each time. This is free because `K1, K2` are `w`-independent.

**Predicted (must verify): a wider belief → lower truncation error.** Averaging over many
hypotheses smooths the objective. So the method works best exactly when robustness matters
most, and degenerates gracefully to nominal MPPI as the belief concentrates.

---

## 3. Environment spec

### 3.1 Simulator

MuJoCo. 7-DOF arm (Franka Panda model is fine). Payload rigidly attached at the end-effector,
its inertial properties set from `θ`.

```
state x   = (q, q̇)                 ∈ R^14
control u = joint velocity command  ∈ R^7
dt = 0.05,  H = 16                  (0.8 s horizon)
U ∈ R^(16×7) = R^112
```

### 3.2 Unknown parameters and their ranges

| param | symbol | range | note |
|---|---|---|---|
| payload mass | `m` | `[0.2, 2.5]` kg | |
| CoM offset | `c_x,c_y,c_z` | `[-0.06, 0.06]` m | relative to grasp frame |
| grasp friction coeff | `μ` | `[0.2, 0.8]` | governs slip |

Prior `p_0(θ)`: uniform over the box (log-uniform for `m` is also acceptable).

### 3.3 Cost terms

`c(U, θ) = Σ_{t=1..H} ℓ(x_t, u_t; θ)` with

```
ℓ = w_goal  · ||p_eef(x_t) - p_goal||²
  + w_tau   · Σ_j max(0, |τ_j| - τ_max_j)²          # joint torque limit (depends on θ!)
  + w_slip  · max(0, F_tangential - μ·F_normal)²     # slip  (depends on θ!)
  + w_tilt  · max(0, |tilt(x_t)| - tilt_max)²        # payload tilt
  + w_jlim  · Σ_j max(0, |q_j| - q_max_j)²
  + w_vel   · Σ_j max(0, |q̇_j| - q̇_max_j)²
  + w_u     · ||u_t||²
```

Suggested starting scalars: `w_goal=1, w_tau=50, w_slip=200, w_tilt=100, w_jlim=50, w_vel=10, w_u=0.01`.
These are *fixed* problem weights — not to be confused with the belief `w_m`.

**Note:** `τ`, `F_tangential`, `F_normal`, `tilt` all depend on `θ`. That is the whole point.

### 3.4 Belief (particle filter)

```
particles: {θ_m}_{m=1..M},  weights {w_m}
M = 32  (sweep 1..64)

every control step:
  measure  z_t = wrist F/T sensor reading (6-dim) + joint torques
  predict  ẑ_t(θ_m) = analytic wrench from θ_m given (q, q̇, q̈)
  update   w_m ← w_m · N(z_t ; ẑ_t(θ_m), R)
  normalize; resample if ESS_pf < M/2 (systematic resampling)
  add small jitter to θ after resampling
```

Belief starts near-uniform on grasp and concentrates over ~0.5–1.5 s of motion.

---

## 4. Data generation (offline)

### 4.1 Core trick — Stein identity

```
∇_U E_ε[ f(U+ε) ] = Σ⁻¹ E_ε[ ε · f(U+ε) ]
```

Lets us estimate all kernels from one noise batch, with **no differentiation of the simulator**.

### 4.2 Collection loop

```python
# one data sample
x0    = sample_state()                 # (14,)  30% near constraint margins
U     = sample_control_seq()           # (112,) mixture, see 4.3
sigma = sample_noise_level()           # log-uniform in [σ_min, σ_max]
theta = sample_params(N_theta)         # (N_theta, 5), N_theta = 48, from prior

eps   = randn(K, 112) * sigma          # K = 1024

# ---- ONLY expensive part -------------------------------------------------
c = rollout_cost(x0, U + eps, theta)   # (K, N_theta)   K*N_theta rollouts
# --------------------------------------------------------------------------

# ---- labels, all from the same `c` matrix --------------------------------
cbar = c.mean(0)                                    # (N_theta,)
cc   = c - cbar                                     # centered, variance reduction

K1_hat = -(1/lam) * (eps.T @ cc) / (K * sigma**2)   # (112, N_theta) -> transpose
K1_hat = K1_hat.T                                   # (N_theta, 112)

# K2 (only needed for a subset of pairs; sample P pairs per sample, P≈64)
for (i,j) in sampled_pairs:
    t1 = (eps * (cc[:,i]*cc[:,j])[:,None]).mean(0)
    t2 = (eps * cc[:,i][:,None]).mean(0) * cbar[j]
    t3 = (eps * cc[:,j][:,None]).mean(0) * cbar[i]
    K2_hat[i,j] = (t1 - t2 - t3) / (2*lam**2 * sigma**2)     # (112,)

# ---- ground-truth scores for MANY beliefs, NO extra rollouts -------------
for n in range(N_BELIEF):              # N_BELIEF = 64
    w  = sample_belief(N_theta)        # (N_theta,), see 4.4
    Cw = c @ w                         # (K,)      matrix-vector product only
    om = softmax(-(Cw - Cw.min())/lam) # (K,)
    G_star = (om[:,None] * eps).sum(0) / sigma**2   # (112,)
    ess    = 1.0 / (om**2).sum()
    store(x0, U, sigma, theta, w, G_star, ess)
```

**Budget note.** One rollout batch (`K × N_theta` rollouts) yields `N_BELIEF = 64` training
targets. This is the main reason data collection is cheap here.

### 4.3 `U` sampling distribution (critical for OOD)

Mix in roughly equal parts:
1. smooth random sequences (OU process or low-pass filtered Gaussian)
2. solutions from running oracle MPPI with a random belief
3. (2) perturbed with noise at several magnitudes
4. deliberately bad sequences (violate limits) — needed so `K1` is learned in violation regions

### 4.4 Belief sampling distribution `W`

Mix:
- `Dirichlet(α=1)` over `N_theta` — broad
- `Dirichlet(α=0.2)` — peaked, near-delta (concentrated belief case)
- realistic PF posteriors replayed from logged runs
- beliefs with many zeros (subset of hypotheses only)

### 4.5 Target scale

Start with 200k stored samples. Report a data-scaling curve.

---

## 5. Network

```
INPUT
  x0    : (14,)
  U     : (112,)
  sigma : scalar   -> Fourier features (16,)
  theta : (M, 5)   -> Fourier features per dim (M, 60)

TRUNK                                    # θ-independent, computed ONCE
  h = MLP([x0, U, ff(sigma)]) -> (256,)
  arch: 4 × Linear(512) + SiLU + LayerNorm

TOKEN                                    # per-θ, cheap, cacheable
  z = MLP(ff(theta))          -> (M, 128)
  arch: 2 × Linear(256) + SiLU

HEADS
  K1   = Head1(h, z)          -> (M, 112)      # 1st-order kernel
  a    = Head2(h, z)          -> (M, R=8)      # 2nd-order, θ-dependent scalars
  v    = Head3(h)             -> (R=8, 112)    # 2nd-order, θ-independent
  cbar = Head4(h, z)          -> (M,)          # aux: smoothed cost value
  arch: Head_i = MLP([h ⊕ z], 2 × Linear(512) + SiLU)
```

**Why Fourier features on `θ` and `σ`:** raw scalars make the MLP learn only slowly-varying
functions of `θ`; beliefs can be sharply peaked, so the kernels must resolve fine `θ` structure.

**Cost:** trunk `O(1)`, heads `O(M)`. `M` can be changed at runtime freely.

---

## 6. Loss

```
L = L_score + 0.3 * L_K1 + 0.1 * L_cbar + 0.01 * L_smooth
```

```python
# 1) assembled-score loss (primary — this is what inference uses)
ws   = w                                        # (M,)
g    = ws @ K1 + ((ws @ a)**2) @ v              # (112,)
L_score = mahalanobis_sq(g - G_star, Sigma)     # ||·||²_{Σ}

# 2) anchor the 1st-order kernel directly (else 1st/2nd split is unidentifiable)
L_K1 = mse(K1, K1_hat)

# 3) aux value head, log-compressed (costs are near-zero most of the time, huge sometimes)
L_cbar = huber(log1p(cbar_pred) - log1p(cbar_hat))

# 4) smoothness in θ (improves interpolation to unseen θ)
L_smooth = || ∂K1/∂θ ||²        # finite-difference or autograd on θ input
```

**Training:** AdamW, lr `3e-4`, cosine decay, batch 256, ~300k steps. EMA of weights for eval.

**Important:** weight samples in the batch by `ESS` or drop samples with `ESS/K < 0.02`
(their `G_star` label is too noisy to be a useful target).

---

## 7. Inference

```python
def control_step(x0, belief, U_bar):
    theta, w = belief.particles, belief.weights        # (M,5), (M,)

    for sigma in SIGMA_SCHEDULE:                       # e.g. [0.4, 0.2, 0.1] annealing
        h = trunk(x0, U_bar, sigma)                    # once
        z = token(theta)                               # cacheable across steps
        K1, a, v, cbar = heads(h, z)

        for gamma in CONTINUATION:                     # e.g. [0.3, 1.0]  (free)
            ws = gamma * w
            g  = ws @ K1 + ((ws @ a)**2) @ v
            U_bar = U_bar + eta * Sigma @ g

    u0 = U_bar[0]
    U_bar = shift(U_bar)                               # warm start next step
    return u0, U_bar
```

### 7.1 Risk slider (same network, different `w`)

The aux head `cbar` makes this possible — it tells which particles are in the bad tail.

| `w` fed to assembly | Behaviour |
|---|---|
| `w = belief` | risk-neutral |
| `w ∝ belief^β` | tempered; `β` slides conservatism continuously |
| `w ∝ belief · 1[cbar ≥ VaR_α]` | CVaR — hedge against the worst `α` fraction |
| `w = δ(argmax_m cbar_m)` | minimax / worst case |
| `w = δ(argmax_m belief_m)` | aggressive, MAP-only |

No retraining for any of these.

### 7.2 Fallback

If belief entropy < threshold (belief has converged), fall back to plain nominal MPPI with the
MAP `θ`. The method should degrade gracefully, not be forced everywhere.

---

## 8. Baselines

| # | Name | Description | Expected failure mode |
|---|---|---|---|
| B1 | Nominal MPPI | full rollouts, MAP `θ` only | ignores uncertainty magnitude |
| B2 | **Robust MPPI (oracle)** | full rollouts over all `M` particles | correct but `M×` slower — this is the quality ceiling |
| B3 | DR-RL | SAC/PPO trained with randomized `θ` | permanently conservative; cannot exploit a converged belief |
| B4 | Context-conditioned RL (RMA-style) | policy conditioned on point estimate `θ̂` | cannot express *how uncertain* it is |
| B5 | Risk-sensitive MPPI | fixed risk parameter | not runtime-adjustable |
| A1 | **Ours, 1st order only** | `K1` mixing, no `K2` | ablation: no hedging |
| A2 | **Ours, 1st + 2nd order** | full method | — |

B2 is the most important baseline: we are *approximating* it, so the claim is
**"same quality, `M×` cheaper"**, not "better control".
B4 is the most important RL baseline: the distinction is *conditioning on a measure vs. a point estimate*.

---

## 9. Experiments

### E0 — GATING EXPERIMENT (run this first, before any learning)

**Question:** does particle count actually matter?

Run B2 (oracle robust MPPI) with `M ∈ {1, 2, 4, 8, 16, 32, 64}`. Plot success rate and
wall-clock vs `M`.

- If success saturates at `M ≤ 8` → the "too expensive" premise is weak. **Stop and reconsider
  the task design** (make CoM range wider, add a slip-critical phase, shorten the time budget).
- If success keeps improving to `M ≥ 32` → premise confirmed; proceed.

### E1 — Formulation check (no learning)

Compute `K1`, `K2` empirically from rollouts (not learned), assemble the score, compare with
brute-force MPPI ground truth.

- 1st-order only vs 1st+2nd: cosine similarity and relative norm error to `G_star`
- Plot error vs `ESS/K` → **this is Figure 1 of the paper**
- Expected: 2nd order clearly better when `ESS/K` moderate; both fail when `ESS/K → 0`

### E2 — Main result

- x-axis: `M`; y-axis: wall-clock per control step. B2 linear, ours ~flat.
- Overlay success rate. Claim: **robustness quality decoupled from compute**.

### E3 — Generalization

- Beliefs never seen in training (sharply peaked, multi-modal, partially zero)
- `M` changed at runtime (4 → 128) with no retraining
- `θ` outside the training box (mild extrapolation)

### E4 — Belief-width prediction

Plot truncation error vs belief entropy. **Predicted: error decreases as belief widens.**
A confirmed prediction is strong evidence the theory is right.

### E5 — Risk slider demo

Sweep `β` continuously; show conservatism varying smoothly, single trained network.

### E6 — Qualitative rollout

Grasp unknown box → belief wide → slow, body-hugging motion → belief converges → accelerates.
Then force a slip event → belief re-widens → controller backs off immediately. No retraining.

### Metrics

```
success          = reached goal ∧ no slip ∧ tilt < limit ∧ no torque violation
per-constraint violation profile (report as a vector, not a scalar)
wall-clock per control step (ms)
ESS/K distribution
belief entropy over time
```

**Always compare under equal wall-clock or equal rollout budget.** Sampling MPC can buy
performance with more samples; fairness objections are the most common review complaint.

---

## 10. Implementation order

| Phase | Deliverable | Acceptance criterion |
|---|---|---|
| **P0** | MuJoCo env + payload params + cost terms | costs respond sensibly to `θ` (heavy → torque violation) |
| **P1** | Oracle robust MPPI (B2) | solves the task; **run E0** and record the `M` curve |
| **P2** | Particle filter | belief concentrates within ~1 s of motion; correct `θ` retained |
| **P3** | Empirical kernel assembly (no NN) | **E1 passes**: assembled score matches brute-force MPPI when using exact `K1,K2` |
| **P4** | Data pipeline | 200k samples; verify `K1_hat` matches finite-difference `∇_U E[c]` on a few points |
| **P5** | Train `K1` only (+ `cbar`) | A1 approaches B2 on easy (wide-belief) cases |
| **P6** | Add `K2` low-rank | A2 > A1, gap grows as hypotheses disagree (E1 predicts where) |
| **P7** | Continuation + annealing + risk slider | E5 works; stiff cases recovered |
| **P8** | Baselines B3/B4 + full eval | E2–E6 |

**P3 is a hard gate.** If the assembled score with *exact* kernels does not reproduce
brute-force MPPI, the bug is in the assembly math, not the network. Do not train before this passes.

---

## 11. Pitfalls

1. **Do not skip E0.** The entire motivation rests on `M` mattering.
2. **`cbar` head is not optional** — without it the risk slider (CVaR, minimax) is impossible.
3. **Center costs before the Stein estimator** (`cc = c - c.mean(0)`). Large constant offsets
   destroy the signal-to-noise ratio.
4. **Anchor `L_K1`.** Without it, the network can move responsibility arbitrarily between 1st
   and 2nd order and training becomes unstable.
5. **Log-compress cost targets.** Constraint costs are zero most of the time and huge
   occasionally; plain MSE chases outliers only.
6. **Sample `w` with zeros.** If every particle always has weight, violation regions of
   individual hypotheses are never observed.
7. **`θ` is 5-D, not 1-D.** Include a scaling study over `dim(θ)`; do not assume it interpolates
   as easily as a scalar index.
8. **Honest ablation vs. `ψ`-style learning.** Learning per-hypothesis cost-to-go
   `ψ(U,θ) ≈ c(U,θ)` and doing softmax at runtime is *exact* in `w` but costs `O(K·M)` network
   evals instead of `O(M·T)`. Measure both; the score formulation is justified only by the
   compute gap (expect ~100–200×), not by accuracy.
9. **Report where it fails.** Low `ESS/K`, concentrated belief, `θ` far outside training range.
   Graceful fallback to nominal MPPI is a feature, not an admission.

---

## 12. Suggested hyperparameters (starting point)

```
H = 16, dt = 0.05
K = 1024 (data collection),  T = 3..5 ascent iters (inference)
M = 32 (default), swept 1..64
lam = 1.0                      # tune so ESS/K ≈ 0.2–0.4 at nominal belief
sigma schedule = [0.4, 0.2, 0.1]
eta = 0.5                      # ascent step, tune per sigma
R = 8                          # K2 rank
N_theta = 48 per data sample,  N_BELIEF = 64 per rollout batch
P = 64 sampled (i,j) pairs per sample for K2 labels
```

---

## 13. Key references to position against

- Touati & Ollivier, *Forward-Backward representations* — zero-shot RL for arbitrary reward
  functions; strongest RL competitor. Weakness: linear task encoding; weak on constrained tasks.
- Luo, Sun, Tenenbaum, Du, *Potential Based Diffusion Motion Planning* (ICML 2024) — composes
  per-constraint learned potentials at test time; needs a refinement step. Our `K2` analysis
  explains *why* that refinement is needed.
- Du et al., *Reduce, Reuse, Recycle* (ICML 2023) — score composition is wrong, fixed with MCMC.
- Sacks & Boots, *Learning to Optimize in MPC* / *Learning Sampling Distributions for MPC*;
  *Deep Model Predictive Optimization* — prior art on learning the MPPI update. Difference:
  their cost function is **fixed**; ours varies at runtime.
