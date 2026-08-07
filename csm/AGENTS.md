# AGENTS.md — DIAL-MPC Compositional Score Matching

## Scope

`csm` is a JAX/Brax-native extension of this repository.  Do not add imports
from the prototype's former `hydrax`, `gpc`, Warp, or Torch projects.

## Mathematical targets

For normalized DIAL action sequences `U` in `[-1, 1]`, data collection uses
plain isotropic MPPI:

```
V = clip(U + sigma * epsilon, -1, 1)
w = softmax(-(C(V) @ omega) / temperature)
score = (sum(w * V) - U) / sigma**2
```

The deployment model is an objective-compositional trajectory energy:

```
E(o, U) = [E_tracking(o,U), E_stability(o,U), E_gait(o,U)]
E_omega(o, U) = omega @ E(o, U)
U <- project(U - eta * grad_U E_omega(o, U))
```

Compose raw scalar energies, never finite-sample MPPI bounded updates.  The
Gibbs normalization and action clipping make the latter nonlinear in omega.
Exact `MBDPI.reverse_once` updates are permitted only as derivative-direction
supervision for the composed energy.

The preferred zero-shot controller is the anchor-factorized Gibbs score model.
It learns clean candidate logits at a full-rank set of anchor weights, composes
an arbitrary preference in logit space, and only then applies the nonlinear
Gibbs softmax:

```
W.T @ alpha = omega
logit_omega(V) = sum_a alpha[a] * logit_anchor[a](V)
weight(V) = softmax_V(logit_omega(V))
U <- sum_V weight(V) * V
```

Never linearly compose normalized Gibbs weights or already-noised scores.
AFGS data must use the proposal of `DIALTCMPPI` and a scale shared across all
anchors in a query; per-weight reward standardization breaks logit linearity.

## DIAL integration

- `data_collection.DIALScoreCollector` rolls a Brax `State` through
  `env.step` with `jax.lax.scan` and batches samples with `jax.vmap`.
- An objective function has signature `cost(state_after_step, action) -> (k,)`
  and must return raw, unweighted stage costs.
- `envs.get_objective_spec` supplies the built-in Go2 trot decomposition:
  tracking, stability, and gait.  Unknown environments fall back to
  the scalar `-state.reward` objective.  Add new DIAL decompositions there.
- DIAL actions, collector queries, model targets, and policy sequences all use
  the same normalized `[-1, 1]` coordinates.

## Training and inference

- `architectures.CompositionalEnergyMLP` is a shared encoder with one scalar
  trajectory-energy head per raw cost objective.
- `energy_training.fit_compositional_energy` regresses raw rollout costs and
  uses exact DIAL updates as sparse gradient direction and log-RMS magnitude
  supervision.  It also matches every objective head to the exact Brax rollout
  cost Jacobian in normalized coordinates (Sobolev supervision).  Base and
  each DAgger round must be sampled equally.
- Fit `StandardNormalizer` on raw observations and preserve per-objective cost
  mean/std in `CompositionalEnergyPolicy`.
- `CompositionalEnergyPolicy.apply` performs projected gradient descent on
  `omega @ E` inside an RMS trust region around the warm-start plan; `shift`
  warm-starts the next DIAL control step.

## Verification

Run the complete minimal integration test on GPU:

```bash
source .venv/bin/activate
dial-csm --example unitree_go2_trot --smoke
```

It must create a loadable `policy.pkl`, `scores.npz`, and
`visualization.html` under the ignored `csm_runs/` directory.  Also run
`python -m compileall -q csm`, `uv pip check`, and `git diff --check` after
changes.
