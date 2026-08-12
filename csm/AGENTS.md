# AGENTS.md — DIAL-MPC Compositional Score Matching

## Scope

`csm` is a JAX/Brax-native extension of this repository.  Do not add imports
from the prototype's former `hydrax`, `gpc`, Warp, or Torch projects.

## Mathematical target

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
  uses exact DIAL updates as sparse gradient direction and bounded final-update
  supervision.  It also matches every objective head to the exact Brax rollout
  cost Jacobian in normalized coordinates (Sobolev supervision).  Hard-query
  influence is capped without clipping its target, but only for unbounded raw
  gradient/Sobolev terms.  Never apply that cap to trust-bounded deployment
  targets.  Magnitude, cosine-direction, and vector deployment losses must act
  on the result of the complete differentiable inference unroll, not directly
  on the magnitude-head scalar.  Draw that sub-batch as a 25/25/50 mixture of
  low, boundary, and saturated teacher magnitudes.  Base and each DAgger round
  must be sampled equally.
- DAgger stores boundary-safe temporal windows with a per-round horizon
  curriculum.  Multi-step training rolls the student plan forward through
  `apply -> shift -> apply` while treating the recorded observations as
  stop-gradient environment states.
- Final beta=0 DAgger rounds use their own lower learning rate and rollout
  early stopping.  Split them into short relabel cycles, restore the best
  validation checkpoint, and recollect exact DIAL targets from that current
  student before every cycle; never continue collection from the last
  optimizer iterate by default.
- Fit `StandardNormalizer` on raw observations and preserve per-objective cost
  mean/std in `CompositionalEnergyPolicy`.
- `CompositionalEnergyPolicy.apply` normalizes the composed energy gradient
  direction and predicts one learned total RMS displacement budget in
  `[0, trust_radius]` for the complete inference call.  The allowed radius
  grows progressively over the inner optimizer steps; never multiply the
  magnitude once per inner step.  `shift` warm-starts the next DIAL step.
- Rollout checkpoint selection uses the same reset/command seeds for every
  training anchor and ranks a checkpoint by its worst anchor score.  Unseen
  weights remain final holdouts and must not participate in selection.

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
