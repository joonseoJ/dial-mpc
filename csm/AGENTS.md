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

## Single-weight score matching baseline

`dial_score` is the deliberately minimal branch of this study: one MLP, one
preference weight, score regression for training, denoising for inference.  It
is not compositional and must stay that way — it exists to prove a single
network can reproduce DIAL before composition is reintroduced.

- Targets come from `MBDPI.reverse_once` itself, averaged over repeats.  Do not
  reimplement MPPI, the reward normalization, or the first-node lock there.
- DIAL's noise is a fixed per-node profile `sigma_control` times one scalar
  annealing factor.  Condition the network on that factor alone (log-uniform
  `t` in `[0, 1]`), and keep the collection and inference schedules identical.
- The network emits the bounded update and `score` divides by `sigma(t)**2`.
  Keep that preconditioning: an unscaled score head has to span DIAL's whole
  `1/sigma**2` range and will not fit.  Node 0 is hard-locked to a zero update.
- Train with `sigma**4`-weighted score regression, which equals mean-squared
  error of the applied control update.  Report scale-free `relative_rms` and
  `cosine` per annealing level, not just the loss.
- That objective under-serves the fine level, whose updates are intrinsically
  smaller and so contribute less squared error.  `level_loss_weights` trades
  update-MSE for per-level relative error; keep it opt-in via
  `--level-loss-balance` and keep reported metrics unweighted so weighted and
  unweighted runs stay comparable on one yardstick.
- `--episode-steps` bounds the *horizon* present in the data, not just the
  number of initial conditions.  Short episodes were adopted for start-state
  diversity and silently removed every long-horizon state: with 100-step
  episodes the fitted policy walked well for ~300 steps and then fell every
  ~6 s, while its 400-step metrics looked excellent.  Keep episodes at least as
  long as the deployment horizon.
- `--episode-steps` and `--collect-steps` must move together.  At a fixed
  `--collect-steps` the episode count is `collect-steps / episode-steps`, so
  buying horizon spends start-state diversity one for one.  Raising episodes
  from 100 to 500 steps at 800 collect-steps cut a round from 8 initial
  conditions to 2 and regressed everything it was meant to fix: first fall
  294 -> 230 steps, on-policy cosine 0.822 -> 0.720, fine-level cosine
  0.764 -> 0.580.  Longer episodes also collect more of a *failing* student, so
  raise the step budget instead of reallocating it.
- Evaluate over the horizon that matters.  Joint-limit termination was acting as
  an unintended safety net: resetting the student every ~167 steps kept it near
  the training distribution, so a 400-step eval reported cosine 0.961 / 1.15x
  where a 3000-step eval on the same policy reported 0.720 / 1.74x.
- To compare training variants, collect once with `--collect-only` and fit each
  variant from the same `--init-policy` on that one dataset.  Letting each
  variant collect its own DAgger data confounds the loss change with the data.
- Fit `StandardNormalizer` on round-0 observations only; refitting it between
  DAgger rounds silently shifts the network's input distribution.
- Round 0 collection is a real DIAL rollout, so its mean reward is the baseline.
  Always compare the deployed denoising rollout against it.
- Training validation is measured on teacher-visited states and therefore cannot
  detect distribution shift.  Verify with `dial_score_eval`, which reruns DIAL
  from the same reset seeds and re-queries the exact teacher at *student*-visited
  states.  A large gap between the two score numbers means more DAgger rounds
  and wider query coverage, not more gradient steps.

## Compositional score matching

`dial_score_compose` extends the single-weight branch to `k` preference weights.

- The exact Gibbs score is linear in `omega` because `log Z(omega)` has no `U`
  dependence.  DIAL supplies that score *smoothed* at `sigma` and estimated with
  a finite softmax, so linearity is approximate — measured at the deployment
  levels, a least-squares composition reproduces the exact update with cosine
  0.994 or better, and degrades as sigma shrinks because the softmax sharpens.
- `reverse_once` divides sample rewards by their own standard deviation, so the
  update is **invariant to the scale of omega**: `delta(omega) == delta(a*omega)`
  to six digits.  Least squares alone therefore mis-scales any target whose
  coefficients do not sum to one — `(1,1,1)` composed with cosine 0.9997 but 40%
  relative error.  `ComposedDialScorePolicy.coefficients` rescales the
  coefficients to sum to one, which fixes the magnitude (40% -> 2.4%) and cannot
  change the direction.  Never drop that normalisation.
- Each basis weight gets its own complete `DialScoreMLP`, never a head on a
  shared trunk, so every field stays independently testable with
  `dial-score-eval` and `dial-score-serve` and a shared encoder is never a
  candidate explanation for a composition failure.
- Label every query under all basis weights with one rng
  (`DialScoreTeacher.targets_basis`).  `reward_weights` enters only the scalar
  reward, so common random numbers give identical samples and physics across
  rows: one rollout pays for `k` labels and the rows are directly comparable.
- Carry the data across a regime change or the run measures the wrong variable.
  Moving from per-field collection to `rollout_basis` changed the label layout,
  the old datasets could not be reused, and each field went from ~120-155k
  samples to 48,840 while still training 20k steps from a warm start.  Every
  field regressed 4-6x in falls, so that run measured the data drop, not the
  omega randomization it was meant to test.  Relabel the stored
  `(state, query, factor)` triples under the missing weights instead of
  recollecting from scratch.
- Score the starting weights with `selection_fn` before the first gradient step.
  Checkpoint selection only picks the best point *within* a fit; without the
  initial score there is no way to tell that every checkpoint was already worse
  than what training started from.
- Uniform Dirichlet sampling over the basis rows starves the vertices: drawn
  weights averaged [0.37, 0.42, 0.22] on the simplex, and the composed policy
  ended up best at interior targets ((2,1,1) 1.53x, (2,2,1) 1.62x) and worst at
  the `(1,1,1)` basis row itself (2.65x).  Mix in the basis rows explicitly if
  they are deployment targets.
- Per-field accuracy is only meaningful against a stated state distribution.
  Rolling one field's policy and scoring all `k` fields at those states measures
  the others *off*-policy, and the two numbers diverge sharply: after DAgger the
  `(1,3,1)` field improved on its own states (fine-level cosine 0.650 -> 0.693)
  while getting worse on `(1,1,1)`'s (0.332 -> 0.295).  Drive with the field's
  own policy, or say plainly which distribution the number came from.
- Per-field DAgger and composition quality pull against each other.  DAgger pulls
  each field toward its own visited states, but composition evaluates all fields
  at one shared state that belongs to neither — the composed policy at a target
  omega visits a third distribution.  Improving fields individually therefore
  degraded composition (`(1,2,1)` went 1.79x -> 1.99x while its field's own
  validation improved 30%).  Collect for composition with the *composed* policy
  driving, over the omega range it will be deployed at.
- Basis weight vectors must be linearly independent and no more numerous than
  the reward rows, or `coefficients` cannot reduce.  With `B` the k-by-n matrix
  of basis weights, `pinv` returns `e_j` for a target equal to row `j` only when
  `B` has full row rank; short of that it returns the minimum-norm spread and
  the composed field mixes in networks that should have had coefficient zero.
  The earlier 2.00 reduction error was exactly this: four basis weights in a
  three-row reward space.  The Go2 push-recovery basis
  `{(3,1,1,1), (1,3,1,1), (1,1,3,1), (1,1,1,3)}` has rank 4 and condition
  number 3.0, and reduces to 1e-16.
- Temperature is not a free parameter alongside omega -- the exact Gibbs score
  depends only on `omega / T`, so per-weight temperatures are legitimate and the
  composition solve simply moves to `nu = omega / T` space.  Rank, and therefore
  reduction, is unaffected by rescaling each basis vector.
- Pick that temperature by measured concentration, not by inheriting DIAL's
  0.05.  At 0.05 the update is a hard argmax -- effective sample size 1.1 out of
  2049 -- so a score label is one lucky sample, not an average.  Because DIAL
  divides returns by their own standard deviation, the temperature that yields a
  given concentration depends on the weight vector: on the push-recovery basis a
  single 0.30 spans 2.9% to 16.2% ESS across the four fields (5.6x).
  Equalising at ~10% needs (0.25, 0.30, 0.37, 0.39) for boost0-3, which lands
  every field within 1.05x.  Push magnitude barely moves ESS (4.4% to 4.9%
  across 0.2-0.9 m/s), so one temperature per weight covers the whole
  disturbance range.
- Raising the temperature buys label quality and spends behavioural contrast.
  Between the same four basis weights, the mean MPPI update distance fell 0.364
  to 0.103 sigma going from 0.05 to 0.30, and the omega-spread of torso tilt
  fell to 0.25x.  Elite-set overlap and the diagonal check are unchanged, since
  those rank samples rather than weight them.  Screen discriminativeness at the
  temperature you will actually collect at.
- Choose basis rows that all keep DIAL walking; a weight the planner cannot
  locomote under yields a field fitted to garbage states.  Judge conditioning on
  the simplex (direction only), since scale is ignored.

## Training and inference

- Residual-MPPI locomotion priors use a command-conditioned SAC actor with a
  state-dependent tanh-Gaussian action distribution.  Preserve exact bounded
  action log likelihoods, including the tanh Jacobian; a deterministic mean
  squared action penalty is not an equivalent prior.  The prior reward must
  remain survival-first while retaining randomized command and gait terms so
  standing still is not an optimum.

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
dial-score --example unitree_go2_trot --smoke
dial-score-eval --run csm_runs/<latest dial-score run> --episodes 1 --steps 20
python -m unittest tests.test_dial_score
```

The Go2 plant has two structural knobs that matter more than the PD gains.
`timestep` sets the physics substep (`n_frames = dt / timestep`); at the stock
20 ms single step MuJoCo's soft joint-limit constraint does not converge, and
the residual shows up as `done` firing on a sub-degree limit violation while the
robot is walking normally.  `pd_substep` recomputes the PD torque from the
running state at each substep instead of holding the start-of-step torque, which
is what a real motor driver does.  Both default to the original behaviour; only
change them together with a full retrain, since either one changes the plant and
therefore the score the student was fit to.

`dial-score-serve` streams a policy live over MJPEG at `/stream.mjpg`, matching
the endpoints `dial-mpc-sim --web-viewer-port` already uses so one `ssh -L`
habit covers both.  Build the MuJoCo renderer inside the rollout thread: an EGL
context belongs to the thread that created it, and binding it from another one
fails with `EGL_BAD_ACCESS`.  Reuse one renderer across frames — `env.render`
constructs a new one per call and costs ~130 ms against ~14 ms reused.

They must create a loadable `policy.pkl`, dataset NPZ, and
`visualization.html` under the ignored `csm_runs/` directory.  Also run
`python -m compileall -q csm`, `uv pip check`, and `git diff --check` after
changes.
