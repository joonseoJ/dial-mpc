"""DIAL-MPC-native Monte-Carlo score collection.

For a query action sequence ``U`` and noise scale ``sigma``, this module uses
the same estimator that motivates MPPI::

    V_k = clip(U + sigma * eps_k, -1, 1)
    w_k proportional to exp(-omega.T @ C(V_k) / temperature)
    score(U) = (sum_k w_k V_k - U) / sigma**2

Unlike the prototype this package was copied from, the implementation rolls
out :class:`brax.envs.base.State` values through a DIAL environment directly.
It therefore has no Hydrax, GPC, Warp, or Torch dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ScoreDataPoint(NamedTuple):
    """One MPPI score-regression sample in normalized action space."""

    u_point: jax.Array
    sigma: float
    score_target: jax.Array
    obs: jax.Array
    mode_idx: int


class DIALTeacherPoint(NamedTuple):
    """One exact ``MBDPI.reverse_once`` bounded-update target."""

    u_point: jax.Array
    sigma: float
    update_target: jax.Array
    obs: jax.Array
    mode_idx: int
    target_variance: float
    ess: float


class ExactDIALTeacher:
    """Average repeated calls to DIAL-MPC's real ``reverse_once`` update.

    No MPPI equations are reimplemented here.  Reward normalization, query
    inclusion, spline interpolation, first-node locking, clipping, and Gibbs
    weighting therefore remain exactly those of :class:`MBDPI`.
    """

    def __init__(self, planner: object, repeats: int = 4) -> None:
        if repeats < 1:
            raise ValueError("teacher repeats must be at least 1")
        self.planner = planner
        self.repeats = int(repeats)

    @staticmethod
    def state_for_mode(state: object, mode_weights: jax.Array) -> object:
        info = dict(state.info)
        info["reward_weights"] = jnp.asarray(mode_weights, dtype=jnp.float32)
        return state.replace(info=info)

    def estimate_update(
        self,
        state: object,
        query: jax.Array,
        diffusion_factor: float | jax.Array,
        rng: jax.Array,
        mode_weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return mean update, target variance, ESS, and next RNG.

        ``diffusion_factor`` is the scalar trajectory factor used by DIAL.
        Its per-node noise is ``planner.sigma_control * diffusion_factor``.
        """

        state = self.state_for_mode(state, mode_weights)
        query = jnp.asarray(query, dtype=jnp.float32)
        noise_scale = self.planner.sigma_control * jnp.asarray(diffusion_factor)
        updates = []
        esses = []
        for _ in range(self.repeats):
            rng, call_rng = jax.random.split(rng)
            _, refined, info = self.planner.reverse_once(
                state, call_rng, query, noise_scale
            )
            updates.append(refined - query)
            esses.append(1.0 / jnp.sum(jnp.square(info["weights"])))
        stacked = jnp.stack(updates)
        mean_update = jnp.mean(stacked, axis=0)
        target_variance = jnp.mean(jnp.var(stacked, axis=0))
        ess = jnp.mean(jnp.stack(esses))
        return mean_update, target_variance, ess, rng

    def point(
        self,
        state: object,
        query: jax.Array,
        diffusion_factor: float,
        rng: jax.Array,
        mode_weights: jax.Array,
        mode_idx: int,
    ) -> tuple[DIALTeacherPoint, jax.Array, jax.Array]:
        update, variance, ess, rng = self.estimate_update(
            state, query, diffusion_factor, rng, mode_weights
        )
        point = DIALTeacherPoint(
            u_point=query,
            sigma=float(diffusion_factor),
            update_target=update,
            obs=state.obs,
            mode_idx=int(mode_idx),
            target_variance=float(variance),
            ess=float(ess),
        )
        return point, jnp.clip(query + update, -1.0, 1.0), rng


ObjectiveFn = Callable[[object, jax.Array], jax.Array]


def scalar_reward_cost(state: object, action: jax.Array) -> jax.Array:
    """Fallback single-objective cost for any DIAL environment."""

    del action
    return jnp.atleast_1d(-state.reward)


class DIALScoreCollector:
    """Collect compositional MPPI scores with a DIAL/Brax environment.

    Args:
        env: DIAL environment exposing ``step`` and ``action_size``.
        horizon: Number of environment actions in each planned sequence.
        num_samples: Monte-Carlo samples per score query.
        temperature: Gibbs/MPPI temperature.
        objective_fn: Returns an unweighted cost vector for a post-step state
            and action.  By default ``env.objective_cost`` is used when the
            environment exposes it, otherwise ``[-state.reward]`` is used.

    DIAL actions are already normalized to ``[-1, 1]``.  Consequently all
    queries, scores, demonstrations, and policy outputs share one coordinate
    system and no task-specific actuator scaling is needed.
    """

    def __init__(
        self,
        env: object,
        horizon: int,
        num_samples: int = 1024,
        temperature: float = 0.05,
        objective_fn: ObjectiveFn | None = None,
        action_transform: Callable[[jax.Array], jax.Array] | None = None,
        lock_first_action: bool = False,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.env = env
        self.horizon = int(horizon)
        self.action_size = int(env.action_size)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.objective_fn = objective_fn or getattr(
            env, "objective_cost", scalar_reward_cost
        )
        self.action_transform = action_transform or (lambda actions: actions)
        self.lock_first_action = bool(lock_first_action)

        step = env.step
        objective = self.objective_fn

        def rollout_one(state, actions):
            actions = self.action_transform(actions)
            def scan_step(carry, action):
                next_state = step(carry, action)
                cost = jnp.asarray(objective(next_state, action))
                return next_state, cost

            _, costs = jax.lax.scan(scan_step, state, actions)
            return jnp.sum(costs, axis=0)

        self._rollout_batch = jax.jit(
            jax.vmap(rollout_one, in_axes=(None, 0))
        )

        def estimate(state, query, sigma, rng, mode_weights):
            eps = jax.random.normal(
                rng,
                (self.num_samples, self.horizon, self.action_size),
            )
            samples = jnp.clip(query[None] + sigma * eps, -1.0, 1.0)
            if self.lock_first_action:
                # Match DIAL-MPC: the control already entering the action
                # buffer is fixed while future spline nodes are optimized.
                samples = samples.at[:, 0].set(query[0])
            objective_costs = self._rollout_batch(state, samples)
            weighted_costs = objective_costs @ mode_weights.T
            weights = jax.nn.softmax(
                -weighted_costs / self.temperature, axis=0
            )
            means = jnp.einsum("sm,shu->mhu", weights, samples)
            scores = (means - query[None]) / jnp.square(sigma)
            return scores, means, objective_costs

        self._estimate = jax.jit(estimate)

    def estimate(
        self,
        state: object,
        query: jax.Array,
        sigma: float | jax.Array,
        rng: jax.Array,
        mode_weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Estimate every mode's score at one query point.

        Returns ``(scores, refined_means, per_sample_objective_costs)`` with
        shapes ``(N,H,nu)``, ``(N,H,nu)``, and ``(S,k)`` respectively.
        """

        query = jnp.asarray(query, dtype=jnp.float32)
        mode_weights = jnp.asarray(mode_weights, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        expected = (self.horizon, self.action_size)
        if query.shape != expected:
            raise ValueError(f"query must have shape {expected}, got {query.shape}")
        if mode_weights.ndim != 2:
            raise ValueError("mode_weights must have shape (num_modes, objectives)")
        return self._estimate(state, query, sigma, rng, mode_weights)

    def collect(
        self,
        state: object,
        obs: jax.Array,
        sigma: float,
        rng: jax.Array,
        mode_weights: jax.Array,
        num_perturbations: int = 3,
        perturb_scale: float = 1.0,
        base_mean: jax.Array | None = None,
        drive_mode: int = 0,
    ) -> tuple[list[list[ScoreDataPoint]], jax.Array]:
        """Collect the base query and nearby score queries for every mode.

        The returned refined mean belongs to ``drive_mode`` and can warm-start
        the next (smaller) sigma level.  Scores for every mode are still
        collected from the shared rollouts.
        """

        if num_perturbations < 0:
            raise ValueError("num_perturbations cannot be negative")
        mode_weights = jnp.asarray(mode_weights, dtype=jnp.float32)
        num_modes = int(mode_weights.shape[0])
        if not 0 <= drive_mode < num_modes:
            raise ValueError(
                f"drive_mode must be in [0, {num_modes}), got {drive_mode}"
            )
        base = (
            jnp.zeros((self.horizon, self.action_size), dtype=jnp.float32)
            if base_mean is None
            else jnp.clip(jnp.asarray(base_mean), -1.0, 1.0)
        )
        obs = jnp.asarray(obs)

        rng, base_rng = jax.random.split(rng)
        base_scores, base_means, _ = self.estimate(
            state, base, sigma, base_rng, mode_weights
        )
        centre = base_means[drive_mode]
        data: list[list[ScoreDataPoint]] = [[] for _ in range(num_modes)]

        def append(query, scores):
            for mode_idx in range(num_modes):
                data[mode_idx].append(
                    ScoreDataPoint(
                        u_point=query,
                        sigma=float(sigma),
                        score_target=scores[mode_idx],
                        obs=obs,
                        mode_idx=mode_idx,
                    )
                )

        append(base, base_scores)
        for _ in range(num_perturbations):
            rng, perturb_rng, score_rng = jax.random.split(rng, 3)
            perturb = jax.random.normal(perturb_rng, centre.shape)
            query = jnp.clip(
                centre + perturb_scale * float(sigma) * perturb, -1.0, 1.0
            )
            scores, _, _ = self.estimate(
                state, query, sigma, score_rng, mode_weights
            )
            append(query, scores)

        return data, centre


def collect_scores_all_modes(
    state: object,
    obs: jax.Array,
    collector: DIALScoreCollector,
    sigma: float,
    rng: jax.Array,
    mode_weights: jax.Array,
    M: int = 3,
    perturb_scale: float = 1.0,
    base_mean: jax.Array | None = None,
    drive_mode: int = 0,
):
    """Functional compatibility wrapper around :meth:`DIALScoreCollector.collect`."""

    return collector.collect(
        state,
        obs,
        sigma,
        rng,
        mode_weights,
        num_perturbations=M,
        perturb_scale=perturb_scale,
        base_mean=base_mean,
        drive_mode=drive_mode,
    )


def stack_score_data(
    data_per_mode: Sequence[Sequence[ScoreDataPoint]],
) -> list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Stack collector output into arrays accepted by score training."""

    result = []
    for mode_data in data_per_mode:
        if not mode_data:
            raise ValueError("every mode needs at least one score data point")
        result.append(
            (
                jnp.stack([point.u_point for point in mode_data]),
                jnp.asarray([point.sigma for point in mode_data]),
                jnp.stack([point.score_target for point in mode_data]),
                jnp.stack([point.obs for point in mode_data]),
            )
        )
    return result


def stack_teacher_data(
    data_per_mode: Sequence[Sequence[DIALTeacherPoint]],
) -> list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Stack exact-teacher points using the existing NPZ four-array schema.

    The third array contains bounded updates (not scores).  Run metadata marks
    this explicitly, so the trainer never multiplies these targets by sigma².
    """

    result = []
    for mode_data in data_per_mode:
        if not mode_data:
            raise ValueError("every mode needs at least one teacher point")
        result.append(
            (
                jnp.stack([point.u_point for point in mode_data]),
                jnp.asarray([point.sigma for point in mode_data]),
                jnp.stack([point.update_target for point in mode_data]),
                jnp.stack([point.obs for point in mode_data]),
            )
        )
    return result


def build_sigma_schedule(
    sigma_min: float = 1e-2,
    sigma_max: float = 1.0,
    n_levels: int = 20,
) -> jax.Array:
    """Return a log-spaced coarse-to-fine noise schedule."""

    if not 0 < sigma_min <= sigma_max:
        raise ValueError("expected 0 < sigma_min <= sigma_max")
    if n_levels < 1:
        raise ValueError("n_levels must be at least 1")
    return jnp.exp(
        jnp.linspace(jnp.log(sigma_max), jnp.log(sigma_min), n_levels)
    )


def save_score_dataset(
    path: str,
    arrays_per_mode: Sequence[tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
) -> None:
    """Save a stacked score dataset without pickling JAX runtime objects."""

    payload: dict[str, np.ndarray] = {"num_modes": np.asarray(len(arrays_per_mode))}
    for i, (u, sigma, score, obs) in enumerate(arrays_per_mode):
        payload[f"u_{i}"] = np.asarray(u)
        payload[f"sigma_{i}"] = np.asarray(sigma)
        payload[f"score_{i}"] = np.asarray(score)
        payload[f"obs_{i}"] = np.asarray(obs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # Passing an open file prevents numpy from appending another ".npz".
    # os.replace-style Path.replace keeps the previous snapshot intact if the
    # process is interrupted while writing the new one.
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(destination)


def load_score_dataset(
    path: str,
) -> list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Load a dataset written by :func:`save_score_dataset`."""

    with np.load(path) as payload:
        count = int(payload["num_modes"])
        return [
            tuple(
                jnp.asarray(payload[f"{name}_{i}"])
                for name in ("u", "sigma", "score", "obs")
            )
            for i in range(count)
        ]
