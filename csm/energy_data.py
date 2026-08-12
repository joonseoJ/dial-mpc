"""Exact rollout-cost data for compositional trajectory-energy learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class EnergyDataset:
    """Value samples plus sparse DIAL direction anchors."""

    plans: jax.Array
    observations: jax.Array
    costs: jax.Array
    sigmas: jax.Array
    guidance_plans: jax.Array
    guidance_observations: jax.Array
    guidance_omegas: jax.Array
    guidance_updates: jax.Array
    guidance_objective_gradients: jax.Array
    guidance_gradient_valid: jax.Array
    guidance_recovery_difficulty: jax.Array


@dataclass
class ClosedLoopDataset:
    """Short student-plan rollouts collected on DAgger state sequences.

    Environment observations are treated as fixed, stop-gradient states during
    training.  Plans remain autoregressive: the optimized plan is shifted and
    becomes the next step's warm start, exactly as it does during deployment.
    """

    initial_plans: jax.Array
    observations: jax.Array
    omegas: jax.Array
    teacher_plans: jax.Array
    valid: jax.Array


def concatenate_energy_datasets(datasets: Sequence[EnergyDataset]) -> EnergyDataset:
    if not datasets:
        raise ValueError("at least one energy dataset is required")
    fields = EnergyDataset.__dataclass_fields__
    return EnergyDataset(
        **{
            name: jnp.concatenate([getattr(dataset, name) for dataset in datasets])
            for name in fields
        }
    )


def concatenate_closed_loop_datasets(
    datasets: Sequence[ClosedLoopDataset],
) -> ClosedLoopDataset:
    if not datasets:
        raise ValueError("at least one closed-loop dataset is required")
    return ClosedLoopDataset(
        **{
            name: jnp.concatenate([getattr(dataset, name) for dataset in datasets])
            for name in ClosedLoopDataset.__dataclass_fields__
        }
    )


def make_closed_loop_windows(
    warm_start_plans: jax.Array,
    observations: jax.Array,
    omegas: jax.Array,
    teacher_plans: jax.Array,
    sequence_length: int,
) -> ClosedLoopDataset:
    """Create overlapping, boundary-safe temporal windows for one rollout."""
    if sequence_length < 1:
        raise ValueError("closed-loop sequence length must be positive")
    count = int(observations.shape[0])
    if count < 1:
        raise ValueError("a closed-loop rollout must contain at least one step")
    windows = []
    for start in range(count):
        stop = min(start + sequence_length, count)
        length = stop - start
        pad = sequence_length - length

        def padded(values):
            chunk = values[start:stop]
            if not pad:
                return chunk
            padding = jnp.repeat(chunk[-1:], pad, axis=0)
            return jnp.concatenate([chunk, padding], axis=0)

        windows.append(
            (
                warm_start_plans[start],
                padded(observations),
                padded(omegas),
                padded(teacher_plans),
                jnp.concatenate(
                    [jnp.ones(length), jnp.zeros(pad)], axis=0
                ),
            )
        )
    fields = zip(*windows)
    return ClosedLoopDataset(*(jnp.stack(field) for field in fields))


def save_energy_dataset(path: Path | str, dataset: EnergyDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            name: np.asarray(getattr(dataset, name))
            for name in EnergyDataset.__dataclass_fields__
        },
    )


def load_energy_dataset(path: Path | str) -> EnergyDataset:
    with np.load(path) as archive:
        arrays = {
            name: jnp.asarray(archive[name])
            for name in EnergyDataset.__dataclass_fields__
            if name in archive
        }
        # Old energy datasets predate exact objective-gradient collection.
        # Keep them usable for value/direction training while masking their
        # synthetic zero gradients out of the Sobolev objective.
        if "guidance_objective_gradients" not in arrays:
            plans = arrays["guidance_plans"]
            objectives = arrays["guidance_omegas"].shape[-1]
            arrays["guidance_objective_gradients"] = jnp.zeros(
                (plans.shape[0], objectives, plans.shape[1], plans.shape[2]),
                dtype=plans.dtype,
            )
            arrays["guidance_gradient_valid"] = jnp.zeros(
                (plans.shape[0],), dtype=jnp.float32
            )
        if "guidance_recovery_difficulty" not in arrays:
            arrays["guidance_recovery_difficulty"] = jnp.zeros(
                (arrays["guidance_plans"].shape[0],), dtype=jnp.float32
            )
        return EnergyDataset(**arrays)


def save_closed_loop_dataset(
    path: Path | str, dataset: ClosedLoopDataset
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            name: np.asarray(getattr(dataset, name))
            for name in ClosedLoopDataset.__dataclass_fields__
        },
    )


def load_closed_loop_dataset(path: Path | str) -> ClosedLoopDataset:
    with np.load(path) as archive:
        return ClosedLoopDataset(
            **{
                name: jnp.asarray(archive[name])
                for name in ClosedLoopDataset.__dataclass_fields__
            }
        )


class ExactEnergyCollector:
    """Label DIAL candidate nodes with raw per-objective rollout costs.

    The real :meth:`MBDPI.reverse_once` supplies candidate nodes and the
    bounded teacher refinement.  Objective labels are recomputed without
    weights, so one candidate supervises every compositional energy head.
    """

    def __init__(
        self,
        planner: object,
        objective_fn: Callable,
        num_candidates: int = 64,
        teacher_repeats: int = 8,
    ) -> None:
        if num_candidates < 2:
            raise ValueError("num_candidates must be at least two")
        if teacher_repeats < 1:
            raise ValueError("teacher_repeats must be positive")
        self.planner = planner
        self.num_candidates = int(num_candidates)
        self.teacher_repeats = int(teacher_repeats)
        env_step = planner.env.step
        node2u = planner.node2u_vmap

        def rollout_cost(state, nodes):
            actions = node2u(nodes)

            def scan_step(carry, action):
                next_state = env_step(carry, action)
                return next_state, jnp.asarray(objective_fn(next_state, action))

            _, costs = jax.lax.scan(scan_step, state, actions)
            return jnp.sum(costs, axis=0)

        self._rollout_costs = jax.jit(jax.vmap(rollout_cost, in_axes=(None, 0)))
        # Output Jacobian shape: (num_objectives, num_nodes, action_size).
        # MJX's constraint solver contains a dynamic lax.while_loop, for which
        # JAX does not define reverse-mode differentiation.  Forward mode is
        # therefore required even though there are roughly sixty node-action
        # inputs and only three scalar outputs.
        self._objective_gradient = jax.jit(
            jax.jacfwd(rollout_cost, argnums=1)
        )

    @staticmethod
    def _set_weights(state, omega):
        return state.replace(
            info={**state.info, "reward_weights": jnp.asarray(omega)}
        )

    def collect_query(self, state, query, factor, rng, omega) -> tuple[EnergyDataset, jax.Array, jax.Array]:
        """Collect value labels and an averaged exact-DIAL direction anchor."""
        return self.collect_query_with_difficulty(
            state, query, factor, rng, omega, recovery_difficulty=0.0
        )

    def collect_query_with_difficulty(
        self,
        state,
        query,
        factor,
        rng,
        omega,
        recovery_difficulty,
    ) -> tuple[EnergyDataset, jax.Array, jax.Array]:
        """Collect one query and retain its state-level recovery priority."""
        state = self._set_weights(state, omega)
        noise_scale = self.planner.sigma_control * jnp.asarray(factor)
        refined = []
        first_info = None
        for _ in range(self.teacher_repeats):
            rng, call_rng = jax.random.split(rng)
            _, next_query, info = self.planner.reverse_once(
                state, call_rng, query, noise_scale
            )
            refined.append(next_query)
            if first_info is None:
                first_info = info
        mean_refined = jnp.mean(jnp.stack(refined), axis=0)

        # Mix high-Gibbs-weight candidates with uniform candidates.  This
        # teaches both the useful basin and enough surrounding landscape for
        # gradients to remain meaningful away from the optimum.
        assert first_info is not None
        candidates = first_info["Y0s"]
        weights = first_info["weights"]
        elite_count = self.num_candidates // 2
        random_count = self.num_candidates - elite_count
        elite_idx = jax.lax.top_k(weights, elite_count)[1]
        rng, sample_rng = jax.random.split(rng)
        random_idx = jax.random.randint(
            sample_rng, (random_count,), 0, candidates.shape[0]
        )
        selected = candidates[jnp.concatenate([elite_idx, random_idx])]
        costs = self._rollout_costs(state, selected)
        objective_gradients = self._objective_gradient(state, query)
        observations = jnp.broadcast_to(
            state.obs, (self.num_candidates, state.obs.shape[-1])
        )
        dataset = EnergyDataset(
            plans=selected,
            observations=observations,
            costs=costs,
            sigmas=jnp.full((self.num_candidates,), factor),
            guidance_plans=query[None],
            guidance_observations=state.obs[None],
            guidance_omegas=jnp.asarray(omega)[None],
            guidance_updates=(mean_refined - query)[None],
            guidance_objective_gradients=objective_gradients[None],
            guidance_gradient_valid=jnp.ones((1,), dtype=jnp.float32),
            guidance_recovery_difficulty=jnp.asarray(
                [recovery_difficulty], dtype=jnp.float32
            ),
        )
        return dataset, mean_refined, rng
