"""Training objective for objective-compositional trajectory energies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm.auto import tqdm

from csm.energy_data import ClosedLoopDataset, EnergyDataset
from csm.energy_policy import normalize_preference_weights


def _project_rms(update: jax.Array, radius: float | None) -> jax.Array:
    """Project plan updates into the RMS trust region used at deployment."""
    if radius is None:
        return update
    rms = jnp.sqrt(
        jnp.mean(jnp.square(update), axis=(-2, -1), keepdims=True) + 1e-8
    )
    return update * jnp.minimum(1.0, jnp.asarray(radius) / rms)


def _sample_influence(
    exact_direction: jax.Array, cap: float | None
) -> jax.Array:
    """Bound a hard query's influence without modifying its gradient target."""
    if cap is None:
        return jnp.ones(exact_direction.shape[0])
    rms = jnp.sqrt(
        jnp.mean(jnp.square(exact_direction), axis=(-2, -1)) + 1e-8
    )
    return jnp.minimum(1.0, jnp.asarray(cap) / rms)


def _deployment_strata(
    updates: jax.Array, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split guidance queries into low, boundary, and saturated targets."""
    values = np.asarray(updates).copy()
    values[:, 0] = 0.0
    rms = np.sqrt(np.mean(np.square(values), axis=(-2, -1)) + 1e-8)
    all_indices = np.arange(len(values), dtype=np.int32)
    masks = (
        rms < 0.8 * radius,
        (rms >= 0.8 * radius) & (rms < radius),
        rms >= radius,
    )
    result = []
    for mask in masks:
        indices = np.flatnonzero(mask).astype(np.int32)
        result.append(indices if len(indices) else all_indices)
    return tuple(result)


def _recovery_strata(
    difficulty: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank queries into an exact 25/25/50 easy/boundary/hard split."""
    values = np.asarray(difficulty).reshape(-1)
    indices = np.argsort(values, kind="stable").astype(np.int32)
    count = len(indices)
    if count == 0:
        raise ValueError("recovery difficulty cannot be empty")
    quarter = count // 4
    cuts = (quarter, 2 * quarter)
    groups = (
        indices[: cuts[0]],
        indices[cuts[0] : cuts[1]],
        indices[cuts[1] :],
    )
    all_indices = np.arange(count, dtype=np.int32)
    return tuple(group if len(group) else all_indices for group in groups)


def _joint_deployment_strata(
    updates: jax.Array,
    difficulty: jax.Array,
    radius: float,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Cross recovery rank with teacher-update magnitude strata."""
    magnitude = _deployment_strata(updates, radius)
    recovery = _recovery_strata(difficulty)
    all_indices = np.arange(len(difficulty), dtype=np.int32)
    rows = []
    for recovery_pool in recovery:
        cells = []
        for magnitude_pool in magnitude:
            intersection = np.intersect1d(
                recovery_pool, magnitude_pool, assume_unique=False
            ).astype(np.int32)
            if len(intersection):
                cells.append(intersection)
            elif len(recovery_pool):
                cells.append(recovery_pool)
            elif len(magnitude_pool):
                cells.append(magnitude_pool)
            else:
                cells.append(all_indices)
        rows.append(tuple(cells))
    return tuple(rows)


def _stratum_counts(batch_size: int) -> tuple[int, int, int]:
    quarter = batch_size // 4
    return quarter, quarter, batch_size - 2 * quarter


def _joint_stratum_counts(batch_size: int) -> np.ndarray:
    """Integer 3x3 allocation with 25/25/50 row and column marginals."""
    marginals = np.asarray(_stratum_counts(batch_size), dtype=np.int32)
    expected = np.outer(marginals, marginals) / max(batch_size, 1)
    counts = np.floor(expected).astype(np.int32)
    row_remaining = marginals - counts.sum(axis=1)
    column_remaining = marginals - counts.sum(axis=0)
    while int(row_remaining.sum()):
        candidates = [
            (expected[row, column] - counts[row, column], row, column)
            for row in range(3)
            for column in range(3)
            if row_remaining[row] and column_remaining[column]
        ]
        _, row, column = max(candidates)
        counts[row, column] += 1
        row_remaining[row] -= 1
        column_remaining[column] -= 1
    return counts


def _sample_stratified_deployment_indices(
    rng: jax.Array,
    strata: tuple[tuple[np.ndarray, ...], ...],
    batch_size: int,
) -> jax.Array:
    """Sample joint 25/25/50 recovery and magnitude marginals."""
    counts = _joint_stratum_counts(batch_size)
    keys = iter(jax.random.split(rng, 9))
    chunks = []
    for row in range(3):
        for column in range(3):
            key = next(keys)
            count = int(counts[row, column])
            if count:
                pool = jnp.asarray(strata[row][column])
                selected = jax.random.randint(key, (count,), 0, len(pool))
                chunks.append(pool[selected])
    return jnp.concatenate(chunks)


def fit_compositional_energy(
    dataset: EnergyDataset | Sequence[EnergyDataset],
    model: nnx.Module,
    normalizer: nnx.Module,
    optimizer: nnx.Optimizer,
    cost_mean: jax.Array,
    cost_std: jax.Array,
    batch_size: int,
    num_iters: int,
    rng: jax.Array,
    guidance_weight: float = 0.2,
    calibration_weight: float = 0.1,
    sobolev_weight: float = 0.1,
    deployment_weight: float = 0.2,
    deployment_direction_weight: float = 0.3,
    conditional_magnitude_weight: float = 0.1,
    conditional_magnitude_cosine: float = 0.7,
    conditional_magnitude_temperature: float = 0.1,
    deployment_batch_size: int = 8,
    sobolev_influence_cap: float | None = 2.0,
    trust_radius: float | None = 0.05,
    closed_loop_dataset: ClosedLoopDataset | Sequence[ClosedLoopDataset] | None = None,
    closed_loop_weight: float = 0.05,
    closed_loop_batch_size: int = 8,
    closed_loop_every: int = 4,
    closed_loop_discount: float = 0.9,
    energy_steps: int = 8,
    energy_step_size: float = 1.0,
    energy_momentum: float = 0.5,
    shift_matrix: jax.Array | None = None,
    preference_weight_sum: float | None = None,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, jax.Array], bool | None] | None = None,
) -> tuple[jax.Array, int]:
    """Fit raw objective energies and their DIAL descent directions.

    Scalar energy regression identifies the landscape.  Sparse direction and
    final unrolled-update losses constrain ``-grad_U(omega @ E)`` with the
    exact bounded DIAL update.  When multiple datasets are supplied, every
    base or DAgger round is selected equally often.
    """
    groups = [dataset] if isinstance(dataset, EnergyDataset) else list(dataset)
    if not groups:
        raise ValueError("at least one energy dataset is required")
    if getattr(model, "magnitude_head", None) is not None and trust_radius is None:
        raise ValueError("bounded magnitude models require a positive trust radius")
    if deployment_batch_size < 1:
        raise ValueError("deployment batch size must be positive")
    if conditional_magnitude_temperature <= 0.0:
        raise ValueError("conditional magnitude temperature must be positive")
    if closed_loop_every < 1:
        raise ValueError("closed-loop frequency must be positive")
    if closed_loop_batch_size < 1:
        raise ValueError("closed-loop batch size must be positive")
    if closed_loop_dataset is None:
        closed_loop_groups = []
    elif isinstance(closed_loop_dataset, ClosedLoopDataset):
        closed_loop_groups = [closed_loop_dataset]
    else:
        closed_loop_groups = list(closed_loop_dataset)
    strata_radius = float(trust_radius) if trust_radius is not None else np.inf
    deployment_strata = [
        _joint_deployment_strata(
            group.guidance_updates,
            group.guidance_recovery_difficulty,
            strata_radius,
        )
        for group in groups
    ]

    def normalized_objective_jacobian(model, plan, obs):
        return jax.jacrev(lambda candidate: model(candidate, obs))(plan)

    def optimize_plan(model, plan, raw_observation, omega):
        """Differentiable copy of deployment inference in normalized actions."""
        observation = normalizer(raw_observation, use_running_average=True)
        if preference_weight_sum is not None:
            omega = normalize_preference_weights(omega, preference_weight_sum)
        anchor = plan

        def energy(candidate):
            normalized = model(candidate, observation)
            raw = normalized * cost_std + cost_mean
            return jnp.dot(omega, raw)

        use_total_budget = getattr(model, "magnitude_head", None) is not None
        total_budget = (
            model.predict_update_magnitude(
                anchor, observation, omega, float(trust_radius)
            )
            if use_total_budget
            else None
        )

        def optimize(carry, step_index):
            current, velocity = carry
            gradient = jax.grad(energy)(current).at[0].set(0.0)
            gradient_rms = jnp.sqrt(jnp.mean(jnp.square(gradient)) + 1e-8)
            if use_total_budget:
                direction = -gradient / gradient_rms
            else:
                scale = jnp.exp(model.log_update_scale.value)
                direction = -scale * gradient
            velocity = (
                energy_momentum * velocity
                + (1.0 - energy_momentum) * direction
            )
            proposed = current + energy_step_size * velocity
            radius = (
                total_budget
                * (step_index.astype(total_budget.dtype) + 1.0)
                / float(energy_steps)
                if use_total_budget
                else trust_radius
            )
            displacement = _project_rms(proposed - anchor, radius)
            current = jnp.clip(anchor + displacement, -1.0, 1.0)
            return (current, velocity), None

        (result, _), _ = jax.lax.scan(
            optimize,
            (plan, jnp.zeros_like(plan)),
            jnp.arange(energy_steps),
        )
        return result

    def closed_loop_loss_fn(
        model,
        initial_plans,
        sequence_observations,
        sequence_omegas,
        sequence_teacher_plans,
        sequence_valid,
    ):
        discounts = jnp.power(
            jnp.asarray(closed_loop_discount),
            jnp.arange(sequence_observations.shape[1]),
        )

        def rollout(initial, observations, omegas, targets, valid):
            def one_step(plan, inputs):
                observation, omega, target, mask, discount = inputs
                predicted = optimize_plan(model, plan, observation, omega)
                error = optax_huber(
                    (predicted - target) / max(float(trust_radius or 1.0), 1e-3)
                )
                loss = mask * discount * jnp.mean(error)
                if shift_matrix is None:
                    shifted = jnp.roll(predicted, -1, axis=0)
                    shifted = shifted.at[-1].set(jnp.zeros_like(shifted[-1]))
                else:
                    shifted = jnp.einsum("ij,ja->ia", shift_matrix, predicted)
                return shifted, loss

            _, losses = jax.lax.scan(
                one_step,
                initial,
                (observations, omegas, targets, valid, discounts),
            )
            return jnp.sum(losses) / (jnp.sum(valid * discounts) + 1e-6)

        return jnp.mean(
            jax.vmap(rollout)(
                initial_plans,
                sequence_observations,
                sequence_omegas,
                sequence_teacher_plans,
                sequence_valid,
            )
        )

    def loss_fn(
        model,
        plans,
        observations,
        costs,
        gplans,
        gobs,
        omegas,
        updates,
        objective_gradients,
        gradient_valid,
        deployment_plans,
        deployment_observations,
        deployment_omegas,
        deployment_updates,
    ):
        observations = normalizer(observations, use_running_average=True)
        target = (costs - cost_mean) / cost_std
        prediction = model(plans, observations)
        residual = prediction - target
        value_loss = jnp.mean(optax_huber(residual))

        gobs = normalizer(gobs, use_running_average=True)

        normalized_jacobians = jax.vmap(
            lambda plan, obs: normalized_objective_jacobian(model, plan, obs)
        )(gplans, gobs)
        raw_jacobians = normalized_jacobians * cost_std[None, :, None, None]
        if preference_weight_sum is not None:
            omegas = normalize_preference_weights(omegas, preference_weight_sum)
        raw_directions = -jnp.einsum("bk,bkha->bha", omegas, raw_jacobians)
        raw_directions = raw_directions.at[:, 0].set(0.0)
        updates = updates.at[:, 0].set(0.0)
        exact_raw_directions = -jnp.einsum(
            "bk,bkha->bha", omegas, objective_gradients
        ).at[:, 0].set(0.0)
        influence = _sample_influence(
            exact_raw_directions, sobolev_influence_cap
        )
        dot = jnp.sum(raw_directions * updates, axis=(-2, -1))
        denom = (
            jnp.linalg.norm(
                raw_directions.reshape(raw_directions.shape[0], -1), axis=-1
            )
            * jnp.linalg.norm(updates.reshape(updates.shape[0], -1), axis=-1)
            + 1e-6
        )
        direction_loss = jnp.mean(1.0 - dot / denom)
        # Supervise what deployment actually returns after all energy-gradient,
        # momentum, clipping, and progressive-budget steps.  The independent
        # sub-batch is stratified by teacher magnitude below.
        predicted_deployment_plans = jax.vmap(
            lambda plan, obs, omega: optimize_plan(model, plan, obs, omega)
        )(
            deployment_plans,
            deployment_observations,
            deployment_omegas,
        )
        predicted_deployment_update = (
            predicted_deployment_plans - deployment_plans
        )
        deployment_updates = deployment_updates.at[:, 0].set(0.0)
        target_deployment_update = _project_rms(deployment_updates, trust_radius)
        predicted_final_rms = jnp.sqrt(
            jnp.mean(
                jnp.square(predicted_deployment_update), axis=(-2, -1)
            )
            + 1e-8
        )
        target_final_rms = jnp.sqrt(
            jnp.mean(jnp.square(target_deployment_update), axis=(-2, -1))
            + 1e-8
        )
        calibration_loss = jnp.mean(
            optax_huber(
                (predicted_final_rms - target_final_rms)
                / max(float(trust_radius), 1e-3)
            )
        )
        final_dot = jnp.sum(
            predicted_deployment_update * target_deployment_update,
            axis=(-2, -1),
        )
        final_denom = (
            jnp.sqrt(
                jnp.sum(
                    jnp.square(predicted_deployment_update), axis=(-2, -1)
                )
                + 1e-8
            )
            * jnp.sqrt(
                jnp.sum(
                    jnp.square(target_deployment_update), axis=(-2, -1)
                )
                + 1e-8
            )
        )
        deployment_direction_loss = jnp.mean(1.0 - final_dot / final_denom)
        final_cosine = final_dot / final_denom
        direction_ready = jax.lax.stop_gradient(
            jax.nn.sigmoid(
                (final_cosine - conditional_magnitude_cosine)
                / conditional_magnitude_temperature
            )
        )
        saturated = jax.lax.stop_gradient(
            (target_final_rms >= 0.8 * float(trust_radius)).astype(
                target_final_rms.dtype
            )
        )
        conditional_weights = direction_ready * saturated
        magnitude_underprediction = jnp.square(
            jax.nn.relu(target_final_rms - predicted_final_rms)
            / max(float(trust_radius), 1e-3)
        )
        conditional_magnitude_loss = jnp.sum(
            conditional_weights * magnitude_underprediction
        ) / (jnp.sum(conditional_weights) + 1e-6)
        deployment_per_sample = jnp.mean(
            optax_huber(
                (predicted_deployment_update - target_deployment_update)
                / max(float(trust_radius or 1.0), 1e-3)
            ),
            axis=(-2, -1),
        )
        # This target is already bounded by the trust region.  Do not suppress
        # hard queries with the raw-gradient Sobolev influence cap.
        deployment_loss = jnp.mean(deployment_per_sample)

        normalized_targets = objective_gradients / cost_std[None, :, None, None]
        # The best-performing formulation used one direct pointwise Sobolev
        # regression term.  Cost normalization supplies the objective units;
        # no separate gradient rescaling, clipping, or direction/magnitude
        # decomposition is applied.
        sobolev_per_sample = jnp.mean(
            optax_huber(normalized_jacobians - normalized_targets),
            axis=(-3, -2, -1),
        )
        gradient_valid = gradient_valid.astype(sobolev_per_sample.dtype)
        sobolev_weights = gradient_valid * influence
        sobolev_loss = jnp.sum(sobolev_weights * sobolev_per_sample) / (
            jnp.sum(sobolev_weights) + 1e-6
        )
        return (
            value_loss
            + guidance_weight * direction_loss
            + calibration_weight * calibration_loss
            + sobolev_weight * sobolev_loss
            + deployment_direction_weight * deployment_direction_loss
            + conditional_magnitude_weight * conditional_magnitude_loss
            + deployment_weight * deployment_loss
        )

    @nnx.jit
    def train_step(model, optimizer, *batch):
        loss, gradients = nnx.value_and_grad(loss_fn)(model, *batch)
        optimizer.update(gradients)
        return loss

    @nnx.jit
    def train_step_closed_loop(model, optimizer, batch, sequence_batch):
        def combined_loss(model, *batch):
            return loss_fn(model, *batch) + closed_loop_weight * closed_loop_loss_fn(
                model, *sequence_batch
            )

        loss, gradients = nnx.value_and_grad(combined_loss)(model, *batch)
        optimizer.update(gradients)
        return loss

    last_loss = jnp.asarray(0.0)
    bar = tqdm(range(num_iters), desc="Compositional energy training", unit="step")
    completed_steps = 0
    for step in bar:
        rng, value_rng, guide_rng, deployment_rng = jax.random.split(rng, 4)
        group_index = step % len(groups)
        current = groups[group_index]
        n_values = int(current.plans.shape[0])
        n_guidance = int(current.guidance_plans.shape[0])
        value_idx = jax.random.randint(value_rng, (batch_size,), 0, n_values)
        guide_batch = min(batch_size, n_guidance)
        guide_idx = jax.random.randint(guide_rng, (guide_batch,), 0, n_guidance)
        deployment_idx = _sample_stratified_deployment_indices(
            deployment_rng,
            deployment_strata[group_index],
            min(deployment_batch_size, n_guidance),
        )
        batch = (
            current.plans[value_idx],
            current.observations[value_idx],
            current.costs[value_idx],
            current.guidance_plans[guide_idx],
            current.guidance_observations[guide_idx],
            current.guidance_omegas[guide_idx],
            current.guidance_updates[guide_idx],
            current.guidance_objective_gradients[guide_idx],
            current.guidance_gradient_valid[guide_idx],
            current.guidance_plans[deployment_idx],
            current.guidance_observations[deployment_idx],
            current.guidance_omegas[deployment_idx],
            current.guidance_updates[deployment_idx],
        )
        use_closed_loop = (
            bool(closed_loop_groups)
            and closed_loop_weight > 0.0
            and step % closed_loop_every == 0
        )
        if use_closed_loop:
            sequence_rng, rng = jax.random.split(rng)
            sequence_group = closed_loop_groups[
                (step // closed_loop_every) % len(closed_loop_groups)
            ]
            sequence_count = int(sequence_group.initial_plans.shape[0])
            sequence_idx = jax.random.randint(
                sequence_rng,
                (min(closed_loop_batch_size, sequence_count),),
                0,
                sequence_count,
            )
            sequence_batch = tuple(
                getattr(sequence_group, name)[sequence_idx]
                for name in ClosedLoopDataset.__dataclass_fields__
            )
            last_loss = train_step_closed_loop(
                model, optimizer, batch, sequence_batch
            )
        else:
            last_loss = train_step(model, optimizer, *batch)
        if step % 100 == 0:
            bar.set_postfix(loss=f"{float(last_loss):.6f}")
        completed = step + 1
        completed_steps = completed
        if (
            checkpoint_every
            and completed % checkpoint_every == 0
            and checkpoint_callback
        ):
            if checkpoint_callback(completed, last_loss):
                break
    bar.close()
    return last_loss, completed_steps


def optax_huber(error: jax.Array, delta: float = 1.0) -> jax.Array:
    """Elementwise Huber loss without introducing another public dependency."""
    absolute = jnp.abs(error)
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * quadratic**2 + delta * (absolute - quadratic)
