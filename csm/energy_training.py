"""Training objective for objective-compositional trajectory energies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm.auto import tqdm

from csm.energy_data import EnergyDataset
from csm.energy_policy import normalize_preference_weights


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
    preference_weight_sum: float | None = None,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, jax.Array], None] | None = None,
) -> jax.Array:
    """Fit raw objective energies and their DIAL descent directions.

    Scalar energy regression identifies the landscape.  Sparse direction and
    log-RMS calibration losses constrain ``-grad_U(omega @ E)`` with the exact
    bounded DIAL update.  When multiple datasets are supplied, every base or
    DAgger round is selected equally often.
    """
    groups = [dataset] if isinstance(dataset, EnergyDataset) else list(dataset)
    if not groups:
        raise ValueError("at least one energy dataset is required")

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
    ):
        observations = normalizer(observations, use_running_average=True)
        target = (costs - cost_mean) / cost_std
        prediction = model(plans, observations)
        residual = prediction - target
        value_loss = jnp.mean(optax_huber(residual))

        gobs = normalizer(gobs, use_running_average=True)

        def one_objective_jacobian(plan, obs):
            # (k, H, action_size): one action-gradient for every normalized
            # scalar energy head.  Parameter differentiation through this
            # jacrev below supplies the mixed derivative required by Sobolev
            # training.
            return jax.jacrev(lambda candidate: model(candidate, obs))(plan)

        normalized_jacobians = jax.vmap(one_objective_jacobian)(gplans, gobs)
        raw_jacobians = normalized_jacobians * cost_std[None, :, None, None]
        if preference_weight_sum is not None:
            omegas = normalize_preference_weights(omegas, preference_weight_sum)
        raw_directions = -jnp.einsum("bk,bkha->bha", omegas, raw_jacobians)
        raw_directions = raw_directions.at[:, 0].set(0.0)
        updates = updates.at[:, 0].set(0.0)
        dot = jnp.sum(raw_directions * updates, axis=(-2, -1))
        denom = (
            jnp.linalg.norm(
                raw_directions.reshape(raw_directions.shape[0], -1), axis=-1
            )
            * jnp.linalg.norm(updates.reshape(updates.shape[0], -1), axis=-1)
            + 1e-6
        )
        direction_loss = jnp.mean(1.0 - dot / denom)
        # Cost-gradient units and bounded-control-update units are different.
        # Keep the Sobolev-constrained energy untouched and learn only this
        # controller conversion scale from the update magnitudes.
        update_scale = jnp.exp(model.log_update_scale.value)
        calibrated_directions = update_scale * jax.lax.stop_gradient(
            raw_directions
        )
        direction_rms = jnp.sqrt(
            jnp.mean(jnp.square(calibrated_directions), axis=(-2, -1)) + 1e-8
        )
        target_rms = jnp.sqrt(
            jnp.mean(jnp.square(updates), axis=(-2, -1)) + 1e-8
        )
        calibration_loss = jnp.mean(
            optax_huber(jnp.log(direction_rms) - jnp.log(target_rms))
        )

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
        sobolev_loss = jnp.sum(
            gradient_valid * sobolev_per_sample
        ) / (jnp.sum(gradient_valid) + 1e-6)
        return (
            value_loss
            + guidance_weight * direction_loss
            + calibration_weight * calibration_loss
            + sobolev_weight * sobolev_loss
        )

    @nnx.jit
    def train_step(model, optimizer, *batch):
        loss, gradients = nnx.value_and_grad(loss_fn)(model, *batch)
        optimizer.update(gradients)
        return loss

    last_loss = jnp.asarray(0.0)
    bar = tqdm(range(num_iters), desc="Compositional energy training", unit="step")
    for step in bar:
        rng, value_rng, guide_rng = jax.random.split(rng, 3)
        current = groups[step % len(groups)]
        n_values = int(current.plans.shape[0])
        n_guidance = int(current.guidance_plans.shape[0])
        value_idx = jax.random.randint(value_rng, (batch_size,), 0, n_values)
        guide_batch = min(batch_size, n_guidance)
        guide_idx = jax.random.randint(guide_rng, (guide_batch,), 0, n_guidance)
        last_loss = train_step(
            model,
            optimizer,
            current.plans[value_idx],
            current.observations[value_idx],
            current.costs[value_idx],
            current.guidance_plans[guide_idx],
            current.guidance_observations[guide_idx],
            current.guidance_omegas[guide_idx],
            current.guidance_updates[guide_idx],
            current.guidance_objective_gradients[guide_idx],
            current.guidance_gradient_valid[guide_idx],
        )
        if step % 100 == 0:
            bar.set_postfix(loss=f"{float(last_loss):.6f}")
        completed = step + 1
        if checkpoint_every and completed % checkpoint_every == 0 and checkpoint_callback:
            checkpoint_callback(completed, last_loss)
    return last_loss


def optax_huber(error: jax.Array, delta: float = 1.0) -> jax.Array:
    """Elementwise Huber loss without introducing another public dependency."""
    absolute = jnp.abs(error)
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * quadratic**2 + delta * (absolute - quadratic)
