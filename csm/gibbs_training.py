"""Distribution-level training for anchor-factorized Gibbs scores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm.auto import tqdm

from csm.energy_training import optax_huber
from csm.gibbs_data import GibbsDataset


def fit_anchor_gibbs(
    dataset: GibbsDataset | Sequence[GibbsDataset],
    model: nnx.Module,
    normalizer: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_iters: int,
    rng: jax.Array,
    anchor_cost_scale: jax.Array,
    temperature: float,
    cost_weight: float = 1.0,
    kl_weight: float = 1.0,
    score_weight: float = 0.2,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, jax.Array], None] | None = None,
) -> jax.Array:
    """Fits raw anchor costs, exact DIAL distributions, and their updates."""

    groups = [dataset] if isinstance(dataset, GibbsDataset) else list(dataset)
    if not groups:
        raise ValueError("at least one Gibbs dataset is required")

    cost_scale = jnp.asarray(anchor_cost_scale)

    def loss_fn(model, observations, queries, candidates, target_costs, updates, factors):
        observations = normalizer(observations, use_running_average=True)
        predicted_normalized = model(queries, candidates, observations, factors)
        predicted_normalized = predicted_normalized - jnp.mean(
            predicted_normalized, axis=1, keepdims=True
        )
        target_costs = target_costs - jnp.mean(
            target_costs, axis=1, keepdims=True
        )
        target_normalized = target_costs / cost_scale

        cost_loss = jnp.mean(
            optax_huber(predicted_normalized - target_normalized)
        )
        predicted_costs = predicted_normalized * cost_scale
        target_std = jnp.maximum(jnp.std(target_costs, axis=1), 1e-6)
        predicted_std = jnp.maximum(jnp.std(predicted_costs, axis=1), 1e-6)
        target_logits = -target_costs / (target_std[:, None, :] * temperature)
        predicted_logits = -predicted_costs / (
            predicted_std[:, None, :] * temperature
        )
        target_log_prob = jax.nn.log_softmax(target_logits, axis=1)
        predicted_log_prob = jax.nn.log_softmax(predicted_logits, axis=1)
        target_prob = jnp.exp(target_log_prob)
        kl_loss = jnp.mean(
            jnp.sum(
                jax.lax.stop_gradient(target_prob)
                * (target_log_prob - predicted_log_prob),
                axis=1,
            )
        )

        predicted_means = jnp.einsum(
            "bma,bmhd->bahd", jnp.exp(predicted_log_prob), candidates
        )
        predicted_updates = predicted_means - queries[:, None]
        # Compare the MPPI displacement rather than an energy gradient.  The
        # factor supplies a stable dimensionless scale across diffusion steps.
        update_scale = jnp.maximum(factors[:, None, None, None], 0.05)
        score_loss = jnp.mean(
            optax_huber((predicted_updates - updates) / update_scale)
        )
        return (
            cost_weight * cost_loss
            + kl_weight * kl_loss
            + score_weight * score_loss
        )

    @nnx.jit
    def train_step(model, optimizer, *batch):
        loss, gradients = nnx.value_and_grad(loss_fn)(model, *batch)
        optimizer.update(gradients)
        return loss

    last_loss = jnp.asarray(0.0)
    bar = tqdm(range(num_iters), desc="Anchor Gibbs training", unit="step")
    for step in bar:
        rng, index_rng = jax.random.split(rng)
        current = groups[step % len(groups)]
        count = int(current.queries.shape[0])
        probabilities = current.priorities / jnp.sum(current.priorities)
        indices = jax.random.choice(
            index_rng,
            count,
            shape=(batch_size,),
            replace=True,
            p=probabilities,
        )
        last_loss = train_step(
            model,
            optimizer,
            current.observations[indices],
            current.queries[indices],
            current.candidates[indices],
            current.anchor_costs[indices],
            current.anchor_updates[indices],
            current.factors[indices],
        )
        if step % 100 == 0:
            bar.set_postfix(loss=f"{float(last_loss):.6f}")
        completed = step + 1
        if checkpoint_every and completed % checkpoint_every == 0 and checkpoint_callback:
            checkpoint_callback(completed, last_loss)
    return last_loss
