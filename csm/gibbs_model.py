"""Neural anchor factors for compositional Gibbs scores."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from csm.architectures import MLP


class AnchorLogitMLP(nnx.Module):
    """Predicts one clean candidate logit for every anchor preference.

    Weight composition is intentionally absent from this network.  It occurs
    analytically on the output logits before Gibbs normalization.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        num_anchors: int,
        hidden_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ) -> None:
        self.action_size = int(action_size)
        self.observation_size = int(observation_size)
        self.horizon = int(horizon)
        self.num_anchors = int(num_anchors)
        plan_size = self.horizon * self.action_size
        # observation, nominal plan, candidate displacement, and log noise
        input_size = self.observation_size + 2 * plan_size + 1
        self.network = MLP(
            [input_size, *list(hidden_sizes), self.num_anchors], rngs=rngs
        )

    def __call__(
        self,
        query: jax.Array,
        candidates: jax.Array,
        observation: jax.Array,
        factor: jax.Array,
    ) -> jax.Array:
        """Returns logits with shape ``(..., candidates, anchors)``."""

        batch_shape = candidates.shape[:-3]
        count = candidates.shape[-3]
        query_flat = query.reshape(batch_shape + (-1,))
        displacement = (candidates - query[..., None, :, :]).reshape(
            batch_shape + (count, -1)
        )
        query_features = jnp.broadcast_to(
            query_flat[..., None, :], batch_shape + (count, query_flat.shape[-1])
        )
        observation_features = jnp.broadcast_to(
            observation[..., None, :],
            batch_shape + (count, observation.shape[-1]),
        )
        log_factor = jnp.log(jnp.maximum(factor, 1e-6))
        factor_features = jnp.broadcast_to(
            log_factor[..., None, None], batch_shape + (count, 1)
        )
        inputs = jnp.concatenate(
            [observation_features, query_features, displacement, factor_features],
            axis=-1,
        )
        return self.network(inputs)
