"""Standalone policy induced by anchor-factorized Gibbs logits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp

from csm.energy_policy import normalize_preference_weights
from csm.gibbs_data import anchor_decoder, compose_anchor_logits


@dataclass
class AnchorGibbsPolicy:
    """Runs learned DIAL-TC-MPPI reweighting without physics rollouts."""

    model: object
    normalizer: object
    anchor_weights: jax.Array
    covariance_sqrt: jax.Array
    sigma_control: jax.Array
    diffuse_factors: jax.Array
    shift_matrix: jax.Array
    candidate_count: int = 64
    preference_weight_sum: float = 5.0

    def __post_init__(self):
        self.decoder = anchor_decoder(self.anchor_weights)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            cloudpickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("rb") as stream:
            policy = cloudpickle.load(stream)
        if not isinstance(policy, cls):
            raise TypeError("checkpoint is not an anchor Gibbs policy")
        return policy

    @property
    def mode_weights(self):
        """Compatibility with the common CSM evaluator."""

        return self.anchor_weights

    def _sample_candidates(self, query, factor, rng):
        random_count = self.candidate_count - 1
        epsilon = jax.random.normal(
            rng,
            (random_count, self.model.horizon, self.model.action_size),
        )
        correlated = jnp.einsum("ts,msa->mta", self.covariance_sqrt, epsilon)
        noise_scale = self.sigma_control * factor
        candidates = query[None] + correlated * noise_scale[None, :, None]
        # DIAL's stable TC mode locks the first action node.
        candidates = candidates.at[:, 0].set(query[0])
        candidates = jnp.clip(candidates, -1.0, 1.0)
        return jnp.concatenate([candidates, query[None]], axis=0)

    def apply(self, prev, y, rng, warm_start_level=1.0, *, omega):
        """Applies several learned TC-MPPI Gibbs updates."""

        omega = normalize_preference_weights(omega, self.preference_weight_sum)
        query = jnp.asarray(prev)
        warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        rng, warm_rng = jax.random.split(rng)
        query = jnp.clip(
            query
            + (1.0 - warm_start_level)
            * jax.random.normal(warm_rng, query.shape),
            -1.0,
            1.0,
        )
        observation = self.normalizer(y, use_running_average=True)

        def update(carry, factor):
            current, key = carry
            key, sample_key = jax.random.split(key)
            candidates = self._sample_candidates(current, factor, sample_key)
            anchor_logits = self.model(
                current, candidates, observation, factor
            )
            logits = compose_anchor_logits(anchor_logits, omega, self.decoder)
            weights = jax.nn.softmax(logits)
            refined = jnp.einsum("m,mha->ha", weights, candidates)
            return (refined, key), None

        (query, _), _ = jax.lax.scan(
            update, (query, rng), self.diffuse_factors
        )
        return query

    def shift(self, sequence):
        return jnp.einsum("ij,ja->ia", self.shift_matrix, sequence)
