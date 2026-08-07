"""Standalone controller that optimizes a composed learned energy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp


def normalize_preference_weights(omega, target_sum):
    """Make positive scalar multiples of a preference vector equivalent."""
    omega = jnp.asarray(omega, dtype=jnp.float32)
    total = jnp.sum(omega, axis=-1, keepdims=True)
    return omega * (jnp.asarray(target_sum, dtype=omega.dtype) / (total + 1e-8))


@dataclass
class CompositionalEnergyPolicy:
    model: object
    normalizer: object
    cost_mean: jax.Array
    cost_std: jax.Array
    mode_weights: jax.Array
    u_min: jax.Array
    u_max: jax.Array
    shift_matrix: jax.Array | None = None
    num_steps: int = 8
    step_size: float = 0.08
    momentum: float = 0.5
    lock_first_action: bool = True
    trust_radius: float | None = 0.05
    preference_weight_sum: float | None = None

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
            raise TypeError("checkpoint is not a compositional energy policy")
        # Checkpoints created before trust-region inference retain their
        # original unit-RMS update semantics unless explicitly migrated.
        if "trust_radius" not in getattr(policy, "__dict__", {}):
            policy.trust_radius = None
        if "preference_weight_sum" not in getattr(policy, "__dict__", {}):
            policy.preference_weight_sum = None
        return policy

    def energies(self, plan, observation):
        observation = self.normalizer(observation, use_running_average=True)
        normalized = self.model(plan, observation)
        return normalized * self.cost_std + self.cost_mean

    def apply(self, prev, y, rng, warm_start_level=1.0, *, omega):
        """Minimize ``omega @ E(o,U)`` with bounded projected momentum steps."""
        if self.preference_weight_sum is not None:
            omega = normalize_preference_weights(
                omega, self.preference_weight_sum
            )
        act_mean = (self.u_max + self.u_min) / 2.0
        act_scale = (self.u_max - self.u_min) / 2.0
        plan = (prev - act_mean) / act_scale
        warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        plan = jnp.clip(
            plan + (1.0 - warm_start_level) * jax.random.normal(rng, plan.shape),
            -1.0,
            1.0,
        )
        observation = self.normalizer(y, use_running_average=True)

        def energy(candidate):
            normalized = self.model(candidate, observation)
            raw = normalized * self.cost_std + self.cost_mean
            return jnp.dot(omega, raw)

        trust_anchor = plan

        def optimize(carry, _):
            current, velocity = carry
            gradient = jax.grad(energy)(current)
            if self.lock_first_action:
                gradient = gradient.at[0].set(0.0)
            log_update_scale = getattr(self.model, "log_update_scale", None)
            update_scale = (
                jnp.exp(log_update_scale.value)
                if log_update_scale is not None
                else jnp.asarray(1.0)
            )
            direction = -update_scale * gradient
            trust_radius = getattr(self, "trust_radius", None)
            rms = jnp.sqrt(jnp.mean(jnp.square(direction)) + 1e-8)
            if trust_radius is None:
                # Backward-compatible behavior for old checkpoints.
                direction = direction / rms
            velocity = self.momentum * velocity + (1.0 - self.momentum) * direction
            proposed = current + self.step_size * velocity
            if trust_radius is not None:
                # Project the *total* displacement from this inference call's
                # warm-start plan into an RMS ball.  Repeated optimizer steps
                # therefore cannot accumulate beyond the trusted local region.
                displacement = proposed - trust_anchor
                displacement_rms = jnp.sqrt(
                    jnp.mean(jnp.square(displacement)) + 1e-8
                )
                displacement = displacement * jnp.minimum(
                    1.0, trust_radius / displacement_rms
                )
                proposed = trust_anchor + displacement
            current = jnp.clip(proposed, -1.0, 1.0)
            return (current, velocity), None

        (plan, _), _ = jax.lax.scan(
            optimize, (plan, jnp.zeros_like(plan)), xs=None, length=self.num_steps
        )
        return plan * act_scale + act_mean

    def shift(self, sequence):
        if self.shift_matrix is not None:
            return jnp.einsum("ij,ja->ia", self.shift_matrix, sequence)
        shifted = jnp.roll(sequence, -1, axis=0)
        return shifted.at[-1].set(jnp.zeros_like(shifted[-1]))
