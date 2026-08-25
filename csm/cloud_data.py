"""Per-query sample costs: a collection you can relabel instead of repeat.

DIAL's expensive step is the rollout, and a rollout does not depend on the
weight vector, the temperature, or how many repeats you average.  Only the
softmax does.  So the durable thing to store is not a label but the raw material
a label is made of: for every query, the per-row cost of every sampled action
sequence.

With that on disk, changing the basis, the temperature, the number of teacher
repeats, or whether DIAL's standard-deviation normalisation is applied is a
matmul and a softmax -- no physics at all.  The alternative, which this project
has already paid for once, is recollecting from scratch and then measuring the
data change instead of the variable you meant to test.

The sampled node trajectories themselves are *not* stored.  They are a
deterministic function of the query, the annealing factor and one rng key, so
keeping the key costs 8 bytes where keeping the nodes would cost ~700 KB.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from csm.dial_lean import make_sampler
from csm.omega import normalize_omega


class DialCloudData(NamedTuple):
    """One entry per (state, query, annealing level).

    Attributes:
        u: ``(Q, H, nu)`` query plans in normalized node coordinates.
        factor: ``(Q,)`` annealing factor each query was posed at.
        level: ``(Q,)`` annealing level index.
        obs: ``(Q, obs_size)`` observations, for training the student.
        qpos/qvel/step: physics state, so a query can also be re-rolled if the
            plant itself changes.
        sample_rng: ``(Q, R, 2)`` one key per repeat; regenerates the cloud.
        terms: ``(Q, R, N, k)`` horizon-mean cost rows for every sample.
        noise: ``(Q, H)`` per-node sampling sigma, the other half of what the
            sampler needs.
    """

    u: jax.Array
    factor: jax.Array
    level: jax.Array
    obs: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    step: jax.Array
    sample_rng: jax.Array
    terms: jax.Array
    noise: jax.Array

    @property
    def size(self) -> int:
        return int(self.u.shape[0])

    @property
    def repeats(self) -> int:
        return int(self.terms.shape[1])

    @property
    def num_rows(self) -> int:
        return int(self.terms.shape[-1])

    def take(self, index) -> "DialCloudData":
        return DialCloudData(*(jnp.asarray(field)[index] for field in self))


def save_clouds(path: str | Path, data: DialCloudData) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in zip(data._fields, data)}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez(stream, **payload)
    temporary.replace(destination)


def load_clouds(path: str | Path) -> DialCloudData:
    with np.load(path) as payload:
        return DialCloudData(
            **{name: jnp.asarray(payload[name]) for name in DialCloudData._fields}
        )


def concat_clouds(parts) -> DialCloudData:
    parts = list(parts)
    if not parts:
        raise ValueError("nothing to concatenate")
    return DialCloudData(
        *(jnp.concatenate([jnp.asarray(getattr(p, name)) for p in parts])
          for name in DialCloudData._fields)
    )


def make_relabeler(mbdpi, dial_config):
    """Turn stored clouds into DIAL updates for any weight and temperature.

    Regenerating the nodes from ``sample_rng`` reproduces the exact proposal the
    costs were measured on, so this is not an approximation of the collector --
    it is the same arithmetic with the rollout skipped.
    """

    sample = make_sampler(mbdpi, dial_config)

    def one(u, noise, key, terms, omega, temp, std_normalize):
        nodes = sample(key, u, noise)
        returns = terms @ omega
        reference = returns[-1]
        scale = jnp.maximum(returns.std(), 1e-6) if std_normalize else 1.0
        weights = jax.nn.softmax((returns - reference) / scale / temp)
        return jnp.einsum("n,nij->ij", weights, nodes) - u

    def relabel(data: DialCloudData, omega, temperature, std_normalize=False,
                repeats: int | None = None):
        """Mean update over ``repeats`` clouds, plus its Monte-Carlo spread."""

        omega = normalize_omega(omega)
        temperature = jnp.asarray(temperature, dtype=jnp.float32)
        used = data.repeats if repeats is None else min(int(repeats), data.repeats)

        def per_query(u, noise, keys, terms):
            deltas = jax.vmap(
                lambda key, term: one(u, noise, key, term, omega, temperature,
                                      std_normalize)
            )(keys[:used], terms[:used])
            return deltas.mean(axis=0), jnp.sqrt(jnp.mean(jnp.var(deltas, axis=0)))

        return jax.vmap(per_query)(
            data.u, data.noise, data.sample_rng, data.terms
        )

    def ess(data: DialCloudData, omega, temperature, std_normalize=False):
        """Effective sample size of the weighting each label came from."""

        omega = normalize_omega(omega)

        def per_cloud(terms):
            returns = terms @ omega
            scale = jnp.maximum(returns.std(), 1e-6) if std_normalize else 1.0
            weights = jax.nn.softmax((returns - returns[-1]) / scale / temperature)
            return 1.0 / jnp.sum(weights**2)

        return jax.vmap(jax.vmap(per_cloud))(data.terms)

    return relabel, ess
