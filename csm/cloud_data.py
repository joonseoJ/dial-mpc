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
from csm.dial_score import DialScoreData
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
        """Mean update over ``repeats`` clouds, plus its Monte-Carlo spread.

        ``temperature`` may be a scalar or one value per query, which is what a
        per-level profile needs: the temperature that puts the fine level at a
        usable effective sample size leaves the coarse level near an argmax, so
        the two levels are labelled at different sharpness.
        """

        omega = normalize_omega(omega)
        temperature = jnp.broadcast_to(
            jnp.asarray(temperature, dtype=jnp.float32), (data.size,)
        )
        used = data.repeats if repeats is None else min(int(repeats), data.repeats)

        def per_query(u, noise, keys, terms, temp):
            deltas = jax.vmap(
                lambda key, term: one(u, noise, key, term, omega, temp,
                                      std_normalize)
            )(keys[:used], terms[:used])
            return deltas.mean(axis=0), jnp.sqrt(jnp.mean(jnp.var(deltas, axis=0)))

        return jax.vmap(per_query)(
            data.u, data.noise, data.sample_rng, data.terms, temperature
        )

    def ess(data: DialCloudData, omega, temperature, std_normalize=False):
        """Effective sample size of the weighting each label came from."""

        omega = normalize_omega(omega)
        temperature = jnp.broadcast_to(
            jnp.asarray(temperature, dtype=jnp.float32), (data.size,)
        )

        def per_cloud(terms, temp):
            returns = terms @ omega
            scale = jnp.maximum(returns.std(), 1e-6) if std_normalize else 1.0
            weights = jax.nn.softmax((returns - returns[-1]) / scale / temp)
            return 1.0 / jnp.sum(weights**2)

        return jax.vmap(
            lambda terms, temp: jax.vmap(per_cloud, in_axes=(0, None))(terms, temp)
        )(data.terms, temperature)

    return relabel, ess


def chunked(fn, data: DialCloudData, chunk: int, *args):
    """Apply a per-query function in blocks, accumulating on the host.

    Relabelling regenerates every sampled node trajectory, which is ~700 KB per
    cloud.  Vmapping that across a whole collection would ask for terabytes of
    device memory even though the stored costs are only tens of gigabytes, so
    the sweep has to be blocked.
    """

    outputs = None
    for start in range(0, data.size, chunk):
        stop = min(start + chunk, data.size)
        piece = jax.tree.map(lambda x: jnp.asarray(x[start:stop]), data)
        block = fn(piece, *[
            jnp.asarray(a[start:stop]) if getattr(a, "ndim", 0) and
            getattr(a, "shape", (0,))[0] == data.size else a for a in args
        ])
        block = jax.tree.map(np.asarray, block)
        if outputs is None:
            outputs = [[] for _ in block]
        for store, value in zip(outputs, block):
            store.append(value)
    return tuple(np.concatenate(store) for store in outputs)


def query_temperatures(clouds: DialCloudData, temperature: float, level_scales):
    """The per-query temperature implied by a per-level profile."""

    scales = jnp.asarray(level_scales, dtype=jnp.float32)
    return scales[jnp.asarray(clouds.level, dtype=jnp.int32)] * temperature


def to_score_data(clouds: DialCloudData, delta: jax.Array) -> DialScoreData:
    """Package relabelled updates as a training set.

    Everything except the label already lives in the cloud file, so switching
    weight, temperature or repeat count produces a new training set without
    touching the environment.
    """

    return DialScoreData(
        u=clouds.u, factor=clouds.factor, level=clouds.level, delta=delta,
        obs=clouds.obs, qpos=clouds.qpos, qvel=clouds.qvel, step=clouds.step,
    )
