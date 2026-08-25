"""A memory-lean restatement of DIAL's MPPI update.

`MBDPI.reverse_once` returns the full pipeline state of every sample at every
step so that downstream code can read `qbar`/`qdbar`/`xbar`.  For a Go2 cloud
that is roughly 200 floats per sample-step against the single float the update
itself needs, and the difference is what stops several DIAL instances from
being vmapped together on one device.

`lean_reverse_once` reproduces the same update exactly -- same proposal, same
locked first node, same clipping, same reference sample, same
standard-deviation normalisation and softmax -- while the rollout scan carries
only what the caller asks for.  It is valid for the untime-correlated sampler
(`time_correlated: false`), which is what every configuration here uses; the
TC-MPPI path keeps its own importance ratio and must go through `reverse_once`.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def make_sampler(mbdpi, dial_config):
    """Draws DIAL's proposal cloud around a plan, including the centre sample."""

    n_sample = dial_config.Nsample
    n_node = dial_config.Hnode + 1
    nu = mbdpi.nu

    def sample(rng, plan, noise_scale):
        eps = jax.random.normal(rng, (n_sample, n_node, nu))
        nodes = eps * noise_scale[None, :, None] + plan
        # DIAL holds the first node at the current plan and evaluates the
        # centre of the proposal as the reference the softmax is measured
        # against, so both have to survive into the cloud.
        nodes = nodes.at[:, 0].set(plan[0])
        nodes = jnp.concatenate([nodes, plan[None]], axis=0)
        return jnp.clip(nodes, -1.0, 1.0)

    return sample


def make_rollout(env, per_step: Callable | None = None):
    """Rolls one action sequence, carrying only `per_step(state)` per step.

    `per_step` returns whatever the caller needs from each visited state; the
    default carries the scalar reward alone, which is all the MPPI update uses.
    """

    if per_step is None:
        per_step = lambda state: state.reward

    def rollout(state, us):
        def step(carry, u):
            nxt = env.step(carry, u)
            return nxt, per_step(nxt)

        _, out = jax.lax.scan(step, state, us)
        return out

    return rollout


def make_lean_update(env, mbdpi, dial_config):
    """One DIAL annealing pass, identical to `reverse_once` but state-free."""

    sample = make_sampler(mbdpi, dial_config)
    rollout = make_rollout(env)
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))
    temp = dial_config.temp_sample

    def update(state, rng, plan, noise_scale):
        rng, sample_rng = jax.random.split(rng)
        nodes = sample(sample_rng, plan, noise_scale)
        us = mbdpi.node2u_vvmap(nodes)
        returns = rollout_vmap(state, us).mean(axis=-1)
        reference = returns[-1]
        scale = jnp.maximum(returns.std(), 1e-6)
        weights = jax.nn.softmax((returns - reference) / scale / temp)
        return rng, jnp.einsum("n,nij->ij", weights, nodes)

    return update


def make_dial_step(env, mbdpi, dial_config, n_diffuse: int | None = None):
    """One closed-loop DIAL control step: shift, anneal, act."""

    update = make_lean_update(env, mbdpi, dial_config)
    sigma = mbdpi.sigma_control
    n_diffuse = dial_config.Ndiffuse if n_diffuse is None else n_diffuse
    factors = dial_config.traj_diffuse_factor ** jnp.arange(n_diffuse)

    def control_step(state, rng, plan):
        plan = mbdpi.shift(plan)

        def anneal(carry, factor):
            key, current = carry
            key, current = update(state, key, current, sigma * factor)
            return (key, current), None

        (rng, plan), _ = jax.lax.scan(anneal, (rng, plan), factors)
        return env.step(state, plan[0]), rng, plan

    return control_step
