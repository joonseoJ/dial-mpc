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


def mppi_logits(returns, temp, std_normalize: bool = True):
    """DIAL's softmax exponent, with its scale normalisation optional.

    With `std_normalize` on this is `reverse_once` verbatim.  Off, the exponent
    is `nu . (-C)` for `nu = omega / temp`, i.e. a genuine Gibbs distribution at
    a known temperature -- which is what makes the score linear in the weights
    and the composition exact.  The price is that the temperature no longer
    adapts to the spread of the cloud on its own.
    """

    reference = returns[-1]
    scale = jnp.maximum(returns.std(), 1e-6) if std_normalize else 1.0
    return (returns - reference) / scale / temp


def make_lean_update(env, mbdpi, dial_config, std_normalize: bool = True):
    """One DIAL annealing pass, identical to `reverse_once` but state-free.

    The temperature is an argument rather than a constant so a caller can give
    each annealing level its own.
    """

    sample = make_sampler(mbdpi, dial_config)
    rollout = make_rollout(env)
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))
    default_temp = dial_config.temp_sample

    def update(state, rng, plan, noise_scale, temp=None):
        temp = default_temp if temp is None else temp
        rng, sample_rng = jax.random.split(rng)
        nodes = sample(sample_rng, plan, noise_scale)
        us = mbdpi.node2u_vvmap(nodes)
        returns = rollout_vmap(state, us).mean(axis=-1)
        weights = jax.nn.softmax(mppi_logits(returns, temp, std_normalize))
        return rng, jnp.einsum("n,nij->ij", weights, nodes)

    return update


def make_dial_step(env, mbdpi, dial_config, n_diffuse: int | None = None,
                   std_normalize: bool = True, level_scales=None):
    """One closed-loop DIAL control step: shift, anneal, act.

    `level_scales` multiplies the temperature per annealing level.  Without the
    spread normalisation a fixed temperature means very different sharpness at
    each level, because a coarse level perturbs the plan far enough to knock the
    robot over and its returns spread accordingly -- measured on this plant, the
    temperature that puts the fine level at 40% effective sample size leaves the
    coarse level near an argmax, a factor of 3.2 apart.  Scaling the temperature
    with the level is the fix, and because the same profile applies to every
    weight vector it cancels out of the composition coefficients entirely.
    """

    update = make_lean_update(env, mbdpi, dial_config, std_normalize)
    sigma = mbdpi.sigma_control
    n_diffuse = dial_config.Ndiffuse if n_diffuse is None else n_diffuse
    factors = dial_config.traj_diffuse_factor ** jnp.arange(n_diffuse)
    if level_scales is None:
        scales = jnp.ones_like(factors)
    else:
        scales = jnp.asarray(level_scales, dtype=factors.dtype)
        if scales.shape != factors.shape:
            raise ValueError(
                f"level_scales must have {factors.shape[0]} entries, one per level"
            )
    temps = scales * dial_config.temp_sample

    def control_step(state, rng, plan):
        plan = mbdpi.shift(plan)

        def anneal(carry, level):
            key, current = carry
            factor, temp = level
            key, current = update(state, key, current, sigma * factor, temp)
            return (key, current), None

        (rng, plan), _ = jax.lax.scan(anneal, (rng, plan), (factors, temps))
        return env.step(state, plan[0]), rng, plan

    return control_step
