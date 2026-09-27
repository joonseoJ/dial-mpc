"""DIAL-MPC whose rollouts are averaged over M hypothesised terrains.

`reverse_once` is `MBDPI.reverse_once` with one change: every sampled plan is
rolled out under each of the M terrains in `parts`, and its return is the
weighted mean over them.  With M = 1 and the true terrain it *is* DIAL
(checked in `check.py`), so every condition in the gate is the same planner
with a different set of worlds in its head.

Everything else -- node splines, the locked first node, the per-level noise
profile, the return normalised by its own spread, the temperature, the shift
-- is DIAL's, taken from an `MBDPI` instance rather than re-implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI

from csm.terrain import stones as S


@dataclass(frozen=True)
class Plan:
    Nsample: int = 1024
    Hsample: int = 16
    Hnode: int = 4
    Ndiffuse: int = 2
    Ndiffuse_init: int = 10
    temp_sample: float = 0.05
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5


def dial_config(p: Plan) -> DialConfig:
    return DialConfig(Nsample=p.Nsample, Hsample=p.Hsample, Hnode=p.Hnode,
                      Ndiffuse=p.Ndiffuse, Ndiffuse_init=p.Ndiffuse_init,
                      temp_sample=p.temp_sample,
                      horizon_diffuse_factor=p.horizon_diffuse_factor,
                      traj_diffuse_factor=p.traj_diffuse_factor)


class Robust:
    def __init__(self, env: S.StonesEnv, p: Plan):
        self.env, self.p = env, p
        self.mb = MBDPI(dial_config(p), env)
        self.nu = env.action_size

    def rollout_rewards(self, tops, state, us):
        """(H,) rewards of one control sequence under one terrain."""
        def body(s, u):
            s = self.env.step_in(tops, s, u)
            return s, s.reward
        return jax.lax.scan(body, state, us)[1]

    def reverse_once(self, state, rng, Ybar, noise, parts, pw):
        p = self.p
        rng, k = jax.random.split(rng)
        eps = jax.random.normal(k, (p.Nsample, p.Hnode + 1, self.nu))
        Y0s = eps * noise[None, :, None] + Ybar
        Y0s = Y0s.at[:, 0].set(Ybar[0])
        Y0s = jnp.concatenate([Y0s, Ybar[None]], 0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        us = self.mb.node2u_vvmap(Y0s)
        per = jax.vmap(lambda tops: jax.vmap(
            lambda u: self.rollout_rewards(tops, state, u))(us))(parts)   # (M, N+1, H)
        # A rollout that diverges (a foot driven into a stone edge can blow up
        # the contact solve) returns NaN, and one NaN return poisons the whole
        # softmax, the executed action and then the world.  Price it below
        # every finite plan instead, and count it so the rate stays visible.
        bad = ~jnp.isfinite(per)
        nbad = jnp.sum(jnp.any(bad, -1))
        worst = jnp.min(jnp.where(bad, jnp.inf, per))
        per = jnp.where(bad, worst - 10.0, per)
        rews = jnp.einsum("m,mn->n", pw, per.mean(-1))
        scale = jnp.maximum(rews.std(), 1e-6)
        logp = (rews - rews[-1]) / scale / p.temp_sample
        w = jax.nn.softmax(logp)
        return rng, jnp.einsum("n,nij->ij", w, Y0s), rews, nbad

    def factors(self, n):
        return self.mb.sigma_control * self.p.traj_diffuse_factor ** jnp.arange(n)[:, None]

    def plan(self, state, rng, Y, parts, pw, n):
        def lvl(c, f):
            rng, Y = c
            rng, Y, rews, nbad = self.reverse_once(state, rng, Y, f, parts, pw)
            return (rng, Y), nbad
        (rng, Y), nbad = jax.lax.scan(lvl, (rng, Y), self.factors(n))
        return rng, Y, jnp.sum(nbad)


class Trace(NamedTuple):
    reward: jax.Array      # (T,)
    done: jax.Array        # (T,)
    x: jax.Array           # (T,) base x
    z: jax.Array           # (T,) base z
    feet_low: jax.Array    # (T,) feet below the walking surface by > 5 cm
    nbad: jax.Array        # (T,) diverged planner rollouts (priced as worst) this step


def episode(ctl: Robust, tops_true, parts, pw, rng, T: int):
    """One closed-loop walk.  The world steps under `tops_true`; the planner
    under `parts` with weights `pw`.  Fixed particles for the whole episode:
    the belief does not update (nothing new is seen while walking)."""
    env = ctl.env
    rng, kr = jax.random.split(rng)
    state = env.reset(kr)
    Y = jnp.zeros((ctl.p.Hnode + 1, ctl.nu))
    rng, Y, _ = ctl.plan(state, rng, Y, parts, pw, ctl.p.Ndiffuse_init)

    def body(c, _):
        state, Y, rng = c
        state = env.step_in(tops_true, state, Y[0])
        Y = ctl.mb.shift(Y)
        rng, Y, nbad = ctl.plan(state, rng, Y, parts, pw, ctl.p.Ndiffuse)
        ps = state.pipeline_state
        zf = ps.site_xpos[env._feet_site_id][:, 2]
        return (state, Y, rng), Trace(state.reward, state.done, ps.x.pos[0, 0], ps.x.pos[0, 2],
                                      jnp.sum(zf < -0.05), nbad)
    (_, _, _), tr = jax.lax.scan(body, (state, Y, rng), None, length=T)
    return tr


def validate_shape(ctl: Robust, bel, M: int, B: int, tol=1e-3):
    """Is the vmapped (B episodes x M particles) program the same computation
    as the single-episode one?  On this GPU stack XLA miscompiled B = 2 at
    M = 32 and 48 -- returns off by a median 150, with no NaN to give it away
    -- while B = 1 and 4 at M = 32 and every other shape tried agreed to 1e-5.
    Compare one annealing level's returns; median, because contact makes the
    odd falling rollout differ at float precision in any two programs."""
    env = ctl.env
    st = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(5), B))
    Y = 0.3 * jax.random.normal(jax.random.PRNGKey(2), (B, ctl.p.Hnode + 1, ctl.nu))
    parts = jax.vmap(lambda k, b: S.sample_particles(k, b, M))(
        jax.random.split(jax.random.PRNGKey(9), B), bel[:B])
    f = ctl.factors(1)[0]
    one = lambda s, y, p: ctl.reverse_once(s, jax.random.PRNGKey(3), y, f, p, jnp.full((M,), 1.0 / M))[2]
    rb = jax.jit(jax.vmap(one))(st, Y, parts)
    n = min(B, 2)
    r1 = jnp.stack([jax.jit(one)(jax.tree.map(lambda x: x[i], st), Y[i], parts[i]) for i in range(n)])
    return float(jnp.median(jnp.abs(rb[:n] - r1))) < tol
