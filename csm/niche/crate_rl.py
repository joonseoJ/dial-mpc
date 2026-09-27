"""Crate climb: does PPO on DIAL's own reward climb where DIAL does?

The DIAL-MPC paper's crate climb: the Go2 puts its head on a 0.87 m target on
top of a crate, from a single reset.  Reward, observation (full qpos, so the
crate's fixed position is known) and plant are the stock ones.  Two plants:

  stock  20 ms single physics step, PD torque held for the step (the paper's)
  fine   10 ms substeps, PD recomputed per substep (the walking tasks' plant)

On the stock plant PPO's random exploration drives MJX non-finite; a guard
prices a non-finite step at -10 and zeroes the observation, so the learner can
learn to avoid it rather than crash.  Success: head within 5 cm of the target
at the end of the 150-step (3 s) episode.
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import jax
import jax.numpy as jnp
import brax.envs as brax_envs
from brax import math

import dial_mpc.envs  # noqa: F401
from dial_mpc.core.dial_core import make_controller
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step, make_lean_update

TARGET = np.array([1.45, 0.0, 0.87])


class Guard(brax_envs.Wrapper):
    def step(self, state, action):
        nxt = self.env.step(state, action)
        bad = ~jnp.isfinite(nxt.reward) | ~jnp.all(jnp.isfinite(nxt.obs))
        return nxt.replace(reward=jnp.where(bad, -10.0, jnp.nan_to_num(nxt.reward)),
                           obs=jnp.nan_to_num(nxt.obs, posinf=0.0, neginf=0.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["dial", "ppo"])
    ap.add_argument("--plant", choices=["stock", "fine"], default="stock")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--ppo-steps", type=float, default=5e8)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    dc, ec = _load_config("unitree_go2_crate_climb", None)
    if a.plant == "fine":
        ec = dataclasses.replace(ec, timestep=0.01, pd_substep=True)
    base = brax_envs.get_environment(dc.env_name, config=ec)
    ti = base._torso_idx - 1

    def head(st):
        x = st.pipeline_state.x
        return x.pos[ti] + math.quat_to_3x3(x.rot[ti]) @ jnp.array([0.285, 0.0, 0.0])

    t0 = time.time()
    if a.arm == "dial":
        env = base
        mbdpi = make_controller(dc, env)
        step = make_dial_step(env, mbdpi, dc, std_normalize=True)
        upd = make_lean_update(env, mbdpi, dc, True)
        factors = dc.traj_diffuse_factor ** jnp.arange(dc.Ndiffuse_init)

        @jax.jit
        def run(state, rng):
            plan = jnp.zeros((dc.Hnode + 1, mbdpi.nu))
            def warm(c, f):
                k, p = c
                return upd(state, k, p, mbdpi.sigma_control * f), None
            (rng, plan), _ = jax.lax.scan(warm, (rng, plan), factors)
            def body(c, _):
                st, k, p = c
                st, k, p = step(st, k, p)
                return (st, k, p), (head(st), st.reward)
            return jax.lax.scan(body, (state, rng, plan), None, length=a.steps)[1]
    else:
        from brax.training.agents.ppo import train as ppo
        env = Guard(base)
        make_inf, params, _ = ppo.train(
            environment=env, num_timesteps=int(a.ppo_steps), episode_length=a.steps,
            num_envs=2048, batch_size=1024, unroll_length=20, num_minibatches=32,
            num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
            discounting=0.97, num_evals=6, seed=0,
            progress_fn=lambda n, m: print(f"  {n:>13,} steps  eval reward "
                                           f"{m['eval/episode_reward']:.2f}", flush=True))
        inf = make_inf(params, deterministic=True)

        @jax.jit
        def run(state, rng):
            def body(st, _):
                act, _ = inf(st.obs, rng)
                st = env.step(st, act)
                return st, (head(st), st.reward)
            return jax.lax.scan(body, state, None, length=a.steps)[1]

    ok = 0
    for seed in range(a.seeds):
        h, r = run(jax.jit(env.reset)(jax.random.PRNGKey(seed)), jax.random.PRNGKey(seed))
        h = np.asarray(h)
        err = float(np.linalg.norm(h[-1] - TARGET))
        ok += err < 0.05
        print(f"{a.arm} {a.plant} seed {seed}: final head {np.round(h[-1], 3)}  error "
              f"{err:.3f} m  max head z {h[:, 2].max():.3f}  mean reward "
              f"{np.mean(np.asarray(r)):.3f}", flush=True)
    print(f"{a.arm} on the {a.plant} plant: {ok}/{a.seeds} climbed  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
