"""Before the gate: is the stones task correct, and can DIAL walk it at all?

  1. identity   M = 1 robust DIAL under the true terrain reproduces DIAL's own
                `reverse_once` on the same state and key (same plans, same
                returns), and a terrain installed by `terrain_sys` gives the
                same physics as a model compiled with those stone heights.
  2. holes bite a foot that steps on a hole drops below the surface: on an
                all-hole strip the robot is down within the episode.
  3. speed      control-step wall time at the gate's sizes.
  4. oracle     the planner that knows the terrain crosses; flat (all solid)
                and random strips, fall rate and progress.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from brax.io import mjcf

from csm.terrain import stones as S
from csm.terrain import robust as R


def make_env(vx=0.6):
    cfg = S.StonesEnvConfig(dt=0.02, timestep=0.01, leg_control="torque", pd_substep=True,
                            action_scale=1.0, terminate_on_joint_limit=False,
                            default_vx=vx, ramp_up_time=1.0, gait="trot")
    return S.StonesEnv(cfg)


def summarise(tr: R.Trace):
    fell = np.asarray(jnp.any(tr.done > 0, -1))
    x_end = np.asarray(tr.x[:, -1])
    cost = -np.asarray(tr.reward.sum(-1))
    return fell, x_end, cost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsample", type=int, default=1024)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args(argv)
    env = make_env(a.vx)
    ctl = R.Robust(env, R.Plan(Nsample=a.nsample))
    lay = S.Layout()

    if "identity" not in a.skip:
        print("=== 1. identity ===")
        tops, _ = S.sample_episode(jax.random.PRNGKey(0), lay)
        st = env.reset(jax.random.PRNGKey(1))
        Y = 0.2 * jax.random.normal(jax.random.PRNGKey(2), (ctl.p.Hnode + 1, ctl.nu))
        f = ctl.factors(1)[0]
        _, Yr, rr, _ = jax.jit(lambda st, Y: ctl.reverse_once(st, jax.random.PRNGKey(3), Y, f,
                                                           tops[None], jnp.ones(1)))(st, Y)
        old = env.sys
        env.sys = env.terrain_sys(tops)
        _, Yd, info = ctl.mb.reverse_once(st, jax.random.PRNGKey(3), Y, f)
        env.sys = old
        # Not bit-exact: two differently fused XLA programs round differently,
        # and contact makes a 1e-7 difference grow in the rollouts that fall.
        # The logic is identical when the median error is at float precision.
        d = np.abs(np.asarray(rr - info["rews"]))
        print(f"  robust(M=1, truth) vs MBDPI.reverse_once: returns median |d| {np.median(d):.1e}, "
              f"{100 * np.mean(d > 1e-3):.1f}% of {d.size} plans > 1e-3 (chaotic ones), "
              f"max |dY| {float(jnp.max(jnp.abs(Yr - Yd))):.1e}")
        mj2 = mujoco.MjModel.from_xml_path(str(S.get_model_path("unitree_go2", S.SCENE)))
        for i, t in enumerate(np.asarray(tops)):
            mj2.geom_pos[int(env.stone_ids[i]), 2] = t - S.HALF_Z
        sys2 = mjcf.load_model(mj2).tree_replace({"opt.timestep": 0.01})
        us = jnp.tile(ctl.mb.node2u_vmap(Yd), (4, 1))[:60]

        def roll(sys_or_tops, direct):
            def b(s, u):
                if direct:
                    o = env.sys; env.sys = sys_or_tops
                    try: s = env.step(s, u)
                    finally: env.sys = o
                else:
                    s = env.step_in(sys_or_tops, s, u)
                return s, s.pipeline_state.q
            return jax.lax.scan(b, st, us)[1]
        qa = jax.jit(lambda t: roll(t, False))(tops)
        qb = roll(sys2, True)
        print(f"  terrain_sys vs compiled model, 60 steps: max |dq| {float(jnp.max(jnp.abs(qa - qb))):.2e}")

    run = jax.jit(jax.vmap(lambda tt, parts, pw, k: R.episode(ctl, tt, parts, pw, k, a.steps),
                           in_axes=(0, 0, 0, 0)))
    B = a.episodes
    keys = jax.random.split(jax.random.PRNGKey(10), B)

    if "holes" not in a.skip:
        print("\n=== 2./3. all-solid vs all-hole strip (oracle), and speed ===")
        for nm, top in (("all solid", S.SOLID), ("all holes", S.HOLE)):
            tt = jnp.full((B, S.N_CELL), top)
            t0 = time.time()
            tr = run(tt, tt[:, None], jnp.ones((B, 1)), keys)
            jax.block_until_ready(tr)
            t1 = time.time()
            tr = run(tt, tt[:, None], jnp.ones((B, 1)), keys)
            jax.block_until_ready(tr)
            t2 = time.time()
            fell, x, c = summarise(tr)
            print(f"  {nm:<10}: fell {fell.sum()}/{B}, x_end {x.mean():.2f} (min {x.min():.2f}), "
                  f"cost {c.mean():.1f} +- {c.std() / np.sqrt(B):.1f};  compile+run {t1 - t0:.0f} s, "
                  f"run {t2 - t1:.1f} s = {1e3 * (t2 - t1) / a.steps:.0f} ms/step for {B} episodes")

    print("\n=== 4. oracle on random strips ===")
    ks = jax.random.split(jax.random.PRNGKey(20), B)
    tt, bel = jax.vmap(lambda k: S.sample_episode(k, lay))(ks)
    print(f"  holes per strip {np.asarray((tt < 0).sum(-1))}")
    tr = run(tt, tt[:, None], jnp.ones((B, 1)), keys)
    fell, x, c = summarise(tr)
    print(f"  oracle    : fell {fell.sum()}/{B}, x_end {x.mean():.2f} (min {x.min():.2f}), "
          f"cost {c.mean():.1f} +- {c.std() / np.sqrt(B):.1f}")
    tr = run(tt, jax.vmap(lambda b: S.representative(b, "optimistic"))(bel)[:, None],
             jnp.ones((B, 1)), keys)
    fell2, x2, c2 = summarise(tr)
    print(f"  optimistic: fell {fell2.sum()}/{B}, x_end {x2.mean():.2f}, cost {c2.mean():.1f}; "
          f"paired vs oracle {np.mean(c2 - c):+.1f} +- {np.std(c2 - c) / np.sqrt(B):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
