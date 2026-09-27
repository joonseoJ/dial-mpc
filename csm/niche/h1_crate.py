"""H1 pushes a 30 kg crate 2 m: does PPO on DIAL's own reward do what DIAL does?

The environment's own comment says this is "exactly the look-ahead a sampling
planner does well and reactive RL does badly" -- never tested.  Nine reward
rows (gait schedule, upright, yaw, velocity, yaw rate, height, energy, foot
contact, crate), uniform weight.  PPO gets the full observation (qpos includes
the crate's slide joint) plus the gait clock the time-indexed gait row needs,
fixed horizon (falls are not an escape from the per-step charge), and a guard
against non-finite steps on the 20 ms plant.

Success: the crate ends within 0.2 m of its target and the torso is up.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import brax.envs as brax_envs

import dial_mpc.envs  # noqa: F401
from dial_mpc.core.dial_core import make_controller
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step, make_lean_update
from csm.niche.crate_rl import Guard
from csm import rl_baseline as RB


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["dial", "ppo"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ppo-steps", type=float, default=5e8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--omega", default=None,
                    help="9 comma-separated row weights (gait, upright, yaw, vel, yaw rate, "
                         "height, energy, contact, crate); default uniform")
    a = ap.parse_args(argv)
    dc, ec = _load_config("unitree_h1_push_crate", None)
    if a.omega:
        import dataclasses
        ec = dataclasses.replace(ec, reward_weights=jnp.asarray(
            [float(v) for v in a.omega.split(",")], jnp.float32))
    base = brax_envs.get_environment(dc.env_name, config=ec)
    ti = base._torso_idx - 1
    rec = lambda st: (st.pipeline_state.qpos[-1] - st.info["crate_tar"],
                      st.pipeline_state.x.pos[ti, 2], st.reward)
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
                return (st, k, p), rec(st)
            return jax.lax.scan(body, (state, rng, plan), None, length=a.steps)[1]

        reset = jax.jit(env.reset)
        prep = lambda st: st
    else:
        from brax.training.agents.ppo import train as ppo
        cadence = float(np.asarray(base._gait_params[base._gait])[1])
        env = RB.GaitClockWrapper(RB.FixedHorizonWrapper(Guard(base)), cadence)
        make_inf, params, _ = ppo.train(
            environment=env, num_timesteps=int(a.ppo_steps), episode_length=a.steps,
            num_envs=2048, batch_size=1024, unroll_length=20, num_minibatches=32,
            num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
            discounting=0.97, num_evals=6, seed=0, normalize_observations=True,
            progress_fn=lambda n, m: print(f"  {n:>13,} steps  eval reward "
                                           f"{m['eval/episode_reward']:.2f}", flush=True))
        inf = make_inf(params, deterministic=True)

        @jax.jit
        def run(state, rng):
            def body(st, _):
                act, _ = inf(st.obs, rng)
                st = env.step(st, act)
                return st, rec(st)
            return jax.lax.scan(body, state, None, length=a.steps)[1]

        reset = jax.jit(env.reset)
        prep = lambda st: st
    ok = 0
    for seed in range(a.seeds):
        err, z, r = (np.asarray(x) for x in run(prep(reset(jax.random.PRNGKey(seed))),
                                                  jax.random.PRNGKey(seed)))
        up = z[-1] > 0.6
        ok += (abs(err[-1]) < 0.2) and up
        print(f"{a.arm} seed {seed}: crate error {err[0]:+.2f} -> {err[-1]:+.2f} m  torso z "
              f"{z[-1]:.2f}  mean reward {r.mean():.3f}", flush=True)
    print(f"{a.arm}: {ok}/{a.seeds} pushed the crate home standing  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
