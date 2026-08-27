"""Closed-loop check that omega changes how DIAL actually rejects a push.

`basis_screen` asks whether omega moves MPPI's update from a shared sample
cloud.  That is the cheap gate.  This is the expensive confirmation: run DIAL
end to end under each omega and measure the recovery strategy it produces.

The headline measurement is a *behavioural* one rather than a reward one.  A
reward margin has to clear DIAL's own seed-to-seed noise before it means
anything, and on this plant that floor is large enough to swallow most
effects.  Whether the robot picks its feet up is not a margin -- it is a
discrete event with a clean probability, and the curve of

    P(the robot steps)  vs  push magnitude

is a sigmoid whose midpoint is exactly the "how hard do I have to be shoved
before I stop trying to stand my ground" threshold that omega is supposed to
control.  Comparing curves is far more sensitive than comparing scalars.

Every omega sees the identical set of pushes (same speeds, same headings, same
reset keys), so the comparison uses common random numbers throughout.  The
whole (omega x speed x seed) grid is one vmapped program driven by the lean
DIAL update in `csm.dial_lean`, because the stock `reverse_once` materialises
every sample's pipeline state and will not fit the outer batch.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

import jax
import jax.numpy as jnp

import brax.envs as brax_envs
from brax import math as brax_math

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import make_controller
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

from csm.basis_screen import build_omegas, _load_config
from csm.dial_lean import make_dial_step


def make_push_reset(env):
    """Reset to the nominal stance with a prescribed push and weight vector."""

    def reset(rng, speed, heading, omega):
        state = env.reset(rng)
        linear = jnp.stack(
            [speed * jnp.cos(heading), speed * jnp.sin(heading), jnp.asarray(0.0)]
        )
        qpos = state.pipeline_state.qpos
        qvel = jnp.zeros_like(state.pipeline_state.qvel).at[:3].set(linear)
        pipeline_state = env.pipeline_init(qpos, qvel)
        state.info["push_linear"] = linear
        state.info["push_speed"] = speed
        state.info["reward_weights"] = omega
        obs = env._get_obs(pipeline_state, state.info)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

    return reset


def make_episode(env, mbdpi, dial_config, n_steps: int, std_normalize=True,
                 level_scales=None):
    control_step = make_dial_step(env, mbdpi, dial_config,
                                  std_normalize=std_normalize,
                                  level_scales=level_scales)
    nominal_feet = env._nominal_foot_pos
    nominal_xy = env._nominal_base_xy
    torso = env._torso_idx - 1
    up = jnp.array([0.0, 0.0, 1.0])

    def observe(state):
        ps = state.pipeline_state
        feet = ps.site_xpos[env._feet_site_id]
        foot_shift = jnp.linalg.norm(feet[:, :2] - nominal_feet[:, :2], axis=-1)
        pos = ps.x.pos[torso]
        tilt = jnp.linalg.norm(brax_math.rotate(up, ps.x.rot[torso]) - up)
        return {
            "foot_shift": foot_shift.max(),
            "lifted": state.info["feet_lifted"],
            "tilt": tilt,
            "base_drift": jnp.linalg.norm(pos[:2] - nominal_xy),
            "height": pos[2],
            "joint_dev": jnp.linalg.norm(ps.q[7:] - env._default_pose),
            "done": state.done,
            "reward_terms": state.info["reward_terms"],
        }

    def episode(state, rng):
        plan = jnp.zeros((dial_config.Hnode + 1, mbdpi.nu))

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control_step(st, key, pl)
            return (st, key, pl), observe(st)

        _, traj = jax.lax.scan(body, (state, rng, plan), None, length=n_steps)
        return traj

    return episode


def summarise(traj, step_threshold: float):
    """Per-episode behaviour summary."""

    fell = traj["done"].max()
    return {
        "stepped": (traj["foot_shift"].max() > step_threshold).astype(jnp.float32),
        "max_foot_shift": traj["foot_shift"].max(),
        "lift_time": traj["lifted"].mean(),
        "mean_tilt": traj["tilt"].mean(),
        "max_tilt": traj["tilt"].max(),
        "mean_drift": traj["base_drift"].mean(),
        "final_drift": traj["base_drift"][-1],
        "mean_joint_dev": traj["joint_dev"].mean(),
        "min_height": traj["height"].min(),
        "fell": fell,
        "terms": traj["reward_terms"].mean(axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument(
        "--speeds", type=float, nargs="+",
        default=[0.4, 0.7, 1.0, 1.3, 1.6],
        help="push magnitudes in m/s; the curve over these is the measurement",
    )
    parser.add_argument(
        "--omegas", type=str, nargs="+", default=["e0", "e1", "e2", "e3", "uniform"],
    )
    parser.add_argument("--step-threshold", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--level-scales", type=float, nargs="+", default=None,
                        help="temperature multiplier per annealing level, "
                             "coarse first")
    parser.add_argument("--std-normalize", dest="std", action="store_true")
    parser.add_argument("--no-std-normalize", dest="std", action="store_false",
                        help="drive DIAL with the raw Gibbs exponent")
    parser.set_defaults(std=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, args.config)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    all_omegas = build_omegas(n_rows)
    missing = [n for n in args.omegas if n not in all_omegas]
    if missing:
        raise ValueError(f"unknown omega names: {missing}")
    omega_names = list(args.omegas)
    omega_vecs = jnp.asarray(np.stack([all_omegas[n] for n in omega_names]))

    speeds = jnp.asarray(args.speeds)
    n_omega, n_speed, n_seed = len(omega_names), len(speeds), args.seeds

    # Common random numbers: the heading and reset key depend on the seed
    # index only, so every omega faces exactly the same pushes.
    rng = jax.random.PRNGKey(args.seed)
    rng, head_rng, reset_rng, roll_rng = jax.random.split(rng, 4)
    headings = jax.random.uniform(
        head_rng, (n_seed,), minval=-jnp.pi, maxval=jnp.pi
    )
    reset_keys = jax.random.split(reset_rng, n_seed)
    roll_keys = jax.random.split(roll_rng, n_seed)

    grid_o, grid_s, grid_d = jnp.meshgrid(
        jnp.arange(n_omega), jnp.arange(n_speed), jnp.arange(n_seed), indexing="ij"
    )
    flat_o, flat_s, flat_d = grid_o.ravel(), grid_s.ravel(), grid_d.ravel()
    total = flat_o.size

    push_reset = make_push_reset(env)
    episode = make_episode(env, mbdpi, dial_config, args.steps, args.std,
                           args.level_scales)

    def one(io, is_, id_):
        state = push_reset(
            reset_keys[id_], speeds[is_], headings[id_], omega_vecs[io]
        )
        traj = episode(state, roll_keys[id_])
        return summarise(traj, args.step_threshold)

    batch = jax.vmap(one)
    chunk = max(min(args.chunk, total), 1)
    pad = (-total) % chunk

    @jax.jit
    def run():
        o = jnp.pad(flat_o, (0, pad))
        s = jnp.pad(flat_s, (0, pad))
        d = jnp.pad(flat_d, (0, pad))
        shaped = jax.tree.map(lambda x: x.reshape(-1, chunk), (o, s, d))
        out = jax.lax.map(lambda c: batch(*c), shaped)
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:])[:total], out)

    print(f"[eval] {n_omega} omegas x {n_speed} speeds x {n_seed} seeds = {total} "
          f"closed-loop DIAL episodes, {args.steps} control steps each, "
          f"chunk={chunk}")
    print(f"[eval] {total * args.steps * dial_config.Ndiffuse * (dial_config.Nsample + 1) * dial_config.Hsample / 1e9:.2f}G "
          f"env-steps of planning")
    t0 = time.time()
    result = run()
    jax.block_until_ready(result)
    print(f"[eval] done in {time.time() - t0:.1f}s")

    res = {k: np.asarray(v) for k, v in result.items()}
    o_idx, s_idx = np.asarray(flat_o), np.asarray(flat_s)

    def grid(key):
        out = np.zeros((n_omega, n_speed))
        for i in range(n_omega):
            for j in range(n_speed):
                sel = (o_idx == i) & (s_idx == j)
                out[i, j] = res[key][sel].mean()
        return out

    stepped, foot, tilt, drift, jdev, fell, lift = (
        grid("stepped"), grid("max_foot_shift"), grid("mean_tilt"),
        grid("mean_drift"), grid("mean_joint_dev"), grid("fell"), grid("lift_time"),
    )

    width = max(len(n) for n in omega_names) + 1
    speed_hdr = "".join(f"{float(v):>9.2f}" for v in speeds)

    def show(title, mat, fmt="{:9.3f}"):
        print(f"\n=== {title} ===")
        print(f"{'omega':<{width}}{speed_hdr}")
        for i, n in enumerate(omega_names):
            print(f"{n:<{width}}" + "".join(fmt.format(mat[i, j]) for j in range(n_speed)))

    show(f"P(step) -- foot moved > {args.step_threshold} m, by push speed", stepped)
    show("max foot displacement (m)", foot)
    show("mean torso tilt", tilt)
    show("mean base drift (m)", drift)
    show("mean joint deviation (rad)", jdev)
    show("fall rate", fell)
    show("fraction of time with a foot in the air", lift)

    print("\n[verdict] P(step) spread across omega, per speed:")
    for j, v in enumerate(speeds):
        col = stepped[:, j]
        lo, hi = int(col.argmin()), int(col.argmax())
        print(f"  push {float(v):.2f} m/s: {col.min():.2f} ({omega_names[lo]}) -> "
              f"{col.max():.2f} ({omega_names[hi]})   spread {col.max() - col.min():.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "env": dial_config.env_name,
                    "omegas": omega_names,
                    "speeds": [float(v) for v in speeds],
                    "seeds": n_seed,
                    "steps": args.steps,
                    "step_threshold": args.step_threshold,
                    "p_step": stepped.tolist(),
                    "max_foot_shift": foot.tolist(),
                    "mean_tilt": tilt.tolist(),
                    "mean_drift": drift.tolist(),
                    "mean_joint_dev": jdev.tolist(),
                    "fall_rate": fell.tolist(),
                    "lift_time": lift.tolist(),
                },
                handle,
                indent=2,
            )
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
