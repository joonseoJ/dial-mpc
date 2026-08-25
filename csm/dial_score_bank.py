"""Fill a state bank with deep, healthy states from a DIAL-MPC rollout.

Student-driven collection cannot reach late states: a fall resets the episode
regardless of ``--episode-steps``, so the horizon in the data is capped by how
long the student survives -- about 500 steps today.  DIAL walks 3000 steps
without falling, so its trajectory is a source of genuinely deep states that can
be handed to the student as episode starts.  That decouples *reaching* a state
from *surviving* to it.

The banked states are DIAL's, not the student's, so they are a seed rather than
a substitute: the student continues from them and those continuations are its
own distribution.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax import envs as brax_envs
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.dial_score import StateBank, dial_factors
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", help="DIAL example name, without .yaml")
    source.add_argument("--config", type=Path, help="DIAL YAML config path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--min-step",
        type=int,
        default=300,
        help="ignore states before this depth; shallow states are already "
        "well covered by ordinary resets",
    )
    parser.add_argument(
        "--stride", type=int, default=25, help="bank every Nth eligible state"
    )
    parser.add_argument("--seed-offset", type=int, default=5000)
    parser.add_argument("--omega", default="1,1,1")
    parser.add_argument("--temp-sample", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--init-passes", type=int, default=5)
    parser.add_argument("--start-noise", type=float, default=0.5)
    parser.add_argument("--capacity", type=int, default=4000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = get_example_path(args.example + ".yaml") if args.example else args.config
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    overrides = {"time_correlated": False, "temp_sample": float(args.temp_sample)}
    if args.samples is not None:
        overrides["Nsample"] = int(args.samples)
    dial_config = dataclasses.replace(
        load_dataclass_from_dict(DialConfig, config), **overrides
    )
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(dial_config.env_name), config, convert_list_to_array=True
    )
    if args.start_noise > 0.0:
        scale = float(args.start_noise)
        env_config = dataclasses.replace(
            env_config,
            randomize_start_state=True,
            start_height_noise=0.02 * scale,
            start_rpy_noise=0.10 * scale,
            start_joint_position_noise=0.10 * scale,
            start_body_linear_velocity_noise=0.20 * scale,
            start_body_angular_velocity_noise=0.20 * scale,
            start_joint_velocity_noise=0.50 * scale,
        )
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    planner = MBDPI(dial_config, env)
    horizon = int(dial_config.Hnode) + 1
    factors = dial_factors(dial_config.traj_diffuse_factor, dial_config.Ndiffuse)
    sigma_control = jnp.asarray(planner.sigma_control, dtype=jnp.float32)
    omega = jnp.asarray(
        [float(v) for v in args.omega.split(",")], dtype=jnp.float32
    )
    reset_env, step_env = jax.jit(env.reset), jax.jit(env.step)
    shift = jax.jit(planner.shift)

    @jax.jit
    def anneal(state, plan, rng, schedule):
        def body(carry, factor):
            plan, rng = carry
            rng, call = jax.random.split(rng)
            _, plan, _ = planner.reverse_once(
                state, call, plan, sigma_control * factor
            )
            return (plan, rng), None

        (plan, rng), _ = jax.lax.scan(body, (plan, rng), schedule)
        return plan, rng

    def with_omega(state):
        info = dict(state.info)
        info["reward_weights"] = omega
        return state.replace(info=info)

    bank = StateBank(capacity=args.capacity)
    first_schedule = jnp.tile(factors, args.init_passes)
    falls, depths = 0, []
    progress = tqdm(
        total=args.episodes * args.steps, desc="DIAL bank", unit="step",
        dynamic_ncols=True,
    )
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        state = with_omega(reset_env(jax.random.PRNGKey(seed)))
        plan = jnp.zeros((horizon, int(env.action_size)), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed + 700_000)
        episode_step = 0
        for _ in range(args.steps):
            schedule = first_schedule if episode_step == 0 else factors
            plan, rng = anneal(state, plan, rng, schedule)
            plan.block_until_ready()
            state = step_env(state, plan[0])
            depth = int(state.info["step"])
            if depth >= args.min_step and depth % args.stride == 0:
                bank.add(state)
                depths.append(depth)
            if float(state.done) > 0.5:
                # DIAL should not fall; if it does the run is not a clean source.
                falls += 1
                state = with_omega(
                    reset_env(jax.random.PRNGKey(seed + 800_000 + falls))
                )
                plan = jnp.zeros_like(plan)
                episode_step = 0
            else:
                plan = shift(plan)
                episode_step += 1
            progress.update()
        progress.set_postfix(banked=len(bank), falls=falls, refresh=False)
    progress.close()

    bank.save(args.out)
    depths = np.asarray(depths)
    print(f"banked {len(bank)} states from {args.episodes} DIAL episodes")
    print(f"DIAL falls during banking: {falls}"
          + ("  <- source is not clean" if falls else "  (clean)"))
    if len(depths):
        print(f"depth: min {depths.min()}  median {int(np.median(depths))}  "
              f"max {depths.max()}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
