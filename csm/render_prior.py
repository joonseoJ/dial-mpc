"""Render a saved RL prior policy and serve the visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs as brax_envs
from brax import math
from brax.io import html

import dial_mpc.envs as dial_envs
from csm.exact_cli import _load_config, _physically_fallen
from csm.rl_prior import (
    RecoveryGo2PriorEnv,
    RecoveryPriorConfig,
    RLPriorPolicy,
    RobustGo2PriorEnv,
    RobustPriorRewardConfig,
)
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import load_dataclass_from_dict


def main():
    parser = argparse.ArgumentParser(description="Render RL prior policy rollout")
    parser.add_argument("policy", type=Path, help="path to prior_policy.pkl")
    parser.add_argument("--example", default="unitree_go2_trot_randomized")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stochastic", action="store_true",
                        help="use stochastic sampling instead of deterministic mode")
    parser.add_argument("--recovery", action="store_true",
                        help="wrap environment with RecoveryGo2PriorEnv")
    parser.add_argument("--reset-mode", type=int, default=0, choices=[0, 1, 2],
                        help="0=nominal, 1=tilted, 2=fallen (only with --recovery)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP port for serving the result")
    parser.add_argument("--no-serve", action="store_true")
    args = parser.parse_args()

    policy = RLPriorPolicy.load(args.policy)
    print(f"Loaded policy: obs={policy.observation_size}, act={policy.action_size}")
    print(f"  hidden={policy.hidden_layer_sizes}, noise_std={policy.init_noise_std}")

    config = _load_config(argparse.Namespace(
        example=args.example, config=None, smoke=False
    ))
    env_name = config["env_name"]
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(env_name), config, convert_list_to_array=True
    )
    base_env = brax_envs.get_environment(env_name, config=env_config)

    if args.recovery:
        env = RecoveryGo2PriorEnv(base_env)
    else:
        env = RobustGo2PriorEnv(base_env)

    if args.recovery:
        state = jax.jit(env.reset_with_mode)(
            jax.random.PRNGKey(args.seed), args.reset_mode
        )
    else:
        state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))

    act_fn = jax.jit(policy.sample if args.stochastic else policy.mode)
    step_fn = jax.jit(env.step)
    torso = int(base_env._torso_idx) - 1

    trajectory = []
    rewards = []
    tracking_errors = []
    heights = []

    rng = jax.random.PRNGKey(args.seed + 1000)
    print(f"Rolling out {args.steps} steps (seed={args.seed}, "
          f"{'stochastic' if args.stochastic else 'deterministic'})...")

    for t in range(args.steps):
        if args.stochastic:
            rng, act_key = jax.random.split(rng)
            action = act_fn(state.obs, act_key)
        else:
            action = act_fn(state.obs)
        state = step_fn(state, action)
        state.reward.block_until_ready()
        trajectory.append(state.pipeline_state)
        rewards.append(float(state.reward))
        heights.append(float(state.pipeline_state.x.pos[torso, 2]))

        body_vel = global_to_body_velocity(
            state.pipeline_state.xd.vel[torso],
            state.pipeline_state.x.rot[torso],
        )
        body_ang = global_to_body_velocity(
            state.pipeline_state.xd.ang[torso] * jnp.pi / 180.0,
            state.pipeline_state.x.rot[torso],
        )
        terr = float(
            jnp.sum(jnp.square(body_vel[:2] - state.info["vel_tar"][:2]))
            + jnp.square(body_ang[2] - state.info["ang_vel_tar"][2])
        )
        tracking_errors.append(terr)

        if not args.recovery and _physically_fallen(env, state):
            print(f"  Fell at step {t+1}")
            break

        if (t + 1) % 200 == 0:
            print(f"  step {t+1}: reward={rewards[-1]:.3f}, "
                  f"height={heights[-1]:.3f}, "
                  f"tracking_err={tracking_errors[-1]:.4f}")

    print(f"\nSummary ({len(trajectory)} steps):")
    print(f"  Mean reward:      {np.mean(rewards):.3f}")
    print(f"  Tracking RMSE:    {np.sqrt(np.mean(tracking_errors)):.3f}")
    print(f"  Mean height:      {np.mean(heights):.3f}")
    print(f"  Min height:       {np.min(heights):.3f}")
    print(f"  Survived:         {len(trajectory)}/{args.steps}")

    output_path = args.output or Path(f"csm_runs/render_seed{args.seed}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html.render(base_env.sys, trajectory))
    print(f"\nVisualization saved to: {output_path}")

    if not args.no_serve:
        import http.server
        import os
        os.chdir(str(output_path.parent))
        filename = output_path.name
        print(f"\nServing on http://0.0.0.0:{args.port}/{filename}")
        print("Press Ctrl+C to stop.")
        handler = http.server.SimpleHTTPRequestHandler
        server = http.server.HTTPServer(("0.0.0.0", args.port), handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
