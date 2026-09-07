"""Reinforcement learning at one fixed weight: the per-omega baseline.

Compositional score matching claims a basis trained once serves every weight in
the cone.  The controller that claim has to beat is not DIAL -- DIAL is the
teacher and wins on quality by construction -- but the obvious alternative:
train a policy with RL at the weight you actually want.  That baseline is
cheap to state and expensive to run, and the whole argument turns on the second
half, so it has to be run honestly.

Two things make it the *same* objective the student and DIAL are scored on.

**The weight.**  `unitree_go2_walk` already computes the objective as
`normalize_omega(info["reward_weights"]) . reward_components`, so pinning omega
needs no new reward code at all -- it is a config field.  The row normalisers
have to come along with it, because two policies fitted under different ones
cannot be compared by cost.

**The horizon.**  Every row is a negative squared error, and `reward_alive` is
computed and then left out of `reward_components`, so `omega . C <= 0`
everywhere.  Paired with a `done` that fires when the torso drops, that makes
falling *optimal*: terminating cuts the bootstrap and stops the bleeding, and a
value-based learner will find that long before it finds walking.  Neither of
the other two arms is allowed that move -- `compose_walk_eval.summarise` scores
the full horizon and the environment keeps stepping after `done`, so lying on
the ground is priced, and DIAL plans a fixed H.  Zeroing `done` is what puts
all three on the same objective.

`--terminate` keeps the termination, which is worth running once: it is the
naive setup, and watching it converge to falling on purpose is the cleanest
demonstration of why the fixed horizon is not a favour done to the baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import time
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs as brax_envs
from brax.envs.base import State, Wrapper
from brax.training.acme import running_statistics

import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from csm.basis_screen import _load_config, build_omegas
from csm.omega import normalize_omega_np


class FixedHorizonWrapper(Wrapper):
    """Hide the environment's own termination from the learner.

    Only the flag is suppressed.  The physics still runs, the torso still hits
    the ground and the objective still charges for it every step -- which is
    the point.  A policy that falls here keeps paying, exactly as it does in
    the evaluation.
    """

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state.replace(done=jnp.zeros_like(state.done))

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        return state.replace(done=jnp.zeros_like(state.done))


def resolve_omega(text: str, n_rows: int) -> np.ndarray:
    catalogue = build_omegas(n_rows)
    if text in catalogue:
        return catalogue[text]
    return normalize_omega_np(np.array([float(v) for v in text.split(",")]))


def build_env(example: str, omega, *, terminate: bool, randomize_start: bool,
              row_scales=None):
    dial_config, env_config = _load_config(example, None)
    if row_scales:
        env_config = dataclasses.replace(
            env_config, track_scale=row_scales[0],
            stability_scale=row_scales[1], gait_scale=row_scales[2])
    env_config = dataclasses.replace(
        env_config,
        reward_weights=jnp.asarray(omega, dtype=jnp.float32),
        randomize_start_state=bool(randomize_start),
    )
    # The student is conditioned on whatever the task varies, so the baseline
    # has to see the same variation or it is answering an easier question.
    # What varies differs by task: the walking environments randomise the
    # velocity command, and the stand-and-resist one randomises the push at
    # reset and has no command at all.
    varies = bool(getattr(env_config, "randomize_tasks", False)) or (
        float(getattr(env_config, "push_linear_velocity", 0.0)) > 0.0
    )
    if not varies:
        raise ValueError(
            "this configuration varies nothing between episodes -- neither a "
            "randomised command nor a push -- so a baseline trained on it "
            "would face one fixed problem the student never does"
        )
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    return (env if terminate else FixedHorizonWrapper(env)), dial_config, env_config


def train(args, env):
    """PPO or SAC, returning `(make_inference_fn, params, metrics)`."""

    progress = []

    def report(step, metrics):
        entry = {"step": int(step),
                 **{k: float(v) for k, v in metrics.items()
                    if np.ndim(v) == 0}}
        progress.append(entry)
        reward = entry.get("eval/episode_reward")
        print(f"[{args.algo}] {int(step):>10} steps  "
              f"reward {reward if reward is None else round(reward, 4)}",
              flush=True)

    hidden = tuple(int(v) for v in args.hidden.split(","))
    if args.algo == "ppo":
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo_train

        factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=hidden,
            value_hidden_layer_sizes=hidden,
        )
        fn, params, metrics = ppo_train.train(
            environment=env,
            num_timesteps=args.num_timesteps,
            episode_length=args.episode_length,
            num_envs=args.num_envs,
            batch_size=args.batch_size,
            unroll_length=args.unroll_length,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            num_evals=args.num_evals,
            learning_rate=args.learning_rate,
            entropy_cost=args.entropy_cost,
            discounting=args.discounting,
            reward_scaling=args.reward_scaling,
            normalize_observations=True,
            # Not `bootstrap_on_timeout`: that path wants the environment to
            # set `info['time_out']` itself, and brax's own `EpisodeWrapper`
            # sets `truncation` instead, which the GAE in `losses.py` already
            # masks on.  With `done` suppressed every episode ends by
            # truncation, so this is the only mechanism that matters here --
            # asking for the other one raises `KeyError: 'time_out'`.
            network_factory=factory,
            seed=args.seed,
            progress_fn=report,
        )
    else:
        from brax.training.agents.sac import networks as sac_networks
        from brax.training.agents.sac import train as sac_train

        factory = functools.partial(
            sac_networks.make_sac_networks,
            hidden_layer_sizes=hidden,
        )
        fn, params, metrics = sac_train.train(
            environment=env,
            num_timesteps=args.num_timesteps,
            episode_length=args.episode_length,
            num_envs=args.num_envs,
            batch_size=args.batch_size,
            num_evals=args.num_evals,
            learning_rate=args.learning_rate,
            discounting=args.discounting,
            reward_scaling=args.reward_scaling,
            min_replay_size=args.min_replay_size,
            max_replay_size=args.max_replay_size,
            grad_updates_per_step=args.grad_updates_per_step,
            normalize_observations=True,
            network_factory=factory,
            seed=args.seed,
            progress_fn=report,
        )
    return fn, params, metrics, progress


def save_policy(path: Path, *, algo, hidden, params, obs_size, act_size,
                omega, omega_name=None, temperature=None):
    """Enough to rebuild the deterministic policy without the trainer.

    The parameters alone are not a policy: the observation normaliser's
    statistics are learned too, and a policy restored without them acts on
    inputs it never saw.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        cloudpickle.dump({
            "algo": algo, "hidden": tuple(hidden), "params": params,
            "observation_size": int(obs_size), "action_size": int(act_size),
            "omega": np.asarray(omega, dtype=float).tolist(),
            # The name as well as the vector, so the evaluation labels the row
            # the way every other arm's table does.
            "omega_name": omega_name,
            "temperature": temperature,
        }, handle)


def load_policy(path):
    """The saved policy as `action = f(obs)`, deterministic."""

    with open(path, "rb") as handle:
        blob = cloudpickle.load(handle)
    normalize = running_statistics.normalize
    if blob["algo"] == "ppo":
        from brax.training.agents.ppo import networks as ppo_networks
        nets = ppo_networks.make_ppo_networks(
            blob["observation_size"], blob["action_size"],
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=blob["hidden"],
            value_hidden_layer_sizes=blob["hidden"],
        )
        make_inference = ppo_networks.make_inference_fn(nets)
    else:
        from brax.training.agents.sac import networks as sac_networks
        nets = sac_networks.make_sac_networks(
            blob["observation_size"], blob["action_size"],
            preprocess_observations_fn=normalize,
            hidden_layer_sizes=blob["hidden"],
        )
        make_inference = sac_networks.make_inference_fn(nets)
    inference = make_inference(blob["params"], deterministic=True)
    return inference, blob


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example", default="unitree_go2_trot_csm")
    p.add_argument("--omega", default="uniform",
                   help="a name from the weight catalogue, or `a,b,c`")
    p.add_argument("--row-scales", type=float, nargs="+", default=None)
    p.add_argument("--algo", choices=("ppo", "sac"), default="ppo")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--terminate", action="store_true",
                   help="keep the environment's fall termination; the naive "
                        "setup, in which falling is the optimal policy")
    p.add_argument("--randomize-start", action="store_true",
                   help="widen the reset distribution.  Leaves the objective "
                        "alone, so the steelman arm may use it")
    p.add_argument("--num-timesteps", type=int, default=100_000_000)
    p.add_argument("--episode-length", type=int, default=1500)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--num-evals", type=int, default=21)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--num-updates-per-batch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-cost", type=float, default=1e-2)
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--min-replay-size", type=int, default=32768)
    p.add_argument("--max-replay-size", type=int, default=1_000_000)
    p.add_argument("--grad-updates-per-step", type=int, default=1)
    p.add_argument("--hidden", default="512,256,128")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.smoke:
        args.num_timesteps = 8192
        args.episode_length = 32
        args.num_envs = 16
        args.num_evals = 1
        args.batch_size = 64
        args.unroll_length = 8
        args.num_minibatches = 2
        args.num_updates_per_batch = 1
        args.min_replay_size = 64
        args.max_replay_size = 4096
        args.hidden = "32,32"

    _, probe_config = _load_config(args.example, None)
    n_rows = int(np.asarray(probe_config.reward_weights).shape[0])
    omega = resolve_omega(args.omega, n_rows)
    env, dial_config, env_config = build_env(
        args.example, omega, terminate=args.terminate,
        randomize_start=args.randomize_start, row_scales=args.row_scales)

    print(f"env {dial_config.env_name}  algo {args.algo}  "
          f"omega {args.omega} = {np.round(omega, 4).tolist()}")
    print(f"row_scales {[env_config.track_scale, env_config.stability_scale, env_config.gait_scale]}  "
          f"terminate {args.terminate}  randomize_start {args.randomize_start}")
    print(f"{args.num_timesteps:,} steps, {args.num_envs} envs, "
          f"episode {args.episode_length}")

    started = time.time()
    _, params, metrics, progress = train(args, env)
    seconds = time.time() - started

    args.output.mkdir(parents=True, exist_ok=True)
    save_policy(args.output / "policy.pkl", algo=args.algo,
                hidden=tuple(int(v) for v in args.hidden.split(",")),
                params=params, obs_size=env.observation_size,
                act_size=env.action_size, omega=omega,
                omega_name=args.omega)
    report = {
        "algo": args.algo, "omega_name": args.omega,
        "omega": np.asarray(omega, dtype=float).tolist(),
        "example": args.example,
        "terminate": bool(args.terminate),
        "randomize_start": bool(args.randomize_start),
        "row_scales": [float(env_config.track_scale),
                       float(env_config.stability_scale),
                       float(env_config.gait_scale)],
        "num_timesteps": int(args.num_timesteps),
        "episode_length": int(args.episode_length),
        "seconds": seconds,
        "final_metrics": {k: float(v) for k, v in metrics.items()
                          if np.ndim(v) == 0},
        "progress": progress,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\ntrained in {seconds / 60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
