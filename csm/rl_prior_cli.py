"""Train a robust stochastic Go2 locomotion prior with Soft Actor-Critic."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs as brax_envs
from brax import math
from brax.io import html
from brax.training.agents.sac import train as sac_train
from brax.training import checkpoint as brax_checkpoint
import orbax.checkpoint as ocp
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.exact_cli import _load_config, _physically_fallen
from csm.rl_prior import (
    RecoveryGo2PriorEnv,
    RecoveryPriorConfig,
    RLPriorPolicy,
    RobustGo2PriorEnv,
    RobustPriorRewardConfig,
    SafetyCritic,
    make_sac_network_factory,
)
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import load_dataclass_from_dict


def _parse_hidden_sizes(text: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must all be positive")
    return sizes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example")
    source.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("csm_runs/rl-prior"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-timesteps", type=int, default=20_000_000)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--num-evals", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--min-replay-size", type=int, default=32_768)
    parser.add_argument("--max-replay-size", type=int, default=1_000_000)
    parser.add_argument("--grad-updates-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward-scaling", type=float, default=0.1)
    parser.add_argument(
        "--hidden-sizes",
        type=_parse_hidden_sizes,
        default=(512, 512, 256),
    )
    parser.add_argument("--init-noise-std", type=float, default=0.3)
    parser.add_argument("--state-independent-std", action="store_true")
    parser.add_argument("--disable-layer-norm", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument(
        "--restore-checkpoint",
        type=Path,
        help="initialize actor and observation normalizer from a Brax SAC checkpoint",
    )
    parser.add_argument(
        "--recovery-prior",
        action="store_true",
        help="train through balanced nominal, tilted, and fallen reset states",
    )

    defaults = RobustPriorRewardConfig()
    for field_name in defaults.__dataclass_fields__:
        parser.add_argument(
            f"--reward-{field_name.replace('_', '-')}",
            type=float,
            default=getattr(defaults, field_name),
        )
    recovery_defaults = RecoveryPriorConfig()
    for field_name in recovery_defaults.__dataclass_fields__:
        parser.add_argument(
            f"--recovery-{field_name.replace('_', '-')}",
            type=type(getattr(recovery_defaults, field_name)),
            default=getattr(recovery_defaults, field_name),
        )
    parser.add_argument("--smoke", action="store_true")
    return parser


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "shape") and np.asarray(value).ndim == 0:
        return float(np.asarray(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_pmapped_restore_checkpoint(source: Path, destination: Path) -> Path:
    """Adds the local-device axis expected by Brax SAC restore.

    Brax saves unreplicated inference parameters but its SAC restore path
    inserts them directly into a pmapped TrainingState.  With one GPU, adding
    a leading singleton axis restores the representation expected internally.
    """
    params = brax_checkpoint.load(source)
    replicated = jax.tree_util.tree_map(lambda value: np.expand_dims(value, 0), params)
    destination.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(destination, replicated, force=True)
    return destination


def _evaluate_policy(env, policy, steps: int, seeds: int):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(policy.mode)
    torso = int(env._torso_idx) - 1
    records = []
    rendered = []
    for seed in range(seeds):
        state = reset(jax.random.PRNGKey(50_000 + seed))
        survived = 0
        tracking_errors = []
        tilts = []
        actions = []
        rewards = []
        trajectory = []
        for _ in range(steps):
            action = act(state.obs)
            state = step(state, action)
            state.reward.block_until_ready()
            if seed == 0:
                trajectory.append(state.pipeline_state)
            actions.append(np.asarray(action))
            rewards.append(float(state.reward))
            body_velocity = global_to_body_velocity(
                state.pipeline_state.xd.vel[torso],
                state.pipeline_state.x.rot[torso],
            )
            body_angular_velocity = global_to_body_velocity(
                state.pipeline_state.xd.ang[torso] * jnp.pi / 180.0,
                state.pipeline_state.x.rot[torso],
            )
            tracking_errors.append(
                float(
                    jnp.sum(
                        jnp.square(body_velocity[:2] - state.info["vel_tar"][:2])
                    )
                    + jnp.square(
                        body_angular_velocity[2] - state.info["ang_vel_tar"][2]
                    )
                )
            )
            tilts.append(
                float(
                    jnp.linalg.norm(
                        math.quat_to_euler(state.pipeline_state.x.rot[torso])[:2]
                    )
                )
            )
            survived += 1
            if _physically_fallen(env, state):
                break
        jerk = (
            float(np.mean(np.square(np.diff(np.asarray(actions), axis=0))))
            if len(actions) > 1
            else 0.0
        )
        records.append(
            {
                "seed": seed,
                "survived_steps": survived,
                "fell": survived < steps,
                "tracking_rmse": float(np.sqrt(np.mean(tracking_errors))),
                "mean_tilt": float(np.mean(tilts)),
                "action_jerk": jerk,
                "mean_reward": float(np.mean(rewards)),
            }
        )
        if seed == 0:
            rendered = trajectory
    return {
        "mean_survived_steps": float(np.mean([r["survived_steps"] for r in records])),
        "survival_rate": float(np.mean([not r["fell"] for r in records])),
        "tracking_rmse": float(np.mean([r["tracking_rmse"] for r in records])),
        "mean_tilt": float(np.mean([r["mean_tilt"] for r in records])),
        "action_jerk": float(np.mean([r["action_jerk"] for r in records])),
        "mean_reward": float(np.mean([r["mean_reward"] for r in records])),
        "rollouts": records,
    }, rendered


def _evaluate_recovery_policy(env, policy, steps: int, seeds: int):
    """Evaluates nominal, tilted and fallen resets without ending at a fall."""
    torso = int(env._prior_torso_idx)

    def rollout(key, mode):
        state = env.reset_with_mode(key, mode)
        start_position = state.pipeline_state.x.pos[torso]

        def one_step(current, unused):
            action = policy.mode(current.obs)
            next_state = env.step(current, action)
            pipeline = next_state.pipeline_state
            health, recovered, up_dot = env._health(pipeline)
            body_velocity = global_to_body_velocity(
                pipeline.xd.vel[torso], pipeline.x.rot[torso]
            )
            body_angular_velocity = global_to_body_velocity(
                pipeline.xd.ang[torso] * jnp.pi / 180.0,
                pipeline.x.rot[torso],
            )
            command = jnp.asarray(
                [
                    next_state.info["vel_tar"][0],
                    next_state.info["vel_tar"][1],
                    next_state.info["ang_vel_tar"][2],
                ]
            )
            actual = jnp.asarray(
                [body_velocity[0], body_velocity[1], body_angular_velocity[2]]
            )
            return next_state, (
                health,
                recovered,
                up_dot,
                pipeline.x.pos[torso, 2],
                command,
                actual,
                action,
                next_state.reward,
                pipeline.x.pos[torso],
            )

        _, outputs = jax.lax.scan(one_step, state, None, length=steps)
        return (start_position, *outputs)

    mode_names = ("nominal", "tilted", "fallen")
    keys = jax.random.split(jax.random.PRNGKey(73_000), len(mode_names) * seeds)
    modes = jnp.repeat(jnp.arange(len(mode_names)), seeds)
    batched = jax.jit(jax.vmap(rollout))(keys, modes)
    batched = jax.tree_util.tree_map(np.asarray, batched)
    (
        starts,
        health,
        recovered,
        up_dot,
        heights,
        commands,
        actual,
        actions,
        rewards,
        positions,
    ) = batched

    scenarios = {}
    for mode, name in enumerate(mode_names):
        selection = slice(mode * seeds, (mode + 1) * seeds)
        mode_recovered = recovered[selection]
        recovery_times = []
        sustained = []
        for trace in mode_recovered:
            consecutive = np.convolve(trace, np.ones(25), mode="valid")
            indices = np.flatnonzero(consecutive >= 25)
            recovery_times.append(int(indices[0] + 24) if len(indices) else steps)
            sustained.append(bool(np.mean(trace[-min(100, steps) :]) >= 0.9))
        mode_commands = commands[selection]
        mode_actual = actual[selection]
        tracking_rmse = np.sqrt(
            np.mean(np.square(mode_commands - mode_actual), axis=(1, 2))
        )
        displacement = (
            positions[selection, -1, :2] - starts[selection, :2]
        )
        scenarios[name] = {
            "sustained_recovery_rate": float(np.mean(sustained)),
            "mean_recovery_steps": float(np.mean(recovery_times)),
            "mean_health": float(np.mean(health[selection])),
            "mean_up_dot": float(np.mean(up_dot[selection])),
            "mean_height": float(np.mean(heights[selection])),
            "tracking_rmse": float(np.mean(tracking_rmse)),
            "mean_speed": float(
                np.mean(np.linalg.norm(mode_actual[..., :2], axis=-1))
            ),
            "mean_displacement": float(np.mean(np.linalg.norm(displacement, axis=-1))),
            "action_jerk": float(
                np.mean(np.square(np.diff(actions[selection], axis=1)))
            ),
            "mean_reward": float(np.mean(rewards[selection])),
        }

    # Render one complete fallen-reset recovery trajectory.  This scan is
    # compiled and executed once instead of issuing thousands of Python calls.
    render_state = env.reset_with_mode(jax.random.PRNGKey(91_000), 2)

    def render_step(current, unused):
        next_state = env.step(current, policy.mode(current.obs))
        return next_state, next_state.pipeline_state

    _, rendered_tree = jax.jit(
        lambda state: jax.lax.scan(render_step, state, None, length=steps)
    )(render_state)
    jax.tree_util.tree_leaves(rendered_tree)[0].block_until_ready()
    rendered = [
        jax.tree_util.tree_map(lambda value: value[index], rendered_tree)
        for index in range(steps)
    ]
    return {"scenarios": scenarios}, rendered


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    if args.smoke:
        args.num_timesteps = 64
        args.episode_length = 8
        args.num_envs = 4
        args.num_eval_envs = 2
        args.num_evals = 1
        args.batch_size = 8
        args.min_replay_size = 8
        args.max_replay_size = 64
        args.hidden_sizes = (32, 32)
        args.eval_steps = 4
        args.eval_seeds = 1

    positive = {
        "num timesteps": args.num_timesteps,
        "episode length": args.episode_length,
        "num envs": args.num_envs,
        "num eval envs": args.num_eval_envs,
        "num evals": args.num_evals,
        "batch size": args.batch_size,
        "min replay size": args.min_replay_size,
        "max replay size": args.max_replay_size,
        "gradient updates per step": args.grad_updates_per_step,
        "evaluation steps": args.eval_steps,
        "evaluation seeds": args.eval_seeds,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_replay_size > args.max_replay_size:
        raise ValueError("min replay size cannot exceed max replay size")
    rounded_prefill = (
        (args.min_replay_size + args.num_envs - 1) // args.num_envs
    ) * args.num_envs
    if (
        args.min_replay_size >= args.num_timesteps
        or rounded_prefill > args.num_timesteps
    ):
        raise ValueError("replay prefill must be smaller than total timesteps")
    if args.init_noise_std <= 0.0:
        raise ValueError("initial noise std must be positive")
    if not 0.0 < args.discounting <= 1.0:
        raise ValueError("discounting must be in (0, 1]")
    if args.learning_rate <= 0.0 or args.reward_scaling <= 0.0:
        raise ValueError("learning rate and reward scaling must be positive")

    env_name = config["env_name"]
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(env_name), config, convert_list_to_array=True
    )
    if not env_config.randomize_tasks or not env_config.randomize_start_state:
        raise ValueError(
            "RL prior training requires command and start-state randomization"
        )
    if args.recovery_prior and hasattr(env_config, "enable_body_collisions"):
        env_config = dataclasses.replace(env_config, enable_body_collisions=True)
    base_env = brax_envs.get_environment(env_name, config=env_config)
    reward_values = {
        field_name: getattr(args, f"reward_{field_name}")
        for field_name in RobustPriorRewardConfig.__dataclass_fields__
    }
    temperatures = {
        key: value
        for key, value in reward_values.items()
        if key.endswith("temperature")
    }
    scales = {
        key: value for key, value in reward_values.items() if key not in temperatures
    }
    if any(value <= 0.0 for value in temperatures.values()):
        raise ValueError("all reward temperatures must be positive")
    if any(value < 0.0 for value in scales.values()):
        raise ValueError("all reward scales must be non-negative")
    reward_config = RobustPriorRewardConfig(**reward_values)
    recovery_values = {
        field_name: getattr(args, f"recovery_{field_name}")
        for field_name in RecoveryPriorConfig.__dataclass_fields__
    }
    if args.recovery_prior:
        probabilities = [
            recovery_values["nominal_reset_probability"],
            recovery_values["tilted_reset_probability"],
            recovery_values["fallen_reset_probability"],
        ]
        if any(value < 0.0 for value in probabilities) or sum(probabilities) <= 0.0:
            raise ValueError("recovery reset probabilities must be non-negative")
        if recovery_values["push_interval_steps"] < 1:
            raise ValueError("recovery push interval must be positive")
        if not 0.0 <= recovery_values["push_probability"] <= 1.0:
            raise ValueError("recovery push probability must be in [0, 1]")
        train_env = RecoveryGo2PriorEnv(
            base_env, RecoveryPriorConfig(**recovery_values)
        )
    else:
        train_env = RobustGo2PriorEnv(base_env, reward_config)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Orbax/TensorStore requires absolute checkpoint paths.
    run_dir = (args.output / f"{env_name}-{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    restore_checkpoint = None
    if args.restore_checkpoint is not None:
        restore_checkpoint = _make_pmapped_restore_checkpoint(
            args.restore_checkpoint.resolve(), run_dir / "restore_pmapped"
        )
    history = []
    bar = tqdm(total=args.num_timesteps, desc="SAC robust prior", unit="step")
    progress_steps = 0

    def progress(step_count, metrics):
        nonlocal progress_steps
        step_count = int(step_count)
        bar.update(max(0, step_count - progress_steps))
        progress_steps = step_count
        record = {"step": step_count, **_jsonable(dict(metrics))}
        history.append(record)
        (run_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        reward = record.get("eval/episode_reward")
        if reward is not None:
            bar.set_postfix(eval_reward=f"{reward:.3f}")

    state_dependent_std = not args.state_independent_std
    layer_norm = not args.disable_layer_norm
    network_factory = make_sac_network_factory(
        args.hidden_sizes,
        args.init_noise_std,
        state_dependent_std,
        layer_norm,
    )

    from brax.training import pmap as brax_pmap
    _captured_ts = [None]
    _orig_assert = brax_pmap.assert_is_replicated

    def _capturing_assert(ts):
        _captured_ts[0] = ts
        return _orig_assert(ts)

    brax_pmap.assert_is_replicated = _capturing_assert

    _, params, final_metrics = sac_train.train(
        environment=train_env,
        num_timesteps=args.num_timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        learning_rate=args.learning_rate,
        discounting=args.discounting,
        seed=args.seed,
        batch_size=args.batch_size,
        num_evals=args.num_evals,
        normalize_observations=True,
        reward_scaling=args.reward_scaling,
        min_replay_size=args.min_replay_size,
        max_replay_size=args.max_replay_size,
        grad_updates_per_step=args.grad_updates_per_step,
        deterministic_eval=True,
        network_factory=network_factory,
        progress_fn=progress,
        checkpoint_logdir=str(run_dir / "checkpoints"),
        restore_checkpoint_path=(
            str(restore_checkpoint) if restore_checkpoint is not None else None
        ),
        max_devices_per_host=1,
    )
    bar.update(max(0, args.num_timesteps - progress_steps))
    bar.close()

    brax_pmap.assert_is_replicated = _orig_assert

    policy = RLPriorPolicy(
        params=params,
        observation_size=int(base_env.observation_size),
        action_size=base_env.action_size,
        hidden_layer_sizes=tuple(args.hidden_sizes),
        init_noise_std=args.init_noise_std,
        state_dependent_std=state_dependent_std,
        layer_norm=layer_norm,
    )
    policy.save(run_dir / "prior_policy.pkl")
    loaded = RLPriorPolicy.load(run_dir / "prior_policy.pkl")
    reset_state = base_env.reset(jax.random.PRNGKey(args.seed + 91))
    test_action = loaded.mode(reset_state.obs)
    test_log_prob = loaded.log_prob(reset_state.obs, test_action)
    if test_action.shape != (base_env.action_size,):
        raise RuntimeError("saved RL prior produced an invalid action shape")
    if not bool(jnp.isfinite(test_log_prob)):
        raise RuntimeError("saved RL prior produced a non-finite log probability")

    if _captured_ts[0] is not None:
        ts = _captured_ts[0]
        target_q_params = jax.tree_util.tree_map(
            lambda x: x[0], ts.target_q_params,
        )
        critic = SafetyCritic(
            q_params=(params[0], target_q_params),
            observation_size=int(base_env.observation_size),
            action_size=base_env.action_size,
            hidden_layer_sizes=tuple(args.hidden_sizes),
            layer_norm=layer_norm,
        )
        critic.save(run_dir / "safety_critic.pkl")
        loaded_critic = SafetyCritic.load(run_dir / "safety_critic.pkl")
        test_q = loaded_critic.q_value(reset_state.obs, test_action)
        if not bool(jnp.isfinite(test_q)):
            raise RuntimeError("saved critic produced a non-finite Q-value")
        print(f"critic Q(s, mode(s)) = {float(test_q):.3f}")
    else:
        print("WARNING: could not capture Q-params from training state")

    if args.recovery_prior:
        evaluation, rendered = _evaluate_recovery_policy(
            train_env, loaded, args.eval_steps, args.eval_seeds
        )
    else:
        evaluation, rendered = _evaluate_policy(
            train_env, loaded, args.eval_steps, args.eval_seeds
        )
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    (run_dir / "visualization.html").write_text(
        html.render(base_env.sys, rendered), encoding="utf-8"
    )
    run_config = {
        **vars(args),
        "algorithm": "soft_actor_critic",
        "policy_distribution": "state_dependent_tanh_gaussian",
        "reward_config": reward_values,
        "recovery_config": recovery_values if args.recovery_prior else None,
        "environment_randomization": {
            key: value
            for key, value in config.items()
            if key == "randomize_tasks"
            or key == "randomize_start_state"
            or key.startswith("command_")
            or key.startswith("start_")
        },
        "final_training_metrics": _jsonable(final_metrics),
        "evaluation": evaluation,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(_jsonable(run_config), indent=2), encoding="utf-8"
    )
    print(f"policy={run_dir / 'prior_policy.pkl'}")
    if _captured_ts[0] is not None:
        print(f"critic={run_dir / 'safety_critic.pkl'}")
    print(f"evaluation={json.dumps(evaluation)}")
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
