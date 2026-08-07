"""Compare DIAL-MPC Go2 experts under several objective-weight vectors."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax import envs as brax_envs
from brax import math
from brax.io import html

import dial_mpc.envs as dial_envs
from csm.envs import get_objective_spec
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


DEFAULT_WEIGHTS = ((2.0, 2.0, 1.0), (2.0, 1.0, 2.0), (1.0, 2.0, 2.0))


def _parse_weights(text: str) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(
        tuple(float(value) for value in row.split(","))
        for row in text.split(";")
    )
    if not rows or any(len(row) != 3 for row in rows):
        raise argparse.ArgumentTypeError(
            "expected semicolon-separated tracking,stability,gait triples"
        )
    return rows


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _metrics(records: dict[str, list], dt: float, warmup_steps: int) -> dict:
    all_position = np.asarray(records["position"])
    all_euler = np.asarray(records["euler"])
    all_done = np.asarray(records["done"])
    all_tilt = np.linalg.norm(all_euler[:, :2], axis=1)
    physical_fall = (all_position[:, 2] < 0.18) | (all_tilt > np.pi / 2.0)
    fall_indices = np.flatnonzero(physical_fall)
    done_indices = np.flatnonzero(all_done > 0.5)
    start = min(warmup_steps, max(len(records["velocity"]) - 3, 0))
    velocity = np.asarray(records["velocity"])[start:]
    target_velocity = np.asarray(records["target_velocity"])[start:]
    position = np.asarray(records["position"])[start:]
    euler = np.asarray(records["euler"])[start:]
    angular_velocity = np.asarray(records["angular_velocity"])[start:]
    actions = np.asarray(records["action"])[start:]
    controls = np.asarray(records["control"])[start:]
    costs = np.asarray(records["cost"])[start:]
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    action_rate = np.diff(actions, axis=0) / dt

    return {
        "evaluated_seconds": float(len(velocity) * dt),
        "forward_distance_m": float(position[-1, 0] - position[0, 0]),
        "forward_speed_mean_mps": float(np.mean(velocity[:, 0])),
        "forward_speed_std_mps": float(np.std(velocity[:, 0])),
        "forward_tracking_rmse_mps": _rms(
            velocity[:, 0] - target_velocity[:, 0]
        ),
        "planar_tracking_rmse_mps": _rms(
            velocity[:, :2] - target_velocity[:, :2]
        ),
        "lateral_speed_rms_mps": _rms(velocity[:, 1]),
        "height_rmse_m": _rms(position[:, 2] - 0.3),
        "minimum_height_m": float(np.min(all_position[:, 2])),
        "tilt_rms_rad": _rms(euler[:, :2]),
        "maximum_tilt_rad": float(np.max(all_tilt)),
        "yaw_rate_rms_radps": _rms(angular_velocity[:, 2]),
        "acceleration_rms_mps2": _rms(acceleration),
        "jerk_rms_mps3": _rms(jerk),
        "action_rate_rms_per_s": _rms(action_rate),
        "joint_torque_rms_nm": _rms(controls),
        "tracking_cost_mean": float(np.mean(costs[:, 0])),
        "stability_cost_mean": float(np.mean(costs[:, 1])),
        "gait_cost_mean": float(np.mean(costs[:, 2])),
        "fell": bool(fall_indices.size),
        "fall_time_s": (
            float((fall_indices[0] + 1) * dt) if fall_indices.size else None
        ),
        "env_done_fraction": float(np.mean(all_done > 0.5)),
        "first_env_done_s": (
            float((done_indices[0] + 1) * dt) if done_indices.size else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="unitree_go2_trot")
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        default=DEFAULT_WEIGHTS,
        help='weight rows, e.g. "2,2,1;2,1,2;1,2,2"',
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("csm_runs/expert_benchmark"))
    parser.add_argument("--no-html", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    with open(get_example_path(args.example + ".yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dial_config = load_dataclass_from_dict(DialConfig, config)
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(dial_config.env_name),
        config,
        convert_list_to_array=True,
    )
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    objective_spec = get_objective_spec(env)
    if objective_spec.num_objectives != 3:
        raise ValueError(f"expected three Go2 objectives, got {objective_spec.names}")

    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    controller = MBDPI(dial_config, env)
    torso_idx = int(env._torso_idx) - 1

    def reverse_scan(carry, factor):
        rng, plan, state = carry
        rng, plan, info = controller.reverse_once(state, rng, plan, factor)
        return (rng, plan, state), info

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode_idx, weight_tuple in enumerate(args.weights):
        weight = jnp.asarray(weight_tuple, dtype=jnp.float32)
        reset_key = jax.random.PRNGKey(args.seed)
        state = reset_env(reset_key)
        state = state.replace(info={**state.info, "reward_weights": weight})
        rng = jax.random.PRNGKey(args.seed + 1)
        plan = jnp.zeros((dial_config.Hnode + 1, env.action_size))
        records = {name: [] for name in (
            "velocity", "target_velocity", "position", "euler",
            "angular_velocity", "action", "control", "cost", "done"
        )}
        rollout = []
        planning_times = []

        print(f"mode {mode_idx + 1}/{len(args.weights)} weights={weight_tuple}")
        for step_idx in range(args.steps):
            action = plan[0]
            state = step_env(state, action)
            pipeline = state.pipeline_state
            rotation = pipeline.x.rot[torso_idx]
            records["velocity"].append(pipeline.xd.vel[torso_idx])
            records["target_velocity"].append(state.info["vel_tar"])
            records["position"].append(pipeline.x.pos[torso_idx])
            records["euler"].append(math.quat_to_euler(rotation))
            records["angular_velocity"].append(
                pipeline.xd.ang[torso_idx] * jnp.pi / 180.0
            )
            records["action"].append(action)
            records["control"].append(pipeline.ctrl)
            records["cost"].append(objective_spec.cost(state, action))
            records["done"].append(state.done)
            rollout.append(pipeline)

            plan = controller.shift(plan)
            n_diffuse = dial_config.Ndiffuse_init if step_idx == 0 else dial_config.Ndiffuse
            factors = controller.sigma_control * (
                dial_config.traj_diffuse_factor ** jnp.arange(n_diffuse)
            )[:, None]
            started = time.perf_counter()
            (rng, plan, _), _ = jax.lax.scan(
                reverse_scan, (rng, plan, state), factors
            )
            plan.block_until_ready()
            planning_times.append(time.perf_counter() - started)
            if (step_idx + 1) % 50 == 0:
                print(f"  step {step_idx + 1}/{args.steps}")

        host_records = {
            name: [np.asarray(value) for value in values]
            for name, values in records.items()
        }
        report = {
            "mode": mode_idx + 1,
            "weights": list(weight_tuple),
            **_metrics(host_records, env.dt, args.warmup_steps),
            # The first calls for both Ndiffuse_init and Ndiffuse include JIT
            # compilation; exclude them from steady-state latency metrics.
            "planning_time_mean_ms": float(1e3 * np.mean(planning_times[2:])),
            "planning_time_p95_ms": float(
                1e3 * np.percentile(planning_times[2:], 95)
            ),
        }
        reports.append(report)
        if not args.no_html:
            rendered = html.render(
                env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 720, True
            )
            (args.output / f"mode_{mode_idx + 1}.html").write_text(
                rendered, encoding="utf-8"
            )

    (args.output / "metrics.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    with open(args.output / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=reports[0].keys())
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps(reports, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
