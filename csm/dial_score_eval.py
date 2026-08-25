"""Verify a trained single-weight DIAL score policy against DIAL-MPC itself.

The training run already reports validation metrics on its own dataset.  That is
not verification: it says the network fits the queries the teacher was asked
about, not that the denoising controller behaves like DIAL when it drives the
robot.  This tool answers the three questions that matter.

1. **Closed loop** — run the student and real DIAL-MPC from the *same* reset
   seeds with the same control-loop ordering, and compare mean reward, survival,
   and termination count.  This is the verdict.
2. **On-policy score agreement** — at states the *student* actually visits, ask
   the exact teacher what the update should have been and compare.  Training
   metrics are measured on teacher-visited states, so this is the number that
   exposes distribution shift.
3. **Contractivity** — a genuine score field should be stable under extra
   annealing passes.  If the plan keeps moving as passes are added, the learned
   field is not a converged denoiser and the rollout will drift.

Both rollouts are rendered so they can be watched side by side.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax import envs as brax_envs
from brax.io import html
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.dial_score import (
    DialScorePolicy,
    DialScoreTeacher,
    build_shift_matrix,
    dial_factors,
    factor_to_t,
)
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run",
        type=Path,
        help="training run directory holding policy.pkl and report.json",
    )
    target.add_argument("--policy", type=Path, help="path to a policy.pkl")
    parser.add_argument(
        "--example",
        default=None,
        help="DIAL example name; defaults to the one recorded in report.json",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="reset seeds evaluated; both controllers see the same ones",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1000,
        help="reset seeds are seed_offset + episode index; keep it away from "
        "the training seed so these are unseen initial states",
    )
    parser.add_argument(
        "--init-passes",
        type=int,
        default=None,
        help="annealing repeats on the first control step; defaults to the "
        "value the policy was trained with",
    )
    parser.add_argument(
        "--temp-sample",
        type=float,
        default=None,
        help="teacher temperature; defaults to the training value so the "
        "baseline is the controller the student was actually imitating",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--omega",
        type=str,
        default=None,
        help="reward weight vector; defaults to the training value",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help=(
            "override the physics timestep.  The control rate is unchanged, so "
            "this only subdivides each control step into more physics steps; "
            "the PD torque is still computed once per control step and held "
            "across them.  Use it to evaluate a policy under a modified plant"
        ),
    )
    parser.add_argument(
        "--start-noise",
        type=float,
        default=0.0,
        help=(
            "enable randomized initial states at this scale.  The stock Go2 "
            "config has randomize_start_state=False and a fixed command, so "
            "reset ignores its seed and the deterministic denoising policy "
            "produces the *same* trajectory for every episode; without this "
            "flag --episodes only varies DIAL's sampling"
        ),
    )
    parser.add_argument(
        "--randomize-command",
        action="store_true",
        help=(
            "resample the velocity command, matching a run trained with "
            "--randomize-command.  Note that mean reward is a poor metric in "
            "this regime: the command ramps up from zero after every reset, so "
            "a controller that falls often collects more cheap near-zero-error "
            "steps.  Read terminations and steps_before_first_termination"
        ),
    )
    parser.add_argument(
        "--score-check",
        type=int,
        default=40,
        help="student-visited states where the exact teacher is queried; "
        "0 disables the on-policy score comparison",
    )
    parser.add_argument(
        "--score-check-repeats",
        type=int,
        default=8,
        help="teacher repeats per on-policy score query",
    )
    parser.add_argument(
        "--contractivity-passes",
        type=int,
        default=8,
        help="extra annealing passes used for the stability check",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser


def _resolve_sources(args) -> tuple[Path, dict]:
    if args.run is not None:
        policy_path = args.run / "policy.pkl"
        if not policy_path.exists():
            raise FileNotFoundError(f"{policy_path} does not exist")
        report_path = args.run / "report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {}
        )
        return policy_path, report
    report_path = args.policy.parent / "report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    return args.policy, report


def _load_config(args, report: dict) -> dict:
    if args.config is not None:
        path = args.config
    else:
        example = args.example or report.get("arguments", {}).get("example")
        if example is None:
            raise ValueError(
                "could not infer the environment; pass --example or --config"
            )
        path = get_example_path(str(example) + ".yaml")
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class JointLimitWatch:
    """Counts joint-limit contacts without ending the episode.

    With ``terminate_on_joint_limit`` off these no longer stop a rollout, but
    how close a controller runs to its limits is still worth reporting: it is
    the difference between walking with margin and walking on the edge.
    """

    def __init__(self, env) -> None:
        self.low = np.asarray(env.joint_range[:, 0])
        self.high = np.asarray(env.joint_range[:, 1])
        self.contacts = 0
        self.worst = 0.0
        self.margin = float("inf")

    def observe(self, pipeline_state) -> None:
        q = np.asarray(pipeline_state.q[7:])
        over = float(max(np.max(q - self.high), np.max(self.low - q)))
        self.margin = min(self.margin, -over)
        if over > 0.0:
            self.contacts += 1
            self.worst = max(self.worst, over)

    def summary(self) -> dict[str, float]:
        return {
            "joint_limit_contacts": self.contacts,
            "worst_joint_overshoot": self.worst,
            "min_joint_margin": (
                None if not np.isfinite(self.margin) else self.margin
            ),
        }


class RolloutResult:
    """Per-episode rollout summary plus the states needed for rendering."""

    def __init__(
        self,
        rewards: list[float],
        terminations: list[int],
        limits: "JointLimitWatch | None" = None,
    ) -> None:
        self.rewards = rewards
        self.terminations = terminations
        self.limits = limits

    def summary(self) -> dict[str, float]:
        steps = len(self.rewards)
        first = self.terminations[0] if self.terminations else steps
        extra = self.limits.summary() if self.limits is not None else {}
        return {
            **extra,
            "mean_reward": float(np.mean(self.rewards)),
            "total_reward": float(np.sum(self.rewards)),
            "steps": steps,
            "terminations": len(self.terminations),
            "steps_before_first_termination": int(first),
        }


def run_student(
    env,
    policy: DialScorePolicy,
    teacher: DialScoreTeacher,
    horizon: int,
    seed: int,
    steps: int,
    init_passes: int,
    collect_states: bool,
) -> tuple[RolloutResult, list, list[tuple[int, jax.Array, jax.Array, int]]]:
    """Run the denoising controller; also log visited annealing queries."""

    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    apply_first = jax.jit(lambda plan, obs: policy.apply(plan, obs, init_passes))
    apply_step = jax.jit(lambda plan, obs: policy.apply(plan, obs, 1))
    shift_plan = jax.jit(policy.shift)

    state = teacher.with_reward_weights(reset_env(jax.random.PRNGKey(seed)))
    plan = jnp.zeros((horizon, int(env.action_size)), dtype=jnp.float32)
    rewards: list[float] = []
    terminations: list[int] = []
    states = []
    visited: list[tuple[int, jax.Array, jax.Array, int]] = []
    limits = JointLimitWatch(env)

    # An episode-local counter, so a restart after a fall gets the same warm-up
    # annealing budget as step 0 instead of a single pass from a zero plan.
    episode_step = 0
    for step_idx in tqdm(
        range(steps), desc=f"student seed {seed}", unit="step", dynamic_ncols=True
    ):
        # Log the query the student is about to refine, at the coarsest level.
        visited.append((step_idx, plan, state, 0))
        plan = (apply_first if episode_step == 0 else apply_step)(plan, state.obs)
        plan.block_until_ready()
        state = step_env(state, plan[0])
        rewards.append(float(state.reward))
        limits.observe(state.pipeline_state)
        if collect_states:
            states.append(state.pipeline_state)
        if float(state.done) > 0.5:
            terminations.append(step_idx)
            state = teacher.with_reward_weights(
                reset_env(jax.random.PRNGKey(seed + 10_000 + len(terminations)))
            )
            plan = jnp.zeros_like(plan)
            episode_step = 0
        else:
            plan = shift_plan(plan)
            episode_step += 1

    return RolloutResult(rewards, terminations, limits), states, visited


def run_dial(
    env,
    planner,
    teacher: DialScoreTeacher,
    factors: jax.Array,
    shift_matrix: jax.Array,
    horizon: int,
    seed: int,
    steps: int,
    init_passes: int,
    collect_states: bool,
) -> tuple[RolloutResult, list]:
    """Run real DIAL-MPC with the student's control-loop ordering."""

    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    shift_plan = jax.jit(lambda plan: jnp.einsum("ij,ja->ia", shift_matrix, plan))
    sigma_control = jnp.asarray(planner.sigma_control, dtype=jnp.float32)

    @jax.jit
    def anneal(state, plan, rng, schedule):
        def body(carry, factor):
            plan, rng = carry
            rng, call_rng = jax.random.split(rng)
            _, plan, _ = planner.reverse_once(
                state, call_rng, plan, sigma_control * factor
            )
            return (plan, rng), None

        (plan, rng), _ = jax.lax.scan(body, (plan, rng), schedule)
        return plan, rng

    state = teacher.with_reward_weights(reset_env(jax.random.PRNGKey(seed)))
    plan = jnp.zeros((horizon, int(env.action_size)), dtype=jnp.float32)
    rng = jax.random.PRNGKey(seed + 500_000)
    rewards: list[float] = []
    terminations: list[int] = []
    states = []
    limits = JointLimitWatch(env)

    first_schedule = jnp.tile(factors, init_passes)
    episode_step = 0
    for step_idx in tqdm(
        range(steps), desc=f"dial seed {seed}", unit="step", dynamic_ncols=True
    ):
        schedule = first_schedule if episode_step == 0 else factors
        plan, rng = anneal(state, plan, rng, schedule)
        plan.block_until_ready()
        state = step_env(state, plan[0])
        rewards.append(float(state.reward))
        limits.observe(state.pipeline_state)
        if collect_states:
            states.append(state.pipeline_state)
        if float(state.done) > 0.5:
            terminations.append(step_idx)
            state = teacher.with_reward_weights(
                reset_env(jax.random.PRNGKey(seed + 10_000 + len(terminations)))
            )
            plan = jnp.zeros_like(plan)
            episode_step = 0
        else:
            plan = shift_plan(plan)
            episode_step += 1

    return RolloutResult(rewards, terminations, limits), states


def check_on_policy_score(
    policy: DialScorePolicy,
    teacher: DialScoreTeacher,
    visited: list,
    factors: jax.Array,
    num_checks: int,
    rng: jax.Array,
) -> dict[str, float]:
    """Compare student and exact-teacher updates at student-visited states."""

    if num_checks < 1 or not visited:
        return {}
    stride = max(len(visited) // num_checks, 1)
    picked = visited[::stride][:num_checks]
    factor_min = float(jnp.min(factors))
    factor_max = float(jnp.max(factors))

    errors: list[float] = []
    targets: list[float] = []
    cosines: list[float] = []
    per_level: dict[int, list[float]] = {}
    for _, plan, state, _ in tqdm(
        picked, desc="on-policy score check", unit="query", dynamic_ncols=True
    ):
        for level in range(int(factors.shape[0])):
            factor = float(factors[level])
            t = factor_to_t(factor, factor_min, factor_max).reshape(1)
            rng, query_rng = jax.random.split(rng)
            target, _, _ = teacher.targets(state, plan, factor, query_rng)
            predicted = policy.delta(plan, state.obs, t)
            error = float(jnp.mean(jnp.square(predicted - target)))
            power = float(jnp.mean(jnp.square(target)))
            cosine = float(
                jnp.sum(predicted * target)
                / (
                    jnp.linalg.norm(predicted) * jnp.linalg.norm(target)
                    + 1e-12
                )
            )
            errors.append(error)
            targets.append(power)
            cosines.append(cosine)
            per_level.setdefault(level, []).append(cosine)

    result = {
        "queries": len(errors),
        "relative_rms": float(
            np.sqrt(np.mean(errors) / (np.mean(targets) + 1e-12))
        ),
        "cosine": float(np.mean(cosines)),
    }
    for level, values in per_level.items():
        result[f"cosine_level_{level}"] = float(np.mean(values))
    return result


def check_contractivity(
    policy: DialScorePolicy,
    visited: list,
    extra_passes: int,
    num_checks: int = 16,
) -> dict[str, float]:
    """Measure how much the plan keeps moving under extra annealing passes."""

    if not visited or extra_passes < 2:
        return {}
    stride = max(len(visited) // num_checks, 1)
    picked = visited[::stride][:num_checks]
    first: list[float] = []
    last: list[float] = []
    for _, plan, state, _ in picked:
        one = policy.apply(plan, state.obs, 1)
        few = policy.apply(plan, state.obs, extra_passes)
        more = policy.apply(plan, state.obs, extra_passes + 1)
        first.append(float(jnp.sqrt(jnp.mean(jnp.square(one - plan)))))
        last.append(float(jnp.sqrt(jnp.mean(jnp.square(more - few)))))
    return {
        "first_pass_rms": float(np.mean(first)),
        "final_pass_rms": float(np.mean(last)),
        # < 1 means the field settles; ~1 or more means it never converges.
        "ratio": float(np.mean(last) / (np.mean(first) + 1e-12)),
        "extra_passes": int(extra_passes),
    }


def main() -> None:
    args = _parser().parse_args()
    policy_path, report = _resolve_sources(args)
    trained = report.get("arguments", {})
    config = _load_config(args, report)

    init_passes = args.init_passes or int(trained.get("init_passes", 5))
    temp_sample = args.temp_sample
    if temp_sample is None and trained.get("temp_sample") is not None:
        temp_sample = float(trained["temp_sample"])
    samples = args.samples
    if samples is None and trained.get("samples") is not None:
        samples = int(trained["samples"])
    omega_text = args.omega or trained.get("omega", "1,1,1")
    omega = jnp.asarray(
        [float(value) for value in str(omega_text).split(",")], dtype=jnp.float32
    )

    dial_config = load_dataclass_from_dict(DialConfig, config)
    overrides = {"time_correlated": False}
    if samples is not None:
        overrides["Nsample"] = int(samples)
    if temp_sample is not None:
        overrides["temp_sample"] = float(temp_sample)
    dial_config = dataclasses.replace(dial_config, **overrides)

    env_name = dial_config.env_name
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(env_name), config, convert_list_to_array=True
    )
    if args.timestep is not None:
        env_config = dataclasses.replace(env_config, timestep=float(args.timestep))
    if args.start_noise > 0.0:
        scale = float(args.start_noise)
        randomized = {
            "randomize_start_state": True,
            "start_height_noise": 0.02 * scale,
            "start_rpy_noise": 0.10 * scale,
            "start_joint_position_noise": 0.10 * scale,
            "start_body_linear_velocity_noise": 0.20 * scale,
            "start_body_angular_velocity_noise": 0.20 * scale,
            "start_joint_velocity_noise": 0.50 * scale,
        }
        missing = [
            key for key in randomized if not hasattr(env_config, key)
        ]
        if missing:
            raise ValueError(
                f"{env_name} has no randomized-start fields {missing}; "
                "--start-noise is not supported here"
            )
        env_config = dataclasses.replace(env_config, **randomized)
    if args.randomize_command:
        if not hasattr(env_config, "randomize_tasks"):
            raise ValueError(f"{env_name} has no randomize_tasks field")
        env_config = dataclasses.replace(env_config, randomize_tasks=True)
    env = brax_envs.get_environment(env_name, config=env_config)
    planner = MBDPI(dial_config, env)
    horizon = int(dial_config.Hnode) + 1

    policy = DialScorePolicy.load(policy_path)
    num_levels = int(jnp.asarray(policy.factors).shape[0])
    factors = dial_factors(dial_config.traj_diffuse_factor, num_levels)
    if not np.allclose(np.asarray(factors), np.asarray(policy.factors), atol=1e-5):
        raise ValueError(
            "the policy's annealing schedule does not match this config; "
            f"policy={np.asarray(policy.factors).tolist()} "
            f"config={np.asarray(factors).tolist()}"
        )
    teacher = DialScoreTeacher(planner, omega, repeats=args.score_check_repeats)
    shift_matrix = build_shift_matrix(planner)

    output = args.output or (
        (args.run or policy_path.parent)
        / f"verify-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)

    print(f"policy={policy_path}")
    print(
        f"environment={env_name} omega={np.asarray(omega).tolist()} "
        f"levels={num_levels} temp_sample={dial_config.temp_sample} "
        f"Nsample={dial_config.Nsample} init_passes={init_passes}"
    )

    results: dict[str, object] = {
        "policy": str(policy_path),
        "environment": env_name,
        "omega": [float(value) for value in omega],
        "temp_sample": float(dial_config.temp_sample),
        "nsample": int(dial_config.Nsample),
        "init_passes": init_passes,
        "steps": args.steps,
        "episodes": [],
    }

    student_summaries = []
    dial_summaries = []
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        render = episode == 0 and not args.no_render
        student, student_states, visited = run_student(
            env, policy, teacher, horizon, seed, args.steps, init_passes, render
        )
        student_summary = student.summary()
        student_summaries.append(student_summary)
        entry = {"seed": seed, "student": student_summary}
        print(
            f"seed {seed} student: mean_reward="
            f"{student_summary['mean_reward']:.4f} "
            f"terminations={student_summary['terminations']} "
            f"first_fall_at={student_summary['steps_before_first_termination']}"
        )

        if not args.no_baseline:
            dial_result, dial_states = run_dial(
                env,
                planner,
                teacher,
                factors,
                shift_matrix,
                horizon,
                seed,
                args.steps,
                init_passes,
                render,
            )
            dial_summary = dial_result.summary()
            dial_summaries.append(dial_summary)
            entry["dial"] = dial_summary
            print(
                f"seed {seed} dial:    mean_reward="
                f"{dial_summary['mean_reward']:.4f} "
                f"terminations={dial_summary['terminations']} "
                f"first_fall_at={dial_summary['steps_before_first_termination']}"
            )
        else:
            dial_states = []

        if episode == 0:
            if args.score_check > 0:
                score = check_on_policy_score(
                    policy,
                    teacher,
                    visited,
                    factors,
                    args.score_check,
                    jax.random.PRNGKey(args.seed_offset),
                )
                results["on_policy_score"] = score
                if score:
                    print(
                        f"on-policy score: relative_rms={score['relative_rms']:.4f} "
                        f"cosine={score['cosine']:.4f} "
                        f"({score['queries']} queries)"
                    )
                    for level in range(num_levels):
                        key = f"cosine_level_{level}"
                        if key in score:
                            print(
                                f"  level {level} "
                                f"(factor={float(factors[level]):.3g}): "
                                f"cosine={score[key]:.4f}"
                            )
            stability = check_contractivity(
                policy, visited, args.contractivity_passes
            )
            results["contractivity"] = stability
            if stability:
                print(
                    f"contractivity: first_pass_rms="
                    f"{stability['first_pass_rms']:.4f} "
                    f"final_pass_rms={stability['final_pass_rms']:.4f} "
                    f"ratio={stability['ratio']:.3f}"
                )
            if render:
                system = env.sys.tree_replace({"opt.timestep": env.dt})
                (output / "student.html").write_text(
                    html.render(system, student_states, 720, True),
                    encoding="utf-8",
                )
                print(f"rendered {output / 'student.html'}")
                if dial_states:
                    (output / "dial.html").write_text(
                        html.render(system, dial_states, 720, True),
                        encoding="utf-8",
                    )
                    print(f"rendered {output / 'dial.html'}")

        results["episodes"].append(entry)

    student_mean = float(np.mean([s["mean_reward"] for s in student_summaries]))
    student_falls = int(np.sum([s["terminations"] for s in student_summaries]))
    student_rewards = [s["mean_reward"] for s in student_summaries]
    deterministic = len(student_rewards) > 1 and float(
        np.ptp(student_rewards)
    ) < 1e-9
    aggregate = {
        "student_mean_reward": student_mean,
        "student_terminations": student_falls,
        "student_episodes_identical": deterministic,
    }
    print("")
    if deterministic:
        print(
            "note: every student episode was identical -- this config resets "
            "deterministically and denoising has no sampling, so the seeds "
            "changed nothing.  Pass --start-noise to test unseen states."
        )
    print(f"student: mean_reward={student_mean:.4f} terminations={student_falls}")
    if dial_summaries:
        dial_mean = float(np.mean([s["mean_reward"] for s in dial_summaries]))
        dial_falls = int(np.sum([s["terminations"] for s in dial_summaries]))
        aggregate.update(
            {
                "dial_mean_reward": dial_mean,
                "dial_terminations": dial_falls,
                "reward_ratio": student_mean / dial_mean if dial_mean else None,
            }
        )
        print(f"dial:    mean_reward={dial_mean:.4f} terminations={dial_falls}")
        print(
            "verdict: student reward is "
            f"{student_mean / dial_mean:.2f}x DIAL's "
            "(1.0 means matched; higher means worse for negative costs)"
        )
    results["aggregate"] = aggregate

    (output / "verification.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
