"""Single-weight DIAL-MPC score matching: collect, train, deploy.

One MLP, one preference weight, score regression for training and denoising for
inference.  See :mod:`csm.dial_score` for the mathematics.

The pipeline is:

1. **Collect** — run DIAL-MPC's own control loop and record, at every annealing
   level it visits, the exact ``MBDPI.reverse_once`` update averaged over
   several sample batches.  Round 0 is a real DIAL rollout, so its mean reward
   is the baseline the student must reach.
2. **Train** — regress one MLP onto those scores with ``sigma**4`` weighting,
   keeping the best validation checkpoint.
3. **DAgger** (optional) — rerun the loop with the network driving the
   annealing chain while DIAL keeps supplying the targets, then keep training on
   the union of both datasets.
4. **Deploy** — replace ``reverse_once`` with the network and run the same
   annealing schedule as a denoiser.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from brax import envs as brax_envs
from brax.io import html
from flax import nnx
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.architectures import StandardNormalizer
from csm.dial_score import (
    DialScoreCollector,
    StateBank,
    mixed_reset_fn,
    DialScoreMLP,
    DialScorePolicy,
    DialScoreTeacher,
    build_shift_matrix,
    concat_dial_score_data,
    dial_factors,
    fit_dial_score,
    level_loss_weights,
    load_dial_score_data,
    save_dial_score_data,
)
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
    parser.add_argument("--output", type=Path, default=Path("csm_runs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--omega",
        type=str,
        default="1,1,1",
        help="the single reward weight vector this model is trained for",
    )

    collection = parser.add_argument_group("teacher and collection")
    collection.add_argument(
        "--samples",
        type=int,
        default=None,
        help="override DIAL Nsample for the teacher; default keeps the config",
    )
    collection.add_argument(
        "--diffuse-steps",
        type=int,
        default=None,
        help="annealing levels per control step; default keeps DIAL Ndiffuse",
    )
    collection.add_argument(
        "--init-passes",
        type=int,
        default=5,
        help="annealing schedule repeats on the first control step",
    )
    collection.add_argument(
        "--teacher-repeats",
        type=int,
        default=4,
        help="reverse_once batches averaged into each score target",
    )
    collection.add_argument(
        "--temp-sample",
        type=float,
        default=None,
        help=(
            "override DIAL temp_sample for the teacher; the default 0.05 makes "
            "the Gibbs weights nearly winner-take-all, so a larger value trades "
            "exact DIAL imitation for a much less noisy score target"
        ),
    )
    collection.add_argument("--collect-steps", type=int, default=400)
    collection.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help=(
            "reset every this many control steps during collection.  One long "
            "episode covers a single state trajectory, which is the usual "
            "reason the fitted score field collapses off that corridor"
        ),
    )
    collection.add_argument(
        "--start-noise",
        type=float,
        default=0.0,
        help=(
            "scale of randomized initial states during collection.  The stock "
            "Go2 config resets deterministically, so without this every episode "
            "starts from the identical pose"
        ),
    )
    collection.add_argument(
        "--randomize-command",
        action="store_true",
        help=(
            "resample the velocity command during collection so the score field "
            "covers a command range; the observation already carries the target"
        ),
    )
    collection.add_argument(
        "--perturbations",
        type=int,
        default=4,
        help="extra queries per level, sampled around the visited plan",
    )
    collection.add_argument(
        "--perturb-scale",
        type=float,
        default=1.0,
        help="perturbation size in units of the level's per-node sigma",
    )
    collection.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="reuse a saved dataset.npz and skip teacher collection",
    )
    collection.add_argument(
        "--state-bank",
        type=Path,
        default=None,
        help="restart a share of episodes from this saved state bank; the "
        "horizon reachable by student-driven collection is capped by how long "
        "the student survives, so late states have to be seeded",
    )
    collection.add_argument(
        "--bank-restart-frac",
        type=float,
        default=0.5,
        help="fraction of episode restarts drawn from --state-bank; keeping the "
        "rest fresh stops the data collapsing onto recovery states",
    )
    collection.add_argument(
        "--failure-bank-out",
        type=Path,
        default=None,
        help="write the states preceding each fall here, for the next round",
    )
    collection.add_argument(
        "--failure-lookback", type=int, default=30,
        help="steps before a fall to bank",
    )
    collection.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "collect and save the dataset, then stop.  Use it to gather one "
            "shared dataset before training several variants on identical data"
        ),
    )

    training = parser.add_argument_group("training")
    training.add_argument(
        "--hidden",
        type=str,
        default="512,512,512",
        help="comma-separated hidden widths of the single MLP",
    )
    training.add_argument("--train-iters", type=int, default=20_000)
    training.add_argument("--batch-size", type=int, default=512)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--warmup-steps", type=int, default=500)
    training.add_argument(
        "--level-loss-balance",
        type=float,
        default=0.0,
        help=(
            "rebalance the annealing levels in the loss.  0 keeps the plain "
            "update-MSE objective; 1 divides each level's error by that level's "
            "target power so the fine level stops being under-served for having "
            "intrinsically smaller updates"
        ),
    )
    training.add_argument("--val-frac", type=float, default=0.1)
    training.add_argument("--eval-every", type=int, default=500)
    training.add_argument(
        "--init-policy",
        type=Path,
        default=None,
        help="continue from an existing compatible policy.pkl",
    )

    dagger = parser.add_argument_group("dagger")
    dagger.add_argument(
        "--dagger-rounds",
        type=int,
        default=1,
        help="student-driven collection rounds after the first fit; 0 disables",
    )
    dagger.add_argument(
        "--dagger-steps",
        type=int,
        default=None,
        help="control steps per DAgger round; default matches --collect-steps",
    )
    dagger.add_argument(
        "--dagger-iters",
        type=int,
        default=None,
        help="gradient steps per DAgger round; default matches --train-iters",
    )
    dagger.add_argument(
        "--dagger-learning-rate",
        type=float,
        default=None,
        help="learning rate for DAgger rounds; default is --learning-rate / 3",
    )

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--eval-steps", type=int, default=400)
    evaluation.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="denoising step-size multiplier; 1.0 matches DIAL exactly",
    )
    evaluation.add_argument(
        "--no-render",
        action="store_true",
        help="skip visualization.html",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smallest end-to-end integration check",
    )
    return parser


def _load_config(args) -> dict:
    path = get_example_path(args.example + ".yaml") if args.example else args.config
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _randomize_env_config(
    env_config, env_name: str, start_noise: float, randomize_command: bool
):
    """Turn on the environment's own state/command randomization.

    The mapping is deliberately a single scale so the caller has one knob:
    orientation and joint pose get the largest share, velocities less, height
    least, because a Go2 that starts tipped over teaches nothing useful.
    """

    updates: dict[str, object] = {}
    if start_noise > 0.0:
        scale = float(start_noise)
        updates.update(
            randomize_start_state=True,
            start_height_noise=0.02 * scale,
            start_rpy_noise=0.10 * scale,
            start_joint_position_noise=0.10 * scale,
            start_body_linear_velocity_noise=0.20 * scale,
            start_body_angular_velocity_noise=0.20 * scale,
            start_joint_velocity_noise=0.50 * scale,
        )
    if randomize_command:
        updates["randomize_tasks"] = True
    if not updates:
        return env_config
    missing = [key for key in updates if not hasattr(env_config, key)]
    if missing:
        raise ValueError(
            f"{env_name} has no randomization fields {missing}; drop "
            "--start-noise / --randomize-command for this environment"
        )
    return dataclasses.replace(env_config, **updates)


def _apply_smoke_defaults(args) -> None:
    args.samples = 8
    args.diffuse_steps = 2
    args.init_passes = 1
    args.teacher_repeats = 1
    args.collect_steps = 2
    args.perturbations = 1
    args.train_iters = 4
    args.batch_size = 2
    args.eval_every = 2
    args.dagger_rounds = 1
    args.dagger_steps = 1
    args.dagger_iters = 2
    args.eval_steps = 2


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    if args.smoke:
        _apply_smoke_defaults(args)

    omega = jnp.asarray(
        [float(value) for value in args.omega.split(",")], dtype=jnp.float32
    )
    hidden = tuple(int(value) for value in args.hidden.split(","))

    dial_config = load_dataclass_from_dict(DialConfig, config)
    if dial_config.time_correlated:
        print(
            "warning: forcing time_correlated=False; this study imitates plain "
            "DIAL annealing"
        )
    overrides = {"time_correlated": False}
    if args.samples is not None:
        overrides["Nsample"] = int(args.samples)
    if args.temp_sample is not None:
        overrides["temp_sample"] = float(args.temp_sample)
    dial_config = dataclasses.replace(dial_config, **overrides)

    env_name = dial_config.env_name
    env_config_type = dial_envs.get_config(env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config, convert_list_to_array=True
    )
    default_weights = getattr(env_config, "reward_weights", None)
    if default_weights is not None and omega.shape != jnp.asarray(
        default_weights
    ).shape:
        raise ValueError(
            f"--omega needs {jnp.asarray(default_weights).shape[0]} values for "
            f"{env_name}"
        )
    env_config = _randomize_env_config(
        env_config, env_name, args.start_noise, args.randomize_command
    )
    env = brax_envs.get_environment(env_name, config=env_config)

    planner = MBDPI(dial_config, env)
    horizon = int(dial_config.Hnode) + 1
    num_levels = args.diffuse_steps or int(dial_config.Ndiffuse)
    factors = dial_factors(dial_config.traj_diffuse_factor, num_levels)
    factor_min = float(jnp.min(factors))
    factor_max = float(jnp.max(factors))
    teacher = DialScoreTeacher(planner, omega, repeats=args.teacher_repeats)
    collector = DialScoreCollector(
        env,
        planner,
        teacher,
        factors,
        perturbations=args.perturbations,
        perturb_scale=args.perturb_scale,
    )
    shift_matrix = build_shift_matrix(planner)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output / f"dial-score-{env_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "environment": env_name,
        "omega": [float(value) for value in omega],
        "horizon_nodes": horizon,
        "action_size": int(env.action_size),
        "nsample": int(dial_config.Nsample),
        "temp_sample": float(dial_config.temp_sample),
        "sigma_control": np.asarray(planner.sigma_control).tolist(),
        "annealing_factors": np.asarray(factors).tolist(),
        "hidden": list(hidden),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }

    def write_report(status: str, **updates) -> None:
        report["status"] = status
        report.update(updates)
        destination = run_dir / "report.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(destination)

    write_report("collecting")
    print(
        f"environment={env_name} nodes={horizon} actions={env.action_size} "
        f"levels={num_levels} factors={np.asarray(factors).tolist()}"
    )
    print(f"sigma_control={np.asarray(planner.sigma_control).tolist()}")

    rng = jax.random.PRNGKey(args.seed)
    seed_bank = (
        StateBank.load(args.state_bank) if args.state_bank is not None else None
    )
    if seed_bank is not None:
        print(f"seed bank: {len(seed_bank)} states, restart fraction "
              f"{args.bank_restart_frac}")
    failure_bank = StateBank() if args.failure_bank_out is not None else None
    collect_kwargs = dict(
        reset_fn=mixed_reset_fn(
            collector._reset, seed_bank, args.bank_restart_frac
        ),
        failure_bank=failure_bank,
        failure_lookback=args.failure_lookback,
    )

    # ------------------------------------------------------------------ #
    # Round 0: teacher-driven collection (a real DIAL-MPC rollout)
    # ------------------------------------------------------------------ #
    if args.dataset is not None:
        data = load_dial_score_data(args.dataset)
        teacher_stats = None
        print(f"loaded {data.size} score targets from {args.dataset}")
    else:
        rng, collect_rng = jax.random.split(rng)
        data, teacher_stats = collector.rollout(
            collect_rng,
            args.collect_steps,
            student=None,
            init_passes=args.init_passes,
            episode_steps=args.episode_steps,
            **collect_kwargs,
            desc="teacher collection",
        )
        save_dial_score_data(run_dir / "dataset.npz", data)
        print(
            f"teacher rollout mean reward={teacher_stats.mean_reward:.4f} "
            f"resets={teacher_stats.num_resets} "
            f"episodes={teacher_stats.num_episodes} "
            f"ess={teacher_stats.mean_effective_samples:.1f}"
        )
        if teacher_stats.relative_label_noise is None:
            print(
                f"target rms={teacher_stats.target_rms:.4f} "
                "(--teacher-repeats 1 cannot estimate the label noise floor)"
            )
        else:
            print(
                f"target rms={teacher_stats.target_rms:.4f} "
                f"label noise={teacher_stats.label_noise:.4f} -> validation "
                f"relative_rms cannot beat "
                f"{teacher_stats.relative_label_noise:.3f} "
                f"(raise --teacher-repeats to lower it)"
            )
        write_report("training", teacher=teacher_stats._asdict())

    observation_size = int(data.obs.shape[-1])

    # ------------------------------------------------------------------ #
    # The single network
    # ------------------------------------------------------------------ #
    if args.init_policy is not None:
        initial = DialScorePolicy.load(args.init_policy)
        model = initial.model
        normalizer = initial.normalizer
        if (
            model.horizon != horizon
            or model.action_size != int(env.action_size)
            or model.observation_size != observation_size
        ):
            raise ValueError("--init-policy architecture is incompatible")
    else:
        normalizer = StandardNormalizer(observation_size)
        # Frozen after round 0 on purpose: refitting it between DAgger rounds
        # would silently shift the network's input distribution.
        normalizer.fit(data.obs)
        model = DialScoreMLP(
            action_size=int(env.action_size),
            observation_size=observation_size,
            horizon=horizon,
            sigma_control=planner.sigma_control,
            hidden=hidden,
            rngs=nnx.Rngs(args.seed),
            factor_min=factor_min,
            factor_max=factor_max,
        )

    def make_policy() -> DialScorePolicy:
        return DialScorePolicy(
            model=model,
            normalizer=normalizer,
            factors=factors,
            shift_matrix=shift_matrix,
            dt=args.dt,
        )

    def make_optimizer(learning_rate: float, num_iters: int) -> nnx.Optimizer:
        warmup = min(args.warmup_steps, max(num_iters // 10, 1))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 10.0,
            peak_value=learning_rate,
            warmup_steps=warmup,
            decay_steps=num_iters,
            end_value=learning_rate / 50.0,
        )
        return nnx.Optimizer(model, optax.adam(schedule))

    def train(num_iters: int, learning_rate: float, label: str) -> dict:
        nonlocal rng
        rng, fit_rng = jax.random.split(rng)
        weights = level_loss_weights(data, args.level_loss_balance)
        if args.level_loss_balance > 0.0:
            print(
                f"{label}: level loss weights="
                + ", ".join(
                    f"{i}:{float(w):.3f}" for i, w in enumerate(weights)
                )
            )
        result = fit_dial_score(
            model,
            make_optimizer(learning_rate, num_iters),
            data,
            normalizer=normalizer,
            batch_size=args.batch_size,
            num_iters=num_iters,
            rng=fit_rng,
            validation_fraction=args.val_frac,
            eval_every=args.eval_every,
            level_weights=(
                weights if args.level_loss_balance > 0.0 else None
            ),
            desc=label,
        )
        print(
            f"{label}: best val_loss={result.best['val_loss']:.4e} "
            f"relative_rms={result.best['val_relative_rms']:.4f} "
            f"cosine={result.best['val_cosine']:.4f} "
            f"(step {result.best['step']})"
        )
        for level, metrics in result.per_level.items():
            print(
                f"  level {level} (factor={float(factors[level]):.3g}): "
                f"relative_rms={metrics['relative_rms']:.4f} "
                f"cosine={metrics['cosine']:.4f}"
            )
        return {
            "samples": data.size,
            "level_loss_weights": [float(value) for value in weights],
            "best": result.best,
            "final": result.final,
            "per_level": {str(k): v for k, v in result.per_level.items()},
            "history": result.history,
        }

    if args.collect_only:
        if args.init_policy is None and args.dagger_rounds > 0:
            raise ValueError(
                "--collect-only with --dagger-rounds needs --init-policy to "
                "drive the student rollouts"
            )
        fits = []
    else:
        fits = [train(args.train_iters, args.learning_rate, "score regression")]
        make_policy().save(run_dir / "policy.pkl")
        write_report("training", fits=fits)

    # ------------------------------------------------------------------ #
    # DAgger: the student visits the queries, DIAL still labels them
    # ------------------------------------------------------------------ #
    dagger_steps = args.dagger_steps or args.collect_steps
    dagger_iters = args.dagger_iters or args.train_iters
    dagger_lr = args.dagger_learning_rate or args.learning_rate / 3.0
    dagger_stats = []
    for round_idx in range(max(args.dagger_rounds, 0)):
        policy = make_policy()
        student = jax.jit(
            lambda plan, obs, t: policy.delta(plan, obs, t)
        )
        rng, collect_rng = jax.random.split(rng)
        round_data, round_stats = collector.rollout(
            collect_rng,
            dagger_steps,
            student=student,
            init_passes=args.init_passes,
            episode_steps=args.episode_steps,
            **collect_kwargs,
            desc=f"dagger collection {round_idx + 1}/{args.dagger_rounds}",
        )
        print(
            f"dagger round {round_idx + 1} student mean reward="
            f"{round_stats.mean_reward:.4f} resets={round_stats.num_resets} "
            f"episodes={round_stats.num_episodes}"
        )
        dagger_stats.append(round_stats._asdict())
        data = concat_dial_score_data([data, round_data])
        save_dial_score_data(run_dir / "dataset.npz", data)
        if args.collect_only:
            write_report("collected", dagger=dagger_stats, samples=data.size)
            continue
        fits.append(
            train(dagger_iters, dagger_lr, f"dagger fit {round_idx + 1}")
        )
        make_policy().save(run_dir / "policy.pkl")
        write_report("training", fits=fits, dagger=dagger_stats)

    if args.collect_only:
        write_report("collected", dagger=dagger_stats, samples=data.size)
        print(f"collected {data.size} samples -> {run_dir / 'dataset.npz'}")
        print(f"saved={run_dir}")
        return

    if failure_bank is not None and len(failure_bank) > 0:
        failure_bank.save(args.failure_bank_out)
        print(f"banked {len(failure_bank)} pre-fall states -> "
              f"{args.failure_bank_out}")
    policy = make_policy()
    policy.save(run_dir / "policy.pkl")
    write_report("evaluating", fits=fits, dagger=dagger_stats)

    # ------------------------------------------------------------------ #
    # Deployment: denoising in place of reverse_once
    # ------------------------------------------------------------------ #
    print("evaluating the denoising policy")
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    apply_first = jax.jit(
        functools.partial(policy.apply, passes=args.init_passes)
    )
    apply_step = jax.jit(functools.partial(policy.apply, passes=1))
    shift_plan = jax.jit(policy.shift)

    rng, reset_rng = jax.random.split(rng)
    state = teacher.with_reward_weights(reset_env(reset_rng))
    plan = jnp.zeros((horizon, int(env.action_size)), dtype=jnp.float32)
    rollout = []
    rewards = []
    dones = 0
    for step_idx in tqdm(
        range(args.eval_steps),
        desc="denoising rollout",
        unit="step",
        dynamic_ncols=True,
    ):
        plan = (apply_first if step_idx == 0 else apply_step)(plan, state.obs)
        plan.block_until_ready()
        state = step_env(state, plan[0])
        rollout.append(state.pipeline_state)
        rewards.append(float(state.reward))
        if float(state.done) > 0.5:
            dones += 1
            rng, reset_rng = jax.random.split(rng)
            state = teacher.with_reward_weights(reset_env(reset_rng))
            plan = jnp.zeros_like(plan)
        else:
            plan = shift_plan(plan)

    student_mean = float(np.mean(rewards))
    print(f"student mean reward={student_mean:.4f} terminations={dones}")
    if teacher_stats is not None:
        print(
            f"teacher mean reward={teacher_stats.mean_reward:.4f} "
            f"ratio={student_mean / teacher_stats.mean_reward:.3f}"
        )

    if not args.no_render:
        rendered = html.render(
            env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 720, True
        )
        (run_dir / "visualization.html").write_text(rendered, encoding="utf-8")

    write_report(
        "complete",
        fits=fits,
        dagger=dagger_stats,
        evaluation={
            "mean_reward": student_mean,
            "total_reward": float(np.sum(rewards)),
            "num_steps": int(args.eval_steps),
            "terminations": dones,
        },
    )
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
