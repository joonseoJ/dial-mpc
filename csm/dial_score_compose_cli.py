"""Train a compositional DIAL score model: k independent fields, one basis each.

The single-weight pipeline is reused verbatim.  Each basis weight gets its own
complete :class:`~csm.dial_score.DialScoreMLP` trained by the same score
regression, so every field can be evaluated and watched on its own with
``dial-score-eval`` and ``dial-score-serve``.  Nothing is shared between fields
except the rollouts that produced their labels — if the composed controller
misbehaves, a shared trunk is never a candidate explanation.

One collection pass labels every query under all basis weights with common
random numbers (see ``DialScoreTeacher.targets_basis``), so k fields cost one
set of environment rollouts rather than k.
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
import optax
import yaml
from brax import envs as brax_envs
from flax import nnx

import dial_mpc.envs as dial_envs
from csm.architectures import StandardNormalizer
from csm.dial_score import (
    ComposedDialScorePolicy,
    DialScoreCollector,
    DialScoreMLP,
    DialScorePolicy,
    DialScoreTeacher,
    build_shift_matrix,
    concat_dial_score_data,
    dial_factors,
    factor_to_t,
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
        "--basis",
        type=str,
        default="1,1,1;3,1,1;1,3,1",
        help="basis weights as 'w1,w2,w3;w1,w2,w3;...'; rows must be "
        "well conditioned and each must keep DIAL walking",
    )
    parser.add_argument(
        "--drive-row",
        type=int,
        default=0,
        help="basis row that advances the environment during collection",
    )
    parser.add_argument(
        "--init-policies",
        type=str,
        default=None,
        help="comma-separated policy.pkl paths, one per basis row, to warm start",
    )

    collection = parser.add_argument_group("teacher and collection")
    collection.add_argument("--samples", type=int, default=None)
    collection.add_argument("--temp-sample", type=float, default=None)
    collection.add_argument("--teacher-repeats", type=int, default=8)
    collection.add_argument("--collect-steps", type=int, default=1200)
    collection.add_argument("--episode-steps", type=int, default=250)
    collection.add_argument("--init-passes", type=int, default=5)
    collection.add_argument("--perturbations", type=int, default=4)
    collection.add_argument("--perturb-scale", type=float, default=2.0)
    collection.add_argument("--start-noise", type=float, default=0.5)
    collection.add_argument(
        "--dataset-prefix",
        type=Path,
        default=None,
        help="reuse datasets saved as <prefix>_row{i}.npz",
    )
    collection.add_argument("--collect-only", action="store_true")

    training = parser.add_argument_group("training")
    training.add_argument("--hidden", type=str, default="512,512,512")
    training.add_argument("--train-iters", type=int, default=8000)
    training.add_argument("--batch-size", type=int, default=512)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--warmup-steps", type=int, default=500)
    training.add_argument("--val-frac", type=float, default=0.1)
    training.add_argument("--eval-every", type=int, default=500)
    training.add_argument("--level-loss-balance", type=float, default=0.0)

    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument(
        "--randomize-drive-omega",
        action="store_true",
        help="draw a fresh omega per episode during DAgger so collection covers "
        "the states the composed controller actually reaches",
    )
    parser.add_argument(
        "--selection-steps",
        type=int,
        default=300,
        help="closed-loop rollout length used to pick the best checkpoint; "
        "0 falls back to validation loss",
    )
    parser.add_argument(
        "--selection-every",
        type=int,
        default=2000,
        help="gradient steps between checkpoint-selection rollouts",
    )
    parser.add_argument(
        "--selection-omegas",
        type=str,
        default="1,1,1;2,1,1;1,2,1",
        help="omegas scored during checkpoint selection",
    )
    parser.add_argument("--dagger-steps", type=int, default=None)
    parser.add_argument("--dagger-iters", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _parse_basis(text: str) -> np.ndarray:
    rows = [
        [float(value) for value in row.split(",")]
        for row in text.split(";")
        if row.strip()
    ]
    basis = np.asarray(rows, dtype=np.float32)
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix of weight rows")
    rank = int(np.linalg.matrix_rank(basis))
    if rank < basis.shape[0]:
        raise ValueError(
            f"basis rows are not independent (rank {rank} of {basis.shape[0]})"
        )
    return basis


def _randomize(env_config, start_noise: float):
    if start_noise <= 0.0:
        return env_config
    scale = float(start_noise)
    return dataclasses.replace(
        env_config,
        randomize_start_state=True,
        start_height_noise=0.02 * scale,
        start_rpy_noise=0.10 * scale,
        start_joint_position_noise=0.10 * scale,
        start_body_linear_velocity_noise=0.20 * scale,
        start_body_angular_velocity_noise=0.20 * scale,
        start_joint_velocity_noise=0.50 * scale,
    )


def main() -> None:
    args = _parser().parse_args()
    path = get_example_path(args.example + ".yaml") if args.example else args.config
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.smoke:
        args.samples, args.teacher_repeats = 8, 2
        args.collect_steps, args.episode_steps, args.init_passes = 2, 2, 1
        args.perturbations, args.train_iters, args.batch_size = 0, 2, 2
        args.eval_every, args.dagger_rounds = 1, 1
        args.dagger_steps, args.dagger_iters = 1, 2

    basis = _parse_basis(args.basis)
    num_rows = int(basis.shape[0])
    hidden = tuple(int(value) for value in args.hidden.split(","))

    dial_config = load_dataclass_from_dict(DialConfig, config)
    overrides = {"time_correlated": False}
    if args.samples is not None:
        overrides["Nsample"] = int(args.samples)
    if args.temp_sample is not None:
        overrides["temp_sample"] = float(args.temp_sample)
    dial_config = dataclasses.replace(dial_config, **overrides)

    env_name = dial_config.env_name
    env_config = _randomize(
        load_dataclass_from_dict(
            dial_envs.get_config(env_name), config, convert_list_to_array=True
        ),
        args.start_noise,
    )
    env = brax_envs.get_environment(env_name, config=env_config)
    planner = MBDPI(dial_config, env)
    horizon = int(dial_config.Hnode) + 1
    factors = dial_factors(dial_config.traj_diffuse_factor, dial_config.Ndiffuse)
    factor_min, factor_max = float(jnp.min(factors)), float(jnp.max(factors))
    shift_matrix = build_shift_matrix(planner)
    teacher = DialScoreTeacher(
        planner, jnp.asarray(basis[args.drive_row]), repeats=args.teacher_repeats
    )
    collector = DialScoreCollector(
        env,
        planner,
        teacher,
        factors,
        perturbations=args.perturbations,
        perturb_scale=args.perturb_scale,
    )

    run_dir = args.output / f"compose-{env_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    condition = float(np.linalg.cond(basis))
    report: dict[str, object] = {
        "environment": env_name,
        "basis": basis.tolist(),
        "basis_condition_number": condition,
        "drive_row": args.drive_row,
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

    print(f"environment={env_name} basis=\n{basis}")
    print(f"basis condition number={condition:.3f}  drive row={args.drive_row}")
    write_report("collecting")

    rng = jax.random.PRNGKey(args.seed)
    if args.dataset_prefix is not None:
        datasets = [
            load_dial_score_data(f"{args.dataset_prefix}_row{row}.npz")
            for row in range(num_rows)
        ]
        stats = None
        print(f"loaded {datasets[0].size} targets per basis row")
    else:
        rng, collect_rng = jax.random.split(rng)
        datasets, stats = collector.rollout_basis(
            collect_rng,
            args.collect_steps,
            basis,
            student=None,
            init_passes=args.init_passes,
            episode_steps=args.episode_steps,
            drive_row=args.drive_row,
            desc="teacher basis collection",
        )
        print(
            f"teacher rollout reward={stats.mean_reward:.4f} "
            f"resets={stats.num_resets} episodes={stats.num_episodes} "
            f"ess={stats.mean_effective_samples:.0f}"
        )
        if stats.relative_label_noise is not None:
            print(
                f"target rms={stats.target_rms:.4f} label noise="
                f"{stats.label_noise:.4f} -> validation relative_rms floor "
                f"{stats.relative_label_noise:.3f}"
            )
        write_report("training", teacher=stats._asdict())

    def save_datasets() -> None:
        for row, data in enumerate(datasets):
            save_dial_score_data(run_dir / f"dataset_row{row}.npz", data)

    save_datasets()
    observation_size = int(datasets[0].obs.shape[-1])

    # ------------------------------------------------------------------ #
    # k independent fields
    # ------------------------------------------------------------------ #
    if args.init_policies:
        loaded = [
            DialScorePolicy.load(item.strip())
            for item in args.init_policies.split(",")
        ]
        if len(loaded) != num_rows:
            raise ValueError("--init-policies needs one path per basis row")
        models = [policy.model for policy in loaded]
        normalizers = [policy.normalizer for policy in loaded]
    else:
        shared_normalizer = StandardNormalizer(observation_size)
        shared_normalizer.fit(datasets[0].obs)
        normalizers = [shared_normalizer] * num_rows
        models = [
            DialScoreMLP(
                action_size=int(env.action_size),
                observation_size=observation_size,
                horizon=horizon,
                sigma_control=planner.sigma_control,
                hidden=hidden,
                rngs=nnx.Rngs(args.seed + 1000 * row),
                factor_min=factor_min,
                factor_max=factor_max,
            )
            for row in range(num_rows)
        ]

    def make_field(row: int) -> DialScorePolicy:
        return DialScorePolicy(
            model=models[row],
            normalizer=normalizers[row],
            factors=factors,
            shift_matrix=shift_matrix,
        )

    def make_policy() -> ComposedDialScorePolicy:
        return ComposedDialScorePolicy(
            policies=tuple(make_field(row) for row in range(num_rows)),
            mode_weights=jnp.asarray(basis),
            pinv_mode_weights=jnp.asarray(np.linalg.pinv(basis)),
        )

    def save_all() -> None:
        make_policy().save(run_dir / "policy.pkl")
        for row in range(num_rows):
            make_field(row).save(run_dir / f"field_row{row}.pkl")

    selection_omegas = [
        jnp.asarray([float(v) for v in item.split(",")], dtype=jnp.float32)
        for item in args.selection_omegas.split(";")
        if item.strip()
    ]
    reset_env, step_env = jax.jit(env.reset), jax.jit(env.step)
    shift_plan = jax.jit(lambda plan: jnp.einsum("ij,ja->ia", shift_matrix, plan))
    pinv_basis = jnp.asarray(np.linalg.pinv(basis))

    def _coefficients(omega):
        c = jnp.asarray(omega, dtype=jnp.float32) @ pinv_basis
        return c / jnp.sum(c)

    ramp_steps = int(round(getattr(env_config, "ramp_up_time", 1.0) / env.dt))

    # nnx.jit takes the modules as arguments and splits their state, so the
    # weights can change between calls without triggering a recompile.  Baking
    # them into a plain jax.jit closure would recompile on every selection
    # rollout, and running eagerly costs ~1 s per control step.
    @nnx.jit
    def _denoise_step(models, normalizers, plan, obs, coefficients, t, sigma_sq):
        score = None
        for index, (model, normalizer) in enumerate(zip(models, normalizers)):
            term = coefficients[index] * model.score(
                plan, normalizer(obs, use_running_average=True), t
            )
            score = term if score is None else score + term
        return jnp.clip(plan + sigma_sq * score, -1.0, 1.0)

    def closed_loop_score(step: int) -> float:
        """Score the current weights by short rollouts, not by validation loss.

        The command ramps from zero after every reset, so a controller that
        falls often harvests cheap near-zero-error steps.  Only steps past the
        ramp count toward the reward, and falls are penalised explicitly.
        """
        fields = tuple(models[row] for row in range(num_rows))
        norms = tuple(normalizers[row] for row in range(num_rows))
        schedule = np.asarray(factors)
        totals = []
        for index, omega in enumerate(selection_omegas):
            coefficients = _coefficients(omega)
            state = reset_env(jax.random.PRNGKey(90_000 + index))
            info = dict(state.info); info["reward_weights"] = omega
            state = state.replace(info=info)
            plan = jnp.zeros((horizon, int(env.action_size)), dtype=jnp.float32)
            mature, falls, episode_step = [], 0, 0
            for _ in range(args.selection_steps):
                passes = args.init_passes if episode_step == 0 else 1
                for _ in range(passes):
                    for factor in schedule:
                        t = factor_to_t(
                            float(factor), factor_min, factor_max
                        ).reshape(1)
                        sigma_sq = jnp.square(fields[0].sigma(t))
                        plan = _denoise_step(
                            fields, norms, plan, state.obs, coefficients,
                            t, sigma_sq,
                        )
                state = step_env(state, plan[0])
                if episode_step >= ramp_steps:
                    mature.append(float(state.reward))
                if float(state.done) > 0.5:
                    falls += 1
                    state = reset_env(jax.random.PRNGKey(90_000 + index + 500 * falls))
                    info = dict(state.info); info["reward_weights"] = omega
                    state = state.replace(info=info)
                    plan = jnp.zeros_like(plan); episode_step = 0
                else:
                    plan = shift_plan(plan); episode_step += 1
            reward = float(np.mean(mature)) if mature else -1.0
            totals.append(reward - 0.05 * falls / max(args.selection_steps, 1) * 100)
        return float(np.mean(totals))

    use_selection = args.selection_steps > 0 and args.selection_every > 0
    if use_selection and args.selection_steps <= ramp_steps:
        raise ValueError(
            f"--selection-steps must exceed the {ramp_steps}-step command ramp; "
            "shorter rollouts contain no steps at full command and score -1"
        )

    def train(num_iters: int, label: str) -> list[dict]:
        nonlocal rng
        fits = []
        for row in range(num_rows):
            weights = level_loss_weights(datasets[row], args.level_loss_balance)
            warmup = min(args.warmup_steps, max(num_iters // 10, 1))
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=args.learning_rate / 10.0,
                peak_value=args.learning_rate,
                warmup_steps=warmup,
                decay_steps=num_iters,
                end_value=args.learning_rate / 50.0,
            )
            rng, fit_rng = jax.random.split(rng)
            result = fit_dial_score(
                models[row],
                nnx.Optimizer(models[row], optax.adam(schedule)),
                datasets[row],
                normalizer=normalizers[row],
                batch_size=args.batch_size,
                num_iters=num_iters,
                rng=fit_rng,
                validation_fraction=args.val_frac,
                eval_every=args.eval_every,
                level_weights=(
                    weights if args.level_loss_balance > 0.0 else None
                ),
                selection_fn=closed_loop_score if use_selection else None,
                selection_every=args.selection_every if use_selection else 0,
                desc=f"{label} row{row} {basis[row].tolist()}",
            )
            print(
                f"{label} row{row} {basis[row].tolist()}: "
                f"relative_rms={result.best['val_relative_rms']:.4f} "
                f"cosine={result.best['val_cosine']:.4f}"
            )
            fits.append(
                {
                    "row": row,
                    "weights": basis[row].tolist(),
                    "samples": datasets[row].size,
                    "best": result.best,
                    "per_level": {
                        str(k): v for k, v in result.per_level.items()
                    },
                }
            )
        return fits

    if args.collect_only:
        write_report("collected", samples=datasets[0].size)
        print(f"saved={run_dir}")
        return

    history = [train(args.train_iters, "fit")]
    save_all()
    write_report("training", fits=history)

    # ------------------------------------------------------------------ #
    # DAgger driven by the composed policy at the drive weight
    # ------------------------------------------------------------------ #
    dagger_steps = args.dagger_steps or args.collect_steps
    dagger_iters = args.dagger_iters or args.train_iters
    dagger_stats = []
    drive_omega = jnp.asarray(basis[args.drive_row])
    for round_idx in range(max(args.dagger_rounds, 0)):
        policy = make_policy()
        if args.randomize_drive_omega:
            student = jax.jit(
                lambda plan, obs, t, omega: sum(
                    policy.coefficients(omega)[i]
                    * policy.policies[i].delta(plan, obs, t)
                    for i in range(num_rows)
                )
            )
        else:
            coefficients = policy.coefficients(drive_omega)
            student = jax.jit(
                lambda plan, obs, t: sum(
                    coefficients[i] * policy.policies[i].delta(plan, obs, t)
                    for i in range(num_rows)
                )
            )
        rng, collect_rng = jax.random.split(rng)
        new_data, round_stats = collector.rollout_basis(
            collect_rng,
            dagger_steps,
            basis,
            student=student,
            init_passes=args.init_passes,
            episode_steps=args.episode_steps,
            drive_row=args.drive_row,
            randomize_drive_omega=args.randomize_drive_omega,
            desc=f"dagger basis collection {round_idx + 1}/{args.dagger_rounds}",
        )
        print(
            f"dagger round {round_idx + 1} composed-student reward="
            f"{round_stats.mean_reward:.4f} resets={round_stats.num_resets}"
        )
        dagger_stats.append(round_stats._asdict())
        datasets = [
            concat_dial_score_data([datasets[row], new_data[row]])
            for row in range(num_rows)
        ]
        save_datasets()
        history.append(train(dagger_iters, f"dagger fit {round_idx + 1}"))
        save_all()
        write_report("training", fits=history, dagger=dagger_stats)

    save_all()
    write_report("complete", fits=history, dagger=dagger_stats)
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
