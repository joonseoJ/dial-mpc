"""End-to-end CSM data collection, training, and evaluation for DIAL-MPC."""

from __future__ import annotations

import argparse
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
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.architectures import CompositionalDenoisingMLP, StandardNormalizer
from csm.data_collection import (
    DIALScoreCollector,
    build_sigma_schedule,
    load_score_dataset,
    save_score_dataset,
    stack_score_data,
)
from csm.envs import get_objective_spec
from csm.policy import CompositionalPolicy
from csm.training import fit_score_regression
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
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="reuse an existing scores.npz and skip expensive MPPI collection",
    )
    parser.add_argument(
        "--onpolicy-dataset",
        type=Path,
        default=None,
        help="DAgger correction scores collected at learner-visited queries",
    )
    parser.add_argument(
        "--onpolicy-frac",
        type=float,
        default=0.5,
        help="fraction of each batch drawn from --onpolicy-dataset",
    )
    parser.add_argument(
        "--init-policy",
        type=Path,
        default=None,
        help="continue training an existing compatible policy",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Gibbs temperature for raw decomposed costs; 1.0 avoids the "
            "near-single-sample targets produced by DIAL's reward-normalized "
            "temperature of 0.05"
        ),
    )
    parser.add_argument("--collect-steps", type=int, default=400)
    parser.add_argument("--sigma-levels", type=int, default=10)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--perturbations", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--score-loss-weight-power",
        type=float,
        default=4.0,
        help="sigma exponent in score loss; 4 minimizes MPPI update MSE",
    )
    parser.add_argument(
        "--predict-score",
        action="store_true",
        help="legacy: predict delta_U/sigma^2 instead of MPPI delta_U directly",
    )
    parser.add_argument(
        "--data-checkpoint-every",
        type=int,
        default=25,
        help="save partial collected data every N environment states; 0 disables",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="save an intermediate policy every N total gradient steps; 0 disables",
    )
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument(
        "--mode-weights",
        type=Path,
        default=None,
        help="JSON matrix of expert weights saved by dial-mpc-weights",
    )
    parser.add_argument(
        "--omega",
        type=str,
        default=None,
        help="Comma-separated inference objective weights",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smallest end-to-end integration check",
    )
    return parser


def _load_config(args) -> dict:
    path = get_example_path(args.example + ".yaml") if args.example else args.config
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _extend(target, new_data) -> None:
    for mode_idx, mode_data in enumerate(new_data):
        target[mode_idx].extend(mode_data)


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)

    if args.data_checkpoint_every < 0 or args.checkpoint_every < 0:
        raise ValueError("checkpoint intervals cannot be negative")

    if args.smoke:
        args.samples = 8
        args.collect_steps = 1
        args.sigma_levels = 1
        args.perturbations = 0
        args.train_iters = 1
        args.batch_size = 1
        args.inference_steps = 2
        args.eval_steps = 2
        args.horizon = 2

    env_name = config["env_name"]
    env_config_type = dial_envs.get_config(env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config, convert_list_to_array=True
    )
    env = brax_envs.get_environment(env_name, config=env_config)
    objective_spec = get_objective_spec(env)
    objective_dim = objective_spec.num_objectives
    if args.mode_weights is None:
        mode_weights = jnp.eye(objective_dim, dtype=jnp.float32)
    else:
        raw_weights = np.asarray(
            json.loads(args.mode_weights.read_text(encoding="utf-8")),
            dtype=np.float32,
        )
        if raw_weights.ndim != 2 or raw_weights.shape[1] != objective_dim:
            raise ValueError(
                f"mode-weight matrix must have shape (modes, {objective_dim})"
            )
        if raw_weights.shape[0] < objective_dim:
            raise ValueError(
                f"at least {objective_dim} mode vectors are required"
            )
        rank = int(np.linalg.matrix_rank(raw_weights))
        if rank < objective_dim:
            raise ValueError(
                f"mode-weight matrix rank is {rank}; expected {objective_dim}"
            )
        mode_weights = jnp.asarray(raw_weights)
    num_modes = int(mode_weights.shape[0])
    # DIAL-MPC optimizes sparse spline nodes, not Hsample independent actions.
    # Matching that representation reduces Go2 plans from 192 to 60 dimensions.
    horizon = args.horizon or (int(config.get("Hnode", 4)) + 1)
    hsample = int(config.get("Hsample", 16))
    control_dt = float(config.get("dt", 0.02))
    step_us = jnp.linspace(0.0, control_dt * hsample, hsample + 1)
    step_nodes = jnp.linspace(0.0, control_dt * hsample, horizon)

    def interpolate_nodes(nodes):
        def interpolate_one(values):
            return InterpolatedUnivariateSpline(
                step_nodes, values, k=min(2, horizon - 1)
            )(step_us)

        return jax.vmap(interpolate_one, in_axes=1, out_axes=1)(nodes)

    def shift_one(values):
        actions = InterpolatedUnivariateSpline(
            step_nodes, values, k=min(2, horizon - 1)
        )(step_us)
        actions = jnp.roll(actions, -1).at[-1].set(0.0)
        return InterpolatedUnivariateSpline(
            step_us, actions, k=min(2, horizon - 1)
        )(step_nodes)

    # Linear map from node plan to DIAL's interpolate-shift-project warm start.
    shift_matrix = jax.vmap(shift_one)(jnp.eye(horizon)).T
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output / f"{env_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"

    def write_run_config(status: str, **updates) -> None:
        run_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        run_config.update(
            {
                "status": status,
                "environment": env_name,
                "objectives": list(objective_spec.names),
                "mode_weights": np.asarray(mode_weights).tolist(),
                "resolved_horizon": horizon,
                "expected_samples_per_mode": (
                    args.collect_steps
                    * num_modes
                    * args.sigma_levels
                    * (args.perturbations + 1)
                    if args.dataset is None
                    else None
                ),
                "total_gradient_steps": args.train_iters * num_modes,
                **updates,
            }
        )
        destination = run_dir / "run_config.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        temporary.replace(destination)

    write_run_config(
        "collecting" if args.dataset is None else "loading_dataset",
        completed_collect_steps=0,
    )

    print(
        f"environment={env_name} actions={env.action_size} "
        f"objectives={objective_spec.names} modes={num_modes} horizon={horizon}"
    )
    rng = jax.random.PRNGKey(args.seed)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    if args.dataset is not None:
        arrays = load_score_dataset(str(args.dataset))
        if len(arrays) != num_modes:
            raise ValueError(
                f"dataset has {len(arrays)} modes, expected {num_modes}"
            )
        # Evaluation starts from a fresh state, independent of the dataset.
        rng, reset_rng = jax.random.split(rng)
        state = reset_env(reset_rng)
    else:
        collector = DIALScoreCollector(
            env,
            horizon=horizon,
            num_samples=args.samples,
            temperature=args.temperature,
            objective_fn=objective_spec.cost,
            action_transform=interpolate_nodes,
            lock_first_action=True,
        )
        sigma_schedule = build_sigma_schedule(
            args.sigma_min, args.sigma_max, args.sigma_levels
        )
        # Each expert mode now drives its own rollout.  Previously only mode 0
        # advanced the environment, leaving the other heads without their own
        # on-policy state distribution.
        states = []
        base_means = []
        for mode_idx in range(num_modes):
            rng, reset_rng = jax.random.split(rng)
            states.append(reset_env(reset_rng))
            base_means.append(jnp.zeros((horizon, env.action_size)))
        collected = [[] for _ in range(num_modes)]
        total_states = args.collect_steps * num_modes

        print("collecting MPPI score targets from every expert mode")
        collection_bar = tqdm(
            total=total_states * (len(sigma_schedule) + 1),
            desc="MPPI data collection",
            unit="work",
            dynamic_ncols=True,
        )
        completed_states = 0
        for step_idx in range(args.collect_steps):
            for mode_idx in range(num_modes):
                state = states[mode_idx]
                base_mean = base_means[mode_idx]
                for sigma in sigma_schedule:
                    rng, collect_rng = jax.random.split(rng)
                    points, base_mean = collector.collect(
                        state,
                        state.obs,
                        float(sigma),
                        collect_rng,
                        mode_weights,
                        num_perturbations=args.perturbations,
                        base_mean=base_mean,
                        drive_mode=mode_idx,
                    )
                    base_mean.block_until_ready()
                    _extend(collected, points)
                    collection_bar.set_postfix(
                        mode=mode_idx,
                        state=f"{step_idx + 1}/{args.collect_steps}",
                        sigma=f"{float(sigma):.3g}",
                        refresh=False,
                    )
                    collection_bar.update()
                state = step_env(state, base_mean[0])
                state.reward.block_until_ready()
                states[mode_idx] = state
                base_means[mode_idx] = jnp.roll(
                    base_mean, -1, axis=0
                ).at[-1].set(0.0)
                completed_states += 1
                collection_bar.update()
                if (
                    args.data_checkpoint_every > 0
                    and completed_states % args.data_checkpoint_every == 0
                    and completed_states < total_states
                ):
                    partial_arrays = stack_score_data(collected)
                    save_score_dataset(
                        str(run_dir / "scores.partial.npz"), partial_arrays
                    )
                    write_run_config(
                        "collecting",
                        completed_collect_steps=completed_states,
                        partial_samples_per_mode=int(
                            partial_arrays[0][0].shape[0]
                        ),
                    )
        collection_bar.close()
        arrays = stack_score_data(collected)
        state = states[0]

    u_per_mode = [entry[0] for entry in arrays]
    sigma_per_mode = [entry[1] for entry in arrays]
    score_per_mode = [entry[2] for entry in arrays]
    obs_per_mode = [entry[3] for entry in arrays]
    save_score_dataset(str(run_dir / "scores.npz"), arrays)
    (run_dir / "scores.partial.npz").unlink(missing_ok=True)
    write_run_config(
        "training",
        completed_collect_steps=args.collect_steps,
        samples_per_mode=int(u_per_mode[0].shape[0]),
        completed_gradient_steps=0,
    )

    observation_size = int(state.obs.shape[0])
    if args.init_policy is None:
        normalizer = StandardNormalizer(observation_size)
        normalizer.fit(jnp.concatenate(obs_per_mode, axis=0))
        model = CompositionalDenoisingMLP(
            action_size=env.action_size,
            observation_size=observation_size,
            horizon=horizon,
            num_objectives=num_modes,
            encoder_hidden=(256, 256, 256),
            head_hidden=(128, 128),
            rngs=nnx.Rngs(args.seed),
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
    else:
        initial_policy = CompositionalPolicy.load(args.init_policy)
        model = initial_policy.model
        normalizer = initial_policy.normalizer
        if (
            model.action_size != env.action_size
            or model.observation_size != observation_size
            or model.horizon != horizon
            or model.num_objectives != num_modes
        ):
            raise ValueError("--init-policy architecture is incompatible")
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate))

    pinv_mode_weights = jnp.asarray(np.linalg.pinv(np.asarray(mode_weights)))

    def make_policy() -> CompositionalPolicy:
        return CompositionalPolicy(
            model=model,
            normalizer=normalizer,
            u_min=-jnp.ones((env.action_size,)),
            u_max=jnp.ones((env.action_size,)),
            mode_weights=mode_weights,
            pinv_mode_weights=pinv_mode_weights,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            num_steps=args.inference_steps,
            predicts_update=not args.predict_score,
            shift_matrix=shift_matrix,
            lock_first_action=True,
        )

    def save_training_checkpoint(completed_steps: int, checkpoint_loss) -> None:
        checkpoint_dir = checkpoints_dir / f"step_{completed_steps:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        make_policy().save(checkpoint_dir / "policy.pkl")
        metadata = {
            "completed_gradient_steps": completed_steps,
            "total_gradient_steps": args.train_iters * num_modes,
            "loss": float(checkpoint_loss),
        }
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        write_run_config("training", **metadata)
        tqdm.write(f"checkpoint={checkpoint_dir}")

    print("training compositional score model")
    onpolicy_arrays = (
        load_score_dataset(str(args.onpolicy_dataset))
        if args.onpolicy_dataset is not None
        else None
    )
    if onpolicy_arrays is not None and len(onpolicy_arrays) != num_modes:
        raise ValueError("on-policy dataset mode count is incompatible")
    rng, train_rng = jax.random.split(rng)
    loss = fit_score_regression(
        u_per_mode,
        sigma_per_mode,
        score_per_mode,
        obs_per_mode,
        model,
        optimizer,
        batch_size=args.batch_size,
        num_iters=args.train_iters,
        rng=train_rng,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        normalizer=normalizer,
        sigma_weight_power=args.score_loss_weight_power,
        predict_update=not args.predict_score,
        onpolicy_u_points_per_obj=(
            [entry[0] for entry in onpolicy_arrays]
            if onpolicy_arrays is not None
            else None
        ),
        onpolicy_sigmas_per_obj=(
            [entry[1] for entry in onpolicy_arrays]
            if onpolicy_arrays is not None
            else None
        ),
        onpolicy_scores_per_obj=(
            [entry[2] for entry in onpolicy_arrays]
            if onpolicy_arrays is not None
            else None
        ),
        onpolicy_obs_per_obj=(
            [entry[3] for entry in onpolicy_arrays]
            if onpolicy_arrays is not None
            else None
        ),
        onpolicy_frac=(
            args.onpolicy_frac if onpolicy_arrays is not None else 0.0
        ),
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=save_training_checkpoint,
    )

    policy = make_policy()
    policy.save(run_dir / "policy.pkl")
    write_run_config(
        "evaluating",
        completed_collect_steps=args.collect_steps,
        samples_per_mode=int(u_per_mode[0].shape[0]),
        completed_gradient_steps=args.train_iters * num_modes,
        final_loss=float(loss),
    )

    print("evaluating learned policy")
    if args.omega is None:
        omega = jnp.mean(mode_weights, axis=0)
    else:
        omega = jnp.asarray(
            [float(value) for value in args.omega.split(",")], dtype=jnp.float32
        )
        if omega.shape != (objective_dim,):
            raise ValueError(
                f"--omega needs {objective_dim} values for {objective_spec.names}"
            )
    plan = jnp.zeros((horizon, env.action_size))
    rollout = []
    apply_policy = jax.jit(
        lambda previous, observation, key: policy.apply(
            previous,
            observation,
            key,
            warm_start_level=1.0,
            omega=omega,
        )
    )
    for _ in tqdm(
        range(args.eval_steps),
        desc="Policy evaluation",
        unit="step",
        dynamic_ncols=True,
    ):
        rng, policy_rng = jax.random.split(rng)
        plan = apply_policy(plan, state.obs, policy_rng)
        plan.block_until_ready()
        state = step_env(state, plan[0])
        rollout.append(state.pipeline_state)
        plan = policy.shift(plan)

    rendered = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 720, True
    )
    with open(run_dir / "visualization.html", "w", encoding="utf-8") as stream:
        stream.write(rendered)

    write_run_config(
        "complete",
        completed_collect_steps=args.collect_steps,
        samples_per_mode=int(u_per_mode[0].shape[0]),
        completed_gradient_steps=args.train_iters * num_modes,
        final_loss=float(loss),
    )

    print(f"loss={float(loss):.6g}")
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
