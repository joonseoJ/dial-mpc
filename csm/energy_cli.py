"""Train a standalone Go2 controller with compositional trajectory energies."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs as brax_envs
from brax.io import html
from flax import nnx
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.architectures import CompositionalEnergyMLP, StandardNormalizer
from csm.energy_data import (
    EnergyDataset,
    ExactEnergyCollector,
    concatenate_energy_datasets,
    load_energy_dataset,
    save_energy_dataset,
)
from csm.energy_policy import (
    CompositionalEnergyPolicy,
    normalize_preference_weights,
)
from csm.energy_training import fit_compositional_energy
from csm.envs import get_objective_spec
from csm.exact_cli import (
    _load_config,
    _physically_fallen,
    _rollout_metrics,
    _set_mode,
)
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import load_dataclass_from_dict


def _parse_episode_lengths(text: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "episode lengths must be comma-separated integers"
        ) from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("episode lengths must all be positive")
    return lengths


def _sample_episode_limit(rng, episode_lengths):
    if not episode_lengths:
        return rng, None
    rng, key = jax.random.split(rng)
    index = int(jax.random.randint(key, (), 0, len(episode_lengths)))
    return rng, int(episode_lengths[index])


def _dagger_beta_schedule(rounds, mixed_beta, student_only_rounds):
    if not 0 <= student_only_rounds <= rounds:
        raise ValueError(
            "student-only DAgger rounds must be between zero and dagger rounds"
        )
    return [mixed_beta] * (rounds - student_only_rounds) + [
        0.0
    ] * student_only_rounds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example")
    source.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("csm_runs/energy"))
    parser.add_argument("--mode-weights", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--collect-steps", type=int, default=400)
    parser.add_argument(
        "--collection-episode-steps",
        type=int,
        default=0,
        help="legacy fixed reset interval; use --collection-episode-lengths",
    )
    parser.add_argument(
        "--collection-episode-lengths",
        type=_parse_episode_lengths,
        default=None,
        help=(
            "randomly sample a reset interval from this comma-separated pool; "
            "duplicates control the short/long episode mixture"
        ),
    )
    parser.add_argument("--energy-candidates", type=int, default=64)
    parser.add_argument("--teacher-repeats", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=30_000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-steps", type=int, default=200)
    parser.add_argument("--dagger-train-iters", type=int, default=15_000)
    parser.add_argument("--dagger-beta", type=float, default=0.5)
    parser.add_argument(
        "--student-only-dagger-rounds",
        type=int,
        default=1,
        help="number of final DAgger rounds collected with beta=0",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--guidance-weight", type=float, default=0.3)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--sobolev-weight", type=float, default=0.2)
    parser.add_argument("--energy-steps", type=int, default=8)
    parser.add_argument("--energy-step-size", type=float, default=1.0)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--minimum-mean-vx", type=float, default=0.5)
    parser.add_argument(
        "--selection-track-command",
        action="store_true",
        help="score tracking against each randomized velocity/yaw command",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--selection-steps", type=int, default=300)
    parser.add_argument("--selection-seeds", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _collect_base(
    env,
    planner,
    collector,
    reset,
    step,
    mode_weights,
    initial_factors,
    regular_factors,
    collect_steps,
    episode_lengths,
    rng,
):
    chunks = []
    states = []
    plans = []
    episode_ages = []
    episode_limits = []
    horizon = planner.args.Hnode + 1
    for omega in mode_weights:
        rng, key = jax.random.split(rng)
        states.append(_set_mode(reset(key), omega))
        plans.append(jnp.zeros((horizon, env.action_size)))
        episode_ages.append(0)
        rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
        episode_limits.append(episode_limit)
    bar = tqdm(
        total=collect_steps * len(mode_weights),
        desc="Compositional energy collection",
        unit="state",
    )
    for state_idx in range(collect_steps):
        for mode_idx, omega in enumerate(mode_weights):
            state, plan = states[mode_idx], plans[mode_idx]
            episode_limit = episode_limits[mode_idx]
            if episode_limit is not None and episode_ages[mode_idx] >= episode_limit:
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_ages[mode_idx] = 0
                rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
                episode_limits[mode_idx] = episode_limit
            state = step(_set_mode(state, omega), plan[0])
            state.reward.block_until_ready()
            if _physically_fallen(env, state):
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_ages[mode_idx] = 0
                rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
                episode_limits[mode_idx] = episode_limit
            plan = planner.shift(plan)
            factors = initial_factors if state_idx == 0 else regular_factors
            for factor in factors:
                chunk, plan, rng = collector.collect_query(
                    state, plan, float(factor), rng, omega
                )
                chunk.costs.block_until_ready()
                chunks.append(chunk)
            states[mode_idx], plans[mode_idx] = state, plan
            episode_ages[mode_idx] += 1
            bar.update()
    bar.close()
    return concatenate_energy_datasets(chunks), rng


def _collect_dagger(
    env,
    planner,
    collector,
    policy,
    reset,
    step,
    mode_weights,
    regular_factors,
    dagger_steps,
    episode_lengths,
    beta,
    rng,
    round_idx,
):
    chunks = []
    apply = jax.jit(
        lambda plan, obs, key, omega: policy.apply(
            plan, obs, key, warm_start_level=1.0, omega=omega
        )
    )
    bar = tqdm(
        total=dagger_steps * len(mode_weights),
        desc=f"Energy DAgger round {round_idx}",
        unit="state",
    )
    for omega in mode_weights:
        rng, key = jax.random.split(rng)
        state = _set_mode(reset(key), omega)
        plan = jnp.zeros((planner.args.Hnode + 1, env.action_size))
        episode_age = 0
        rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
        for state_idx in range(dagger_steps):
            if episode_limit is not None and episode_age >= episode_limit:
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_age = 0
                rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
            state = step(_set_mode(state, omega), plan[0])
            state.reward.block_until_ready()
            if _physically_fallen(env, state):
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_age = 0
                rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
            shifted = planner.shift(plan)
            rng, policy_rng = jax.random.split(rng)
            learner = apply(shifted, state.obs, policy_rng, omega)
            teacher_plan = learner
            for factor in regular_factors:
                chunk, teacher_plan, rng = collector.collect_query(
                    state, teacher_plan, float(factor), rng, omega
                )
                chunk.costs.block_until_ready()
                chunks.append(chunk)
            plan = jnp.clip((1.0 - beta) * learner + beta * teacher_plan, -1.0, 1.0)
            episode_age += 1
            bar.update()
    bar.close()
    return concatenate_energy_datasets(chunks), rng


def _balanced_cost_statistics(datasets):
    """Mean/std of an equal mixture of base and every DAgger round."""
    means = jnp.stack([jnp.mean(dataset.costs, axis=0) for dataset in datasets])
    second_moments = jnp.stack(
        [jnp.mean(jnp.square(dataset.costs), axis=0) for dataset in datasets]
    )
    mean = jnp.mean(means, axis=0)
    variance = jnp.maximum(jnp.mean(second_moments, axis=0) - mean**2, 1e-6)
    return mean, jnp.maximum(jnp.sqrt(variance), 1e-3)


def _reparameterize_energy_normalization(
    model, old_mean, old_std, new_mean, new_std
):
    """Change normalized output coordinates while preserving raw energies."""
    for objective_idx in range(model.num_objectives):
        head = getattr(model, f"energy_head{objective_idx}")
        output_layer = getattr(head, f"l{head.num_hidden}")
        scale = old_std[objective_idx] / new_std[objective_idx]
        shift = (old_mean[objective_idx] - new_mean[objective_idx]) / new_std[
            objective_idx
        ]
        output_layer.kernel.value = output_layer.kernel.value * scale
        output_layer.bias.value = output_layer.bias.value * scale + shift


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    if args.smoke:
        args.samples = 8
        args.collect_steps = 1
        args.energy_candidates = 4
        args.teacher_repeats = 1
        args.train_iters = 2
        args.dagger_rounds = 1
        args.dagger_steps = 1
        args.dagger_train_iters = 2
        args.batch_size = 2
        args.energy_steps = 2
        args.checkpoint_every = 1
        args.selection_steps = 2
        args.selection_seeds = 1
        args.eval_steps = 2

    episode_lengths = args.collection_episode_lengths
    if episode_lengths is None:
        episode_lengths = (
            (args.collection_episode_steps,)
            if args.collection_episode_steps > 0
            else ()
        )

    env_name = config["env_name"]
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(env_name), config, convert_list_to_array=True
    )
    env = brax_envs.get_environment(env_name, config=env_config)
    objective_spec = get_objective_spec(env)
    mode_path = args.mode_weights or Path(__file__).with_name("go2_modes.json")
    mode_weights = jnp.asarray(
        json.loads(mode_path.read_text(encoding="utf-8")), dtype=jnp.float32
    )
    if mode_weights.ndim != 2 or jnp.any(mode_weights < 0.0):
        raise ValueError("mode weights must be a non-negative matrix")
    if mode_weights.shape[1] != objective_spec.num_objectives:
        raise ValueError("mode weights do not match objective count")
    mode_sums = np.asarray(jnp.sum(mode_weights, axis=1))
    if np.any(mode_sums <= 0.0):
        raise ValueError("every mode weight vector must have positive sum")
    preference_weight_sum = float(np.median(mode_sums))
    mode_weights = normalize_preference_weights(
        mode_weights, preference_weight_sum
    )
    if not 0.0 <= args.dagger_beta <= 1.0:
        raise ValueError("dagger beta must be in [0, 1]")
    if args.collection_episode_steps < 0:
        raise ValueError("collection episode steps must be non-negative")
    dagger_beta_schedule = _dagger_beta_schedule(
        args.dagger_rounds,
        args.dagger_beta,
        args.student_only_dagger_rounds,
    )

    dial_config = load_dataclass_from_dict(DialConfig, config)
    if args.samples is not None:
        dial_config.Nsample = args.samples
    if args.temperature is not None:
        dial_config.temp_sample = args.temperature
    if args.energy_candidates > dial_config.Nsample + 1:
        raise ValueError("energy candidates cannot exceed DIAL samples + query")
    planner = MBDPI(dial_config, env)
    collector = ExactEnergyCollector(
        planner,
        objective_spec.cost,
        num_candidates=args.energy_candidates,
        teacher_repeats=args.teacher_repeats,
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    horizon = dial_config.Hnode + 1
    node_basis = jnp.eye(horizon)

    def shift_basis(basis):
        plan = jnp.zeros((horizon, env.action_size))
        return planner.shift(plan.at[:, 0].set(basis))[:, 0]

    shift_matrix = jax.vmap(shift_basis)(node_basis).T
    regular_factors = jnp.asarray(
        dial_config.traj_diffuse_factor ** np.arange(dial_config.Ndiffuse),
        dtype=jnp.float32,
    )
    initial_factors = jnp.asarray(
        dial_config.traj_diffuse_factor ** np.arange(dial_config.Ndiffuse_init),
        dtype=jnp.float32,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output / f"{env_name}-{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        base = load_energy_dataset(args.dataset)
    else:
        base, rng = _collect_base(
            env,
            planner,
            collector,
            reset,
            step,
            mode_weights,
            initial_factors,
            regular_factors,
            args.collect_steps,
            episode_lengths,
            rng,
        )
    save_energy_dataset(run_dir / "energy_data.npz", base)
    # Retain the conventional artifact name expected by existing automation.
    save_energy_dataset(run_dir / "scores.npz", base)

    normalizer = StandardNormalizer(int(base.observations.shape[-1]))
    normalizer.fit(base.observations)
    cost_mean, cost_std = _balanced_cost_statistics([base])
    model = CompositionalEnergyMLP(
        action_size=env.action_size,
        observation_size=int(base.observations.shape[-1]),
        horizon=horizon,
        num_objectives=objective_spec.num_objectives,
        encoder_hidden=(512, 512, 512),
        head_hidden=(256, 256),
        rngs=nnx.Rngs(args.seed),
    )
    def make_optimizer():
        return nnx.Optimizer(model, optax.adam(args.learning_rate))

    optimizer = make_optimizer()

    def make_policy():
        return CompositionalEnergyPolicy(
            model=model,
            normalizer=normalizer,
            cost_mean=cost_mean,
            cost_std=cost_std,
            mode_weights=mode_weights,
            u_min=-jnp.ones(env.action_size),
            u_max=jnp.ones(env.action_size),
            shift_matrix=shift_matrix,
            num_steps=args.energy_steps,
            step_size=args.energy_step_size,
            lock_first_action=True,
            trust_radius=args.trust_radius,
            preference_weight_sum=preference_weight_sum,
        )

    checkpoints = []
    global_step = 0
    phase = "base"

    def checkpoint(local_step, loss):
        path = checkpoints_dir / f"step_{global_step + local_step:06d}_{phase}" / "policy.pkl"
        make_policy().save(path)
        path.with_name("metadata.json").write_text(
            json.dumps({"step": global_step + local_step, "loss": float(loss), "phase": phase}, indent=2),
            encoding="utf-8",
        )
        checkpoints.append(path)

    def train(dataset, iterations):
        nonlocal rng, global_step
        rng, train_rng = jax.random.split(rng)
        loss = fit_compositional_energy(
            dataset,
            model,
            normalizer,
            optimizer,
            cost_mean,
            cost_std,
            batch_size=args.batch_size,
            num_iters=iterations,
            rng=train_rng,
            guidance_weight=args.guidance_weight,
            calibration_weight=args.calibration_weight,
            sobolev_weight=args.sobolev_weight,
            preference_weight_sum=preference_weight_sum,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=checkpoint,
        )
        global_step += iterations
        return loss

    loss = train([base], args.train_iters)
    base_path = checkpoints_dir / f"step_{global_step:06d}_base_final" / "policy.pkl"
    make_policy().save(base_path)
    checkpoints.append(base_path)

    dagger_sets = []
    for round_zero in range(args.dagger_rounds):
        phase = f"dagger_{round_zero + 1}"
        dagger, rng = _collect_dagger(
            env,
            planner,
            collector,
            make_policy(),
            reset,
            step,
            mode_weights,
            regular_factors,
            args.dagger_steps,
            episode_lengths,
            dagger_beta_schedule[round_zero],
            rng,
            round_zero + 1,
        )
        dagger_sets.append(dagger)
        save_energy_dataset(run_dir / f"dagger_round_{round_zero + 1}.npz", dagger)
        aggregated = concatenate_energy_datasets([base, *dagger_sets])
        save_energy_dataset(run_dir / "dagger_aggregated.npz", aggregated)
        balanced_groups = [base, *dagger_sets]
        new_mean, new_std = _balanced_cost_statistics(balanced_groups)
        _reparameterize_energy_normalization(
            model, cost_mean, cost_std, new_mean, new_std
        )
        cost_mean, cost_std = new_mean, new_std
        # Adam moments live in the old normalized coordinates.  Restarting the
        # optimizer avoids carrying incompatible moments across the exact head
        # reparameterization above.
        optimizer = make_optimizer()
        loss = train(balanced_groups, args.dagger_train_iters)
        final_path = checkpoints_dir / f"step_{global_step:06d}_{phase}_final" / "policy.pkl"
        make_policy().save(final_path)
        checkpoints.append(final_path)

    selection = []
    best = None
    for path in tqdm(
        list(dict.fromkeys(checkpoints)),
        desc="Energy rollout checkpoint selection",
        unit="checkpoint",
    ):
        candidate = CompositionalEnergyPolicy.load(path)
        metrics = _rollout_metrics(
            env,
            candidate,
            mode_weights,
            args.selection_steps,
            args.selection_seeds,
            minimum_mean_vx=args.minimum_mean_vx,
            track_command=args.selection_track_command,
        )
        metrics["policy"] = str(path)
        selection.append(metrics)
        if best is None or metrics["selection_score"] > best[0]:
            best = (metrics["selection_score"], path, metrics)
    assert best is not None
    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    shutil.copy2(best[1], run_dir / "policy.pkl")

    policy = CompositionalEnergyPolicy.load(run_dir / "policy.pkl")
    rng, key = jax.random.split(rng)
    state = _set_mode(reset(key), mode_weights[0])
    plan = jnp.zeros((horizon, env.action_size))
    apply = jax.jit(
        lambda p, o, k: policy.apply(
            p, o, k, warm_start_level=1.0, omega=mode_weights[0]
        )
    )
    rollout = []
    for _ in tqdm(range(args.eval_steps), desc="Selected energy policy rendering", unit="step"):
        rng, key = jax.random.split(rng)
        plan = apply(plan, state.obs, key)
        state = step(_set_mode(state, mode_weights[0]), plan[0])
        state.reward.block_until_ready()
        rollout.append(state.pipeline_state)
        plan = policy.shift(plan)
        if _physically_fallen(env, state):
            break
    (run_dir / "visualization.html").write_text(
        html.render(env.sys, rollout), encoding="utf-8"
    )
    config_out = {
        **vars(args),
        "framework": "objective_compositional_energy",
        "objective_names": list(objective_spec.names),
        "mode_weights": np.asarray(mode_weights).tolist(),
        "cost_mean": np.asarray(cost_mean).tolist(),
        "cost_std": np.asarray(cost_std).tolist(),
        "preference_weight_sum": preference_weight_sum,
        "selected": str(best[1]),
        "selected_metrics": best[2],
        "completed_gradient_steps": global_step,
        "resolved_collection_episode_lengths": list(episode_lengths),
        "dagger_beta_schedule": dagger_beta_schedule,
        "environment_randomization": {
            key: value
            for key, value in config.items()
            if key == "randomize_tasks"
            or key == "randomize_start_state"
            or key == "include_foot_height_observation"
            or key.startswith("command_")
            or key.startswith("start_")
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in config_out.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"selected={best[1]}")
    print(f"metrics={json.dumps(best[2])}")
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
