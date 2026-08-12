"""Train a standalone Go2 controller with compositional trajectory energies."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs as brax_envs
from brax import math as brax_math
from brax.io import html
from flax import nnx
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.architectures import CompositionalEnergyMLP, StandardNormalizer
from csm.energy_data import (
    EnergyDataset,
    ExactEnergyCollector,
    concatenate_closed_loop_datasets,
    concatenate_energy_datasets,
    load_energy_dataset,
    make_closed_loop_windows,
    save_closed_loop_dataset,
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


def _parse_closed_loop_horizons(text: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "closed-loop horizons must be comma-separated integers"
        ) from exc
    if not horizons or any(value < 1 for value in horizons):
        raise argparse.ArgumentTypeError(
            "closed-loop horizons must all be positive"
        )
    return horizons


def _resolve_closed_loop_horizons(rounds, curriculum, override=None):
    if override is not None:
        if override < 1:
            raise ValueError("closed-loop steps must be positive")
        return [int(override)] * rounds
    return [int(curriculum[min(i, len(curriculum) - 1)]) for i in range(rounds)]


def _split_integer(total: int, parts: int) -> list[int]:
    """Split work nearly equally while preserving the exact total."""
    if total < 1 or parts < 1:
        raise ValueError("split total and parts must be positive")
    quotient, remainder = divmod(total, parts)
    return [
        quotient + (i < remainder)
        for i in range(parts)
        if quotient or i < remainder
    ]


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


def _state_recovery_difficulty(env, state) -> tuple[float, bool]:
    """Return a dimensionless physical-risk score and recoverability flag."""
    torso = int(env._torso_idx) - 1
    rotation = state.pipeline_state.x.rot[torso]
    euler = brax_math.quat_to_euler(rotation)
    tilt = float(jnp.linalg.norm(euler[:2]))
    height = float(state.pipeline_state.x.pos[torso, 2])
    angular_speed = float(
        jnp.linalg.norm(state.pipeline_state.xd.ang[torso] * jnp.pi / 180.0)
    )
    tilt_risk = np.clip(tilt / (0.5 * np.pi), 0.0, 1.5)
    height_risk = np.clip((0.30 - height) / 0.12, 0.0, 1.5)
    angular_risk = np.clip(angular_speed / 4.0, 0.0, 1.5)
    score = 0.55 * tilt_risk + 0.25 * height_risk + 0.20 * angular_risk
    recoverable = height > 0.20 and tilt < 1.25
    return float(score), bool(recoverable)


def _annotate_recovery_difficulty(dataset, state_score, trust_radius):
    """Combine physical risk with the bounded teacher correction size."""
    update = dataset.guidance_updates.at[:, 0].set(0.0)
    update_rms = jnp.sqrt(jnp.mean(jnp.square(update), axis=(-2, -1)) + 1e-8)
    correction = jnp.clip(update_rms / max(float(trust_radius), 1e-6), 0.0, 2.0)
    return replace(
        dataset,
        guidance_recovery_difficulty=jnp.asarray(state_score) + correction,
    )


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
    parser.add_argument(
        "--student-only-learning-rate",
        type=float,
        default=2e-5,
        help="lower learning rate for beta=0 DAgger rounds",
    )
    parser.add_argument(
        "--student-only-train-iters",
        type=int,
        default=10_000,
        help="maximum optimizer steps in each beta=0 DAgger round",
    )
    parser.add_argument(
        "--student-only-eval-every",
        type=int,
        default=2_500,
        help="rollout-validation interval for beta=0 early stopping",
    )
    parser.add_argument(
        "--student-only-early-stop-patience",
        type=int,
        default=2,
        help="number of non-improving rollout validations before stopping",
    )
    parser.add_argument("--guidance-weight", type=float, default=0.3)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--sobolev-weight", type=float, default=0.2)
    parser.add_argument("--deployment-weight", type=float, default=0.2)
    parser.add_argument(
        "--deployment-direction-weight",
        type=float,
        default=0.3,
        help="cosine loss weight for the final unrolled deployment update",
    )
    parser.add_argument(
        "--conditional-magnitude-weight",
        type=float,
        default=0.1,
        help="under-predicted saturated-update magnitude loss weight",
    )
    parser.add_argument(
        "--conditional-magnitude-cosine",
        type=float,
        default=0.7,
        help="direction cosine where hard-query magnitude supervision turns on",
    )
    parser.add_argument(
        "--conditional-magnitude-temperature",
        type=float,
        default=0.1,
        help="soft direction gate temperature for conditional magnitude loss",
    )
    parser.add_argument(
        "--deployment-batch-size",
        type=int,
        default=8,
        help="guidance queries used for differentiable final-update supervision",
    )
    parser.add_argument(
        "--sobolev-influence-cap",
        type=float,
        default=2.0,
        help="raw composed-gradient RMS where per-query influence starts shrinking",
    )
    parser.add_argument("--closed-loop-weight", type=float, default=0.05)
    parser.add_argument(
        "--closed-loop-steps",
        type=int,
        default=None,
        help="deprecated constant-horizon override for the curriculum",
    )
    parser.add_argument(
        "--closed-loop-horizon-curriculum",
        type=_parse_closed_loop_horizons,
        default=(4, 4, 6, 8, 12),
        help="comma-separated DAgger-round temporal horizons",
    )
    parser.add_argument("--closed-loop-batch-size", type=int, default=8)
    parser.add_argument("--closed-loop-every", type=int, default=4)
    parser.add_argument("--closed-loop-discount", type=float, default=0.9)
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
    parser.add_argument("--final-selection-steps", type=int, default=500)
    parser.add_argument("--final-selection-seeds", type=int, default=10)
    parser.add_argument("--selection-finalists", type=int, default=5)
    parser.add_argument("--hard-recovery-steps", type=int, default=500)
    parser.add_argument("--hard-recovery-window", type=int, default=24)
    parser.add_argument("--hard-recovery-queries-per-mode", type=int, default=64)
    parser.add_argument("--hard-recovery-teacher-repeats", type=int, default=16)
    parser.add_argument("--hard-recovery-train-iters", type=int, default=5_000)
    parser.add_argument("--hard-recovery-learning-rate", type=float, default=1e-5)
    parser.add_argument("--hard-recovery-eval-every", type=int, default=1_000)
    parser.add_argument("--hard-recovery-early-stop-patience", type=int, default=2)
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
    trust_radius,
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
            state_difficulty, _ = _state_recovery_difficulty(env, state)
            factors = initial_factors if state_idx == 0 else regular_factors
            for factor in factors:
                chunk, plan, rng = collector.collect_query_with_difficulty(
                    state,
                    plan,
                    float(factor),
                    rng,
                    omega,
                    state_difficulty,
                )
                chunk = _annotate_recovery_difficulty(
                    chunk, state_difficulty, trust_radius
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
    closed_loop_steps,
    trust_radius,
):
    chunks = []
    closed_loop_chunks = []
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

    def finish_sequence(records):
        if not records:
            return
        warm_starts, observations, omegas, teacher_plans = zip(*records)
        closed_loop_chunks.append(
            make_closed_loop_windows(
                jnp.stack(warm_starts),
                jnp.stack(observations),
                jnp.stack(omegas),
                jnp.stack(teacher_plans),
                closed_loop_steps,
            )
        )
        records.clear()

    for omega in mode_weights:
        rng, key = jax.random.split(rng)
        state = _set_mode(reset(key), omega)
        plan = jnp.zeros((planner.args.Hnode + 1, env.action_size))
        episode_age = 0
        sequence_records = []
        rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
        for state_idx in range(dagger_steps):
            if episode_limit is not None and episode_age >= episode_limit:
                finish_sequence(sequence_records)
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_age = 0
                rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
            state = step(_set_mode(state, omega), plan[0])
            state.reward.block_until_ready()
            if _physically_fallen(env, state):
                finish_sequence(sequence_records)
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_age = 0
                rng, episode_limit = _sample_episode_limit(rng, episode_lengths)
            shifted = planner.shift(plan)
            rng, policy_rng = jax.random.split(rng)
            learner = apply(shifted, state.obs, policy_rng, omega)
            teacher_plan = learner
            state_difficulty, _ = _state_recovery_difficulty(env, state)
            for factor in regular_factors:
                chunk, teacher_plan, rng = collector.collect_query_with_difficulty(
                    state,
                    teacher_plan,
                    float(factor),
                    rng,
                    omega,
                    state_difficulty,
                )
                chunk = _annotate_recovery_difficulty(
                    chunk, state_difficulty, trust_radius
                )
                chunk.costs.block_until_ready()
                chunks.append(chunk)
            sequence_records.append(
                (shifted, state.obs, jnp.asarray(omega), teacher_plan)
            )
            plan = jnp.clip((1.0 - beta) * learner + beta * teacher_plan, -1.0, 1.0)
            episode_age += 1
            bar.update()
        finish_sequence(sequence_records)
    bar.close()
    return (
        concatenate_energy_datasets(chunks),
        concatenate_closed_loop_datasets(closed_loop_chunks),
        rng,
    )


def _collect_hard_recovery(
    env,
    planner,
    collector,
    policy,
    reset,
    step,
    mode_weights,
    rollout_steps,
    recovery_window,
    queries_per_mode,
    factor,
    trust_radius,
    rng,
):
    """Relabel recoverable student states nearest to failure with DIAL."""
    apply = jax.jit(
        lambda plan, obs, key, omega: policy.apply(
            plan, obs, key, warm_start_level=1.0, omega=omega
        )
    )
    selected = []
    rollout_bar = tqdm(
        total=rollout_steps * len(mode_weights),
        desc="Hard recovery state search",
        unit="state",
    )
    for omega in mode_weights:
        rng, key = jax.random.split(rng)
        state = _set_mode(reset(key), omega)
        plan = jnp.zeros((planner.args.Hnode + 1, env.action_size))
        episode_records = []
        candidates = []
        for _ in range(rollout_steps):
            state = step(_set_mode(state, omega), plan[0])
            state.reward.block_until_ready()
            if _physically_fallen(env, state):
                recent = episode_records[-recovery_window:]
                for distance, record in enumerate(reversed(recent)):
                    proximity = 1.0 - distance / max(len(recent), 1)
                    record["difficulty"] += 2.0 * proximity
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                episode_records = []
                rollout_bar.update()
                continue
            shifted = planner.shift(plan)
            rng, policy_rng = jax.random.split(rng)
            learner = apply(shifted, state.obs, policy_rng, omega)
            physical_risk, recoverable = _state_recovery_difficulty(env, state)
            if recoverable:
                record = {
                    "state": state,
                    "query": learner,
                    "omega": jnp.asarray(omega),
                    "difficulty": physical_risk,
                }
                candidates.append(record)
                episode_records.append(record)
            plan = learner
            rollout_bar.update()
        candidates.sort(key=lambda record: record["difficulty"], reverse=True)
        selected.extend(candidates[:queries_per_mode])
    rollout_bar.close()
    if not selected:
        raise RuntimeError("hard recovery search found no recoverable student states")

    chunks = []
    relabel_bar = tqdm(selected, desc="Hard recovery DIAL relabel", unit="query")
    for record in relabel_bar:
        chunk, _, rng = collector.collect_query_with_difficulty(
            record["state"],
            record["query"],
            factor,
            rng,
            record["omega"],
            record["difficulty"],
        )
        chunk = _annotate_recovery_difficulty(
            chunk, record["difficulty"], trust_radius
        )
        chunk.costs.block_until_ready()
        chunks.append(chunk)
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
        # Two states exercise two independent student-only relabel cycles.
        args.dagger_steps = 2
        args.dagger_train_iters = 2
        args.student_only_train_iters = 2
        args.student_only_eval_every = 1
        args.student_only_early_stop_patience = 1
        args.batch_size = 2
        args.deployment_batch_size = 1
        args.closed_loop_batch_size = 1
        args.closed_loop_every = 1
        args.energy_steps = 2
        args.checkpoint_every = 1
        args.selection_steps = 2
        args.selection_seeds = 1
        args.final_selection_steps = 2
        args.final_selection_seeds = 1
        args.selection_finalists = 1
        args.hard_recovery_steps = 2
        args.hard_recovery_window = 1
        args.hard_recovery_queries_per_mode = 1
        args.hard_recovery_teacher_repeats = 1
        args.hard_recovery_train_iters = 2
        args.hard_recovery_eval_every = 1
        args.hard_recovery_early_stop_patience = 1
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
    if args.sobolev_influence_cap <= 0.0:
        raise ValueError("Sobolev influence cap must be positive")
    if (
        args.deployment_weight < 0.0
        or args.deployment_direction_weight < 0.0
        or args.conditional_magnitude_weight < 0.0
        or args.closed_loop_weight < 0.0
    ):
        raise ValueError(
            "deployment direction/vector and closed-loop weights must be non-negative"
        )
    if args.deployment_batch_size < 1:
        raise ValueError("deployment batch size must be positive")
    if not -1.0 <= args.conditional_magnitude_cosine <= 1.0:
        raise ValueError("conditional magnitude cosine must be in [-1, 1]")
    if args.conditional_magnitude_temperature <= 0.0:
        raise ValueError("conditional magnitude temperature must be positive")
    if args.trust_radius <= 0.0:
        raise ValueError("trust radius must be positive")
    if args.student_only_learning_rate <= 0.0:
        raise ValueError("student-only learning rate must be positive")
    if args.student_only_train_iters < 1:
        raise ValueError("student-only train iterations must be positive")
    if args.student_only_eval_every < 1:
        raise ValueError("student-only evaluation interval must be positive")
    if args.student_only_early_stop_patience < 1:
        raise ValueError("student-only early-stop patience must be positive")
    if args.closed_loop_steps is not None and args.closed_loop_steps < 1:
        raise ValueError("closed-loop steps must be positive")
    if args.closed_loop_every < 1:
        raise ValueError("closed-loop frequency must be positive")
    if args.closed_loop_batch_size < 1:
        raise ValueError("closed-loop batch size must be positive")
    if not 0.0 < args.closed_loop_discount <= 1.0:
        raise ValueError("closed-loop discount must be in (0, 1]")
    if args.collection_episode_steps < 0:
        raise ValueError("collection episode steps must be non-negative")
    positive_integer_options = {
        "final selection steps": args.final_selection_steps,
        "final selection seeds": args.final_selection_seeds,
        "selection finalists": args.selection_finalists,
        "hard recovery steps": args.hard_recovery_steps,
        "hard recovery window": args.hard_recovery_window,
        "hard recovery queries per mode": args.hard_recovery_queries_per_mode,
        "hard recovery teacher repeats": args.hard_recovery_teacher_repeats,
        "hard recovery train iterations": args.hard_recovery_train_iters,
        "hard recovery evaluation interval": args.hard_recovery_eval_every,
        "hard recovery early-stop patience": args.hard_recovery_early_stop_patience,
    }
    for name, value in positive_integer_options.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.hard_recovery_learning_rate <= 0.0:
        raise ValueError("hard recovery learning rate must be positive")
    dagger_beta_schedule = _dagger_beta_schedule(
        args.dagger_rounds,
        args.dagger_beta,
        args.student_only_dagger_rounds,
    )
    closed_loop_horizons = _resolve_closed_loop_horizons(
        args.dagger_rounds,
        args.closed_loop_horizon_curriculum,
        args.closed_loop_steps,
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
            args.trust_radius,
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

    def make_optimizer(learning_rate=None):
        rate = args.learning_rate if learning_rate is None else learning_rate
        return nnx.Optimizer(model, optax.adam(rate))

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
    early_stop_context = None

    def checkpoint(local_step, loss):
        path = checkpoints_dir / f"step_{global_step + local_step:06d}_{phase}" / "policy.pkl"
        make_policy().save(path)
        metadata = {
            "step": global_step + local_step,
            "loss": float(loss),
            "phase": phase,
        }
        should_stop = False
        if early_stop_context is not None:
            metrics = _rollout_metrics(
                env,
                make_policy(),
                mode_weights,
                early_stop_context.get("selection_steps", args.selection_steps),
                early_stop_context.get("selection_seeds", args.selection_seeds),
                minimum_mean_vx=args.minimum_mean_vx,
                track_command=args.selection_track_command,
                common_seeds=True,
                worst_mode_selection=True,
            )
            metadata["early_stop_metrics"] = metrics
            score = metrics["selection_score"]
            early_stop_context["last_score"] = score
            early_stop_context["last_path"] = path
            if score > early_stop_context["best_score"]:
                early_stop_context["best_score"] = score
                early_stop_context["best_path"] = path
                early_stop_context["stale"] = 0
            else:
                early_stop_context["stale"] += 1
            should_stop = (
                early_stop_context["stale"]
                >= early_stop_context.get(
                    "patience", args.student_only_early_stop_patience
                )
            )
        path.with_name("metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        checkpoints.append(path)
        return should_stop

    def train(dataset, iterations, closed_loop=None, checkpoint_interval=None):
        nonlocal rng, global_step
        rng, train_rng = jax.random.split(rng)
        loss, completed = fit_compositional_energy(
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
            deployment_weight=args.deployment_weight,
            deployment_direction_weight=args.deployment_direction_weight,
            conditional_magnitude_weight=args.conditional_magnitude_weight,
            conditional_magnitude_cosine=args.conditional_magnitude_cosine,
            conditional_magnitude_temperature=(
                args.conditional_magnitude_temperature
            ),
            deployment_batch_size=args.deployment_batch_size,
            sobolev_influence_cap=args.sobolev_influence_cap,
            trust_radius=args.trust_radius,
            closed_loop_dataset=closed_loop,
            closed_loop_weight=args.closed_loop_weight,
            closed_loop_batch_size=args.closed_loop_batch_size,
            closed_loop_every=args.closed_loop_every,
            closed_loop_discount=args.closed_loop_discount,
            energy_steps=args.energy_steps,
            energy_step_size=args.energy_step_size,
            shift_matrix=shift_matrix,
            preference_weight_sum=preference_weight_sum,
            checkpoint_every=(
                args.checkpoint_every
                if checkpoint_interval is None
                else checkpoint_interval
            ),
            checkpoint_callback=checkpoint,
        )
        global_step += completed
        return loss

    loss = train([base], args.train_iters)
    base_path = checkpoints_dir / f"step_{global_step:06d}_base_final" / "policy.pkl"
    make_policy().save(base_path)
    checkpoints.append(base_path)

    dagger_sets = []
    closed_loop_sets = []
    dagger_training_history = []
    for round_zero in range(args.dagger_rounds):
        phase = f"dagger_{round_zero + 1}"
        student_only = dagger_beta_schedule[round_zero] == 0.0
        learning_rate = (
            args.student_only_learning_rate if student_only else args.learning_rate
        )
        iterations = (
            min(args.dagger_train_iters, args.student_only_train_iters)
            if student_only
            else args.dagger_train_iters
        )
        horizon_for_round = closed_loop_horizons[round_zero]
        if student_only:
            cycle_count = min(
                args.dagger_steps,
                max(
                    1,
                    int(np.ceil(iterations / args.student_only_eval_every)),
                ),
            )
            train_splits = _split_integer(iterations, cycle_count)
            collection_splits = _split_integer(args.dagger_steps, cycle_count)
        else:
            train_splits = [iterations]
            collection_splits = [args.dagger_steps]
        early_stop_context = (
            {
                "best_score": -np.inf,
                "best_path": None,
                "last_score": None,
                "last_path": None,
                "stale": 0,
            }
            if student_only
            else None
        )
        round_start_step = global_step
        round_dagger_chunks = []
        round_closed_loop_chunks = []
        cycle_history = []
        for cycle_zero, (cycle_collection_steps, cycle_train_steps) in enumerate(
            zip(collection_splits, train_splits, strict=True)
        ):
            dagger_chunk, closed_loop_chunk, rng = _collect_dagger(
                env,
                planner,
                collector,
                make_policy(),
                reset,
                step,
                mode_weights,
                regular_factors,
                cycle_collection_steps,
                episode_lengths,
                dagger_beta_schedule[round_zero],
                rng,
                round_zero + 1,
                horizon_for_round,
                args.trust_radius,
            )
            round_dagger_chunks.append(dagger_chunk)
            round_closed_loop_chunks.append(closed_loop_chunk)
            round_dagger = concatenate_energy_datasets(round_dagger_chunks)
            round_closed_loop = concatenate_closed_loop_datasets(
                round_closed_loop_chunks
            )
            if student_only:
                save_energy_dataset(
                    run_dir
                    / f"dagger_round_{round_zero + 1}_cycle_{cycle_zero + 1}.npz",
                    dagger_chunk,
                )
                save_closed_loop_dataset(
                    run_dir
                    / (
                        f"dagger_closed_loop_round_{round_zero + 1}"
                        f"_cycle_{cycle_zero + 1}.npz"
                    ),
                    closed_loop_chunk,
                )
            save_energy_dataset(
                run_dir / f"dagger_round_{round_zero + 1}.npz",
                round_dagger,
            )
            save_closed_loop_dataset(
                run_dir / f"dagger_closed_loop_round_{round_zero + 1}.npz",
                round_closed_loop,
            )

            balanced_groups = [base, *dagger_sets, round_dagger]
            new_mean, new_std = _balanced_cost_statistics(balanced_groups)
            _reparameterize_energy_normalization(
                model, cost_mean, cost_std, new_mean, new_std
            )
            cost_mean, cost_std = new_mean, new_std
            # Relabel cycles intentionally restart Adam: the energy output
            # coordinates and current-student query distribution both changed.
            optimizer = make_optimizer(learning_rate)
            loss = train(
                balanced_groups,
                cycle_train_steps,
                closed_loop=[*closed_loop_sets, round_closed_loop],
                checkpoint_interval=(cycle_train_steps if student_only else None),
            )

            if student_only and early_stop_context["best_path"] is not None:
                # The next cycle must be recollected from the best current
                # student, not the last optimizer iterate.
                restored = CompositionalEnergyPolicy.load(
                    early_stop_context["best_path"]
                )
                model = restored.model
                normalizer = restored.normalizer
                cost_mean = restored.cost_mean
                cost_std = restored.cost_std
                cycle_history.append(
                    {
                        "cycle": cycle_zero + 1,
                        "collected_states_per_mode": cycle_collection_steps,
                        "gradient_steps": cycle_train_steps,
                        "closed_loop_horizon": horizon_for_round,
                        "evaluated_checkpoint": str(
                            early_stop_context["last_path"]
                        ),
                        "selection_score": float(
                            early_stop_context["last_score"]
                        ),
                        "restored_best_checkpoint": str(
                            early_stop_context["best_path"]
                        ),
                        "stale_validations": early_stop_context["stale"],
                    }
                )
                if (
                    early_stop_context["stale"]
                    >= args.student_only_early_stop_patience
                ):
                    break

        round_dagger = concatenate_energy_datasets(round_dagger_chunks)
        round_closed_loop = concatenate_closed_loop_datasets(
            round_closed_loop_chunks
        )
        dagger_sets.append(round_dagger)
        closed_loop_sets.append(round_closed_loop)
        aggregated = concatenate_energy_datasets([base, *dagger_sets])
        save_energy_dataset(run_dir / "dagger_aggregated.npz", aggregated)
        best_student_path = (
            early_stop_context["best_path"] if student_only else None
        )
        best_student_score = (
            early_stop_context["best_score"] if student_only else None
        )
        dagger_training_history.append(
            {
                "round": round_zero + 1,
                "beta": dagger_beta_schedule[round_zero],
                "student_only": student_only,
                "closed_loop_horizon": horizon_for_round,
                "learning_rate": learning_rate,
                "requested_gradient_steps": iterations,
                "completed_gradient_steps": global_step - round_start_step,
                "early_stopped": global_step - round_start_step < iterations,
                "restored_best_checkpoint": (
                    str(best_student_path) if best_student_path else None
                ),
                "best_early_stop_score": (
                    float(best_student_score)
                    if best_student_path is not None
                    else None
                ),
                "relabel_cycles": cycle_history,
            }
        )
        early_stop_context = None
        final_path = checkpoints_dir / f"step_{global_step:06d}_{phase}_final" / "policy.pkl"
        make_policy().save(final_path)
        checkpoints.append(final_path)

    phase = "hard_recovery"
    recovery_collector = ExactEnergyCollector(
        planner,
        objective_spec.cost,
        num_candidates=args.energy_candidates,
        teacher_repeats=args.hard_recovery_teacher_repeats,
    )
    recovery_dataset, rng = _collect_hard_recovery(
        env,
        planner,
        recovery_collector,
        make_policy(),
        reset,
        step,
        mode_weights,
        args.hard_recovery_steps,
        args.hard_recovery_window,
        args.hard_recovery_queries_per_mode,
        float(regular_factors[0]),
        args.trust_radius,
        rng,
    )
    save_energy_dataset(run_dir / "hard_recovery_relabel.npz", recovery_dataset)
    recovery_groups = [base, *dagger_sets, recovery_dataset]
    new_mean, new_std = _balanced_cost_statistics(recovery_groups)
    _reparameterize_energy_normalization(
        model, cost_mean, cost_std, new_mean, new_std
    )
    cost_mean, cost_std = new_mean, new_std
    optimizer = make_optimizer(args.hard_recovery_learning_rate)
    early_stop_context = {
        "best_score": -np.inf,
        "best_path": None,
        "last_score": None,
        "last_path": None,
        "stale": 0,
        "patience": args.hard_recovery_early_stop_patience,
        "selection_steps": args.final_selection_steps,
        "selection_seeds": args.final_selection_seeds,
    }
    recovery_start_step = global_step
    train(
        recovery_groups,
        args.hard_recovery_train_iters,
        closed_loop=closed_loop_sets,
        checkpoint_interval=min(
            args.hard_recovery_eval_every, args.hard_recovery_train_iters
        ),
    )
    recovery_best_path = early_stop_context["best_path"]
    recovery_best_score = early_stop_context["best_score"]
    if recovery_best_path is not None:
        restored = CompositionalEnergyPolicy.load(recovery_best_path)
        model = restored.model
        normalizer = restored.normalizer
        cost_mean = restored.cost_mean
        cost_std = restored.cost_std
    early_stop_context = None
    recovery_final_path = (
        checkpoints_dir / f"step_{global_step:06d}_{phase}_final" / "policy.pkl"
    )
    make_policy().save(recovery_final_path)
    checkpoints.append(recovery_final_path)
    recovery_training_history = {
        "rollout_steps_per_mode": args.hard_recovery_steps,
        "queries_per_mode": args.hard_recovery_queries_per_mode,
        "collected_queries": int(recovery_dataset.guidance_plans.shape[0]),
        "teacher_repeats": args.hard_recovery_teacher_repeats,
        "learning_rate": args.hard_recovery_learning_rate,
        "requested_gradient_steps": args.hard_recovery_train_iters,
        "completed_gradient_steps": global_step - recovery_start_step,
        "restored_best_checkpoint": (
            str(recovery_best_path) if recovery_best_path is not None else None
        ),
        "best_500_step_score": (
            float(recovery_best_score)
            if recovery_best_path is not None
            else None
        ),
    }

    screening = []
    for path in tqdm(
        list(dict.fromkeys(checkpoints)),
        desc="Energy checkpoint screening",
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
            common_seeds=True,
            worst_mode_selection=True,
        )
        metrics["policy"] = str(path)
        screening.append(metrics)
    finalists = sorted(
        screening, key=lambda item: item["selection_score"], reverse=True
    )[: min(args.selection_finalists, len(screening))]
    final_selection = []
    best = None
    for screened in tqdm(
        finalists,
        desc="500-step finalist selection",
        unit="checkpoint",
    ):
        path = Path(screened["policy"])
        candidate = CompositionalEnergyPolicy.load(path)
        metrics = _rollout_metrics(
            env,
            candidate,
            mode_weights,
            args.final_selection_steps,
            args.final_selection_seeds,
            minimum_mean_vx=args.minimum_mean_vx,
            track_command=args.selection_track_command,
            common_seeds=True,
            worst_mode_selection=True,
        )
        metrics["policy"] = str(path)
        metrics["screening_score"] = screened["selection_score"]
        final_selection.append(metrics)
        if best is None or metrics["selection_score"] > best[0]:
            best = (metrics["selection_score"], path, metrics)
    assert best is not None
    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "screening": screening,
                "finalists": final_selection,
                "selected": best[2],
            },
            indent=2,
        ),
        encoding="utf-8",
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
        "resolved_closed_loop_horizons": closed_loop_horizons,
        "selection_common_seeds": True,
        "selection_aggregation": "worst_mode",
        "selection_pipeline": {
            "screening_steps": args.selection_steps,
            "screening_seeds": args.selection_seeds,
            "finalists": args.selection_finalists,
            "final_steps": args.final_selection_steps,
            "final_seeds": args.final_selection_seeds,
        },
        "dagger_training_history": dagger_training_history,
        "hard_recovery_training_history": recovery_training_history,
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
