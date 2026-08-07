"""Train an anchor-factorized Gibbs score policy from DIAL-TC-MPPI."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
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
from csm.architectures import StandardNormalizer
from csm.energy_policy import normalize_preference_weights
from csm.envs import get_objective_spec
from csm.exact_cli import (
    _load_config,
    _physically_fallen,
    _rollout_metrics,
    _set_mode,
)
from csm.gibbs_data import (
    DIALTCGibbsTeacher,
    GibbsDataset,
    concatenate_gibbs_datasets,
    load_gibbs_dataset,
    save_gibbs_dataset,
)
from csm.gibbs_model import AnchorLogitMLP
from csm.gibbs_policy import AnchorGibbsPolicy
from csm.gibbs_training import fit_anchor_gibbs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import DIALTCMPPI
from dial_mpc.utils.io_utils import load_dataclass_from_dict


def _parse_betas(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("betas must be comma-separated floats") from error
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("every DAgger beta must be in [0, 1]")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example")
    source.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("csm_runs/anchor-gibbs"))
    parser.add_argument("--anchor-weights", type=Path, default=None)
    parser.add_argument("--validation-weights", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--collect-steps", type=int, default=400)
    parser.add_argument("--model-candidates", type=int, default=128)
    parser.add_argument("--banks-per-query", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=30_000)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-steps", type=int, default=200)
    parser.add_argument("--dagger-train-iters", type=int, default=15_000)
    parser.add_argument(
        "--dagger-betas", type=_parse_betas, default=(0.5, 0.25, 0.0, 0.0, 0.0)
    )
    parser.add_argument("--prefall-window", type=int, default=10)
    parser.add_argument("--prefall-priority", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--selection-steps", type=int, default=300)
    parser.add_argument("--selection-seeds", type=int, default=5)
    parser.add_argument("--minimum-mean-vx", type=float, default=0.5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _zero_history(planner):
    return jnp.zeros(
        (planner.args.tc_history_length, planner.nu), dtype=jnp.float32
    )


def _append_history(history, action):
    return jnp.concatenate([history[1:], action[None]], axis=0)


def _promote_prefall(chunks, recent_steps, priority):
    """Raises replay priority for queries that produced an imminent fall."""

    for step_indices in recent_steps:
        for index in step_indices:
            chunks[index] = replace(
                chunks[index],
                priorities=jnp.full_like(chunks[index].priorities, priority),
            )


def _collect_base(
    env,
    planner,
    teacher,
    reset,
    step,
    anchor_weights,
    initial_factors,
    regular_factors,
    collect_steps,
    prefall_window,
    prefall_priority,
    rng,
):
    chunks = []
    states, plans, histories = [], [], []
    recent_steps = [deque(maxlen=prefall_window) for _ in anchor_weights]
    horizon = planner.args.Hnode + 1
    for omega in anchor_weights:
        rng, key = jax.random.split(rng)
        states.append(_set_mode(reset(key), omega))
        plans.append(jnp.zeros((horizon, env.action_size)))
        histories.append(_zero_history(planner))

    bar = tqdm(
        total=collect_steps * len(anchor_weights),
        desc="DIAL-TC anchor Gibbs collection",
        unit="state",
    )
    for state_idx in range(collect_steps):
        for mode_idx, omega in enumerate(anchor_weights):
            state, plan = states[mode_idx], plans[mode_idx]
            history = histories[mode_idx]
            executed = plan[0]
            state = step(_set_mode(state, omega), executed)
            state.reward.block_until_ready()
            history = _append_history(history, executed)
            if _physically_fallen(env, state):
                _promote_prefall(
                    chunks, recent_steps[mode_idx], prefall_priority
                )
                recent_steps[mode_idx].clear()
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                history = _zero_history(planner)
            plan = planner.shift(plan)
            factors = initial_factors if state_idx == 0 else regular_factors
            step_indices = []
            for factor in factors:
                chunk, plan, rng = teacher.collect_query(
                    state, plan, float(factor), rng, omega, history
                )
                chunk.anchor_costs.block_until_ready()
                chunks.append(chunk)
                step_indices.append(len(chunks) - 1)
            recent_steps[mode_idx].append(step_indices)
            states[mode_idx], plans[mode_idx], histories[mode_idx] = (
                state,
                plan,
                history,
            )
            bar.update()
    bar.close()
    return concatenate_gibbs_datasets(chunks), rng


def _collect_dagger(
    env,
    planner,
    teacher,
    policy,
    reset,
    step,
    anchor_weights,
    regular_factors,
    dagger_steps,
    beta,
    prefall_window,
    prefall_priority,
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
        total=dagger_steps * len(anchor_weights),
        desc=f"Anchor Gibbs DAgger round {round_idx}",
        unit="state",
    )
    for omega in anchor_weights:
        rng, key = jax.random.split(rng)
        state = _set_mode(reset(key), omega)
        plan = jnp.zeros((planner.args.Hnode + 1, env.action_size))
        history = _zero_history(planner)
        recent_steps = deque(maxlen=prefall_window)
        for _ in range(dagger_steps):
            executed = plan[0]
            state = step(_set_mode(state, omega), executed)
            state.reward.block_until_ready()
            history = _append_history(history, executed)
            if _physically_fallen(env, state):
                _promote_prefall(chunks, recent_steps, prefall_priority)
                recent_steps.clear()
                rng, key = jax.random.split(rng)
                state = _set_mode(reset(key), omega)
                plan = jnp.zeros_like(plan)
                history = _zero_history(planner)
            shifted = planner.shift(plan)
            rng, policy_rng = jax.random.split(rng)
            learner = apply(shifted, state.obs, policy_rng, omega)
            expert = learner
            step_indices = []
            for factor in regular_factors:
                chunk, expert, rng = teacher.collect_query(
                    state, expert, float(factor), rng, omega, history
                )
                chunk.anchor_costs.block_until_ready()
                chunks.append(chunk)
                step_indices.append(len(chunks) - 1)
            recent_steps.append(step_indices)
            plan = jnp.clip((1.0 - beta) * learner + beta * expert, -1.0, 1.0)
            bar.update()
    bar.close()
    return concatenate_gibbs_datasets(chunks), rng


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    if args.smoke:
        args.samples = 8
        args.collect_steps = 1
        args.model_candidates = 4
        args.banks_per_query = 2
        args.train_iters = 2
        args.dagger_rounds = 1
        args.dagger_betas = (0.0,)
        args.dagger_steps = 1
        args.dagger_train_iters = 2
        args.batch_size = 2
        args.checkpoint_every = 1
        args.selection_steps = 2
        args.selection_seeds = 1
        args.minimum_mean_vx = None
        args.eval_steps = 2

    if len(args.dagger_betas) != args.dagger_rounds:
        raise ValueError("dagger-betas must contain exactly dagger-rounds values")
    if args.prefall_window < 1 or args.prefall_priority < 1.0:
        raise ValueError("pre-fall window must be positive and priority must be >= 1")
    if min(args.cost_weight, args.kl_weight, args.score_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")

    env_name = config["env_name"]
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(env_name), config, convert_list_to_array=True
    )
    env = brax_envs.get_environment(env_name, config=env_config)
    objective_spec = get_objective_spec(env)
    anchor_path = args.anchor_weights or Path(__file__).with_name("go2_modes.json")
    anchor_weights = jnp.asarray(
        json.loads(anchor_path.read_text(encoding="utf-8")), dtype=jnp.float32
    )
    if anchor_weights.ndim != 2 or anchor_weights.shape[1] != objective_spec.num_objectives:
        raise ValueError("anchor weights must match the objective dimension")
    if np.any(np.asarray(anchor_weights) < 0.0):
        raise ValueError("anchor weights must be non-negative")
    anchor_sums = np.asarray(jnp.sum(anchor_weights, axis=1))
    if np.any(anchor_sums <= 0.0):
        raise ValueError("every anchor must have positive total weight")
    preference_weight_sum = float(np.median(anchor_sums))
    anchor_weights = normalize_preference_weights(
        anchor_weights, preference_weight_sum
    )
    if int(jnp.linalg.matrix_rank(anchor_weights)) != objective_spec.num_objectives:
        raise ValueError("anchor weights must span the objective space")
    validation_path = args.validation_weights or Path(__file__).with_name(
        "go2_validation_weights.json"
    )
    validation_weights = jnp.asarray(
        json.loads(validation_path.read_text(encoding="utf-8")), dtype=jnp.float32
    )
    if (
        validation_weights.ndim != 2
        or validation_weights.shape[1] != objective_spec.num_objectives
        or np.any(np.asarray(validation_weights) < 0.0)
    ):
        raise ValueError("validation weights must be non-negative objective rows")
    validation_weights = normalize_preference_weights(
        validation_weights, preference_weight_sum
    )
    selection_weights = jnp.concatenate(
        [anchor_weights, validation_weights], axis=0
    )

    dial_config = load_dataclass_from_dict(DialConfig, config)
    dial_config = replace(
        dial_config,
        Nsample=args.samples,
        temp_sample=(
            args.temperature
            if args.temperature is not None
            else dial_config.temp_sample
        ),
        time_correlated=True,
        # This retains DIAL's annealed mean and makes the student proposal
        # exactly reproducible without a deployment action-history input.
        tc_mean_mode="dial",
    )
    planner = DIALTCMPPI(dial_config, env)
    teacher = DIALTCGibbsTeacher(
        planner,
        objective_spec.cost,
        anchor_weights,
        num_model_candidates=args.model_candidates,
        banks_per_query=args.banks_per_query,
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
        base = load_gibbs_dataset(args.dataset)
    else:
        base, rng = _collect_base(
            env,
            planner,
            teacher,
            reset,
            step,
            anchor_weights,
            initial_factors,
            regular_factors,
            args.collect_steps,
            args.prefall_window,
            args.prefall_priority,
            rng,
        )
    save_gibbs_dataset(run_dir / "gibbs_data.npz", base)

    normalizer = StandardNormalizer(int(base.observations.shape[-1]))
    normalizer.fit(base.observations)
    anchor_cost_scale = jnp.maximum(
        jnp.std(base.anchor_costs, axis=(0, 1)), 1e-6
    )
    model = AnchorLogitMLP(
        action_size=env.action_size,
        observation_size=int(base.observations.shape[-1]),
        horizon=horizon,
        num_anchors=int(anchor_weights.shape[0]),
        hidden_sizes=(512, 512, 512),
        rngs=nnx.Rngs(args.seed),
    )

    def make_optimizer():
        return nnx.Optimizer(model, optax.adam(args.learning_rate))

    optimizer = make_optimizer()

    def make_policy():
        return AnchorGibbsPolicy(
            model=model,
            normalizer=normalizer,
            anchor_weights=anchor_weights,
            covariance_sqrt=planner.tc_sampler.covariance_sqrt,
            sigma_control=planner.sigma_control,
            diffuse_factors=regular_factors,
            shift_matrix=shift_matrix,
            anchor_cost_scale=anchor_cost_scale,
            candidate_count=args.model_candidates,
            preference_weight_sum=preference_weight_sum,
            temperature=dial_config.temp_sample,
            format_version=2,
        )

    checkpoints = []
    global_step = 0
    phase = "base"

    def checkpoint(local_step, loss):
        path = (
            checkpoints_dir
            / f"step_{global_step + local_step:06d}_{phase}"
            / "policy.pkl"
        )
        make_policy().save(path)
        path.with_name("metadata.json").write_text(
            json.dumps(
                {
                    "step": global_step + local_step,
                    "loss": float(loss),
                    "phase": phase,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        checkpoints.append(path)

    def train(groups, iterations):
        nonlocal rng, global_step
        rng, train_rng = jax.random.split(rng)
        loss = fit_anchor_gibbs(
            groups,
            model,
            normalizer,
            optimizer,
            batch_size=args.batch_size,
            num_iters=iterations,
            rng=train_rng,
            anchor_cost_scale=anchor_cost_scale,
            temperature=dial_config.temp_sample,
            cost_weight=args.cost_weight,
            kl_weight=args.kl_weight,
            score_weight=args.score_weight,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=checkpoint,
        )
        global_step += iterations
        return loss

    loss = train([base], args.train_iters)
    base_path = checkpoints_dir / f"step_{global_step:06d}_base_final" / "policy.pkl"
    make_policy().save(base_path)
    checkpoints.append(base_path)

    dagger_sets: list[GibbsDataset] = []
    for round_zero in range(args.dagger_rounds):
        phase = f"dagger_{round_zero + 1}"
        dagger, rng = _collect_dagger(
            env,
            planner,
            teacher,
            make_policy(),
            reset,
            step,
            anchor_weights,
            regular_factors,
            args.dagger_steps,
            args.dagger_betas[round_zero],
            args.prefall_window,
            args.prefall_priority,
            rng,
            round_zero + 1,
        )
        dagger_sets.append(dagger)
        save_gibbs_dataset(run_dir / f"dagger_round_{round_zero + 1}.npz", dagger)
        save_gibbs_dataset(
            run_dir / "dagger_aggregated.npz",
            concatenate_gibbs_datasets([base, *dagger_sets]),
        )
        optimizer = make_optimizer()
        loss = train([base, *dagger_sets], args.dagger_train_iters)
        final_path = (
            checkpoints_dir
            / f"step_{global_step:06d}_{phase}_final"
            / "policy.pkl"
        )
        make_policy().save(final_path)
        checkpoints.append(final_path)

    selection = []
    best_eligible = None
    best_diagnostic = None
    unique_checkpoints = list(dict.fromkeys(checkpoints))
    phase_finals = [
        path for path in unique_checkpoints if path.parent.name.endswith("_final")
    ]
    selection_candidates = phase_finals or unique_checkpoints
    for path in tqdm(
        selection_candidates,
        desc="Anchor Gibbs checkpoint selection",
        unit="checkpoint",
    ):
        candidate = AnchorGibbsPolicy.load(path)
        metrics = _rollout_metrics(
            env,
            candidate,
            selection_weights,
            args.selection_steps,
            args.selection_seeds,
            minimum_mean_vx=args.minimum_mean_vx,
        )
        metrics["policy"] = str(path)
        speed_passed = (
            args.minimum_mean_vx is None
            or metrics["qualified_survival_rate"] >= 1.0
        )
        metrics["hard_gate_passed"] = bool(
            metrics["survival_rate"] >= 1.0 and speed_passed
        )
        selection.append(metrics)
        candidate_record = (metrics["selection_score"], path, metrics)
        if best_diagnostic is None or candidate_record[0] > best_diagnostic[0]:
            best_diagnostic = candidate_record
        if metrics["hard_gate_passed"] and (
            best_eligible is None or candidate_record[0] > best_eligible[0]
        ):
            best_eligible = candidate_record
    assert best_diagnostic is not None
    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    config_out = {
        **vars(args),
        "framework": "anchor_factorized_raw_cost_gibbs_score_v2",
        "expert": "DIAL-TC-MPPI",
        "objective_names": list(objective_spec.names),
        "anchor_weights": np.asarray(anchor_weights).tolist(),
        "validation_weights": np.asarray(validation_weights).tolist(),
        "anchor_condition_number": float(np.linalg.cond(np.asarray(anchor_weights))),
        "anchor_cost_scale": np.asarray(anchor_cost_scale).tolist(),
        "preference_weight_sum": preference_weight_sum,
        "selection_hard_gate": {
            "steps": args.selection_steps,
            "seeds": args.selection_seeds,
            "all_weights_must_survive": True,
            "minimum_mean_vx": args.minimum_mean_vx,
        },
        "selection_passed": best_eligible is not None,
        "selected": str(best_eligible[1]) if best_eligible else None,
        "selected_metrics": best_eligible[2] if best_eligible else None,
        "best_rejected_metrics": (
            best_diagnostic[2] if best_eligible is None else None
        ),
        "completed_gradient_steps": global_step,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in config_out.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    if best_eligible is None:
        print(f"saved={run_dir}")
        raise RuntimeError(
            "no checkpoint survived the hard deployment gate on every anchor "
            "and unseen validation weight; policy.pkl was intentionally not published"
        )

    best = best_eligible
    shutil.copy2(best[1], run_dir / "policy.pkl")

    policy = AnchorGibbsPolicy.load(run_dir / "policy.pkl")
    rng, key = jax.random.split(rng)
    state = _set_mode(reset(key), anchor_weights[0])
    plan = jnp.zeros((horizon, env.action_size))
    apply = jax.jit(
        lambda p, o, k: policy.apply(
            p, o, k, warm_start_level=1.0, omega=anchor_weights[0]
        )
    )
    rollout = []
    for _ in tqdm(
        range(args.eval_steps), desc="Anchor Gibbs rendering", unit="step"
    ):
        rng, key = jax.random.split(rng)
        plan = apply(plan, state.obs, key)
        state = step(_set_mode(state, anchor_weights[0]), plan[0])
        state.reward.block_until_ready()
        rollout.append(state.pipeline_state)
        plan = policy.shift(plan)
        if _physically_fallen(env, state):
            break
    (run_dir / "visualization.html").write_text(
        html.render(env.sys, rollout), encoding="utf-8"
    )

    print(f"selected={best[1]}")
    print(f"metrics={json.dumps(best[2])}")
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
