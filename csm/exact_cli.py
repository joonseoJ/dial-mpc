"""Train compositional Go2 CSM from exact DIAL-MPC update targets."""

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
import yaml
from brax import envs as brax_envs
from brax import math
from brax.io import html
from flax import nnx
from tqdm.auto import tqdm

import dial_mpc.envs as dial_envs
from csm.architectures import CompositionalDenoisingMLP, StandardNormalizer
from csm.data_collection import (
    DIALTeacherPoint,
    ExactDIALTeacher,
    load_score_dataset,
    save_score_dataset,
    stack_teacher_data,
)
from csm.envs import get_objective_spec
from csm.policy import CompositionalPolicy
from csm.training import fit_score_regression
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example")
    source.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("csm_runs/exact"))
    parser.add_argument("--mode-weights", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--init-policy", type=Path, default=None)
    parser.add_argument("--skip-base-training", action="store_true")
    parser.add_argument("--initial-gradient-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--collect-steps", type=int, default=400)
    parser.add_argument("--teacher-repeats", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=10_000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-steps", type=int, default=200)
    parser.add_argument("--dagger-train-iters", type=int, default=5_000)
    parser.add_argument("--dagger-beta", type=float, default=0.5)
    parser.add_argument("--onpolicy-frac", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--data-checkpoint-every", type=int, default=25)
    parser.add_argument("--selection-steps", type=int, default=300)
    parser.add_argument("--selection-seeds", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _load_config(args: argparse.Namespace) -> dict:
    path = get_example_path(args.example + ".yaml") if args.example else args.config
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _set_mode(state, weights):
    info = dict(state.info)
    info["reward_weights"] = jnp.asarray(weights, dtype=jnp.float32)
    return state.replace(info=info)


def _physically_fallen(env, state) -> bool:
    torso = int(env._torso_idx) - 1
    euler = math.quat_to_euler(state.pipeline_state.x.rot[torso])
    height = float(state.pipeline_state.x.pos[torso, 2])
    tilt = float(jnp.linalg.norm(euler[:2]))
    return height < 0.18 or tilt > np.pi / 2


def _concat_datasets(datasets):
    return [
        tuple(jnp.concatenate([dataset[i][field] for dataset in datasets])
              for field in range(4))
        for i in range(len(datasets[0]))
    ]


def _rollout_metrics(
    env, policy, mode_weights, steps, seeds, minimum_mean_vx=None
):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    torso = int(env._torso_idx) - 1
    records = []
    apply = jax.jit(
        lambda plan, obs, key, weight: policy.apply(
            plan, obs, key, warm_start_level=1.0, omega=weight
        )
    )
    for mode_idx, omega in enumerate(mode_weights):
        for seed in range(seeds):
            rng = jax.random.PRNGKey(10_000 + 97 * mode_idx + seed)
            state = _set_mode(reset(rng), omega)
            plan = jnp.zeros((policy.model.horizon, env.action_size))
            velocities = []
            actions = []
            tilts = []
            survived = 0
            for step_idx in range(steps):
                rng, key = jax.random.split(rng)
                plan = apply(plan, state.obs, key, omega)
                state = step(_set_mode(state, omega), plan[0])
                state.reward.block_until_ready()
                actions.append(np.asarray(plan[0]))
                velocities.append(float(state.pipeline_state.xd.vel[torso, 0]))
                euler = math.quat_to_euler(state.pipeline_state.x.rot[torso])
                tilt = float(jnp.linalg.norm(euler[:2]))
                tilts.append(tilt)
                height = float(state.pipeline_state.x.pos[torso, 2])
                survived = step_idx + 1
                plan = policy.shift(plan)
                # DIAL-MPC does not terminate planning on the environment's
                # joint-limit ``done`` flag.  Only a physical fall ends the
                # validation rollout.
                if height < 0.18 or tilt > np.pi / 2:
                    break
            action_array = np.asarray(actions)
            jerk = (
                float(np.mean(np.square(np.diff(action_array, n=2, axis=0))))
                if len(action_array) > 2
                else 0.0
            )
            records.append(
                {
                    "mode": mode_idx,
                    "seed": seed,
                    "survived_steps": survived,
                    "fell": survived < steps,
                    "mean_vx": float(np.mean(velocities)),
                    "tracking_rmse": float(
                        np.sqrt(np.mean(np.square(np.asarray(velocities) - 0.8)))
                    ),
                    "mean_tilt": float(np.mean(tilts)),
                    "action_jerk": jerk,
                }
            )
    mean_steps = float(np.mean([r["survived_steps"] for r in records]))
    survival_rate = float(np.mean([not r["fell"] for r in records]))
    if minimum_mean_vx is None:
        qualifies = [True for _ in records]
    else:
        qualifies = [r["mean_vx"] >= minimum_mean_vx for r in records]
    for record, qualifies_speed in zip(records, qualifies):
        record["meets_minimum_speed"] = bool(qualifies_speed)
    qualified_mean_steps = float(
        np.mean(
            [r["survived_steps"] if good else 0 for r, good in zip(records, qualifies)]
        )
    )
    qualified_survival_rate = float(
        np.mean([not r["fell"] and good for r, good in zip(records, qualifies)])
    )
    tracking = float(np.mean([r["tracking_rmse"] for r in records]))
    tilt = float(np.mean([r["mean_tilt"] for r in records]))
    jerk = float(np.mean([r["action_jerk"] for r in records]))
    # Survival dominates the ordering; locomotion quality breaks ties.
    selection_score = (
        qualified_mean_steps
        + steps * qualified_survival_rate
        - 10 * tracking
        - tilt
        - jerk
    )
    return {
        "selection_score": selection_score,
        "mean_survived_steps": mean_steps,
        "survival_rate": survival_rate,
        "qualified_mean_survived_steps": qualified_mean_steps,
        "qualified_survival_rate": qualified_survival_rate,
        "minimum_mean_vx": minimum_mean_vx,
        "tracking_rmse": tracking,
        "mean_tilt": tilt,
        "action_jerk": jerk,
        "rollouts": records,
    }


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    if args.smoke:
        args.samples = 8
        args.collect_steps = 1
        args.teacher_repeats = 1
        args.train_iters = 1
        args.dagger_rounds = 1
        args.dagger_steps = 1
        args.dagger_train_iters = 1
        args.batch_size = 1
        args.checkpoint_every = 1
        args.selection_steps = 2
        args.selection_seeds = 1
        args.eval_steps = 2

    env_name = config["env_name"]
    objective_spec = get_objective_spec(
        brax_envs.get_environment(
            env_name,
            config=load_dataclass_from_dict(
                dial_envs.get_config(env_name), config, convert_list_to_array=True
            ),
        )
    )
    mode_path = args.mode_weights or Path(__file__).with_name("go2_modes.json")
    mode_weights = jnp.asarray(
        json.loads(mode_path.read_text(encoding="utf-8")), dtype=jnp.float32
    )
    if mode_weights.ndim != 2 or mode_weights.shape[1] != objective_spec.num_objectives:
        raise ValueError("mode weights do not match the environment objectives")
    if np.linalg.matrix_rank(np.asarray(mode_weights)) < objective_spec.num_objectives:
        raise ValueError("mode weights must span the objective space")

    env_config = load_dataclass_from_dict(
        dial_envs.get_config(env_name), config, convert_list_to_array=True
    )
    env = brax_envs.get_environment(env_name, config=env_config)
    dial_config = load_dataclass_from_dict(DialConfig, config)
    if args.samples is not None:
        dial_config.Nsample = args.samples
    if args.temperature is not None:
        dial_config.temp_sample = args.temperature
    planner = MBDPI(dial_config, env)
    teacher = ExactDIALTeacher(planner, args.teacher_repeats)
    # ``MBDPI.shift`` is linear.  Evaluate it on a node-space basis once and
    # store the exact spline shift/project operator in every learned policy.
    node_basis = jnp.eye(dial_config.Hnode + 1)

    def shift_basis(basis):
        plan = jnp.zeros((dial_config.Hnode + 1, env.action_size))
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
    sigma_min = float(jnp.min(initial_factors))
    sigma_max = float(jnp.max(initial_factors))
    horizon = dial_config.Hnode + 1

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output / f"{env_name}-{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "mode_weights": np.asarray(mode_weights).tolist(),
        "teacher": "dial_mpc.core.dial_core.MBDPI.reverse_once",
        "target_kind": "bounded_update",
        "Nsample": dial_config.Nsample,
        "temp_sample": dial_config.temp_sample,
        "regular_diffusion_factors": np.asarray(regular_factors).tolist(),
        "initial_diffusion_factors": np.asarray(initial_factors).tolist(),
        "status": "collecting",
    }

    def write_config(**updates):
        run_config.update(updates)
        serializable = {
            k: str(v) if isinstance(v, Path) else v for k, v in run_config.items()
        }
        target = run_dir / "run_config.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        temporary.replace(target)

    write_config()
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)

    diagnostics = []
    if args.dataset is None:
        collected: list[list[DIALTeacherPoint]] = [
            [] for _ in range(len(mode_weights))
        ]
        states, plans = [], []
        for omega in mode_weights:
            rng, key = jax.random.split(rng)
            states.append(_set_mode(reset(key), omega))
            plans.append(jnp.zeros((horizon, env.action_size)))
        total = args.collect_steps * len(mode_weights)
        bar = tqdm(total=total, desc="Exact DIAL teacher collection", unit="state")
        completed = 0
        for step_idx in range(args.collect_steps):
            for mode_idx, omega in enumerate(mode_weights):
                state, plan = states[mode_idx], plans[mode_idx]
                state = step(_set_mode(state, omega), plan[0])
                state.reward.block_until_ready()
                if _physically_fallen(env, state):
                    rng, key = jax.random.split(rng)
                    state = _set_mode(reset(key), omega)
                    plan = jnp.zeros_like(plan)
                plan = planner.shift(plan)
                factors = initial_factors if step_idx == 0 else regular_factors
                for factor in factors:
                    point, plan, rng = teacher.point(
                        state, plan, float(factor), rng, omega, mode_idx
                    )
                    plan.block_until_ready()
                    collected[mode_idx].append(point)
                    diagnostics.append(
                        [mode_idx, step_idx, float(factor), point.target_variance, point.ess]
                    )
                states[mode_idx], plans[mode_idx] = state, plan
                completed += 1
                bar.update()
                bar.set_postfix(mode=mode_idx, ess=f"{point.ess:.1f}", refresh=False)
                if args.data_checkpoint_every and completed % args.data_checkpoint_every == 0:
                    save_score_dataset(run_dir / "scores.partial.npz", stack_teacher_data(collected))
                    write_config(completed_collect_states=completed)
        bar.close()
        base_arrays = stack_teacher_data(collected)
        save_score_dataset(run_dir / "scores.npz", base_arrays)
        np.savez_compressed(
            run_dir / "teacher_diagnostics.npz",
            values=np.asarray(diagnostics, dtype=np.float32),
            columns=np.asarray(["mode", "state", "factor", "variance", "ess"]),
        )
    else:
        base_arrays = load_score_dataset(args.dataset)
        rng, key = jax.random.split(rng)
        states = [reset(key)]
    (run_dir / "scores.partial.npz").unlink(missing_ok=True)

    u = [x[0] for x in base_arrays]
    sigma = [x[1] for x in base_arrays]
    targets = [x[2] for x in base_arrays]
    obs = [x[3] for x in base_arrays]
    observation_size = int(obs[0].shape[-1])
    if args.init_policy is None:
        normalizer = StandardNormalizer(observation_size)
        normalizer.fit(jnp.concatenate(obs))
        model = CompositionalDenoisingMLP(
            action_size=env.action_size,
            observation_size=observation_size,
            horizon=horizon,
            num_objectives=len(mode_weights),
            encoder_hidden=(512, 512, 512),
            head_hidden=(256, 256),
            rngs=nnx.Rngs(args.seed),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    else:
        initial_policy = CompositionalPolicy.load(args.init_policy)
        model = initial_policy.model
        normalizer = initial_policy.normalizer
        if (
            model.action_size != env.action_size
            or model.observation_size != observation_size
            or model.horizon != horizon
            or model.num_objectives != len(mode_weights)
        ):
            raise ValueError("--init-policy architecture is incompatible")
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate))
    pinv = jnp.asarray(np.linalg.pinv(np.asarray(mode_weights)))

    def make_policy():
        return CompositionalPolicy(
            model=model,
            normalizer=normalizer,
            u_min=-jnp.ones(env.action_size),
            u_max=jnp.ones(env.action_size),
            mode_weights=mode_weights,
            pinv_mode_weights=pinv,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_steps=len(regular_factors),
            predicts_update=True,
            shift_matrix=shift_matrix,
            lock_first_action=True,
            inference_sigmas=regular_factors,
        )

    checkpoint_paths = [] if args.init_policy is None else [args.init_policy]
    global_steps = args.initial_gradient_steps
    phase = "base"

    def checkpoint(local_steps, loss):
        nonlocal global_steps
        completed = global_steps + local_steps
        directory = checkpoints_dir / f"step_{completed:06d}_{phase}"
        directory.mkdir(parents=True, exist_ok=True)
        make_policy().save(directory / "policy.pkl")
        (directory / "metadata.json").write_text(
            json.dumps({"step": completed, "phase": phase, "loss": float(loss)}, indent=2),
            encoding="utf-8",
        )
        checkpoint_paths.append(directory / "policy.pkl")
        write_config(status="training", completed_gradient_steps=completed, phase=phase)

    def train_phase(iterations, onpolicy=None):
        nonlocal rng, global_steps
        rng, key = jax.random.split(rng)
        loss = fit_score_regression(
            u, sigma, targets, obs, model, optimizer,
            batch_size=args.batch_size,
            num_iters=iterations,
            rng=key,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            predict_update=True,
            targets_are_updates=True,
            normalizer=normalizer,
            onpolicy_u_points_per_obj=None if onpolicy is None else [x[0] for x in onpolicy],
            onpolicy_sigmas_per_obj=None if onpolicy is None else [x[1] for x in onpolicy],
            onpolicy_scores_per_obj=None if onpolicy is None else [x[2] for x in onpolicy],
            onpolicy_obs_per_obj=None if onpolicy is None else [x[3] for x in onpolicy],
            onpolicy_frac=0.0 if onpolicy is None else args.onpolicy_frac,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=checkpoint,
        )
        global_steps += iterations * len(mode_weights)
        return loss

    write_config(status="training", samples_per_mode=[int(x.shape[0]) for x in u])
    if args.skip_base_training:
        if args.init_policy is None:
            raise ValueError("--skip-base-training requires --init-policy")
        loss = jnp.asarray(float("nan"))
    else:
        loss = train_phase(args.train_iters)
        phase_path = checkpoints_dir / f"step_{global_steps:06d}_base_final" / "policy.pkl"
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        make_policy().save(phase_path)
        checkpoint_paths.append(phase_path)

    dagger_datasets = []
    for round_idx in range(args.dagger_rounds):
        phase = f"dagger_{round_idx + 1}"
        beta = args.dagger_beta * (1.0 - round_idx / max(args.dagger_rounds, 1))
        round_points = [[] for _ in range(len(mode_weights))]
        bar = tqdm(
            total=args.dagger_steps * len(mode_weights),
            desc=f"Query DAgger round {round_idx + 1}", unit="state"
        )
        for mode_idx, omega in enumerate(mode_weights):
            rng, key = jax.random.split(rng)
            state = _set_mode(reset(key), omega)
            plan = jnp.zeros((horizon, env.action_size))
            coeff = omega @ pinv
            for _ in range(args.dagger_steps):
                state = step(_set_mode(state, omega), plan[0])
                state.reward.block_until_ready()
                if _physically_fallen(env, state):
                    rng, key = jax.random.split(rng)
                    state = _set_mode(reset(key), omega)
                    plan = jnp.zeros_like(plan)
                plan = planner.shift(plan)
                for factor in regular_factors:
                    query = plan
                    t = jnp.asarray([
                        (jnp.log(factor) - jnp.log(sigma_min))
                        / (jnp.log(sigma_max) - jnp.log(sigma_min))
                    ])
                    normalized_obs = normalizer(state.obs, use_running_average=True)
                    learner_update = model(query, normalized_obs, t, coeff).at[0].set(0.0)
                    point, teacher_plan, rng = teacher.point(
                        state, query, float(factor), rng, omega, mode_idx
                    )
                    round_points[mode_idx].append(point)
                    plan = jnp.clip(
                        query + (1.0 - beta) * learner_update
                        + beta * point.update_target, -1.0, 1.0
                    )
                    plan.block_until_ready()
                bar.update()
        bar.close()
        round_arrays = stack_teacher_data(round_points)
        save_score_dataset(run_dir / f"dagger_round_{round_idx + 1}.npz", round_arrays)
        dagger_datasets.append(round_arrays)
        aggregated = _concat_datasets(dagger_datasets)
        save_score_dataset(run_dir / "dagger_aggregated.npz", aggregated)
        loss = train_phase(args.dagger_train_iters, aggregated)
        phase_path = checkpoints_dir / f"step_{global_steps:06d}_{phase}_final" / "policy.pkl"
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        make_policy().save(phase_path)
        checkpoint_paths.append(phase_path)

    # Rollout validation, rather than regression loss, selects the deployment
    # checkpoint across all modes and held-out reset seeds.
    write_config(status="selecting_checkpoint")
    selection = []
    best = None
    unique_paths = list(dict.fromkeys(checkpoint_paths))
    for path in tqdm(unique_paths, desc="Rollout checkpoint selection", unit="checkpoint"):
        candidate = CompositionalPolicy.load(path)
        metrics = _rollout_metrics(
            env, candidate, mode_weights, args.selection_steps, args.selection_seeds
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
    policy = CompositionalPolicy.load(run_dir / "policy.pkl")

    # Render the selected checkpoint under the first training mode.
    rng, key = jax.random.split(rng)
    state = _set_mode(reset(key), mode_weights[0])
    plan = jnp.zeros((horizon, env.action_size))
    apply = jax.jit(lambda p, o, k: policy.apply(
        p, o, k, warm_start_level=1.0, omega=mode_weights[0]
    ))
    rollout = []
    for _ in tqdm(range(args.eval_steps), desc="Selected policy rendering", unit="step"):
        rng, key = jax.random.split(rng)
        plan = apply(plan, state.obs, key)
        state = step(_set_mode(state, mode_weights[0]), plan[0])
        state.reward.block_until_ready()
        rollout.append(state.pipeline_state)
        plan = policy.shift(plan)
        euler = math.quat_to_euler(state.pipeline_state.x.rot[int(env._torso_idx) - 1])
        height = float(state.pipeline_state.x.pos[int(env._torso_idx) - 1, 2])
        if height < 0.18 or float(jnp.linalg.norm(euler[:2])) > np.pi / 2:
            break
    rendered = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 720, True
    )
    (run_dir / "visualization.html").write_text(rendered, encoding="utf-8")
    write_config(
        status="complete",
        completed_gradient_steps=global_steps,
        final_loss=float(loss),
        selected_checkpoint=str(best[1]),
        selected_metrics=best[2],
    )
    print(f"selected={best[1]}")
    print(f"metrics={json.dumps(best[2])}")
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
