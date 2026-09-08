"""One score network conditioned on omega, instead of k fields composed.

Compositional score matching trains a field per basis row and mixes them
linearly, which is exact because the Gibbs score is linear in `nu = omega / T`.
The obvious alternative needs none of that: hand `omega` to the network and let
it learn the dependence.  If it works, the linearity argument is machinery
nobody needs; if it does not, the reason to build a basis is measured rather
than assumed.

Nothing about the pipeline changes except where omega enters.  The same clouds
are relabelled at several weights, exactly as the basis rows are, and the
resulting labels are stacked into one training set whose observation carries
the weight it was labelled under.  So the network sees `(u, obs, omega, t)` and
predicts the update for that omega, and `fit_dial_score` trains it unchanged.

The comparison is only fair if the two arms see the same number of labels.  The
basis consumed `k * queries` of them across its `k` fields; this keeps that
total and spends it on more distinct weights instead, so the difference under
test is how the weight dependence is represented and not how much supervision
paid for it.
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
from flax import nnx

import brax.envs as brax_envs
from dial_mpc.core.dial_core import make_controller

from csm.architectures import StandardNormalizer
from csm.basis_screen import _load_config, build_omegas
from csm.cloud_data import make_relabeler
from csm.dial_score import (
    ComposedDialScorePolicy, DialScoreData, DialScoreMLP, DialScorePolicy,
    build_shift_matrix, dial_factors, fit_dial_score,
)
from csm.fit_from_clouds import build_datasets, shard_paths
from csm.omega import normalize_omega_np


def sample_weights(count: int, n_rows: int, seed: int) -> list[np.ndarray]:
    """Directions on the unit sphere restricted to the positive orthant.

    The same distribution the conditioned RL baseline samples from, and where
    `normalize_omega` puts everything the planner is ever asked for.
    """

    rng = np.random.default_rng(seed)
    raw = rng.exponential(size=(count, n_rows))
    return [normalize_omega_np(row) for row in raw]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clouds", type=Path, nargs="+", required=True)
    p.add_argument("--example", default="unitree_go2_trot_csm")
    p.add_argument("--weights", nargs="+", default=None,
                   help="catalogue names or literal `a,b,c`; defaults to the "
                        "three boost rows, uniform, and random draws to fill "
                        "--num-weights")
    p.add_argument("--num-weights", type=int, default=6)
    p.add_argument("--label-budget", type=int, default=None,
                   help="total training rows across every weight.  Defaults to "
                        "`basis_size * queries`, the number of labels the basis "
                        "consumed, so neither arm is better supervised.")
    p.add_argument("--basis-size", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.020)
    p.add_argument("--level-scales", type=float, nargs="+", default=[2.625, 1.0])
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--shards", type=int, default=None)
    p.add_argument("--relabel-chunk", type=int, default=4096)
    p.add_argument("--hidden", default="512,512,512")
    p.add_argument("--train-iters", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("csm_runs/conditioned"))
    args = p.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    planner = make_controller(dial_config, env)
    factors = dial_factors(dial_config.traj_diffuse_factor, dial_config.Ndiffuse)
    factor_min, factor_max = float(jnp.min(factors)), float(jnp.max(factors))
    shift_matrix = build_shift_matrix(planner)
    horizon = int(dial_config.Hnode) + 1
    n_rows = int(np.asarray(env_config.reward_weights).shape[0])

    catalogue = build_omegas(n_rows)
    if args.weights:
        weights = [
            catalogue[w] if w in catalogue
            else normalize_omega_np(np.array([float(v) for v in w.split(",")]))
            for w in args.weights
        ]
        names = list(args.weights)
    else:
        fixed = [f"boost{i}" for i in range(n_rows)] + ["uniform"]
        weights = [catalogue[n] for n in fixed]
        names = list(fixed)
        extra = max(args.num_weights - len(weights), 0)
        for i, w in enumerate(sample_weights(extra, n_rows, args.seed)):
            weights.append(w)
            names.append("rand%d" % i)
    stack = np.stack(weights)

    paths = shard_paths(args.clouds, args.shards)
    relabel, ess_fn = make_relabeler(planner, dial_config)
    run_dir = args.output / f"conditioned-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cond] {len(paths)} shards, {len(weights)} weights: {names}")
    datasets, label_stats = build_datasets(
        paths, relabel, ess_fn, stack, names, args.temperature,
        tuple(args.level_scales), args.repeats, args.relabel_chunk,
    )
    n_sample = dial_config.Nsample + 1
    for entry in label_stats:
        entry["ess_share"] = entry.pop("mean_ess") / n_sample
        print(f"  {entry['row']}: rms {entry['rms']:.4f}  "
              f"noise {entry['relative_noise']:.1%}  "
              f"ESS {entry['ess_share']:.1%}")

    queries = int(datasets[0].size)
    budget = args.label_budget or args.basis_size * queries
    per_weight = max(budget // len(datasets), 1)
    print(f"[cond] {queries:,} queries; budget {budget:,} labels "
          f"({args.basis_size} x {queries:,}), {per_weight:,} per weight")

    rng = np.random.default_rng(args.seed)
    parts = {k: [] for k in ("u", "factor", "level", "delta", "obs",
                             "qpos", "qvel", "step")}
    for weight, data in zip(weights, datasets):
        take = rng.choice(queries, size=min(per_weight, queries), replace=False)
        take = np.sort(take)
        omega = np.broadcast_to(np.asarray(weight, dtype=np.float32),
                                (len(take), n_rows))
        parts["obs"].append(
            np.concatenate([np.asarray(data.obs)[take], omega], axis=-1))
        for key in ("u", "factor", "level", "delta", "qpos", "qvel", "step"):
            parts[key].append(np.asarray(getattr(data, key))[take])
    combined = DialScoreData(
        **{k: jnp.asarray(np.concatenate(v)) for k, v in parts.items()}
    )
    del datasets, parts
    print(f"[cond] combined training set {combined.size:,} rows, "
          f"observation {combined.obs.shape[-1]} "
          f"(env {int(combined.obs.shape[-1]) - n_rows} + {n_rows} weight)")

    observation_size = int(combined.obs.shape[-1])
    hidden = tuple(int(v) for v in args.hidden.split(","))
    normalizer = StandardNormalizer(observation_size)
    normalizer.fit(combined.obs)
    model = DialScoreMLP(
        action_size=int(env.action_size), observation_size=observation_size,
        horizon=horizon, sigma_control=planner.sigma_control,
        hidden=hidden, rngs=nnx.Rngs(args.seed),
        factor_min=factor_min, factor_max=factor_max,
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate * 0.1, peak_value=args.learning_rate,
        warmup_steps=max(args.train_iters // 20, 1),
        decay_steps=args.train_iters, end_value=args.learning_rate * 0.05,
    )
    result = fit_dial_score(
        model, nnx.Optimizer(model, optax.adam(schedule)), combined,
        normalizer=normalizer, batch_size=args.batch_size,
        num_iters=args.train_iters, rng=jax.random.PRNGKey(args.seed),
        validation_fraction=args.val_frac, eval_every=args.eval_every,
        desc="conditioned score regression",
    )
    print(f"[cond] best {result.best}")

    field = DialScorePolicy(
        model=model, normalizer=normalizer, factors=factors,
        shift_matrix=shift_matrix, temperature=args.temperature,
        level_scales=jnp.asarray(args.level_scales),
    )
    # A single field with an identity solve: the evaluator must append omega to
    # the observation rather than mix anything, which `--conditioned` selects.
    policy = ComposedDialScorePolicy(
        policies=(field,),
        mode_weights=jnp.asarray(stack[:1]),
        pinv_mode_weights=jnp.asarray(np.linalg.pinv(stack[:1])),
        basis_temperatures=jnp.asarray([args.temperature]),
        temperature=args.temperature,
        pinv_nu_weights=jnp.asarray(np.linalg.pinv(stack[:1] / args.temperature)),
    )
    policy.save(run_dir / "policy.pkl")
    (run_dir / "report.json").write_text(json.dumps({
        "clouds": [str(c) for c in args.clouds], "weights": names,
        "weight_vectors": stack.tolist(), "temperature": args.temperature,
        "level_scales": list(args.level_scales), "queries": queries,
        "label_budget": budget, "per_weight": per_weight,
        "observation_size": observation_size, "n_rows": n_rows,
        "conditioned": True, "labels": label_stats,
        "fit": {"best": result.best, "final": result.final,
                "per_level": {str(k): v for k, v in result.per_level.items()}},
    }, indent=2, default=float))
    print(f"[cond] wrote {run_dir}")


if __name__ == "__main__":
    main()
