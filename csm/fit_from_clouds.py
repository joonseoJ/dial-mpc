"""Fit the basis score fields from a stored cloud collection.

Collection and training are separate here because they have to be.  A cloud
file holds the per-sample costs, so every training set this produces -- one per
basis weight -- is a relabelling of the *same* rollouts under a different
softmax.  The fields therefore see identical states, identical queries and
identical physics, and differ only through the weighting, which is exactly the
comparison composition needs.

Changing the temperature, the per-level profile, the number of repeats averaged
into a label, or the basis itself re-derives the training sets from disk in
seconds.  Nothing here touches the environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import time
from pathlib import Path

import numpy as np
import optax
import yaml

import jax
import jax.numpy as jnp
from flax import nnx

import brax.envs as brax_envs

from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.cloud_data import (
    DialScoreData, chunked, load_clouds, make_relabeler, query_temperatures,
)
from csm.architectures import StandardNormalizer
from csm.dial_score import (
    ComposedDialScorePolicy, DialScoreMLP, DialScorePolicy,
    build_shift_matrix, dial_factors, fit_dial_score, save_dial_score_data,
)
from csm.omega import normalize_omega_np, nu_matrix


def shard_paths(folders, limit: int | None = None) -> list[str]:
    """Every shard across every round, base first.

    Pooling rather than replacing matters: a DAgger round is collected where
    the student goes, which is a narrower distribution than the teacher's, and
    dropping the base round would trade coverage for on-policy accuracy.  This
    project has already measured what happens when a round silently shrinks the
    dataset -- the run ends up reporting the data change, not the variable.
    """

    paths: list[str] = []
    for folder in ([folders] if isinstance(folders, Path) else folders):
        found = sorted(glob.glob(str(Path(folder) / "clouds_*.npz")))
        if not found:
            raise FileNotFoundError(f"no cloud shards under {folder}")
        paths.extend(found[:limit] if limit else found)
    return paths


def build_datasets(paths, relabel, ess_fn, basis, names, temperature,
                   level_scales, repeats, chunk):
    """Relabel every shard under every basis row, streaming through the host.

    A collection is tens of gigabytes of stored costs and relabelling
    regenerates the sampled trajectories on top of that, so shards are processed
    one at a time and only the finished labels are kept.  Every row sees the
    identical queries, which is the property that makes the fields comparable.
    """

    shared = {name: [] for name in ("u", "factor", "level", "obs", "qpos",
                                    "qvel", "step")}
    labels = {name: [] for name in names}
    diagnostics = {name: {"sq": 0.0, "noise": 0.0, "ess": 0.0, "n": 0}
                   for name in names}
    for index, path in enumerate(paths):
        clouds = load_clouds(path)
        temps = query_temperatures(clouds, temperature, level_scales)
        for key in shared:
            shared[key].append(np.asarray(getattr(clouds, key)))
        for row, name in enumerate(names):
            weight = jnp.asarray(basis[row])
            delta, spread = chunked(
                lambda c, t: relabel(c, weight, t, False, repeats),
                clouds, chunk, temps,
            )
            ess = chunked(
                lambda c, t: (ess_fn(c, weight, t, False),), clouds, chunk, temps
            )[0]
            labels[name].append(delta)
            stats = diagnostics[name]
            stats["sq"] += float((delta ** 2).sum())
            stats["noise"] += float(spread.sum())
            # ess is (queries, repeats): every stored cloud has its own.
            stats["ess"] += float(ess.sum())
            stats["clouds"] = stats.get("clouds", 0) + int(ess.size)
            stats["n"] += delta.size
            stats.setdefault("queries", 0)
            stats["queries"] += int(delta.shape[0])
        print(f"  shard {index + 1}/{len(paths)}: {clouds.size} queries", flush=True)
        del clouds

    merged = {key: np.concatenate(value) for key, value in shared.items()}
    datasets, summary = [], []
    for name in names:
        delta = np.concatenate(labels[name])
        datasets.append(DialScoreData(
            u=jnp.asarray(merged["u"]), factor=jnp.asarray(merged["factor"]),
            level=jnp.asarray(merged["level"]), delta=jnp.asarray(delta),
            obs=jnp.asarray(merged["obs"]), qpos=jnp.asarray(merged["qpos"]),
            qvel=jnp.asarray(merged["qvel"]), step=jnp.asarray(merged["step"]),
        ))
        stats = diagnostics[name]
        rms = float(np.sqrt(stats["sq"] / stats["n"]))
        used = repeats or 1
        noise = stats["noise"] / stats["queries"] / np.sqrt(max(used, 1))
        summary.append({"row": name, "rms": rms, "label_noise": noise,
                        "relative_noise": noise / max(rms, 1e-9),
                        "mean_ess": stats["ess"] / stats["clouds"]})
    return datasets, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds", type=Path, nargs="+", required=True,
                        help="one or more cloud directories; DAgger rounds are "
                             "pooled rather than replacing the base round")
    parser.add_argument("--output", type=Path, default=Path("csm_runs"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--basis", nargs="+",
                        default=["boost0", "boost1", "boost2", "boost3"])
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--level-scales", type=float, nargs="+",
                        default=[3.175, 1.0])
    parser.add_argument("--repeats", type=int, default=None,
                        help="clouds averaged per label; defaults to all stored")
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--relabel-chunk", type=int, default=4096)
    parser.add_argument("--hidden", type=str, default="512,512,512")
    parser.add_argument("--train-iters", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--level-loss-balance", type=float, default=0.0)
    parser.add_argument("--save-datasets", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, args.config)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    planner = make_controller(dial_config, env)
    factors = dial_factors(dial_config.traj_diffuse_factor, dial_config.Ndiffuse)
    factor_min, factor_max = float(jnp.min(factors)), float(jnp.max(factors))
    shift_matrix = build_shift_matrix(planner)
    horizon = int(dial_config.Hnode) + 1

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    basis = np.stack([catalogue[name] for name in args.basis])
    if len(args.basis) > n_rows:
        raise ValueError(
            f"{len(args.basis)} basis weights in a {n_rows}-row reward cannot "
            "reduce; the pseudoinverse would spread a basis target over fields "
            "that should have coefficient zero"
        )
    rank = int(np.linalg.matrix_rank(basis))
    if rank < len(args.basis):
        raise ValueError(f"basis rank {rank} < {len(args.basis)} rows")

    paths = shard_paths(args.clouds, args.shards)
    relabel, ess_fn = make_relabeler(planner, dial_config)

    run_dir = args.output / f"clouds-fit-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fit] {len(paths)} shards, basis={args.basis} "
          f"cond={np.linalg.cond(basis):.2f}, T={args.temperature} "
          f"level_scales={args.level_scales}")
    datasets, label_stats = build_datasets(
        paths, relabel, ess_fn, basis, args.basis, args.temperature,
        tuple(args.level_scales), args.repeats, args.relabel_chunk,
    )
    n_sample = dial_config.Nsample + 1
    for entry in label_stats:
        entry["ess_share"] = entry.pop("mean_ess") / n_sample
        print(f"  {entry['row']}: rms {entry['rms']:.4f}  label noise "
              f"{entry['label_noise']:.5f} ({entry['relative_noise']:.1%})  "
              f"ESS {entry['ess_share']:.1%}")
    print(f"[fit] {datasets[0].size:,} training rows per field")
    if args.save_datasets:
        for row, name in enumerate(args.basis):
            save_dial_score_data(run_dir / f"dataset_{name}.npz", datasets[row])

    observation_size = int(datasets[0].obs.shape[-1])
    hidden = tuple(int(v) for v in args.hidden.split(","))
    # One normalizer, fitted once: every field sees the same observations by
    # construction, and refitting per field would shift each network's input
    # distribution for no reason.
    normalizer = StandardNormalizer(observation_size)
    normalizer.fit(datasets[0].obs)

    models, histories = [], []
    for row, name in enumerate(args.basis):
        model = DialScoreMLP(
            action_size=int(env.action_size),
            observation_size=observation_size,
            horizon=horizon,
            sigma_control=planner.sigma_control,
            hidden=hidden,
            rngs=nnx.Rngs(args.seed + 1000 * row),
            factor_min=factor_min,
            factor_max=factor_max,
        )
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=args.learning_rate * 0.1,
            peak_value=args.learning_rate,
            warmup_steps=max(args.train_iters // 20, 1),
            decay_steps=args.train_iters,
            end_value=args.learning_rate * 0.05,
        )
        optimizer = nnx.Optimizer(model, optax.adam(schedule))
        print(f"[fit] field {name}")
        result = fit_dial_score(
            model, optimizer, datasets[row], normalizer=normalizer,
            batch_size=args.batch_size, num_iters=args.train_iters,
            rng=jax.random.PRNGKey(args.seed + row),
            validation_fraction=args.val_frac, eval_every=args.eval_every,
            desc=f"score regression [{name}]",
        )
        models.append(model)
        histories.append({"best": result.best, "final": result.final,
                          "per_level": {str(k): v for k, v in result.per_level.items()}})
        print(f"  best {result.best}")

    def field(row: int) -> DialScorePolicy:
        return DialScorePolicy(
            model=models[row], normalizer=normalizer, factors=factors,
            shift_matrix=shift_matrix, temperature=args.temperature,
            level_scales=jnp.asarray(args.level_scales),
        )

    nu = nu_matrix(basis, [args.temperature] * len(args.basis))
    policy = ComposedDialScorePolicy(
        policies=tuple(field(row) for row in range(len(args.basis))),
        mode_weights=jnp.asarray(basis),
        pinv_mode_weights=jnp.asarray(np.linalg.pinv(basis)),
        basis_temperatures=jnp.asarray([args.temperature] * len(args.basis)),
        temperature=args.temperature,
        pinv_nu_weights=jnp.asarray(np.linalg.pinv(nu)),
    )
    policy.save(run_dir / "policy.pkl")
    for row, name in enumerate(args.basis):
        field(row).save(run_dir / f"field_{name}.pkl")

    check = {
        name: np.asarray(policy.coefficients(jnp.asarray(basis[row]))).tolist()
        for row, name in enumerate(args.basis)
    }
    print("[fit] reduction check (target = each basis row):")
    for name, coefficients in check.items():
        print(f"  {name}: {np.round(coefficients, 6)}")

    (run_dir / "report.json").write_text(json.dumps({
        "clouds": [str(c) for c in args.clouds],
        "shards": [Path(p).name for p in paths],
        "queries": int(datasets[0].size),
        "basis": args.basis,
        "temperature": args.temperature,
        "level_scales": args.level_scales,
        "repeats_used": args.repeats or clouds.repeats,
        "labels": label_stats,
        "fits": {name: histories[row] for row, name in enumerate(args.basis)},
        "reduction": check,
    }, indent=2))
    print(f"[fit] wrote {run_dir}")


if __name__ == "__main__":
    main()
