"""Run a large DIAL collection and write it as reusable clouds.

Everything decided here is either physical (which states get visited) or
structural (how many queries, how far off-plan they sit).  The weight vector,
the temperature, the per-level temperature profile and the number of repeats
averaged into a label are all decided later from the stored clouds, so this is
run once and the training sets are derived from it.

The one knob that is *not* recoverable afterwards is the ladder of perturbation
distances, because it determines which query plans were physically rolled out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

import jax
import jax.numpy as jnp

import brax.envs as brax_envs

from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_score import ComposedDialScorePolicy
from csm.parallel_collect import collect, make_student_driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, required=True,
                        help="directory the cloud shards are written to")
    parser.add_argument("--steps", type=int, default=4000,
                        help="control steps per environment")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--perturb-scales", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 1.5])
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--level-scales", type=float, nargs="+",
                        default=[3.175, 1.0])
    parser.add_argument("--episode-steps", type=int, default=150)
    parser.add_argument("--init-passes", type=int, default=5)
    parser.add_argument("--steps-per-call", type=int, default=25)
    parser.add_argument("--push-interval", type=int, default=40)
    parser.add_argument("--push-probability", type=float, default=0.8)
    parser.add_argument("--push-speed-min", type=float, default=0.25)
    parser.add_argument("--push-speed-max", type=float, default=1.2)
    parser.add_argument("--mix-basis-rows", type=float, default=0.5)
    parser.add_argument("--shard-every", type=int, default=250,
                        help="control steps between shard writes")
    parser.add_argument("--basis", nargs="+",
                        default=["boost0", "boost1", "boost2", "boost3"])
    parser.add_argument("--student-policy", type=Path, default=None,
                        help="composed policy that drives collection (DAgger); "
                             "labels still come from DIAL either way")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, args.config)
    import dataclasses

    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    basis = jnp.stack([catalogue[name] for name in args.basis])
    rank = int(np.linalg.matrix_rank(np.asarray(basis)))
    if rank < len(args.basis):
        raise ValueError(
            f"basis rows are not independent (rank {rank} of {len(args.basis)})"
        )
    if len(args.basis) > n_rows:
        raise ValueError(
            f"{len(args.basis)} basis weights in a {n_rows}-row reward cannot "
            "reduce; composition would mix in fields that should be zero"
        )

    student = None
    if args.student_policy is not None:
        student = make_student_driver(
            ComposedDialScorePolicy.load(args.student_policy)
        )
        print(f"[collect] DAgger: driving with {args.student_policy}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries = (args.steps * args.num_envs * dial_config.Ndiffuse
               * (1 + len(args.perturb_scales)))
    rollouts = queries * args.repeats * (dial_config.Nsample + 1)
    print(f"[collect] env={dial_config.env_name} basis={args.basis} "
          f"cond={np.linalg.cond(np.asarray(basis)):.2f}")
    print(f"[collect] {args.steps} steps x {args.num_envs} envs -> ~{queries:,} "
          f"queries, {rollouts * (dial_config.Hsample + 1) / 1e9:.1f}G env-steps, "
          f"~{queries * args.repeats * (dial_config.Nsample + 1) * n_rows * 4 / 1e9:.1f} GB")

    started = time.time()
    data, stats = collect(
        env, mbdpi, dial_config, basis, jax.random.PRNGKey(args.seed),
        num_steps=args.steps, num_envs=args.num_envs, repeats=args.repeats,
        perturb_scales=tuple(args.perturb_scales), temperature=args.temperature,
        std_normalize=False, level_scales=tuple(args.level_scales),
        episode_steps=args.episode_steps, init_passes=args.init_passes,
        steps_per_call=args.steps_per_call, push_interval=args.push_interval,
        push_probability=args.push_probability,
        push_speed_min=args.push_speed_min, push_speed_max=args.push_speed_max,
        mix_basis_rows=args.mix_basis_rows, student=student,
        shard_dir=out, shard_every=args.shard_every, progress=True,
    )
    elapsed = time.time() - started

    report = {
        "environment": dial_config.env_name,
        "basis": args.basis,
        "basis_weights": np.asarray(basis).tolist(),
        "temperature": args.temperature,
        "level_scales": args.level_scales,
        "perturb_scales": args.perturb_scales,
        "std_normalize": False,
        "student_policy": (None if args.student_policy is None
                           else str(args.student_policy)),
        "nsample": dial_config.Nsample + 1,
        "hsample": dial_config.Hsample,
        "hnode": dial_config.Hnode,
        "ndiffuse": dial_config.Ndiffuse,
        "seconds": elapsed,
        **{k: v for k, v in stats.items() if k != "shards"},
        "shards": [Path(p).name for p in stats["shards"]],
    }
    (out / "collection.json").write_text(json.dumps(report, indent=2))
    print(f"[collect] {stats['queries']:,} queries in {elapsed / 3600:.2f} h "
          f"across {len(stats['shards'])} shards -> {out}")
    print(f"[collect] mean reward {stats['mean_reward']:.4f}, falls {stats['falls']}")


if __name__ == "__main__":
    main()
