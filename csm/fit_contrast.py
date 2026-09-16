"""Fit the gait fields as a shared mean plus a per-gait residual.

The four gait labels share most of their content -- posture, height, the
velocity command -- and differ only in the stepping pattern.  Regressing each
label whole spends the network's error budget on the shared part and leaves
the gait inside it: on the v2 clouds every field reached the same validation
error (relative rms 0.86) and walk, the gait that needs four distinct foot
phases rather than a binary pairing, could not be resolved at that precision.

So split the label exactly,

    label_i = m + r_i,   m = mean_j label_j,   r_i = label_i - m,   sum_i r_i = 0

and fit five networks: one for `m` and one per `r_i`.  Each residual is
scaled up to the mean's power before regression so its error is measured
against itself, and scaled back at inference (csm.contrast_field.ContrastField).
The composed controller mixes fields with coefficients that sum to one, so
`sum_i a_i (m + r_i) = m + sum_i a_i r_i` -- the label for the mixed weight by
the same arithmetic.  Nothing about the composition changes; only what each
network is asked to be accurate about.

Everything up to the datasets is fit_from_clouds' own code, so the two fits
see identical labels and differ only in the decomposition.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax import nnx

import brax.envs as brax_envs
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.cloud_data import make_relabeler
from csm.architectures import StandardNormalizer
from csm.contrast_field import ContrastField
from csm.dial_score import (
    ComposedDialScorePolicy, DialScoreMLP, DialScorePolicy,
    build_shift_matrix, dial_factors, fit_dial_score_stacked,
)
from csm.fit_from_clouds import build_datasets, shard_paths, _slug
from csm.omega import normalize_omega_np, nu_matrix


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clouds", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=Path("csm_runs"))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--example", type=str, default=None)
    src.add_argument("--config", type=str, default=None)
    p.add_argument("--basis", nargs="+", default=["e0", "e1", "e2", "e3"])
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--level-scales", type=float, nargs="+", default=[1.0, 1.0])
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--shards", type=int, default=None)
    p.add_argument("--relabel-chunk", type=int, default=1024)
    p.add_argument("--min-height", type=float, default=0.25)
    p.add_argument("--hidden", type=str, default="512,512,512")
    p.add_argument("--train-iters", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

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
    basis = np.stack([
        catalogue[name] if name in catalogue
        else normalize_omega_np(np.array([float(v) for v in name.split(",")]))
        for name in args.basis
    ])
    k = len(args.basis)

    paths = shard_paths(args.clouds, args.shards)
    relabel, ess_fn = make_relabeler(planner, dial_config)
    run_dir = args.output / f"contrast-fit-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[contrast] {len(paths)} shards, basis={args.basis}, T={args.temperature}")
    datasets, label_stats = build_datasets(
        paths, relabel, ess_fn, basis, args.basis, args.temperature,
        tuple(args.level_scales), args.repeats, args.relabel_chunk,
        args.min_height, False,
    )
    for entry in label_stats:
        print(f"  {entry['row']}: rms {entry['rms']:.4f}  label noise "
              f"{entry['label_noise']:.5f} ({entry['relative_noise']:.1%})")

    # --- the decomposition -------------------------------------------------
    deltas = jnp.stack([d.delta for d in datasets])            # (k, N, H, A)
    mean = deltas.mean(axis=0)
    residuals = deltas - mean[None]
    mean_power = float(jnp.sqrt(jnp.mean(jnp.square(mean))))
    gains = []
    for row, name in enumerate(args.basis):
        res_power = float(jnp.sqrt(jnp.mean(jnp.square(residuals[row]))))
        gains.append(mean_power / max(res_power, 1e-9))
        print(f"  {name}: residual rms {res_power:.4f} = {res_power / mean_power:.1%} "
              f"of the mean's ({mean_power:.4f}); gain {gains[-1]:.2f}")
    base = datasets[0]
    contrast_sets = [base._replace(delta=mean)] + [
        base._replace(delta=residuals[row] * gains[row]) for row in range(k)
    ]
    names = ["mean"] + [f"res_{n}" for n in args.basis]

    observation_size = int(base.obs.shape[-1])
    hidden = tuple(int(v) for v in args.hidden.split(","))
    normalizer = StandardNormalizer(observation_size)
    normalizer.fit(base.obs)

    def new_model(row: int) -> DialScoreMLP:
        return DialScoreMLP(
            action_size=int(env.action_size), observation_size=observation_size,
            horizon=horizon, sigma_control=planner.sigma_control, hidden=hidden,
            rngs=nnx.Rngs(args.seed + 1000 * row),
            factor_min=factor_min, factor_max=factor_max,
        )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate * 0.1, peak_value=args.learning_rate,
        warmup_steps=max(args.train_iters // 20, 1),
        decay_steps=args.train_iters, end_value=args.learning_rate * 0.05,
    )
    models = [new_model(row) for row in range(k + 1)]
    print(f"[contrast] fitting {names} together")
    results = fit_dial_score_stacked(
        models, optax.adam(schedule), contrast_sets, normalizer=normalizer,
        batch_size=args.batch_size, num_iters=args.train_iters,
        rng=jax.random.PRNGKey(args.seed), validation_fraction=args.val_frac,
        eval_every=args.eval_every, names=names,
        desc=f"contrast regression {names}",
    )
    histories = {}
    for name, result in zip(names, results):
        histories[name] = {"best": result.best, "final": result.final}
        print(f"  {name}: best {result.best}")

    def field(row: int) -> DialScorePolicy:
        return DialScorePolicy(
            model=models[row], normalizer=normalizer, factors=factors,
            shift_matrix=shift_matrix, temperature=args.temperature,
            level_scales=jnp.asarray(args.level_scales),
        )

    mean_field = field(0)
    fields = tuple(ContrastField(base=mean_field, residual=field(row + 1),
                                 gain=gains[row]) for row in range(k))
    nu = nu_matrix(basis, [args.temperature] * k)
    policy = ComposedDialScorePolicy(
        policies=fields, mode_weights=jnp.asarray(basis),
        pinv_mode_weights=jnp.asarray(np.linalg.pinv(basis)),
        basis_temperatures=jnp.asarray([args.temperature] * k),
        temperature=args.temperature,
        pinv_nu_weights=jnp.asarray(np.linalg.pinv(nu)),
    )
    policy.save(run_dir / "policy.pkl")
    mean_field.save(run_dir / "field_mean.pkl")
    for row, name in enumerate(args.basis):
        fields[row].save(run_dir / f"field_{_slug(name)}.pkl")
    (run_dir / "report.json").write_text(json.dumps({
        "clouds": [str(c) for c in args.clouds], "basis": args.basis,
        "temperature": args.temperature, "gains": gains,
        "label_stats": label_stats, "fits": histories,
    }, indent=2, default=float))
    print(f"[contrast] saved {run_dir}")


if __name__ == "__main__":
    main()
