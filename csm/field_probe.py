"""Does a fitted field predict the update DIAL actually makes, at deployment?

Validation error is measured on the cloud: stored plans, stored observations,
labels recomputed from stored returns.  It says the network reproduces its
training distribution.  It does not say the field is right at the states a
closed loop reaches, and on this project validation error has never predicted
control.

So this asks the deployment question directly.  From a reset state, run the
student's own refinement loop, and at every control step compare two things at
the *same* plan and observation:

  the field's prediction   sum_i a_i field_i.delta(plan, obs, t)
  DIAL's own update        one lean update at the same plan, weight and level

Both are the increment added to the plan, so they are directly comparable in
size and direction.  Cosine near one with a matching magnitude means the field
is right and any failure is the loop compounding small errors; a magnitude
ratio far from one, or a cosine near zero, means the field is wrong where it is
being asked to work, whatever its validation error said.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import brax.envs as brax_envs
import dial_mpc.envs  # noqa: F401
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_lean_update
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import normalize_omega_np
from csm.push_recover_eval import make_push_reset
from csm.screen import COMMANDS, set_command, set_omega


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--example", default="unitree_go2_push_recover")
    p.add_argument("--target", default="boost0")
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--level-scales", type=float, nargs="+", default=[3.175, 1.0])
    p.add_argument("--speed", type=float, default=0.45,
                   help="push magnitude on the stand-and-resist task")
    p.add_argument("--command", default=None,
                   help="a csm.screen command name; selects the walking\n                        reset instead of a push, so the same probe gives a\n                        healthy baseline from a basis that is known to work")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--init-passes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-cloud", type=Path, default=None,
                   help="a stored shard: compare the field against the label "
                        "on the cloud's own states, through the deployment "
                        "call path.  Separates a training/inference mismatch "
                        "from distribution shift -- the fit already reports a "
                        "cosine here, so a lower one means the two paths "
                        "disagree and a matching one means the field is simply "
                        "being asked to work somewhere it was never fitted.")
    p.add_argument("--cloud-rows", type=int, default=256)
    args = p.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    policy = ComposedDialScorePolicy.load(args.policy)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    omega = (catalogue[args.target] if args.target in catalogue
             else normalize_omega_np(
                 np.array([float(v) for v in args.target.split(",")])))
    coeff = np.asarray(policy.coefficients(jnp.asarray(omega)))
    print(f"target {args.target} -> coefficients {np.round(coeff, 4)}")

    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    lo, hi = float(jnp.min(factors)), float(jnp.max(factors))
    shift = jnp.asarray(fields[0].shift_matrix)
    sigma = mbdpi.sigma_control
    scales = jnp.asarray(args.level_scales, dtype=factors.dtype)
    temps = scales * args.temperature
    update = make_lean_update(env, mbdpi, dial_config, std_normalize=False)

    def field_update(plan, obs, factor):
        t = factor_to_t(factor, lo, hi).reshape(1)
        parts = jnp.stack([f.delta(plan, obs, t) for f in fields])
        return jnp.einsum("k,kij->ij", jnp.asarray(coeff), parts)

    if args.from_cloud is not None:
        from csm.cloud_data import (load_clouds, make_relabeler,
                                    query_temperatures)
        clouds = load_clouds(args.from_cloud)
        clouds = jax.tree.map(lambda x: x[: args.cloud_rows], clouds)
        relabel, _ = make_relabeler(mbdpi, dial_config)
        temps = query_temperatures(clouds, args.temperature,
                                   tuple(args.level_scales))
        label, _ = relabel(clouds, jnp.asarray(omega), temps, False, None)
        t = jax.vmap(lambda f: factor_to_t(f, lo, hi).reshape(1))(clouds.factor)
        pred = jax.vmap(
            lambda u, o, tt: jnp.einsum(
                "k,kij->ij", jnp.asarray(coeff),
                jnp.stack([f.delta(u, o, tt) for f in fields]),
            )
        )(clouds.u, clouds.obs, t)
        a = np.asarray(pred).reshape(len(pred), -1)
        b = np.asarray(label).reshape(len(label), -1)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        cos = (a * b).sum(1) / np.maximum(na * nb, 1e-12)
        print(f"on the cloud's own {len(a)} states, through field.delta:")
        print(f"  |field| {na.mean():.4f}   |label| {nb.mean():.4f}   "
              f"ratio {np.mean(na / np.maximum(nb, 1e-9)):.2f}")
        print(f"  cosine mean {cos.mean():.3f}  median {np.median(cos):.3f}")
        return

    if args.command is not None:
        state = jax.jit(env.reset)(jax.random.PRNGKey(11 + args.seed))
        state = set_command(env, state, COMMANDS[args.command])
        state = set_omega(state, omega)
    else:
        state = make_push_reset(env)(jax.random.PRNGKey(args.seed),
                                     jnp.asarray(args.speed),
                                     jnp.asarray(0.3), jnp.asarray(omega))
    plan = jnp.zeros((dial_config.Hnode + 1, int(env.action_size)))
    for _ in range(args.init_passes):
        for factor in factors:
            plan = jnp.clip(plan + field_update(plan, state.obs, factor),
                            -1.0, 1.0)

    key = jax.random.PRNGKey(args.seed + 1)
    print(f"\n{'step':>4}{'level':>6}{'|field|':>10}{'|DIAL|':>10}"
          f"{'ratio':>8}{'cosine':>8}")
    for step in range(args.steps):
        for level, (factor, temp) in enumerate(zip(factors, temps)):
            mine = field_update(plan, state.obs, factor)
            key, theirs = update(state, key, plan, sigma * factor, temp)
            theirs = theirs - plan
            a, b = np.asarray(mine).ravel(), np.asarray(theirs).ravel()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = float(a @ b / max(na * nb, 1e-12))
            print(f"{step:>4}{level:>6}{na:10.4f}{nb:10.4f}"
                  f"{na / max(nb, 1e-9):8.2f}{cos:8.3f}")
            plan = jnp.clip(plan + mine, -1.0, 1.0)
        state = env.step(state, plan[0])
        plan = jnp.einsum("ij,ja->ia", shift, plan)


if __name__ == "__main__":
    main()
