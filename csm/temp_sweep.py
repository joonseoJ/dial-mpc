"""Pick DIAL's sampling temperature per basis weight, by measured concentration.

DIAL divides sample returns by their own standard deviation before the softmax,
so the temperature that produces a usable weighting depends on the *shape* of
the return distribution -- which depends on the weight vector and on how hard
the robot was just pushed.  At the stock 0.05 the update is a hard argmax:
effective sample size 1.1 out of 2049, meaning the score label handed to a
student is one lucky sample rather than an average.  That is a data-quality
problem before it is a composition problem.

This sweep runs closed-loop DIAL at each candidate temperature, over a range of
push magnitudes, and reports the effective sample size of the final annealing
level -- the distribution the applied action actually came from.  It also
reports the fall rate and cost, because a temperature that produces pretty
weights but drops the robot is not a candidate.

ESS at every evaluation temperature is computed from the same returns vector on
device, so the off-diagonal (driven at one temperature, scored at another) costs
nothing and shows how much of the ESS reading is a property of the temperature
rather than of the states it steers into.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import brax.envs as brax_envs

from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import (
    make_rollout, make_sampler,
    effective_sample_size as _effective_sample_size,
    mppi_weights,
)
from csm.push_recover_eval import make_push_reset


# DIAL's own weighting, then 1 / sum(w^2); the softmax lives in dial_lean.
effective_sample_size = _effective_sample_size


def make_probe_step(env, mbdpi, dial_config, eval_temps, std_normalize=True):
    sample = make_sampler(mbdpi, dial_config)
    rollout_vmap = jax.vmap(make_rollout(env), in_axes=(None, 0))
    sigma = mbdpi.sigma_control
    factors = dial_config.traj_diffuse_factor ** jnp.arange(dial_config.Ndiffuse)

    def anneal(state, rng, plan, noise_scale, temp):
        rng, key = jax.random.split(rng)
        nodes = sample(key, plan, noise_scale)
        returns = rollout_vmap(state, mbdpi.node2u_vvmap(nodes)).mean(axis=-1)
        weights = mppi_weights(returns, temp, std_normalize)
        return rng, jnp.einsum("n,nij->ij", weights, nodes), returns

    def step(state, rng, plan, temp, shift):
        plan = jnp.where(shift, mbdpi.shift(plan), plan)

        def body(carry, factor):
            key, current = carry
            key, current, returns = anneal(state, key, current, sigma * factor, temp)
            return (key, current), returns

        (rng, plan), returns = jax.lax.scan(body, (rng, plan), factors)
        state = env.step(state, plan[0])
        final = returns[-1]
        ess = jax.vmap(
            lambda t: effective_sample_size(final, t, std_normalize)
        )(eval_temps)
        return state, rng, plan, ess, final.std()

    return step


def make_episode(env, mbdpi, dial_config, eval_temps, n_steps: int, std_normalize=True):
    probe = make_probe_step(env, mbdpi, dial_config, eval_temps, std_normalize)

    def episode(state, rng, temp):
        plan = jnp.zeros((dial_config.Hnode + 1, mbdpi.nu))

        def body(carry, index):
            st, key, pl = carry
            st, key, pl, ess, spread = probe(st, key, pl, temp, index > 0)
            return (st, key, pl), (ess, spread, st.reward, st.done)

        _, (ess, spread, reward, done) = jax.lax.scan(
            body, (state, rng, plan), jnp.arange(n_steps)
        )
        return {
            "ess": ess,                       # (steps, n_eval_temps)
            "spread": spread.mean(),
            "cost": -reward.mean(),
            "fell": done.max(),
        }

    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--omegas", nargs="+",
                        default=["boost0", "boost1", "boost2", "boost3"])
    parser.add_argument("--temps", type=float, nargs="+",
                        default=[0.20, 0.25, 0.275, 0.30, 0.35],
                        help="temperatures DIAL is actually driven at")
    parser.add_argument("--eval-temps", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.275, 0.30, 0.35, 0.50],
                        help="temperatures scored post hoc from the same clouds")
    parser.add_argument("--pushes", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.9])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std-normalize", dest="std", action="store_true")
    parser.add_argument("--no-std-normalize", dest="std", action="store_false",
                        help="score the raw Gibbs exponent nu . (-C) instead of "
                             "DIAL's spread-normalised one")
    parser.set_defaults(std=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, args.config)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    n_sample = dial_config.Nsample + 1

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    omega_vecs = jnp.asarray(np.stack([catalogue[n] for n in args.omegas]))
    temps = jnp.asarray(args.temps)
    eval_temps = jnp.asarray(args.eval_temps)
    pushes = jnp.asarray(args.pushes)

    n_omega, n_temp, n_push, n_seed = (
        len(args.omegas), len(args.temps), len(args.pushes), args.seeds
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, head_rng, reset_rng, roll_rng = jax.random.split(rng, 4)
    headings = jax.random.uniform(head_rng, (n_seed,), minval=-jnp.pi, maxval=jnp.pi)
    reset_keys = jax.random.split(reset_rng, n_seed)
    roll_keys = jax.random.split(roll_rng, n_seed)

    push_reset = make_push_reset(env)
    episode = make_episode(env, mbdpi, dial_config, eval_temps, args.steps, args.std)

    grid = jnp.meshgrid(
        jnp.arange(n_omega), jnp.arange(n_temp),
        jnp.arange(n_push), jnp.arange(n_seed), indexing="ij",
    )
    flat = [g.ravel() for g in grid]
    total = flat[0].size

    def one(io, it, ip, id_):
        state = push_reset(reset_keys[id_], pushes[ip], headings[id_], omega_vecs[io])
        return episode(state, roll_keys[id_], temps[it])

    batch = jax.vmap(one)
    chunk = max(min(args.chunk, total), 1)
    pad = (-total) % chunk

    @jax.jit
    def run():
        padded = [jnp.pad(f, (0, pad)).reshape(-1, chunk) for f in flat]
        out = jax.lax.map(lambda c: batch(*c), tuple(padded))
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:])[:total], out)

    print(f"[temp] {n_omega} omegas x {n_temp} temps x {n_push} pushes x {n_seed} "
          f"seeds = {total} episodes, {args.steps} steps, Nsample={n_sample}, "
          f"std_normalize={args.std}")
    t0 = time.time()
    result = run()
    jax.block_until_ready(result)
    print(f"[temp] done in {time.time() - t0:.1f}s")

    ess = np.asarray(result["ess"])        # (total, steps, n_eval)
    cost = np.asarray(result["cost"])
    fell = np.asarray(result["fell"])
    spread = np.asarray(result["spread"])
    io, it, ip = np.asarray(flat[0]), np.asarray(flat[1]), np.asarray(flat[2])
    share = ess / n_sample

    # On-policy: driven at t, scored at t.
    drive_to_eval = [args.eval_temps.index(t) for t in args.temps]

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    width = max(len(n) for n in args.omegas) + 2
    hdr = "".join(f"{t:>9.3f}" for t in args.temps)

    print("\n=== ESS share of the applied update, %% (driven and scored at the same temp) ===")
    print(f"{'omega':<{width}}{hdr}")
    on_policy = np.zeros((n_omega, n_temp))
    worst = np.zeros((n_omega, n_temp))
    for i in range(n_omega):
        for j in range(n_temp):
            sel = (io == i) & (it == j)
            column = share[sel][:, :, drive_to_eval[j]]
            on_policy[i, j] = column.mean() * 100
            worst[i, j] = np.percentile(column, 10) * 100
        print(f"{args.omegas[i]:<{width}}"
              + "".join(f"{on_policy[i, j]:9.2f}" for j in range(n_temp)))
    print(f"{'mean':<{width}}" + "".join(f"{on_policy[:, j].mean():9.2f}" for j in range(n_temp)))

    print("\n=== worst case (10th percentile over steps/pushes/seeds), %% ===")
    print(f"{'omega':<{width}}{hdr}")
    for i in range(n_omega):
        print(f"{args.omegas[i]:<{width}}"
              + "".join(f"{worst[i, j]:9.2f}" for j in range(n_temp)))

    print("\n=== ESS share by push magnitude, %% (rows: push, cols: temp; all omegas) ===")
    print(f"{'push':<{width}}{hdr}")
    for k in range(n_push):
        row = []
        for j in range(n_temp):
            sel = (it == j) & (ip == k)
            row.append(share[sel][:, :, drive_to_eval[j]].mean() * 100)
        print(f"{float(args.pushes[k]):<{width}.2f}" + "".join(f"{v:9.2f}" for v in row))

    print("\n=== fall rate / mean cost ===")
    print(f"{'omega':<{width}}" + "".join(f"{t:>9.3f}" for t in args.temps))
    for i in range(n_omega):
        cells = []
        for j in range(n_temp):
            sel = (io == i) & (it == j)
            cells.append(f"{fell[sel].mean():.2f}/{cost[sel].mean():.2f}")
        print(f"{args.omegas[i]:<{width}}" + "".join(f"{c:>9}" for c in cells))

    print("\n=== ESS share vs evaluation temperature (post hoc, all runs pooled), %% ===")
    print("temp     " + "".join(f"{t:>8.3f}" for t in args.eval_temps))
    print("share    " + "".join(f"{share[:, :, e].mean() * 100:8.2f}"
                                for e in range(len(args.eval_temps))))

    # A usable label needs a weighting, not an argmax, and it has to hold up on
    # the hardest push rather than on average.
    score = worst.mean(axis=0)
    best = int(np.argmax(score))
    print(f"\n[pick] worst-case mean ESS share peaks at temp={args.temps[best]:.3f} "
          f"({score[best]:.2f}%), on-policy mean {on_policy[:, best].mean():.2f}%, "
          f"fall rate {fell[it == best].mean():.2f}")
    print(f"[pick] mean return spread {spread.mean():.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "omegas": args.omegas,
                    "temps": args.temps,
                    "eval_temps": args.eval_temps,
                    "pushes": args.pushes,
                    "n_sample": n_sample,
                    "ess_share_percent": on_policy.tolist(),
                    "ess_share_p10_percent": worst.tolist(),
                    "fall_rate": [
                        [float(fell[(io == i) & (it == j)].mean()) for j in range(n_temp)]
                        for i in range(n_omega)
                    ],
                    "cost": [
                        [float(cost[(io == i) & (it == j)].mean()) for j in range(n_temp)]
                        for i in range(n_omega)
                    ],
                    "pooled_share_by_eval_temp": [
                        float(share[:, :, e].mean()) for e in range(len(args.eval_temps))
                    ],
                    "pick": float(args.temps[best]),
                },
                handle,
                indent=2,
            )
        print(f"[temp] wrote {args.out}")


if __name__ == "__main__":
    main()
