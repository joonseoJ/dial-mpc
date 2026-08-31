"""Screen a reward basis for weight discriminativeness before training anything.

The question a basis has to answer is narrow: from the same state, does
changing omega change which action samples MPPI selects?  If it does not, no
amount of score-field training will produce omega-conditioned behaviour, and
the whole compositional pipeline is measuring noise.

The screen is close to free because a rollout does not depend on omega.  DIAL
draws one cloud of Nsample action sequences per state; the per-row reward terms
of that cloud are all any omega needs.  So we roll out once and re-weight the
stored terms for every omega, which makes the cost O(1) in the number of
weights instead of one full DIAL campaign each.

Two further things keep the GPU busy rather than the host:

  * the rollout collects only the reward-term vector, not the pipeline state.
    DIAL's own `rollout_us` materialises q/qd/x for every sample and step; at
    Nsample 2048 that dominates both memory traffic and peak footprint, and
    none of it is needed here.
  * initial states are swept with `lax.map` so the inner vmap stays at exactly
    Nsample, the width the sampler is already tuned for, instead of spilling
    into a batch the device has to serialise anyway.

Note that omega is scale-invariant here: DIAL divides the sample returns by
their own standard deviation before the softmax, so only the direction of the
weight vector affects the update.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import yaml

import jax
import jax.numpy as jnp

import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import make_controller
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

from csm.dial_lean import (
    make_dial_step, make_rollout, make_sampler,
    mppi_weights as _mppi_weights,
)
from csm.omega import normalize_omega_np


def _load_config(example: str | None, config_path: str | None):
    if example is not None:
        config_dict = yaml.safe_load(open(get_example_path(example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(config_path))
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )
    return dial_config, env_config


def build_omegas(n_rows: int) -> Dict[str, np.ndarray]:
    """Vertices plus the basis rows we would actually train on, unit length.

    Every weight vector is normalised on the way out.  The objective depends on
    omega / T alone, so a free scale on omega would make the temperature
    meaningless; fixing the length here is what lets a temperature be measured,
    quoted and composed.
    """

    raw: Dict[str, np.ndarray] = {}
    for i in range(n_rows):
        vec = np.zeros(n_rows)
        vec[i] = 1.0
        raw[f"e{i}"] = vec
    raw["uniform"] = np.ones(n_rows)
    for i in range(n_rows):
        vec = np.ones(n_rows)
        vec[i] = 3.0
        raw[f"boost{i}"] = vec
    return {name: normalize_omega_np(vec) for name, vec in raw.items()}


def make_pipeline(env, mbdpi, dial_config, warmup_steps: int,
                  std_normalize: bool = True):
    """Warm DIAL up on the shared control weights, then emit one sample cloud.

    Scoring every omega on a cloud drawn around a *warmed-up* plan at the final
    annealing level is the point.  The level-0 proposal is far too wide to say
    anything: at sigma 1.0 on a [-1, 1] action box most samples simply fall
    over, one lucky outlier dominates the softmax, and every omega collapses
    onto it.  What DIAL actually acts on is the narrow, annealed cloud around
    the shifted previous solution, so that is what the screen has to look at.

    The warmup runs at the environment's configured weights for every omega, on
    purpose: comparing omegas on their own visited states measures exam
    difficulty, not the objective.  One shared state distribution keeps the
    comparison honest.
    """

    control_step = make_dial_step(env, mbdpi, dial_config,
                                  std_normalize=std_normalize)
    sample = make_sampler(mbdpi, dial_config)
    final_noise = mbdpi.sigma_control * (
        dial_config.traj_diffuse_factor ** (dial_config.Ndiffuse - 1)
    )
    rollout = make_rollout(
        env,
        lambda s: (s.info["reward_terms"], s.info["feet_lifted"], s.done),
    )
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))

    def pipeline(state, rng):
        plan = jnp.zeros((dial_config.Hnode + 1, mbdpi.nu))

        def warm(carry, _):
            st, key, pl = carry
            st, key, pl = control_step(st, key, pl)
            return (st, key, pl), None

        (state, rng, plan), _ = jax.lax.scan(
            warm, (state, rng, plan), None, length=warmup_steps
        )
        plan = mbdpi.shift(plan)

        rng, cloud_rng = jax.random.split(rng)
        nodes = sample(cloud_rng, plan, final_noise)
        us = mbdpi.node2u_vvmap(nodes)
        terms, lifted, done = rollout_vmap(state, us)
        return (
            terms.mean(axis=1),
            nodes,
            lifted.mean(axis=1),
            done.max(axis=1),
            state.info["push_speed"],
        )

    return pipeline, float(jnp.mean(final_noise))


# The softmax lives in one place; see `csm.dial_lean.mppi_logits`.
mppi_weights = _mppi_weights


def analyse_state(terms, Y0s, lifted, done, omega_mat, temp, elite_frac,
                  std_normalize=True):
    """All omegas, one shared sample cloud."""

    returns = terms @ omega_mat.T  # (n_sample, n_omega)
    weights = jax.vmap(
        lambda r: mppi_weights(r, temp, std_normalize), in_axes=1, out_axes=0
    )(returns)
    # MPPI's actual output: the weighted mean node trajectory.
    updates = jnp.einsum("kn,nij->kij", weights, Y0s)
    ess = 1.0 / jnp.sum(weights**2, axis=1)

    n_elite = max(int(round(elite_frac * terms.shape[0])), 1)
    order = jnp.argsort(-returns, axis=0)[:n_elite]  # (n_elite, n_omega)
    elite_lift = jnp.mean(lifted[order], axis=0)
    elite_terms = jnp.mean(terms[order], axis=0)  # (n_omega, n_rows) after take
    elite_fail = jnp.mean(done[order], axis=0)
    weighted_lift = weights @ lifted
    return {
        "updates": updates,
        "weights": weights,
        "ess": ess,
        "elite_idx": order,
        "elite_lift": elite_lift,
        "elite_terms": elite_terms,
        "elite_fail": elite_fail,
        "weighted_lift": weighted_lift,
        "returns_std": returns.std(axis=0),
    }


def jaccard_matrix(elite_idx: np.ndarray) -> np.ndarray:
    """Overlap of the top-k sets across omegas, from one shared cloud."""

    n_omega = elite_idx.shape[1]
    out = np.zeros((n_omega, n_omega))
    sets = [set(elite_idx[:, k].tolist()) for k in range(n_omega)]
    for i in range(n_omega):
        for j in range(n_omega):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            out[i, j] = inter / max(union, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=4,
        help="DIAL control steps before the cloud is drawn; the push lands at "
             "reset, so this places the screen inside the recovery transient",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=4,
        help="initial states evaluated in one vmapped batch; the inner sampler "
             "is already Nsample wide, so this multiplies device occupancy",
    )
    parser.add_argument("--elite-frac", type=float, default=0.05)
    parser.add_argument("--std-normalize", dest="std", action="store_true")
    parser.add_argument("--no-std-normalize", dest="std", action="store_false",
                        help="drive DIAL with the raw Gibbs exponent")
    parser.set_defaults(std=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, args.config)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    omegas = build_omegas(n_rows)
    names = list(omegas)
    omega_mat = jnp.asarray(np.stack([omegas[k] for k in names]))
    n_omega = len(names)

    reset_vmap = jax.jit(jax.vmap(env.reset))
    pipeline, noise_level = make_pipeline(
        env, mbdpi, dial_config, args.warmup_steps, args.std
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng, cloud_rng = jax.random.split(rng, 3)
    states = reset_vmap(jax.random.split(reset_rng, args.states))
    cloud_rngs = jax.random.split(cloud_rng, args.states)

    chunk = max(min(args.chunk, args.states), 1)
    if args.states % chunk:
        raise ValueError("--states must be a multiple of --chunk")

    def one(state, key):
        terms, nodes, lifted, done, push = pipeline(state, key)
        out = analyse_state(
            terms, nodes, lifted, done, omega_mat,
            dial_config.temp_sample, args.elite_frac, args.std,
        )
        out["push_speed"] = push
        out["cloud_fail"] = done.mean()
        return out

    batch = jax.vmap(one)

    chunk_idx = jnp.arange(args.states).reshape(-1, chunk)

    @jax.jit
    def screen(states, rngs):
        # vmap fills the device, lax.map bounds peak memory.  Splitting the
        # sweep this way keeps one knob (--chunk) between "too small to saturate
        # the GPU" and "too large to fit".  The split is done by gathering on an
        # index array rather than reshaping the state pytree: a Brax pipeline
        # state carries leaves with zero-length axes (empty constraint rows) and
        # those cannot be reshaped.
        def run_chunk(ids):
            return batch(jax.tree.map(lambda x: x[ids], states), rngs[ids])

        out = jax.lax.map(run_chunk, chunk_idx)
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), out)

    print(f"[screen] env={dial_config.env_name} rows={n_rows} "
          f"omegas={n_omega} states={args.states} "
          f"Nsample={dial_config.Nsample} Hsample={dial_config.Hsample} "
          f"warmup={args.warmup_steps} cloud_sigma={noise_level:.3f}")
    t0 = time.time()
    result = screen(states, cloud_rngs)
    jax.block_until_ready(result)
    compiled = time.time() - t0
    t0 = time.time()
    result = screen(states, cloud_rngs)
    jax.block_until_ready(result)
    elapsed = time.time() - t0
    print(f"[screen] compile+run {compiled:.1f}s, warm run {elapsed:.1f}s, "
          f"chunk={chunk}")
    rollouts = args.states * (args.warmup_steps * dial_config.Ndiffuse + 1) * dial_config.Nsample
    print(f"[screen] {rollouts:,} rollouts x {dial_config.Hsample} steps "
          f"in {elapsed:.1f}s ({rollouts * dial_config.Hsample / elapsed / 1e6:.2f}M env-steps/s)")

    updates = np.asarray(result["updates"])          # (S, K, Hnode+1, nu)
    elite_idx = np.asarray(result["elite_idx"])      # (S, n_elite, K)
    elite_lift = np.asarray(result["elite_lift"])    # (S, K)
    elite_terms = np.asarray(result["elite_terms"])  # (S, K, rows)
    elite_fail = np.asarray(result["elite_fail"])
    ess = np.asarray(result["ess"])
    push_speed = np.asarray(result["push_speed"])
    cloud_fail = np.asarray(result["cloud_fail"])

    # Primary metric: how far apart are the MPPI updates that different omegas
    # produce from the same cloud, in units of the cloud's own sigma.
    dist = np.zeros((n_omega, n_omega))
    for s in range(updates.shape[0]):
        flat = updates[s].reshape(n_omega, -1)
        dist += np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    dist /= updates.shape[0]
    dist /= noise_level * np.sqrt(updates.shape[2] * updates.shape[3])

    overlap = np.mean(
        [jaccard_matrix(elite_idx[s]) for s in range(elite_idx.shape[0])], axis=0
    )
    # Two disjoint random top-k sets out of n_sample still overlap by chance;
    # that level, not zero, is what "no discrimination" looks like.
    n_elite = elite_idx.shape[1]
    n_sample = dial_config.Nsample + 1
    chance = (n_elite / n_sample) / (2 - n_elite / n_sample)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    width = max(len(n) for n in names) + 1
    print(f"\n[setup] push speed {push_speed.mean():.2f} m/s "
          f"({push_speed.min():.2f}-{push_speed.max():.2f}), "
          f"cloud fall rate {cloud_fail.mean():.2f}")

    print("\n=== MPPI update distance / cloud sigma (0 = identical action) ===")
    print(" " * width + "".join(f"{n:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{width}}" + "".join(f"{dist[i, j]:9.3f}" for j in range(n_omega)))

    print(f"\n=== elite-set Jaccard overlap (chance = {chance:.3f}, 1.0 = identical) ===")
    print(" " * width + "".join(f"{n:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{width}}" + "".join(f"{overlap[i, j]:9.3f}" for j in range(n_omega)))

    print("\n=== elite behaviour per omega ===")
    print(f"{'omega':<{width}}{'feet_lifted':>13}{'fail_rate':>11}{'ESS':>9}"
          + "".join(f"{'row' + str(r):>9}" for r in range(n_rows)))
    for i, n in enumerate(names):
        print(f"{n:<{width}}{elite_lift[:, i].mean():13.3f}"
              f"{elite_fail[:, i].mean():11.3f}{ess[:, i].mean():9.1f}"
              + "".join(f"{elite_terms[:, i, r].mean():9.2f}" for r in range(n_rows)))

    off = ~np.eye(n_omega, dtype=bool)
    lift_by_omega = elite_lift.mean(axis=0)
    print(f"\n[verdict] mean off-diagonal update distance = {dist[off].mean():.3f} sigma")
    print(f"[verdict] mean off-diagonal elite overlap    = {overlap[off].mean():.3f} "
          f"(chance {chance:.3f})")
    print(f"[verdict] elite feet-lifted spread over omega= "
          f"{lift_by_omega.max() - lift_by_omega.min():.3f} "
          f"[{names[int(lift_by_omega.argmin())]} -> {names[int(lift_by_omega.argmax())]}]")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "env": dial_config.env_name,
                    "warmup_steps": args.warmup_steps,
                    "cloud_sigma": noise_level,
                    "push_speed_mean": float(push_speed.mean()),
                    "names": names,
                    "omegas": {k: v.tolist() for k, v in omegas.items()},
                    "chance_overlap": float(chance),
                    "update_distance": dist.tolist(),
                    "elite_overlap": overlap.tolist(),
                    "elite_lift": lift_by_omega.tolist(),
                    "elite_fail": elite_fail.mean(axis=0).tolist(),
                    "elite_terms": elite_terms.mean(axis=0).tolist(),
                    "ess": ess.mean(axis=0).tolist(),
                },
                handle,
                indent=2,
            )
        print(f"[screen] wrote {args.out}")


if __name__ == "__main__":
    main()
