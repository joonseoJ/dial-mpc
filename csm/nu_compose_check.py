"""Does composing basis fields in nu space reproduce DIAL at a target weight?

The claim compositional score matching rests on is that the score is linear in
the weight vector, so a target can be written as a fixed mixture of a few basis
fields.  This measures that claim on DIAL itself, before any network is trained
-- if DIAL's own updates do not compose, no student fitted to them will.

The measurement uses one shared sample cloud per state.  A rollout does not
depend on the weights, so every basis field and every target can be scored from
the same 2049 trajectories: only the softmax differs.  That removes sampling
noise from the comparison entirely, leaving exactly the question being asked --
is the *weighting* linear?

Two conventions are compared.

  nu space   : coefficients solved so that `sum a_i (omega_i / T_i) = omega* / T*`.
               Correct whenever the fields carry different temperatures, since
               a field trained at `(omega_i, T_i)` learned the score for `nu_i`.
  omega space: the legacy solve, least squares in omega rescaled to sum to one.
               Equivalent to nu space only when all temperatures agree.

And two DIAL variants, because the standard-deviation normalisation inside
`reverse_once` is what breaks linearity in the first place: it divides returns
by their own spread, which is a nonlinear function of the weights, so the
update is the score of a warped objective rather than of `omega . C`.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

import brax.envs as brax_envs

from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step, make_rollout, make_sampler
from csm.omega import normalize_omega_np, nu_coefficients, omega_coefficients


def make_cloud(env, mbdpi, dial_config, warmup_steps: int, drive_temp: float):
    """Warm DIAL up, then hand back one proposal cloud and its per-row costs."""

    control_step = make_dial_step(env, mbdpi, dial_config)
    sample = make_sampler(mbdpi, dial_config)
    final_noise = mbdpi.sigma_control * (
        dial_config.traj_diffuse_factor ** (dial_config.Ndiffuse - 1)
    )
    rollout = make_rollout(env, lambda s: s.info["reward_terms"])
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))

    def cloud(state, rng):
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
        terms = rollout_vmap(state, mbdpi.node2u_vvmap(nodes)).mean(axis=1)
        return terms, nodes, plan

    return cloud


def deltas(terms, nodes, plan, omegas, temps, std_normalize: bool):
    """MPPI's applied update for many (omega, temperature) pairs, one cloud.

    With `std_normalize` off the exponent is literally `nu . (-C)`, so the
    weighting is a Gibbs distribution at a stated temperature.  With it on it
    is `reverse_once` as written, whose effective temperature is `T * std(R)`
    and therefore weight-dependent.
    """

    returns = terms @ omegas.T                       # (n_sample, k)
    reference = returns[-1]
    if std_normalize:
        scale = jnp.maximum(returns.std(axis=0), 1e-6)
        logits = (returns - reference) / scale / temps
    else:
        logits = (returns - reference) / temps
    weights = jax.nn.softmax(logits, axis=0)
    return jnp.einsum("nk,nij->kij", weights, nodes) - plan


def agreement(composed: np.ndarray, actual: np.ndarray) -> dict:
    a, b = composed.reshape(-1), actual.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "cosine": float(a @ b / max(na * nb, 1e-12)),
        "magnitude": float(na / max(nb, 1e-12)),
        "rel_error": float(np.linalg.norm(a - b) / max(nb, 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--basis", nargs="+",
                        default=["boost0", "boost1", "boost2", "boost3"])
    parser.add_argument("--basis-temps", type=float, nargs="+",
                        default=[0.25, 0.30, 0.37, 0.39])
    parser.add_argument("--target-temps", type=float, nargs="+",
                        default=[0.25, 0.30, 0.37])
    parser.add_argument("--targets", nargs="+", default=None,
                        help="weight directions to compose, as comma-separated "
                             "vectors; defaults to a set of interior points")
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--drive-temp", type=float, default=0.30)
    parser.add_argument("--std-normalize", dest="std", action="store_true")
    parser.add_argument("--no-std-normalize", dest="std", action="store_false")
    parser.set_defaults(std=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if len(args.basis) != len(args.basis_temps):
        raise ValueError("--basis and --basis-temps must have the same length")

    dial_config, env_config = _load_config(args.example, args.config)
    import dataclasses

    dial_config = dataclasses.replace(dial_config, temp_sample=args.drive_temp)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    basis_omegas = np.stack([catalogue[n] for n in args.basis])
    basis_temps = np.asarray(args.basis_temps, dtype=float)

    # Targets: basis rows at their own temperature (reduction), basis rows at a
    # different temperature (retempering), and interior directions.
    targets: list[tuple[str, np.ndarray, float]] = []
    for name, temp in zip(args.basis, basis_temps):
        targets.append((f"{name}@own", catalogue[name], float(temp)))
    targets.append((f"{args.basis[0]}@retemper",
                    catalogue[args.basis[0]], float(basis_temps[-1])))
    if args.targets:
        interior = {
            name: np.array([float(v) for v in name.split(",")])
            for name in args.targets
        }
    else:
        interior = {
            "uniform": np.ones(n_rows),
            "(2,1,1,1)": np.array([2.0, 1.0, 1.0, 1.0][:n_rows]),
            "(1,2,2,1)": np.array([1.0, 2.0, 2.0, 1.0][:n_rows]),
            "(1,1,2,2)": np.array([1.0, 1.0, 2.0, 2.0][:n_rows]),
            "(3,2,1,1)": np.array([3.0, 2.0, 1.0, 1.0][:n_rows]),
        }
    for name, vec in interior.items():
        for temp in args.target_temps:
            targets.append((f"{name}@{temp:g}", normalize_omega_np(vec), float(temp)))

    target_omegas = np.stack([t[1] for t in targets])
    target_temps = np.asarray([t[2] for t in targets])

    all_omegas = jnp.asarray(np.concatenate([basis_omegas, target_omegas]))
    all_temps = jnp.asarray(np.concatenate([basis_temps, target_temps]))
    n_basis = len(args.basis)

    cloud = make_cloud(env, mbdpi, dial_config, args.warmup_steps, args.drive_temp)
    reset_vmap = jax.jit(jax.vmap(env.reset))
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng, cloud_rng = jax.random.split(rng, 3)
    states = reset_vmap(jax.random.split(reset_rng, args.states))
    cloud_rngs = jax.random.split(cloud_rng, args.states)

    chunk = max(min(args.chunk, args.states), 1)
    if args.states % chunk:
        raise ValueError("--states must be a multiple of --chunk")
    index = jnp.arange(args.states).reshape(-1, chunk)

    def one(state, key):
        terms, nodes, plan = cloud(state, key)
        return deltas(terms, nodes, plan, all_omegas, all_temps, args.std)

    batch = jax.vmap(one)

    @jax.jit
    def run(states, keys):
        def chunk_fn(ids):
            return batch(jax.tree.map(lambda x: x[ids], states), keys[ids])

        out = jax.lax.map(chunk_fn, index)
        return out.reshape((-1,) + out.shape[2:])

    print(f"[nu] basis={args.basis} temps={list(basis_temps)} "
          f"std_normalize={args.std} states={args.states} "
          f"Nsample={dial_config.Nsample + 1}")
    t0 = time.time()
    updates = np.asarray(run(states, cloud_rngs))   # (states, k, Hnode+1, nu)
    print(f"[nu] {args.states} clouds in {time.time() - t0:.1f}s")

    basis_updates = updates[:, :n_basis]
    rows = []
    for j, (name, omega, temp) in enumerate(targets):
        actual = updates[:, n_basis + j]
        a_nu = nu_coefficients(basis_omegas, basis_temps, omega, temp)
        a_om = omega_coefficients(basis_omegas, omega)
        nu_scores, om_scores = [], []
        for s in range(updates.shape[0]):
            nu_scores.append(
                agreement(np.einsum("k,kij->ij", a_nu, basis_updates[s]), actual[s])
            )
            om_scores.append(
                agreement(np.einsum("k,kij->ij", a_om, basis_updates[s]), actual[s])
            )
        rows.append({
            "target": name,
            "omega_vec": omega.tolist(),
            "temp": temp,
            "coefficients_nu": a_nu.tolist(),
            "coefficients_omega": a_om.tolist(),
            "nu": {k: float(np.mean([d[k] for d in nu_scores]))
                   for k in ("cosine", "magnitude", "rel_error")},
            "omega": {k: float(np.mean([d[k] for d in om_scores]))
                      for k in ("cosine", "magnitude", "rel_error")},
        })

    width = max(len(r["target"]) for r in rows) + 2
    print(f"\n=== composed vs actual DIAL update "
          f"(std_normalize={args.std}) ===")
    print(f"{'target':<{width}}{'nu cos':>9}{'nu mag':>9}{'nu err':>9}"
          f"{'om cos':>10}{'om mag':>9}{'om err':>9}")
    for r in rows:
        print(f"{r['target']:<{width}}"
              f"{r['nu']['cosine']:9.4f}{r['nu']['magnitude']:9.3f}"
              f"{r['nu']['rel_error']:9.3f}"
              f"{r['omega']['cosine']:10.4f}{r['omega']['magnitude']:9.3f}"
              f"{r['omega']['rel_error']:9.3f}")

    reduction = [r for r in rows if r["target"].endswith("@own")]
    interior_rows = [r for r in rows if "@own" not in r["target"]
                     and "retemper" not in r["target"]]
    print(f"\n[reduction]  nu  cos {np.mean([r['nu']['cosine'] for r in reduction]):.5f} "
          f"err {np.mean([r['nu']['rel_error'] for r in reduction]):.2e}")
    print(f"[reduction]  om  cos {np.mean([r['omega']['cosine'] for r in reduction]):.5f} "
          f"err {np.mean([r['omega']['rel_error'] for r in reduction]):.2e}")
    print(f"[interior]   nu  cos {np.mean([r['nu']['cosine'] for r in interior_rows]):.4f} "
          f"mag {np.mean([r['nu']['magnitude'] for r in interior_rows]):.3f} "
          f"err {np.mean([r['nu']['rel_error'] for r in interior_rows]):.3f}")
    print(f"[interior]   om  cos {np.mean([r['omega']['cosine'] for r in interior_rows]):.4f} "
          f"mag {np.mean([r['omega']['magnitude'] for r in interior_rows]):.3f} "
          f"err {np.mean([r['omega']['rel_error'] for r in interior_rows]):.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"basis": args.basis,
                       "basis_temps": basis_temps.tolist(),
                       "std_normalize": bool(args.std),
                       "states": args.states,
                       "rows": rows}, handle, indent=2)
        print(f"[nu] wrote {args.out}")


if __name__ == "__main__":
    main()
