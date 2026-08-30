"""Screen the in-hand basis: does the weight pick how the object gets turned?

The goal orientation turns continuously, so the four rows stay in competition
for the whole episode.  Two measurements, kept apart for the same reason as in
the locomotion screens.

`cloud` is the discriminativeness test: one shared warm-up, one proposal cloud
at the final annealing level, every omega re-weighting those same samples.  It
also measures each row's spread, which is what the row normalizers have to
equalise -- a row whose weight cannot move the argmax is not part of the basis
however large you make it.

`manipulate` is the physical evidence.  Each omega drives a full DIAL loop and
the four channels are read directly: how fast the object actually turned, how
many fingertips stayed on it, how hard they pressed, and how far the hand left
its nominal configuration.  Cost scores can agree for uninteresting reasons; a
fingertip count cannot.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step, make_rollout, make_sampler
from csm.gait_choice_eval import analyse_cloud, make_warm_start

ROWS = ["progress", "security", "force", "posture"]


def set_omega(state, omega):
    info = dict(state.info)
    info["reward_weights"] = jnp.asarray(omega, dtype=jnp.float32)
    return state.replace(info=info)


def channels(env, state):
    """One row of physical evidence per reward row."""

    ps = state.pipeline_state
    index = env._object_body_idx - 1
    return {
        "spin": jnp.dot(ps.xd.ang[index], state.info["goal_axis"]),
        "angle_error": state.info["angle_error"],
        "contact": state.info["n_contact"],
        "drift": jnp.linalg.norm(ps.x.pos[index] - env._nominal_object),
        "squeeze": state.info["squeeze"],
        "posture": jnp.sum(jnp.square(ps.q[7:] - env._nominal_hand)),
        "terms": state.info["reward_terms"],
        "done": state.done,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", default="allegro_in_hand")
    parser.add_argument("--mode", default="both",
                        choices=["cloud", "manipulate", "both"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--elite-frac", type=float, default=0.02)
    parser.add_argument("--no-std-normalize", action="store_true")
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    std_normalize = not args.no_std_normalize
    dial_config, env_config = _load_config(args.example, None)
    if args.nsample is not None:
        dial_config = dataclasses.replace(dial_config, Nsample=args.nsample)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    reset = jax.jit(env.reset)

    catalogue = build_omegas(4)
    names = (args.weights.split(",") if args.weights
             else ["uniform"] + [f"boost{i}" for i in range(4)]
                  + [f"e{i}" for i in range(4)])
    omegas = {n: catalogue[n] for n in names}
    omega_mat = np.stack([omegas[n] for n in names])

    print(f"env {dial_config.env_name}  Hsample {dial_config.Hsample} "
          f"({dial_config.Hsample * env.dt:.2f} s)  Nsample {dial_config.Nsample}  "
          f"temp {dial_config.temp_sample}  goal {env_config.goal_rate} rad/s",
          flush=True)
    print("rows: " + ", ".join(ROWS), flush=True)

    warm = make_warm_start(env, mbdpi, dial_config, std_normalize)
    control = make_dial_step(env, mbdpi, dial_config, std_normalize=std_normalize)

    @jax.jit
    def shared_start(state, rng):
        """Warm up on the shared weights so no omega is judged on its own states."""

        rng, plan = warm(state, rng)

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control(st, key, pl)
            return (st, key, pl), None

        (state, rng, plan), _ = jax.lax.scan(
            body, (state, rng, plan), None, length=args.warmup
        )
        return state, plan

    report = {"example": args.example, "rows": ROWS,
              "weights": {n: omegas[n].tolist() for n in names}}

    # ------------------------------------------------------------------ #
    if args.mode in ("cloud", "both"):
        sample = make_sampler(mbdpi, dial_config)
        final_noise = mbdpi.sigma_control * (
            dial_config.traj_diffuse_factor ** (dial_config.Ndiffuse - 1)
        )
        rollout = make_rollout(env, lambda s: (s.info["reward_terms"], s.done))
        rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))

        @jax.jit
        def cloud(state, plan, rng):
            plan = mbdpi.shift(plan)
            rng, key = jax.random.split(rng)
            nodes = sample(key, plan, final_noise)
            terms, done = rollout_vmap(state, mbdpi.node2u_vvmap(nodes))
            return terms.mean(axis=1), done.max(axis=1)

        all_terms, t0 = [], time.time()
        for k in range(args.states):
            st = set_omega(reset(jax.random.PRNGKey(200 + k)), catalogue["uniform"])
            st, plan = shared_start(st, jax.random.PRNGKey(k))
            terms, done = cloud(st, plan, jax.random.PRNGKey(900 + k))
            all_terms.append(np.asarray(terms))
            print(f"  cloud {k+1}/{args.states}  drop-rate "
                  f"{float(np.mean(np.asarray(done))):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        row_std = np.mean([t.std(axis=0) for t in all_terms], axis=0)
        mult = float(np.mean(row_std)) / np.maximum(row_std, 1e-12)
        # Equalise at constant total magnitude.  Raising the weak rows alone is
        # what put the walking robot on the ground two tasks ago.
        mult = mult / float(np.exp(np.mean(np.log(mult))))
        current = [env_config.progress_scale, env_config.security_scale,
                   env_config.force_scale, env_config.posture_scale]
        print("\n=== row scale calibration ===", flush=True)
        print(f"{'row':<10}{'std':>12}{'x to equalise':>15}{'new scale':>13}")
        for i, r in enumerate(ROWS):
            print(f"{r:<10}{row_std[i]:12.4g}{mult[i]:15.3f}"
                  f"{current[i]/mult[i]:13.4g}")

        per = [analyse_cloud(t, omega_mat, dial_config.temp_sample,
                             args.elite_frac, std_normalize) for t in all_terms]
        cross = np.mean([p["cross"] for p in per], axis=0)
        jac = np.mean([p["jaccard"] for p in per], axis=0)
        head = "".join(f"{n:>11}" for n in names)
        print("\n=== cross-evaluation on the shared cloud ===")
        print(f"{'update':<9}{head}")
        for i, n in enumerate(names):
            print(f"{n:<9}" + "".join(f"{cross[i,j]:11.4f}"
                                     for j in range(len(names))))
        wins = sum(int(np.argmax(cross[:, j]) == j) for j in range(len(names)))
        margins = [
            100.0 * (cross[j, j] - np.max(np.delete(cross[:, j], j)))
            / max(abs(np.max(np.delete(cross[:, j], j))), 1e-9)
            for j in range(len(names))
        ]
        print(f"\ndiagonal wins: {wins}/{len(names)}")
        print("margins %:  " + "  ".join(f"{n}={m:+.2f}"
                                        for n, m in zip(names, margins)))
        print("vs temperature: " + "  ".join(
            f"{t:.3f}:{sum(int(np.argmax(c[:, j]) == j) for j in range(len(names)))}"
            f"/{len(names)}"
            for t, c in ((t, np.mean([analyse_cloud(x, omega_mat, t,
                                                    args.elite_frac,
                                                    std_normalize)["cross"]
                                      for x in all_terms], axis=0))
                         for t in [dial_config.temp_sample * f
                                   for f in (0.2, 0.5, 1, 2, 5)])))
        print(f"\n=== elite overlap (chance ~ {args.elite_frac:.3f}) ===")
        print(f"{'':<9}{head}")
        for i, n in enumerate(names):
            print(f"{n:<9}" + "".join(f"{jac[i,j]:11.3f}"
                                     for j in range(len(names))))
        print(f"\n=== ESS vs temperature (of {dial_config.Nsample+1}) ===")
        print(f"{'temp':<9}" + "".join(f"{n:>10}" for n in names))
        for temp in [dial_config.temp_sample * f for f in (0.2, 0.5, 1, 2, 5)]:
            e = np.mean([analyse_cloud(t, omega_mat, temp, args.elite_frac,
                                       std_normalize)["ess"]
                         for t in all_terms], axis=0)
            print(f"{temp:<9.3f}" + "".join(
                f"{100*v/(dial_config.Nsample+1):9.1f}%" for v in e))
        report["cloud"] = {"row_std": row_std.tolist(), "cross": cross.tolist(),
                           "jaccard": jac.tolist(), "diagonal_wins": wins,
                           "margins_pct": margins}

    # ------------------------------------------------------------------ #
    if args.mode in ("manipulate", "both"):
        @jax.jit
        def run(state, rng, plan):
            def body(carry, _):
                st, key, pl = carry
                st, key, pl = control(st, key, pl)
                return (st, key, pl), channels(env, st)

            _, trace = jax.lax.scan(
                body, (state, rng, plan), None, length=args.steps
            )
            return trace

        print(f"\n=== turning the object (goal {env_config.goal_rate} rad/s) ===")
        print(f"{'weight':<9}{'spin':>8}{'ang':>8}{'contact':>9}{'drift':>8}"
              f"{'squeeze':>11}{'posture':>9}{'alive':>7}{'drop':>6}", flush=True)
        starts, stats = [], {}
        for seed in range(args.seeds):
            st = set_omega(reset(jax.random.PRNGKey(seed)), catalogue["uniform"])
            starts.append(shared_start(st, jax.random.PRNGKey(seed)))
        for n in names:
            runs = []
            for seed, (st, plan) in enumerate(starts):
                trace = run(set_omega(st, omegas[n]),
                            jax.random.PRNGKey(seed), plan)
                done = np.asarray(trace["done"])
                alive = (int(np.argmax(done > 0.5)) if np.any(done > 0.5)
                         else args.steps)
                sl = slice(0, max(alive, 1))
                runs.append({
                    k: float(np.asarray(v)[sl].mean())
                    for k, v in trace.items() if k not in ("terms", "done")
                } | {"alive": alive, "dropped": bool(np.any(done > 0.5)),
                     "row_means": np.asarray(trace["terms"])[sl].mean(0).tolist()})
            stats[n] = runs
            m = lambda k: float(np.mean([r[k] for r in runs]))
            drops = sum(r["dropped"] for r in runs)
            print(f"{n:<9}{m('spin'):8.3f}{m('angle_error'):8.3f}"
                  f"{m('contact'):9.2f}{m('drift'):8.4f}{m('squeeze'):11.2e}"
                  f"{m('posture'):9.3f}{m('alive'):7.0f}"
                  f"{(f'{drops}/{len(runs)}' if drops else '-'):>6}", flush=True)
        report["manipulate"] = stats

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
