"""Gait statistics of the composed walking controller across target weights.

The report's weight table (walking report S8.4) is measured here rather than in
a notebook, so the numbers can be re-derived.  One command, one initial state,
one set of three fields; only the target weight changes.  What is measured is
physical -- how far the robot travelled, how high and how steadily it carried
the torso -- because cost can differ for uninteresting reasons and a distance
cannot.

The weight catalogue mirrors the push-recovery report's: the basis rows that
were actually fitted, interior mixtures, strong emphases, and then the pairs
and pure single-row requests that sit *outside* the basis' convex hull.  The
last two groups are extrapolation and are labelled as such: nothing was
collected at a weight with a zero row.

`--teacher` adds DIAL at the same weight from the same state, which is the only
way to read a cost column across weights -- a pure-row objective is a different
function, so its cost is not comparable to uniform's except through the ratio
to its own teacher.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config
from csm.compose_walk_eval import make_student, make_teacher
from csm.dial_score import ComposedDialScorePolicy
from csm.omega import normalize_omega_np
from csm.screen import COMMANDS, set_command, set_omega


# (name, raw weight, group).  Raw vectors are normalised on the way in, so
# (3,1,1) and (6,2,2) are the same request.
CATALOGUE: list[tuple[str, tuple[float, float, float], str]] = [
    ("(1,1,1)", (1, 1, 1), "uniform"),
    ("(3,1,1)", (3, 1, 1), "basis"),
    ("(1,3,1)", (1, 3, 1), "basis"),
    ("(1,1,3)", (1, 1, 3), "basis"),
    ("(2,1,1)", (2, 1, 1), "interior"),
    ("(1,2,1)", (1, 2, 1), "interior"),
    ("(1,1,2)", (1, 1, 2), "interior"),
    ("(2,2,1)", (2, 2, 1), "interior"),
    ("(1,2,2)", (1, 2, 2), "interior"),
    ("(2,1,2)", (2, 1, 2), "interior"),
    ("(3,1,2)", (3, 1, 2), "interior"),
    ("(5,1,1)", (5, 1, 1), "strong"),
    ("(1,5,1)", (1, 5, 1), "strong"),
    ("(1,1,5)", (1, 1, 5), "strong"),
    ("(1,1,0)", (1, 1, 0), "pair"),
    ("(0,1,1)", (0, 1, 1), "pair"),
    ("(1,0,1)", (1, 0, 1), "pair"),
    ("(1,0,0)", (1, 0, 0), "pure"),
    ("(0,1,0)", (0, 1, 0), "pure"),
    ("(0,0,1)", (0, 0, 1), "pure"),
]


def _summarise(pos, reward, done, terms, n_steps):
    """Distance, torso height and its spread, plus cost and survival.

    Cost is averaged over the whole episode rather than up to the first fall:
    truncating there scores a controller that went down early on the steps
    before it had accumulated any cost, which reports the fallen ones as best.

    The three rows are reported separately as well.  Torso height spread is a
    property of the torso and says nothing about the gait row, which prices
    *foot* height against the nominal trot trajectory -- reading one off the
    other is exactly the confusion this column invites.
    """

    pos = np.asarray(pos)
    reward = np.asarray(reward)
    done = np.asarray(done)
    terms = np.asarray(terms)
    z = pos[:, 2]
    fell = bool(np.any(done > 0.5))
    alive = int(np.argmax(done > 0.5)) if fell else n_steps
    return {
        "distance": float(pos[-1, 0] - pos[0, 0]),
        "lateral": float(abs(pos[-1, 1] - pos[0, 1])),
        "height_mean": float(z.mean()),
        "height_std": float(z.std()),
        "cost": -float(reward.mean()),
        "row_track": -float(terms[:, 0].mean()),
        "row_stab": -float(terms[:, 1].mean()),
        "row_gait": -float(terms[:, 2].mean()),
        "fell": fell,
        "alive": alive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--example", default="unitree_go2_trot_csm")
    parser.add_argument("--command", default="box_fast",
                        help="a key of csm.screen.COMMANDS")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--init-passes", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--level-scales", type=float, nargs="+",
                        default=[2.625, 1.0])
    parser.add_argument("--teacher", action="store_true",
                        help="also run DIAL at each weight, for a cost ratio")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    temperature = (args.temperature if args.temperature is not None
                   else dial_config.temp_sample)
    dial_config = dataclasses.replace(dial_config, temp_sample=temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    reset = jax.jit(env.reset)
    torso = env._torso_idx - 1

    policy = ComposedDialScorePolicy.load(args.policy)

    def record(st):
        return (st.pipeline_state.x.pos[torso], st.reward, st.done,
                st.info["reward_terms"])

    student = make_student(env, policy, dial_config, args.init_passes,
                           args.steps, record=record)
    teacher = None
    if args.teacher:
        teacher = make_teacher(env, mbdpi, dial_config, args.init_passes,
                               False, args.steps, tuple(args.level_scales),
                               record=record)

    command = COMMANDS[args.command]
    print(f"policy {args.policy}")
    print(f"command {args.command} = {command}  T={temperature}  "
          f"steps={args.steps}  seeds={args.seeds}")
    header = (f"{'weight':<9}{'group':<9}{'dist':>7}{'|dy|':>7}{'z':>7}"
              f"{'z std':>8}{'cost':>8}{'track':>8}{'stab':>8}{'gait':>8}"
              f"{'fall':>6}{'alive':>7}")
    if teacher is not None:
        header += f"{'DIAL':>9}{'ratio':>7}"
    print(header)

    rows = []
    for name, raw, group in CATALOGUE:
        omega = normalize_omega_np(np.asarray(raw, dtype=float))
        acc, tacc = [], []
        for seed in range(args.seeds):
            state = reset(jax.random.PRNGKey(11 + seed))
            state = set_command(env, state, command)
            state = set_omega(state, omega)
            acc.append(_summarise(*student(state, jnp.asarray(omega),
                                           temperature), args.steps))
            if teacher is not None:
                tacc.append(_summarise(*teacher(state,
                                                jax.random.PRNGKey(seed)),
                                       args.steps))

        def mean(key, src):
            return float(np.mean([float(r[key]) for r in src]))

        row = {"weight": name, "group": group,
               "omega": [float(v) for v in omega]}
        row.update({k: mean(k, acc) for k in
                    ("distance", "lateral", "height_mean", "height_std",
                     "cost", "row_track", "row_stab", "row_gait", "alive")})
        row["falls"] = int(sum(r["fell"] for r in acc))
        line = (f"{name:<9}{group:<9}{row['distance']:7.3f}"
                f"{row['lateral']:7.3f}{row['height_mean']:7.3f}"
                f"{row['height_std']:8.4f}{row['cost']:8.4f}"
                f"{row['row_track']:8.4f}{row['row_stab']:8.4f}"
                f"{row['row_gait']:8.4f}"
                f"{row['falls']:6d}{row['alive']:7.0f}")
        if teacher is not None:
            row["dial_cost"] = mean("cost", tacc)
            row["dial_falls"] = int(sum(r["fell"] for r in tacc))
            row["ratio"] = row["cost"] / max(row["dial_cost"], 1e-9)
            line += f"{row['dial_cost']:9.4f}{row['ratio']:7.3f}"
        print(line, flush=True)
        rows.append(row)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"policy": str(args.policy), "command": args.command,
             "command_value": list(command), "steps": args.steps,
             "seeds": args.seeds, "temperature": float(temperature),
             "rows": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
