"""Measure the cost rows, then set the weights from what was measured.

A standing rule on this project, and one the previous task needed twice: an
objective whose rows differ by three orders of magnitude is one term with six
decorations, and no temperature can rescue it.  The split follows what each row
is *for*:

  always-active rows (goal, slosh, effort) are balanced by their measured
    spread across the proposal cloud, so the objective is a genuine trade.
    Spread and not mean, because a softmax is blind to a constant offset -- what
    decides between two sampled plans is how much the row *varies*, and a row
    with a large mean and no spread is invisible to the planner while looking
    important in a table.

  constraint rows (spill, freeboard, tilt, vel) keep large weights.  They are
    hinges that sit at zero almost always, and a term that is zero until it is
    violated is supposed to be expensive when it is; scaling one by its own
    spread would make a rare violation cheap, which is the opposite of what a
    limit means.

The frontier measurement that precedes this fixed the one weight the rule above
cannot set -- how much distance is worth how much liquid.  At `w_goal = 1` the
planner will not move a full tray at all (0.367 m short of a 0.60 m carry); at
50 it spills a fifth of the liquid; at 10 it arrives 0.079 m short having
spilled nothing.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P

ALWAYS = ("goal", "slosh", "effort")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=int, default=512)
    ap.add_argument("--thetas", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--anchor", type=float, default=0.30,
                    help="the plan the cloud is centred on: a plain carry at "
                         "this fraction of the acceleration limit, which is "
                         "roughly where the planner operates")
    args = ap.parse_args(argv)

    _, rows_fn = P.make_cost_matrix()
    s0 = P.stage_init()
    th = P.sample_theta(jax.random.PRNGKey(1), args.thetas)

    # centre the cloud on a plausible plan, not on zero: rows are measured
    # where the planner works, and at U = 0 the stage never moves so the goal
    # row is a constant and the spill row is identically zero
    U0 = jnp.zeros((P.N_NODE, P.NU)).at[:P.N_NODE // 3, 0].set(args.anchor) \
        .at[2 * P.N_NODE // 3:, 0].set(-args.anchor).reshape(-1)
    eps = args.sigma * jax.random.normal(jax.random.PRNGKey(0),
                                         (args.plans, P.DIM_U))
    V = jnp.clip(U0[None, :] + eps, -1.0, 1.0)

    f = jax.jit(jax.vmap(jax.vmap(rows_fn, in_axes=(None, 0, None, None)),
                         in_axes=(None, None, 0, None)))
    R = np.asarray(f(s0, V, th, P.GOAL))            # (thetas, plans, 7)
    flat = R.reshape(-1, len(P.ROW_NAMES))

    print(f"{args.plans} plans x {args.thetas} hypotheses, sigma {args.sigma}, "
          f"centred on a {args.anchor:.2f}-amplitude carry\n")
    hdr = (f"{'row':<11}{'mean':>11}{'std':>11}{'max':>11}{'zero %':>9}"
           f"{'kind':>12}")
    print(hdr); print("-" * len(hdr))
    for i, n in enumerate(P.ROW_NAMES):
        c = flat[:, i]
        kind = "always" if n in ALWAYS else "constraint"
        print(f"{n:<11}{c.mean():>11.4f}{c.std():>11.4f}{c.max():>11.4f}"
              f"{100*(c == 0).mean():>8.1f}%{kind:>12}")

    # --- proposed weights ----------------------------------------------------
    # equalise the *spread* of the always-active rows against the goal row,
    # then keep the frontier's goal weight and the constraint weights.
    print("\n=== proposed weights ===")
    idx = {n: i for i, n in enumerate(P.ROW_NAMES)}
    sd = {n: flat[:, idx[n]].std() for n in P.ROW_NAMES}
    w_goal = 10.0            # from the distance-versus-spill frontier
    props = {"goal": w_goal}
    for n in ALWAYS:
        if n == "goal":
            continue
        props[n] = float(w_goal * sd["goal"] / max(sd[n], 1e-12))
    for n in P.ROW_NAMES:
        if n not in ALWAYS:
            props[n] = getattr(P.Costs(), n)
    for n in P.ROW_NAMES:
        contrib = props[n] * sd[n]
        print(f"  {n:<11}{props[n]:>12.4f}   weighted spread {contrib:>10.4f}"
              f"   {'always' if n in ALWAYS else 'constraint'}")
    print("\n  weighted spread is what the softmax actually ranks plans by; the "
          "three\n  always-active rows are equalised on it, the constraint rows "
          "are not and\n  should not be.")
    print(f"\nCosts({', '.join(f'{n}={props[n]:.4g}' for n in P.ROW_NAMES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
