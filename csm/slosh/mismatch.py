"""SUPERSEDED -- plans from `fluid_init`, the t=0 liquid, at every step.

Every result this script produced ran on the liquid-reset bug described in
`plant.make_rollout` and is invalid; `closed_loop.py` and `gate.py` replace it.
Kept only so the numbers quoted in the notes can be traced to their source.

Does planning under the wrong hypothesis actually cost anything here?

This is the gate that the previous task failed, and failing it cheaply is the
whole point of running it before E0.  On the rigid-payload arm the mismatch
penalty was +0.038 m of goal distance -- real but small -- and E0 then took two
hours to say the same thing: success saturated at `M = 4`, because one
compromise plan served the entire prior.

Plan under `theta_a`, execute under `theta_b`, for every pair.  The diagonal
answers "is it solvable when you know", the off-diagonal answers "does knowing
matter", and only the second is what `M` can buy.  A cost-ranking correlation
between hypotheses does not answer it: that stays high even when the optimal
*actions* differ, because most of a plan's cost is the part every hypothesis
agrees about.

The probe set varies the two axes the task was designed around:

  fill depth   moves the slosh frequency as `sqrt(h)` -- a factor of 1.6 across
               the range -- and the freeboard in the opposite direction, so a
               full tray is both slower to settle and less forgiving.  One
               unknown setting both is what makes the constraint threshold
               belong to `theta`.
  slosh phase  where the wave is *right now*.  Two hypotheses half a cycle
               apart want the acceleration pulse at opposite times; that is a
               disagreement in sign, not in magnitude, and it is the kind an
               averaged plan cannot serve.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P


def probe_thetas(n_phase=4, amp=0.04, depths=(0.020, 0.050)):
    """A small deliberate set, not a random draw.

    `amp` is the pre-existing wave as a fraction of the depth.  At 0.04 the
    phase axis produced no planner/truth interaction at all -- the wave the
    carry itself raises swamps the one already there, so "which phase does the
    planner assume" stopped mattering.  It is a flag because that is a
    hypothesis about the task, and the way to settle it is to vary it.
    """
    out, names = [], []
    for h0 in depths:
        for j in range(n_phase):
            ph = 2 * np.pi * j / n_phase
            out.append([h0, 1000.0, 0.02, amp, ph, 0.0, 0.0])
            names.append(f"h{h0:.3f} ph{j}")
    return jnp.asarray(out), names


def make_batched(n_pair, n_sample, w: P.Costs):
    """Every (plan, true) pair advanced in lockstep, in one dispatch."""
    _, cost = P.make_rollout(w)

    @jax.jit
    def update(states, fluids, Us, th_plan, sigma, lam, goal, key):
        eps = sigma * jax.random.normal(key, (n_pair, n_sample, P.DIM_U))
        eps = eps.at[:, -1, :].set(0.0)
        V = jnp.clip(Us[:, None, :] + eps, -1.0, 1.0)

        def per_pair(s, v, th):
            return jax.vmap(cost, in_axes=(None, 0, None, None))(s, v, th, goal)

        C = jax.vmap(per_pair)(states, V, th_plan)
        om = jax.nn.softmax(-(C - C.min(axis=1, keepdims=True)) / lam, axis=1)
        return jnp.clip(Us + jnp.einsum("pk,pkd->pd", om, eps), -1.0, 1.0)

    @jax.jit
    def world(states, fluids, us, th_true, goal):
        def one(s, U, u, th):
            ns, a, al = P.stage_step(s, u)
            nU, lost = P.fluid_step(U, s, a, al, th)
            return ns, nU, P.stage_cost(nU, ns, u, th, goal, lost)
        return jax.vmap(one)(states, fluids, us, th_true)

    return update, world


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--sigma", type=float, nargs="+",
                    default=list(P.SIGMA_SCHEDULE_3))
    ap.add_argument("--lam", type=float, nargs="+", default=list(P.LAM_SCHEDULE_3))
    ap.add_argument("--amp", type=float, default=0.04,
                    help="pre-existing wave as a fraction of the fill depth")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.020, 0.050])
    args = ap.parse_args(argv)

    th, names = probe_thetas(amp=args.amp, depths=tuple(args.depths))
    n = th.shape[0]
    pa, pb = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pa, pb = pa.ravel(), pb.ravel()
    th_plan, th_true = th[jnp.asarray(pa)], th[jnp.asarray(pb)]
    n_pair = n * n

    update, world = make_batched(n_pair, args.samples, P.Costs())
    states = jnp.tile(P.stage_init(), (n_pair, 1))
    fluids = jax.vmap(P.fluid_init)(th_true)
    Us = jnp.zeros((n_pair, P.DIM_U))
    prof = jnp.zeros((n_pair, len(P.ROW_NAMES)))

    print(f"plan under theta_a, execute under theta_b.  {n_pair} pairs in "
          f"lockstep, {args.steps} control steps, {args.samples} plans "
          f"-> {n_pair * args.samples:,} rollouts per dispatch")
    print(f"carry {float(P.GOAL[0]):.2f} m, sigma {args.sigma}, lam {args.lam}, "
          f"initial wave {100*args.amp:.0f}% of depth\n", flush=True)

    t0 = time.time()
    key = jax.random.PRNGKey(0)
    for t in range(args.steps):
        for sg, lm in zip(args.sigma, args.lam):
            key, k = jax.random.split(key)
            Us = update(states, fluids, Us, th_plan, sg, lm, P.GOAL, k)
        states, fluids, rows = world(states, fluids, Us[:, :P.NU], th_true, P.GOAL)
        prof = prof + rows
        Us = jax.vmap(P.shift_nodes)(Us)
        if t == 0:
            print(f"  first step (with compile) {time.time()-t0:.0f}s", flush=True)
    gaps = np.asarray(jnp.linalg.norm(states[:, :2] - P.GOAL, axis=-1)).reshape(n, n)
    prof = np.asarray(prof).reshape(n, n, len(P.ROW_NAMES))
    print(f"  done in {time.time()-t0:.0f}s\n")

    def show(Mx, title, unit):
        print(f"=== {title} ({unit}) ===")
        head = "plan \\ true"
        print(f"{head:<14}" + "".join(f"{x:>13}" for x in names))
        for a in range(n):
            print(f"{names[a]:<14}" + "".join(f"{Mx[a, b]:13.4f}" for b in range(n)))
        d = np.diag(Mx).mean()
        off = (Mx.sum() - np.trace(Mx)) / (n * n - n)
        print(f"  matched {d:.4f}   mismatched {off:.4f}   "
              f"penalty {off - d:+.4f}"
              f"  ({100*(off-d)/max(abs(d),1e-9):+.0f}%)\n")
        return d, off

    dg, og = show(gaps, "distance left", "m")
    ds, os_ = show(prof[:, :, 1], "spilled fraction of the liquid", "-")
    dl, ol = show(prof[:, :, 3], "residual slosh", "row units")

    print("=== verdict ===")
    d0 = float(jnp.linalg.norm(P.GOAL))
    print(f"  solvable:      knowing theta ends {dg:.3f} m from a {d0:.2f} m "
          f"carry, spilling {ds:.4f}")
    print(f"  theta matters: the wrong hypothesis costs {og-dg:+.3f} m, "
          f"{os_-ds:+.4f} spill, {ol-dl:+.2f} slosh")
    print(f"\n  for comparison, the rigid-payload task that failed E0 showed a "
          f"mismatch\n  penalty of +0.038 m (+52%) and saturated at M = 4.")
    if dg > 0.15 or ds > 0.05:
        print("\n  -> the oracle does not do the task.  Fix that before reading "
              "the off-diagonal:\n     hypotheses disagreeing about something "
              "nobody can do is not evidence.")
    elif (os_ - ds) < 0.01 and (og - dg) < 0.02:
        print("\n  -> one plan serves every hypothesis.  M cannot matter here "
              "either, and the\n     premise fails regardless of what is "
              "learned afterwards.")
    else:
        print("\n  -> both hold; E0 is worth running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
