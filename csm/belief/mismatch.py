"""Two questions the task design has to answer before E0 is worth running.

    solvable       does a controller that *knows* theta reach the goal without
                   violating anything?  This is the diagonal of the matrix
                   below, and it is not optional: a task nobody can do makes
                   hypotheses disagree beautifully and is worthless.

    theta matters  does planning under the wrong hypothesis cost real
                   performance?  This is the off-diagonal.

Plan under `theta_a`, execute under `theta_b`, for every pair.  That is the
quantity E0 depends on, and far more directly than a cost-ranking correlation
between hypotheses -- which stays high even when the optimal *actions* differ,
because most of a plan's cost is the part every hypothesis agrees about.  On
this plant the correlation read 0.918 while the question of whether the arm
should tilt its wrist left or right was still wide open.

A mismatch penalty near zero means one plan serves every hypothesis, `M` cannot
matter, and the premise of belief-conditioning fails here regardless of what
any learned model does afterwards.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

from csm.belief import plant as P
from csm.belief.oracle import make_planner, make_world


def probe_thetas():
    """A small deliberate set, not a random draw.

    The axis that decides whether hypotheses can disagree in *direction* is the
    lateral centre of mass: it flips the sign of the wrist torque, so `+c` and
    `-c` ask for opposite corrections.  Mass is varied alongside because it
    scales how much any correction costs, and `mu` is held because it was
    measured to move only a threshold, never a direction.
    """
    out, names = [], []
    for m in (0.4, 2.2):
        for cy in (-0.06, 0.0, 0.06):
            out.append([m, 0.0, cy, 0.0, 0.5])
            names.append(f"m{m:.1f} cy{cy:+.2f}")
    return jnp.asarray(out), names


def make_batched(model, bare, ids, n_pair, n_sample, w: P.Costs):
    """Every (plan, true) pair advanced in lockstep, in one dispatch.

    Run pair by pair this is 512 rollouts at a time, which is squarely in the
    regime where MJX leaves the GPU idle -- measured on this plant, 512
    rollouts run near 120k physics steps/s against 2.1M at 49k rollouts, and
    the sequential version of this diagnostic was killed at forty minutes
    without finishing.  The pairs are independent, so batching them costs
    nothing but a vmap and buys back the order of magnitude.
    """
    _, cost_fn = P.make_rollout(model, bare, ids, w)

    @jax.jit
    def update(datas, Us, thetas, sigma, lam, goal, key):
        eps = sigma * jax.random.normal(key, (n_pair, n_sample, P.DIM_U))
        eps = eps.at[:, -1, :].set(0.0)          # the centre is scored too
        V = jnp.clip(Us[:, None, :] + eps, -1.0, 1.0)

        def per_pair(data, v, th):
            return jax.vmap(cost_fn, in_axes=(None, 0, None, None))(
                data, v, th, goal)

        C = jax.vmap(per_pair)(datas, V, thetas)          # (P, K)
        om = jax.nn.softmax(-(C - C.min(axis=1, keepdims=True)) / lam, axis=1)
        return jnp.clip(Us + jnp.einsum("pk,pkd->pd", om, eps), -1.0, 1.0)

    @jax.jit
    def world(datas, us, thetas_true, goal):
        def one(d, u, th):
            mdl = P.apply_theta(model, ids, th)
            prev = d.cvel[ids["payload"], 3:]
            nd, bias = P.step(mdl, bare, ids, d, u)
            return nd, P.stage_cost(mdl, ids, nd, prev, u, th, goal, bias, w)
        return jax.vmap(one)(datas, us, thetas_true)

    @jax.jit
    def init(q0, thetas_true):
        d0 = mjx.make_data(model).replace(qpos=q0)
        datas = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_pair,) + jnp.shape(x)), d0)
        return jax.vmap(lambda d, th: mjx.forward(
            P.apply_theta(model, ids, th), d))(datas, thetas_true)

    return init, update, world


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--lam", type=float, nargs="+", default=list(P.LAM_SCHEDULE),
                    help="one temperature per annealing level; a single value "
                         "cannot serve both, the cost spread differs 5x between "
                         "them")
    ap.add_argument("--sigma", type=float, nargs="+",
                    default=list(P.SIGMA_SCHEDULE))
    args = ap.parse_args(argv)

    model, bare, ids, mj = P.load()
    w = P.Costs()
    th, names = probe_thetas()
    n = th.shape[0]
    pa, pb = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pa, pb = pa.ravel(), pb.ravel()
    th_plan, th_true = th[jnp.asarray(pa)], th[jnp.asarray(pb)]
    n_pair = n * n

    init, update, world = make_batched(model, bare, ids, n_pair,
                                       args.samples, w)
    goal, q0 = P.GOAL, P.Q_CARRY
    print(f"plan under theta_a, execute under theta_b.  {n_pair} pairs in "
          f"lockstep, {args.steps} control steps ({args.steps * P.DT:.1f} s), "
          f"{args.samples} plans -> {n_pair * args.samples:,} rollouts/dispatch")
    print(f"transport {float(jnp.linalg.norm(P.GOAL - P.P_START)):.2f} m, "
          f"sigma {args.sigma} lam {args.lam}\n", flush=True)

    t0 = time.time()
    datas = init(q0, th_true)
    Us = jnp.zeros((n_pair, P.DIM_U))
    prof = jnp.zeros((n_pair, 7))
    key = jax.random.PRNGKey(0)
    for t in range(args.steps):
        for sg, lm in zip(args.sigma, args.lam):
            key, k = jax.random.split(key)
            Us = update(datas, Us, th_plan, sg, lm, goal, k)
        datas, rows = world(datas, Us[:, :P.NU], th_true, goal)
        prof = prof + rows
        Us = jax.vmap(P.shift_nodes)(Us)
        if t == 0:
            print(f"  first step (with compile) {time.time() - t0:.0f}s", flush=True)
    gaps = np.asarray(jnp.linalg.norm(
        datas.xpos[:, ids["payload"]] - goal, axis=-1)).reshape(n, n)
    prof = np.asarray(prof).reshape(n, n, 7)
    print(f"  done in {time.time() - t0:.0f}s\n")

    def show(M, title, unit):
        print(f"=== {title} ({unit}) ===")
        head = "plan \\ true"
        print(f"{head:<14}" + "".join(f"{x:>12}" for x in names))
        for a in range(n):
            print(f"{names[a]:<14}" + "".join(f"{M[a, b]:12.3f}" for b in range(n)))
        d = np.diag(M).mean()
        off = (M.sum() - np.trace(M)) / (n * n - n)
        print(f"  matched (diagonal) {d:.3f}   mismatched {off:.3f}   "
              f"penalty {off - d:+.3f}\n")
        return d, off

    dg, og = show(gaps, "distance left to the goal", "m")
    ds, os_ = show(prof[:, :, 2], "accumulated slip", "row units")
    dt_, ot = show(prof[:, :, 3], "accumulated tilt", "row units")

    d0 = float(jnp.linalg.norm(P.GOAL - P.P_START))
    print("=== verdict ===")
    print(f"  solvable:      an oracle ends {dg:.3f} m from a goal it started "
          f"{d0:.2f} m from  ({100*(1-dg/d0):.0f}% of the way)")
    print(f"  theta matters: the wrong hypothesis costs {og - dg:+.3f} m of "
          f"goal, {os_ - ds:+.2f} slip, {ot - dt_:+.2f} tilt")
    if dg > 0.25:
        print("  -> the oracle does not do the task.  Fix that first: "
              "hypotheses disagreeing\n     about an impossible task is not "
              "evidence for anything.")
    elif (og - dg) < 0.02 and (os_ - ds) < 0.5:
        print("  -> one plan serves every hypothesis.  M cannot matter here, "
              "and the premise\n     fails regardless of what is learned "
              "afterwards.")
    else:
        print("  -> both hold; E0 is worth running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
