"""Set the annealing ladder by measuring it.

Two things, in the order they depend on each other: how fine the finest
proposal has to be, and one temperature per level.  Both are measured, and the
second is measured *online* -- level `k` acts on the plan level `k-1` produced,
so its cost cloud only exists once the earlier levels have run.  At every
control step each level's cloud is built, `lambda` is bisected to hit a target
effective sample size on that cloud, the update is applied, and the geometric
mean of the chosen temperatures over steps and episodes becomes the schedule.

The target ESS is not assumed either.  On the previous plant the project's
usual 25% turned out to be worse than 50% (0.0631 m against 0.0590 m of
residual distance), so it is swept against the task rather than inherited.

Why a ladder at all: the finest sigma sets the smallest correction the planner
can propose.  On the previous plant a two-level schedule whose finest sigma was
0.25 left a floor of 0.04-0.06 m under every experiment, because each sampled
plan overshot the remaining error in some direction, the softmax saw them as
equally good, and the weighted average of symmetric noise is no update at all.
Here the same failure would show up as an inability to place an acceleration
pulse precisely enough to cancel a wave.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P

N_EP, K, M = 8, 512, 8


def make(w: P.Costs):
    _, cost = P.make_rollout(w)

    @jax.jit
    def cloud(states, fluids, Us, th_b, bel, sigma, goal, key):
        """Belief-weighted cost of one proposal cloud, and the noise that made it."""
        eps = sigma * jax.random.normal(key, (N_EP, K, P.DIM_U))
        eps = eps.at[:, -1, :].set(0.0)
        V = jnp.clip(Us[:, None, :] + eps, -1.0, 1.0)

        def per_ep(s, v):
            return jax.vmap(jax.vmap(cost, in_axes=(None, 0, None, None)),
                            in_axes=(None, None, 0, None), out_axes=1)(
                                s, v, th_b, goal) @ bel

        return jax.vmap(per_ep)(states, V), eps

    @jax.jit
    def apply(Us, Cw, eps, lam):
        om = jax.nn.softmax(-(Cw - Cw.min(axis=1, keepdims=True)) / lam[:, None],
                            axis=1)
        return jnp.clip(Us + jnp.einsum("ek,ekd->ed", om, eps), -1.0, 1.0)

    @jax.jit
    def world(states, fluids, us, th_true, goal):
        def one(s, U, u, th):
            ns, a, al = P.stage_step(s, u)
            nU, lost = P.fluid_step(U, s, a, al, th)
            return ns, nU, P.stage_cost(nU, ns, u, th, goal, lost)
        return jax.vmap(one)(states, fluids, us, th_true)

    return cloud, apply, world


def lam_for_ess(c, target, lo=1e-6, hi=1e6, iters=60):
    """Bisect the temperature so this cloud's ESS fraction hits `target`.

    ESS rises monotonically with lambda, so bisection is exact and costs no
    physics -- the cloud is evaluated once and then re-weighted.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c - c.min()

    def ess(lam):
        wgt = np.exp(-c / lam)
        wgt = wgt / wgt.sum()
        return 1.0 / np.sum(wgt ** 2) / wgt.size

    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if ess(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def episode(cloud, apply, world, sigmas, target, th_true, th_b, bel, steps, key):
    states = jnp.tile(P.stage_init(), (N_EP, 1))
    fluids = jax.vmap(P.fluid_init)(th_true)
    Us = jnp.zeros((N_EP, P.DIM_U))
    prof = jnp.zeros((N_EP, len(P.ROW_NAMES)))
    lam_log = {s: [] for s in sigmas}
    for _ in range(steps):
        for s in sigmas:
            key, k = jax.random.split(key)
            Cw, eps = cloud(states, fluids, Us, th_b, bel, s, P.GOAL, k)
            Cw = np.asarray(Cw)
            lams = np.array([lam_for_ess(Cw[e], target) for e in range(N_EP)])
            lam_log[s].append(lams)
            Us = apply(Us, jnp.asarray(Cw), eps, jnp.asarray(lams))
        states, fluids, rows = world(states, fluids, Us[:, :P.NU], th_true, P.GOAL)
        prof = prof + rows
        Us = jax.vmap(P.shift_nodes)(Us)
    gap = np.asarray(jnp.linalg.norm(states[:, :2] - P.GOAL, axis=-1))
    return gap, np.asarray(prof), lam_log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", type=str, nargs="+",
                    default=["0.6,0.25", "0.6,0.25,0.10", "0.6,0.25,0.10,0.04"])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args(argv)

    w = P.Costs()
    cloud, apply, world = make(w)
    th_true = P.sample_theta(jax.random.PRNGKey(7), N_EP)
    th_b = P.sample_theta(jax.random.PRNGKey(11), M)
    bel = jnp.full((M,), 1.0 / M)

    print(f"{N_EP} episodes x {args.steps} control steps, {K} plans, M={M}, "
          f"lambda solved online per episode and level\n")
    # Selection is on the *weighted total cost*, not on any single row.  The
    # first pass of this script ranked by the final distance alone and picked
    # the schedule that covered the most ground while spilling three times as
    # much liquid as another -- which is the objective's own trade being
    # decided by the diagnostic instead of by the objective.
    wv = np.asarray(P.cost_weights(w))
    hdr = (f"{'ladder':<26}{'ESS':>6}{'total':>9}{'goal':>9}{'spill':>9}"
           f"{'freeb':>9}{'slosh':>9}{'gap m':>8}{'spilled':>9}")
    print(hdr); print("-" * len(hdr))
    best, store = (1e30, None, None), {}
    for ls in args.ladders:
        sig = tuple(float(x) for x in ls.split(","))
        for t in args.targets:
            gap, prof, lam_log = episode(cloud, apply, world, sig, t, th_true,
                                         th_b, bel, args.steps,
                                         jax.random.PRNGKey(0))
            sched = tuple(float(np.exp(np.mean(np.log(np.concatenate(lam_log[s])))))
                          for s in sig)
            store[(ls, t)] = sched
            wr = prof.mean(0) * wv
            tot = float(wr.sum())
            print(f"{ls:<26}{t:>6.2f}{tot:>9.2f}{wr[0]:>9.2f}{wr[1]:>9.2f}"
                  f"{wr[2]:>9.2f}{wr[3]:>9.2f}{gap.mean():>8.3f}"
                  f"{prof[:, 1].mean():>9.4f}", flush=True)
            # printed per row, not only in the summary: the previous run died
            # part way through and took the solved temperatures with it
            print(f"      lambda = {tuple(round(x, 5) for x in sched)}",
                  flush=True)
            if np.isfinite(tot) and tot < best[0]:
                best = (tot, ls, t)

    print(f"\nbest by weighted total cost: ladder {best[1]} at ESS "
          f"{100*best[2]:.0f}% -> {best[0]:.2f}")
    print(f"  SIGMA_SCHEDULE = {tuple(float(x) for x in best[1].split(','))}")
    print(f"  LAM_SCHEDULE   = "
          f"{tuple(round(x, 5) for x in store[(best[1], best[2])])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
