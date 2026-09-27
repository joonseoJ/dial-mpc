"""Is E0 worth running?  Decided before looking, then measured.

E0 sweeps the number of belief particles M.  It is only informative if a whole
belief can do something its mean cannot, so three things are checked first,
each as a paired difference on the same true parameters and seeds:

  (i)   the oracle is the floor.  A planner told the truth must beat every
        planner that is not; if it does not, the harness is wrong and nothing
        else in this table means anything.  (It failed this before the
        liquid-reset and terminal-slosh fixes.)
  (ii)  the truth is worth knowing: oracle clearly below mean-theta.  If not,
        uncertainty does not matter on this task and no M can help.
  (iii) the spread is worth carrying: belief M=8 clearly below the BEST single
        hypothesis -- the better of mean-theta and worst-theta.  Comparing only
        against the mean was a strawman the first time this gate ran: errors
        were asymmetric, the mean was the bold choice, and one particle at the
        fullest tray matched eight.  If any single hypothesis does as well as
        eight, the M curve is flat past M=1 and E0 would only restate that.

Run E0 only if all three hold, "clearly" meaning more than two standard errors.

Two consistency checks ride along:
  oracle_full   the planner on the WORLD grid: the price of planning on PLAN.
  belief(true)  one particle *at the true theta*, its liquid propagated by the
                planner's own model instead of read from the world.  It should
                sit close to the oracle; a large gap means open-loop
                propagation drifts, and every belief condition inherits that.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P, closed_loop as CL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=36)
    args = ap.parse_args(argv)
    E = args.episodes
    th = P.sample_theta(jax.random.PRNGKey(7), E)
    prior = P.sample_theta(jax.random.PRNGKey(11), 8)
    # "worst" by depth is the most spill-prone; on the held-tray carrier the
    # most dangerous is the heaviest load, the largest tipping stiffness
    # rho A g L^2/12 + m g h/2.  Both are single-hypothesis baselines.
    worst = prior[jnp.argmax(prior[:, 0])]
    k_liq = prior[:, 1] * P.LX * P.LY * 9.81 * (P.LX ** 2 / 12 + prior[:, 0] ** 2 / 2)
    heavy = prior[jnp.argmax(k_liq)]
    mean = prior.mean(0)
    kw = dict(seeds=args.seeds, steps=args.steps, sigma=P.SIGMA_SCHEDULE,
              ess_target=P.ESS_TARGET, elite=P.ELITE)
    one = jnp.ones((1,))
    conds = [
        ("oracle", "oracle", None, None),
        ("belief(true theta)", "belief", th[:, None], one),
        ("mean theta", "belief", jnp.tile(mean[None, None], (E, 1, 1)), one),
        ("worst theta", "belief", jnp.tile(worst[None, None], (E, 1, 1)), one),
        ("heaviest theta", "belief", jnp.tile(heavy[None, None], (E, 1, 1)), one),
        ("belief M=8", "belief", jnp.tile(prior[None], (E, 1, 1)), jnp.full((8,), 1 / 8)),
        ("oracle_full (WORLD planner)", "oracle_full", None, None),
    ]
    print(f"{E} true thetas x {args.seeds} seeds, {args.steps} steps "
          f"({args.steps*P.DT:.1f} s); planner {P.PLAN}, world {P.WORLD}; "
          f"sigma {P.SIGMA_SCHEDULE}, ESS {P.ESS_TARGET}, elite {P.ELITE}\n")
    print(f"{'condition':<30}{'cost':>8}{'vs oracle':>16}{'spilled':>9}{'gap m':>8}"
          f"{'improve':>9}")
    print("-" * 80)
    R = {}
    for nm, mode, tb, bel in conds:
        r = CL.run(mode, th, tb, bel, **kw)
        R[nm] = r
        d = "(reference)" if nm == "oracle" else "%+.2f+-%.2f" % CL.paired(r, R["oracle"])
        print(f"{nm:<30}{r.total.mean():>8.2f}{d:>16}{r.spilled.mean():>9.4f}"
              f"{r.gap.mean():>8.3f}{r.improve:>9.2f}", flush=True)

    print("\n=== decision ===")
    def test(a, b, label):
        m, se = CL.paired(R[a], R[b])
        ok = m < -2 * se
        print(f"  {label:<46} {a} - {b} = {m:+.2f} +- {se:.2f}  -> {'YES' if ok else 'no'}")
        return ok
    others = [k for k in R if k not in ("oracle", "oracle_full (WORLD planner)")]
    floor = all(CL.paired(R["oracle"], R[k])[0] < 2 * CL.paired(R["oracle"], R[k])[1]
                for k in others)
    print(f"  (i)   oracle is the floor (not beaten beyond 2 SE by any condition): "
          f"{'YES' if floor else 'NO -- harness is suspect'}")
    info = test("oracle", "mean theta", "(ii)  knowing the truth is worth something")
    best1 = min(("mean theta", "worst theta", "heaviest theta"),
                key=lambda k: R[k].total.mean())
    spread = test("belief M=8", best1, "(iii) the belief beats the best single guess")
    m, se = CL.paired(R["oracle"], R["oracle_full (WORLD planner)"])
    print(f"  price of the PLAN grid (oracle - oracle_full): {m:+.2f} +- {se:.2f}")
    m, se = CL.paired(R["belief(true theta)"], R["oracle"])
    print(f"  open-loop propagation drift (belief(true) - oracle): {m:+.2f} +- {se:.2f}")
    go = floor and info and spread
    print(f"\n  RUN E0: {'YES' if go else 'NO'}")
    return 0 if go else 3


if __name__ == "__main__":
    raise SystemExit(main())
