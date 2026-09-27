"""Closed-loop MPPI on the tray, for every planner this study compares.

One harness, so that the conditions differ only in what they are told.  Each
earlier comparison was its own script, and the liquid-reset bug lived in all of
them because each copied the same wrong call; a condition that is a flag on one
code path cannot drift from the others.

  world    the validated WORLD grid.  Every reported number comes from it.
  planner  the PLAN grid, starting every rollout from a liquid state -- never
           from `fluid_init`, which is only the t=0 state.
             oracle        true theta; the world's liquid, restricted to PLAN.
             oracle_full   the same on the WORLD grid, to price the coarsening.
             belief        M particles (theta_m, liquid_m); each particle's
                           liquid is advanced on PLAN through the command that
                           was actually executed.  That is the predictive
                           belief with no observations: sample the prior,
                           simulate forward under the known command history.
  scoring  identical rows for planner and world: slosh every step, the
           deadline inside the episode.  The previous evaluation ended at 1.2 s,
           before a 1.25 s deadline, and so charged `late` exactly zero.

Two checks run inside every episode, because the last failure was a planner
that was wrong in a way no final cost could reveal:
  improve  after each MPPI update, is the new plan cheaper *under the planner's
           own model* than the plan it started from?  A sampler that does not
           descend its own objective explains nothing downstream.
  ess      effective sample size per annealing level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P


@dataclass
class Result:
    total: np.ndarray        # (seeds, episodes) weighted episode cost
    rows: np.ndarray         # (len(ROW_NAMES),) weighted, mean over all
    spilled: np.ndarray      # (seeds, episodes) fraction of liquid lost
    gap: np.ndarray          # (seeds, episodes) final distance to goal
    spill_t: np.ndarray      # (steps,) spill per step, mean
    ess: np.ndarray          # (levels,) mean ESS fraction
    improve: float           # fraction of updates that lowered the planner's cost
    worsen: float            # mean relative cost increase of an update (0 if none)
    lams: list               # per level, the temperatures used (for calibration)
    raw: np.ndarray = None   # (seeds, episodes, rows) unweighted row sums


def make(w: P.Costs, grid: P.Grid, E: int, K: int, elite: bool = False):
    _, cost = P.make_rollout_state(w, grid)

    def planner_cost(s, fl, v, tb, bel):
        """Belief-weighted cost of one plan: fl (M,..), tb (M,7)."""
        return jax.vmap(lambda f, t: cost(s, f, v, t, P.GOAL))(fl, tb) @ bel

    @jax.jit
    def cloud(states, pf, Us, th_b, bel, sg, key):
        eps = sg * jax.random.normal(key, (E, K, P.DIM_U))
        eps = eps.at[:, -1, :].set(0.0)                  # the current plan itself
        V = jnp.clip(Us[:, None, :] + eps, -1.0, 1.0)
        C = jax.vmap(lambda s, fl, v, tb: jax.vmap(
            lambda vv: planner_cost(s, fl, vv, tb, bel))(v))(states, pf, V, th_b)
        return C, eps

    @jax.jit
    def apply(states, pf, Us, th_b, bel, C, eps, lam):
        om = jax.nn.softmax(-(C - C.min(1, keepdims=True)) / lam[:, None], axis=1)
        Un = jnp.clip(Us + jnp.einsum("ek,ekd->ed", om, eps), -1.0, 1.0)
        Cn = jax.vmap(lambda s, fl, u, tb: planner_cost(s, fl, u, tb, bel))(
            states, pf, Un, th_b)
        ess = 1.0 / jnp.sum(om ** 2, 1) / K
        cur = C[:, -1]
        ok = Cn <= cur + 1e-6 * jnp.abs(cur)
        worse = jnp.maximum(Cn - cur, 0.0)
        if elite:
            # Keep the weighted average only if it beats every sample it was
            # built from; otherwise keep the best sample.  The current plan is
            # one of the samples (eps = 0), so the chosen plan can never be
            # worse than the plan the update started from.
            best = jnp.argmin(C, axis=1)
            Vb = jnp.clip(Us + eps[jnp.arange(E), best], -1.0, 1.0)
            take = Cn > C[jnp.arange(E), best]
            Un = jnp.where(take[:, None], Vb, Un)
        return Un, ess, ok, worse / (jnp.abs(cur) + 1e-9)

    return cloud, apply


@jax.jit
def world_step(states, fluids, us, th_true):
    def one(s, U, u, t):
        nU, ns, lost = P.fluid_step(U, s, u, t, P.WORLD)
        return ns, nU, P.stage_cost(nU, ns, u, t, P.GOAL, lost, 0.0, P.WORLD)
    return jax.vmap(one)(states, fluids, us, th_true)


def _propagate(grid):
    """Each particle's liquid, advanced through the command actually executed,
    from the carrier state the world was actually in -- the carrier is measured,
    only the liquid is hypothesised."""
    @jax.jit
    def prop(pf, prev, us, th_b):
        return jax.vmap(lambda fl, s, u, tb: jax.vmap(
            lambda f, t: P.fluid_step(f, s, u, t, grid)[0])(fl, tb))(pf, prev, us, th_b)
    return prop


_restrict = jax.jit(jax.vmap(lambda u: P.restrict(u, P.WORLD, P.PLAN)))


def lam_for_ess(c, target, lo=1e-6, hi=1e7, iters=60):
    c = np.asarray(c, dtype=np.float64); c = c - c.min()
    def ess(lam):
        wgt = np.exp(-c / lam); wgt /= wgt.sum()
        return 1.0 / np.sum(wgt ** 2) / wgt.size
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if ess(mid) < target else (lo, mid)
    return float(np.sqrt(lo * hi))


def run(mode, th_true, th_b=None, bel=None, *, w=None, seeds=2, steps=36, K=512,
        sigma=None, lam=None, ess_target=None, elite=False) -> Result:
    """Closed-loop episodes, all `E = len(th_true)` in lockstep.

    `lam` fixes the temperatures; `ess_target` instead bisects them online at
    every step and level, which is how the schedule is calibrated.
    """
    w = P.Costs() if w is None else w
    sigma = P.SIGMA_SCHEDULE if sigma is None else sigma
    E = th_true.shape[0]
    grid = P.WORLD if mode == "oracle_full" else P.PLAN
    cloud, apply = make(w, grid, E, K, elite)
    prop = _propagate(grid)
    if mode in ("oracle", "oracle_full"):
        th_b, bel = th_true[:, None], jnp.ones((1,))
    wv = np.asarray(P.cost_weights(w))

    tot, sp, gp, rows_all, st_all, ess_all, imp, lams = [], [], [], [], [], [], [], []
    raw_all = []
    worse_all = []
    lam_log = [[] for _ in sigma]
    for sd in range(seeds):
        states = jnp.tile(P.stage_init(), (E, 1))
        fluids = jax.vmap(lambda t: P.fluid_init(t, P.WORLD))(th_true)
        pf = jax.vmap(jax.vmap(lambda t: P.fluid_init(t, grid)))(th_b)
        Us = jnp.zeros((E, P.DIM_U)); prof = jnp.zeros((E, len(P.ROW_NAMES)))
        key = jax.random.PRNGKey(sd); st = []
        for t in range(steps):
            if mode == "oracle":
                pf = _restrict(fluids)[:, None]
            elif mode == "oracle_full":
                pf = fluids[:, None]
            for li, sg in enumerate(sigma):
                key, k = jax.random.split(key)
                C, eps = cloud(states, pf, Us, th_b, bel, sg, k)
                if ess_target is not None:
                    Cn_ = np.asarray(C)
                    lv = np.array([lam_for_ess(Cn_[e], ess_target) for e in range(E)])
                    lam_log[li].append(lv)
                else:
                    lv = np.full(E, lam[li])
                Us, e, ok, wr = apply(states, pf, Us, th_b, bel, C, eps, jnp.asarray(lv))
                ess_all.append((li, float(e.mean()))); imp.append(float(ok.mean()))
                worse_all.append(float(wr.mean()))
            prev = states
            u_now = Us[:, :P.NU]
            states, fluids, rows = world_step(states, fluids, u_now, th_true)
            if mode == "belief":
                pf = prop(pf, prev, u_now, th_b)
            prof = prof + rows; st.append(float(rows[:, 1].mean()))
            Us = jax.vmap(P.shift_nodes)(Us)
        prof = np.array(prof)
        tot.append((prof * wv).sum(-1)); rows_all.append((prof * wv).mean(0))
        raw_all.append(prof)
        sp.append(prof[:, 1]); st_all.append(st)
        gp.append(np.asarray(jnp.linalg.norm(states[:, :2] - P.GOAL, axis=-1)))
    ess = np.array([np.mean([e for l, e in ess_all if l == li]) for li in range(len(sigma))])
    lams = [float(np.exp(np.mean(np.log(np.concatenate(v))))) if v else None for v in lam_log]
    return Result(np.stack(tot), np.mean(rows_all, 0), np.stack(sp), np.stack(gp),
                  np.mean(st_all, 0), ess, float(np.mean(imp)), float(np.mean(worse_all)),
                  lams, np.stack(raw_all))


def paired(a: Result, b: Result):
    """Mean and standard error of a - b over matched episodes and seeds."""
    n = min(a.total.shape[0], b.total.shape[0])
    d = (a.total[:n] - b.total[:n]).ravel()
    return float(d.mean()), float(d.std() / np.sqrt(d.size))
