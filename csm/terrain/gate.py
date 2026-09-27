"""E0 decision gate for the stones task: does the number of terrain particles
matter, and does it keep mattering past eight?

Every condition is the same robust DIAL (`robust.py`) with a different set of
terrains in its head; the world always steps under the true one.  Episodes
share reset keys and planner keys across conditions (common random numbers),
so every comparison is paired episode by episode.

  oracle        the true terrain (M = 1)                 -- the floor
  optimistic    every unseen stone solid (M = 1)
  conservative  every unseen stone a hole (M = 1)
  likely        each unseen stone its more likely state (M = 1)
  two-branch    {optimistic, conservative}, weighted by the mean unseen hole
                probability -- the contingency-MPC baseline: right marginal on
                average, no independence between stones
  M = 8/32/64   independent draws from the per-stone belief, equal weights

Pre-registered criteria (the same shape as the tray gate, plus the question
the tray gate taught us to ask):
  (i)   the oracle is the floor
  (ii)  knowing the terrain is worth something: oracle beats the best single
        stand-in beyond 2 SE
  (iii) particles beat the best of {single stand-ins, two-branch} beyond 2 SE
  (iv)  and keep improving: M = 64 beats M = 8 beyond 2 SE
E0 is worth running only if (i)-(iv) all hold.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from csm.terrain import stones as S
from csm.terrain import robust as R
from csm.terrain.check import make_env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--nsample", type=int, default=1024)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--ms", type=int, nargs="+", default=[8, 32, 64])
    ap.add_argument("--p-seen", type=float, default=0.4)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--lane-width", type=float, default=0.30)
    ap.add_argument("--only", nargs="*", default=None, help="run only these conditions")
    a = ap.parse_args(argv)
    S.configure(a.lanes, a.lane_width)
    lay = S.Layout(p_seen=a.p_seen)
    env = make_env(a.vx)
    ctl = R.Robust(env, R.Plan(Nsample=a.nsample))
    E = a.episodes
    ks = jax.random.split(jax.random.PRNGKey(123), E)
    tt, bel = jax.vmap(lambda k: S.sample_episode(k, lay))(ks)
    run_keys = jax.random.split(jax.random.PRNGKey(456), E)
    part_keys = jax.random.split(jax.random.PRNGKey(789), E)
    unseen = (bel > 0) & (bel < 1)
    pbar = np.asarray(jnp.sum(jnp.where(unseen, bel, 0), -1) / jnp.maximum(unseen.sum(-1), 1))
    print(f"{a.lanes} lanes x {a.lane_width} m. {E} strips: holes {np.asarray((tt < 0).sum(-1)).mean():.1f}/{S.N_CELL}, unseen "
          f"{np.asarray(unseen.sum(-1)).mean():.1f}, mean unseen hole prob {pbar.mean():.2f}; "
          f"Nsample {a.nsample}, {a.steps} steps ({a.steps * 0.02:.1f} s), vx {a.vx}\n", flush=True)

    single = lambda kind: (jax.vmap(lambda b: S.representative(b, kind))(bel)[:, None],
                           jnp.ones((E, 1)))
    two = (jnp.stack([single("optimistic")[0][:, 0], single("conservative")[0][:, 0]], 1),
           jnp.stack([1 - pbar, pbar], 1).astype(jnp.float32))
    conds = [("oracle", (tt[:, None], jnp.ones((E, 1)))),
             ("optimistic", single("optimistic")),
             ("conservative", single("conservative")),
             ("likely", single("likely")),
             ("two-branch", two)]
    for m in a.ms:
        parts = jax.vmap(lambda k, b: S.sample_particles(k, b, m))(part_keys, bel)
        conds.append((f"M={m}", (parts, jnp.full((E, m), 1.0 / m))))

    runners = {}
    res = {}
    print(f"{'condition':<14}{'cost':>9}{'vs oracle':>18}{'fell':>7}{'x_end':>7}{'min':>6}{'s':>7}")
    print("-" * 68)
    for nm, (parts, pw) in conds:
        if a.only and nm not in a.only and nm != "oracle":
            continue
        m = parts.shape[1]
        if m not in runners:
            runners[m] = jax.jit(jax.vmap(
                lambda t_, p_, w_, k_: R.episode(ctl, t_, p_, w_, k_, a.steps)))
        t0 = time.time()
        cs, fs, xs = [], [], []
        b = a.batch if m <= 8 else max(1, a.batch * 8 // m)
        while not R.validate_shape(ctl, bel, m, b):
            print(f"  ({nm}: batch {b} x M {m} miscompiles; trying {b // 2 if b > 1 else 'none'})")
            if b == 1:
                raise SystemExit(f"{nm}: no batch size gives a correct program")
            b //= 2
        for i in range(0, E, b):
            sl = slice(i, i + b)
            tr = runners[m](tt[sl], parts[sl], pw[sl], run_keys[sl])
            cs.append(-np.asarray(tr.reward.sum(-1))); fs.append(np.asarray(jnp.any(tr.done > 0, -1)))
            xs.append(np.asarray(tr.x[:, -1]))
        c, f, x = np.concatenate(cs), np.concatenate(fs), np.concatenate(xs)
        res[nm] = (c, f, x)
        if nm == "oracle":
            d = "(reference)"
        else:
            dd = c - res["oracle"][0]
            d = f"{dd.mean():+.2f}+-{dd.std() / np.sqrt(E):.2f}"
        print(f"{nm:<14}{c.mean():>9.2f}{d:>18}{int(f.sum()):>4}/{E:<2}{x.mean():>7.2f}"
              f"{x.min():>6.2f}{time.time() - t0:>7.0f}", flush=True)
    np.savez(f"csm_runs/terrain_gate_{a.lanes}lanes_seen{a.p_seen:.2f}_{int(time.time())}.npz",
             **{k: np.stack(v) for k, v in res.items()})

    print("\n=== decision ===")
    def pd(a_, b_):
        d = res[a_][0] - res[b_][0]
        return d.mean(), d.std() / np.sqrt(d.size)
    need = {"oracle", "optimistic", "conservative", "likely", "two-branch"} | {f"M={m}" for m in a.ms}
    if not need <= set(res):
        return 0
    others = [k for k in res if k != "oracle"]
    floor = all(pd("oracle", k)[0] < 2 * pd("oracle", k)[1] for k in others)
    print(f"  (i)   oracle is the floor: {'YES' if floor else 'NO -- harness suspect'}")
    singles = ("optimistic", "conservative", "likely")
    best1 = min(singles, key=lambda k: res[k][0].mean())
    m_, s_ = pd("oracle", best1)
    ok2 = m_ < -2 * s_
    print(f"  (ii)  oracle - best single ({best1}) = {m_:+.2f} +- {s_:.2f} -> {'YES' if ok2 else 'no'}")
    baseline = min(singles + ("two-branch",), key=lambda k: res[k][0].mean())
    m8 = f"M={a.ms[0]}"
    bestM = min((f"M={m}" for m in a.ms), key=lambda k: res[k][0].mean())
    m_, s_ = pd(bestM, baseline)
    ok3 = m_ < -2 * s_
    print(f"  (iii) {bestM} - best baseline ({baseline}) = {m_:+.2f} +- {s_:.2f} -> {'YES' if ok3 else 'no'}")
    mL = f"M={a.ms[-1]}"
    m_, s_ = pd(mL, m8)
    ok4 = m_ < -2 * s_
    print(f"  (iv)  {mL} - {m8} = {m_:+.2f} +- {s_:.2f} -> {'YES' if ok4 else 'no'}")
    print(f"\n  RUN E0: {'YES' if floor and ok2 and ok3 and ok4 else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
