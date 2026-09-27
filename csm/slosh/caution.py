"""Is being too careful now as expensive as being too bold?

Planner told a fill depth `a`, tray actually at depth `b`, for every pair on a
grid.  The diagonal is the planner that knows; the two triangles are the two
kinds of wrong:

  cautious  a > b  plans for a fuller tray than there is -- slow, never spills
  bold      a < b  plans for a shallower one -- fast, spills

On the verified harness, before the time row, the prior mean (bold for full
trays) cost +8.8 over the oracle while the fullest-tray hypothesis cost only
+2.2, so the best hedge over any belief was "assume the worst" and one particle
reproduced it.  The time row prices lateness; this sweeps its weight and reads
both triangles, looking for the point where the two errors cost the same while
the planner that knows still does the task cleanly.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P, closed_loop as CL

DEPTHS = (0.020, 0.030, 0.040, 0.050)


def matrix(w, seeds=2, steps=36):
    """Mismatched and matched costs on the SAME episode layout, cell by cell.

    MPPI's noise is drawn per episode index, so two cells of one matrix see
    different noise.  On the held-tray carrier a couple of steps of arrival
    time moves the cost by 24-48, which made a single diagonal cell a
    reference with +-20-30 of noise: one run put the matched full tray at 368.6
    where a separate run of the identical condition gave 340.3, and every cell
    in that column then looked better than knowing the truth.  So the reference
    for cell (a, b) is the planner told the truth, run at the same episode index
    with the same key -- common random numbers, as the gate pairs its
    conditions.
    """
    n = len(DEPTHS)
    th = jnp.asarray([[d, 1000.0, 0.02, 0.15, 1.0, 0.05, 2.0] for d in DEPTHS])
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a, b = a.ravel(), b.ravel()
    kw = dict(w=w, seeds=seeds, steps=steps, sigma=P.SIGMA_SCHEDULE,
              ess_target=P.ESS_TARGET, elite=P.ELITE)
    true = th[jnp.asarray(b)]
    mis = CL.run("belief", true, th[jnp.asarray(a)][:, None], jnp.ones((1,)), **kw)
    ref = CL.run("belief", true, true[:, None], jnp.ones((1,)), **kw)
    T = mis.total.reshape(seeds, n, n)                       # [seed, assumed, true]
    R = ref.total.reshape(seeds, n, n)
    raw = ref.raw.reshape(seeds, n, n, -1)
    return T - R, raw


def summarise(ex, raw):
    n = ex.shape[1]
    cau = np.stack([ex[:, i, j] for i in range(n) for j in range(n) if i > j], 1)
    bol = np.stack([ex[:, i, j] for i in range(n) for j in range(n) if i < j], 1)
    diag_spill = np.mean([raw[:, i, i, 1] for i in range(n)])
    it = P.ROW_NAMES.index("time")          # by name: rows get appended
    diag_time = np.mean([raw[:, i, i, it] for i in range(n)]) * P.DT
    return (cau.mean(), cau.std() / np.sqrt(cau.size), bol.mean(),
            bol.std() / np.sqrt(bol.size), diag_spill, diag_time)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0, 3.0, 6.0, 10.0])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)
    print(f"assumed x true depth {DEPTHS}, {args.seeds} seeds, deadline {P.T_ARRIVE} s\n")
    print(f"{'w_time':>7}{'too cautious':>16}{'too bold':>16}{'bold/cautious':>15}"
          f"{'matched spill':>15}{'matched arrival s':>19}")
    print("-" * 88)
    for wt in args.weights:
        ex, raw = matrix(P.Costs(time=wt), args.seeds)
        c, cs, b, bs, sp, ta = summarise(ex, raw)
        print(f"{wt:>7.1f}{c:>10.2f}+-{cs:<4.2f}{b:>10.2f}+-{bs:<4.2f}"
              f"{b/max(c,1e-9):>15.2f}{sp:>15.4f}{ta:>19.2f}", flush=True)
    print("\ncautious and bold are the mean excess cost over the planner told the")
    print("truth, paired cell by cell on the same episode index and key;")
    print("'matched arrival' is the time the planner that knows spends outside the")
    print("arrival radius -- the task should still be done, and done cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
