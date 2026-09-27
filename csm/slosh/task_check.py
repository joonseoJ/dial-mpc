"""Is the task the one the design intended?

Three questions, and a "no" to any of them means the plant is wrong rather than
the method:

  1. does naive haste spill?      A bang-bang carry at the acceleration limit
                                  must spill a full tray.  If nothing spills,
                                  the spill row is decoration and this task has
                                  the same defect the rigid-payload one did --
                                  where four of seven rows were identically
                                  zero for every hypothesis.
  2. is the threshold theta's?    It must spill the *full* tray and not the
                                  shallow one, at the same command.  A limit
                                  that fires for everybody is a constant and
                                  cannot make hypotheses disagree.
  3. can it be done at all?       A gentler carry must reach the goal without
                                  spilling.  A task nobody can do makes
                                  hypotheses disagree beautifully and is
                                  worthless.

Then the throughput, because the whole study is priced in rollouts.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P


def open_loop(u_seq, theta, s0=None):
    """Run a fixed command sequence, returning the per-step trace."""
    s = P.stage_init() if s0 is None else s0
    U = P.fluid_init(theta)

    def body(carry, u):
        s, U = carry
        ns, a, al = P.stage_step(s, u)
        nU, lost = P.fluid_step(U, s, a, al, theta)
        h = nU[..., 0]
        wall = jnp.concatenate([h[0, :], h[-1, :], h[:, 0], h[:, -1]])
        return (ns, nU), jnp.array([ns[0], jnp.max(wall),
                                    P.wave_energy(nU, theta),
                                    lost / (P.LX * P.LY * theta[0])])

    (s, U), tr = jax.lax.scan(body, (s, U), u_seq)
    return s, U, tr


def bang_bang(frac):
    """Accelerate then brake at `frac` of the limit, straight along x."""
    u = jnp.zeros((P.H, P.NU))
    half = P.H // 2
    u = u.at[:half, 0].set(frac).at[half:, 0].set(-frac)
    return u


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.6, 0.35, 0.2])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.020, 0.035, 0.050])
    args = ap.parse_args(argv)

    run = jax.jit(open_loop)
    print(f"tray {P.LX}x{P.LY} m, rim {P.H_WALL} m, grid {P.NX}x{P.NY}, "
          f"horizon {P.H} steps ({P.H*P.DT:.2f} s), a_max {P.A_MAX} m/s^2")
    print(f"goal {float(P.GOAL[0]):.2f} m\n")

    print("=== 1-2. does naive haste spill, and only for the full tray? ===")
    print("  (bang-bang along x at a fraction of the acceleration limit, "
          "liquid initially still)")
    hdr = (f"{'accel':>7}{'a m/s2':>8}" +
           "".join(f"{'h=' + format(d, '.3f'):>22}" for d in args.depths))
    print(hdr)
    print(f"{'':>15}" + "".join(f"{'peak/rim':>11}{'spilled':>11}"
                                for _ in args.depths))
    print("-" * len(hdr))
    for f in args.fracs:
        u = bang_bang(f)
        row = f"{f:>7.2f}{f*P.A_MAX:>8.2f}"
        for d in args.depths:
            th = jnp.array([d, 1000.0, 0.02, 0.0, 0.0, 0.0, 0.0])
            s, U, tr = jax.block_until_ready(run(u, th))
            peak = float(jnp.max(tr[:, 1])) / P.H_WALL
            row += f"{peak:>11.2f}{float(jnp.sum(tr[:, 3])):>11.3f}"
        print(row)
    print(f"\n  peak/rim > 1.00 means the liquid went over the wall.")

    # --- 3. is there a command that does the job? ---------------------------
    print("\n=== 3. can a gentler carry reach the goal without spilling? ===")
    print(f"{'accel':>7}{'depth':>7}{'reach m':>9}{'gap m':>8}{'peak/rim':>10}"
          f"{'spilled':>10}{'residual slosh':>16}")
    print("-" * 68)
    for f in (0.5, 0.35, 0.25):
        for d in (0.020, 0.050):
            th = jnp.array([d, 1000.0, 0.02, 0.0, 0.0, 0.0, 0.0])
            # accelerate, coast, brake -- three phases instead of two
            u = jnp.zeros((P.H, P.NU))
            q = P.H // 3
            u = u.at[:q, 0].set(f).at[2 * q:, 0].set(-f)
            s, U, tr = jax.block_until_ready(run(u, th))
            print(f"{f:>7.2f}{d:>7.3f}{float(s[0]):>9.3f}"
                  f"{float(abs(s[0]-P.GOAL[0])):>8.3f}"
                  f"{float(jnp.max(tr[:,1]))/P.H_WALL:>10.2f}"
                  f"{float(jnp.sum(tr[:,3])):>10.3f}"
                  f"{float(P.wave_energy(U, th)):>16.3f}")

    # --- throughput ---------------------------------------------------------
    print("\n=== throughput ===")
    cost_mat, _ = P.make_cost_matrix()
    s0 = P.stage_init()
    for K, M in ((256, 8), (1024, 8), (1024, 32)):
        V = jnp.clip(0.5 * jax.random.normal(jax.random.PRNGKey(0),
                                             (K, P.DIM_U)), -1, 1)
        th = P.sample_theta(jax.random.PRNGKey(1), M)
        f = lambda: cost_mat(s0, V, th, P.GOAL)
        jax.block_until_ready(f())
        t0 = time.time()
        for _ in range(3):
            jax.block_until_ready(f())
        dt = (time.time() - t0) / 3
        cells = K * M * P.H * P.SUBSTEPS * P.NX * P.NY
        print(f"  {K:>5} plans x M={M:<3} {dt:7.3f} s/level   "
              f"{cells/dt/1e9:6.2f}G cell-updates/s")
    print(f"\n  sequential depth per rollout: {P.H} steps x {P.SUBSTEPS} "
          f"substeps x 2 RK stages = {P.H*P.SUBSTEPS*2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
