"""Does the shallow-water solver do what it claims?

Five checks, in the order that a failure in one would invalidate the next:

  1. mass conservation     reflective walls must not leak.  This catches
                           boundary-condition sign errors, which otherwise show
                           up much later as a slow drift nobody attributes to
                           the walls.
  2. mode frequency        a small standing wave must oscillate at
                           `(n pi / L) sqrt(g h)`, the analytic frequency of
                           *these equations*.  This is the check that the flux,
                           the reconstruction and the time stepping are wired
                           together correctly.
  3. numerical damping     with no physical damping the wave must barely decay.
                           This is the check that actually decides the scheme:
                           the fluid's damping ratio is one of the unknown
                           parameters and runs 0.005-0.05, so numerical
                           dissipation has to sit well under 0.005 or the belief
                           is over a quantity the solver invents.
  4. static tilt           held at a tilt `alpha`, the surface must settle to a
                           slope of `tan(alpha)` in the tray frame.  This is the
                           check on the source term, which is the whole
                           mechanism by which tilt controls the slosh.
  5. shallow-water error   the gap between these equations and potential flow,
                           `omega^2 = g k tanh(k h)`.  Not a bug -- the modelling
                           error -- but it has to be reported, because it is what
                           bounds any claim made with this plant.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import swe

G0 = 9.81


def free_run(U0, dx, g_n, g_x, dt, n):
    """`n` steps at fixed `dt`, returning every state."""
    def body(U, _):
        Un = swe.step_1d(U, dt, dx, g_n, g_x)
        return Un, Un
    _, traj = jax.lax.scan(body, U0, None, length=n)
    return traj


def mode_init(N, L, h0, amp, mode=1):
    """A small standing wave: `h = h0 + amp cos(n pi x / L)`, fluid at rest."""
    x = (jnp.arange(N) + 0.5) * (L / N)
    h = h0 + amp * jnp.cos(mode * jnp.pi * x / L)
    return jnp.stack([h, jnp.zeros_like(h)], axis=-1)


def measure_freq_and_decay(traj, dt, L, N):
    """Frequency and damping ratio from the first-mode amplitude history.

    Projecting onto the mode shape rather than tracking a peak: the signal is
    a sum of modes and a peak tracker would report whichever happened to be
    largest.
    """
    x = (np.arange(N) + 0.5) * (L / N)
    shape = np.cos(np.pi * x / L)
    shape = shape / (shape @ shape)
    a = np.asarray(traj[:, :, 0]) @ shape          # modal amplitude per step

    # frequency from zero crossings of the (mean-removed) signal
    s = a - a.mean()
    sign = np.sign(s)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) < 3:
        return float("nan"), float("nan")
    # linear interpolation of each crossing time, then the mean half-period
    t = []
    for i in cross:
        frac = s[i] / (s[i] - s[i + 1])
        t.append((i + frac) * dt)
    t = np.array(t)
    half = np.diff(t).mean()
    omega = np.pi / half

    # damping from the decay of successive extrema magnitudes
    env_i = ((cross[:-1] + cross[1:]) // 2)
    env = np.abs(s[env_i])
    good = env > env.max() * 1e-3
    if good.sum() < 3:
        return omega, float("nan")
    tt = t[:-1][good]
    lg = np.log(env[good])
    slope = np.polyfit(tt, lg, 1)[0]               # env ~ exp(slope t)
    zeta = -slope / omega
    return omega, zeta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--length", type=float, default=0.20, help="tray length, m")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.02, 0.05, 0.08])
    ap.add_argument("--amp-frac", type=float, default=0.10,
                    help="initial wave amplitude as a fraction of depth")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--periods", type=float, default=8.0)
    args = ap.parse_args(argv)
    L = args.length

    print("=== 1-2-3. mass, frequency, numerical damping (no physical damping) ===")
    print(f"tray length {L} m, initial amplitude {100*args.amp_frac:.0f}% of depth, "
          f"CFL {args.cfl}\n")
    hdr = (f"{'cells':>6}{'depth':>7}{'dt ms':>8}{'omega':>9}{'analytic':>10}"
           f"{'err %':>8}{'zeta_num':>10}{'mass drift':>12}")
    print(hdr); print("-" * len(hdr))
    worst_zeta = 0.0
    for N in args.cells:
        dx = L / N
        for h0 in args.depths:
            c = np.sqrt(G0 * h0)
            dt = args.cfl * dx / c
            w_an = float(swe.linear_modes(1, L, h0))
            n = int(args.periods * 2 * np.pi / w_an / dt)
            U0 = mode_init(N, L, h0, args.amp_frac * h0)
            traj = jax.block_until_ready(free_run(U0, dx, G0, 0.0, dt, n))
            w, zeta = measure_freq_and_decay(traj, dt, L, N)
            m0 = float(U0[:, 0].sum())
            drift = float(np.asarray(traj[-1, :, 0]).sum() / m0 - 1.0)
            worst_zeta = max(worst_zeta, 0.0 if np.isnan(zeta) else zeta)
            print(f"{N:>6}{h0:>7.3f}{1000*dt:>8.2f}{w:>9.3f}{w_an:>10.3f}"
                  f"{100*(w/w_an-1):>8.2f}{zeta:>10.5f}{drift:>12.2e}")

    print(f"\n  physical damping ratio range in theta: 0.005 to 0.05")
    print(f"  worst numerical damping measured:      {worst_zeta:.5f}")
    if worst_zeta > 0.005:
        print("  -> TOO DISSIPATIVE.  The scheme invents more damping than the "
              "smallest value\n     the belief is supposed to range over; refine "
              "the grid or the scheme.")
    else:
        print(f"  -> acceptable: {0.005/max(worst_zeta,1e-9):.1f}x below the "
              f"smallest physical value.")

    # --- 4. static tilt ------------------------------------------------------
    print("\n=== 4. static tilt: the surface must settle to slope tan(alpha) ===")
    N, h0 = 64, 0.05
    dx = L / N
    print(f"{'alpha deg':>10}{'measured slope':>16}{'tan(alpha)':>12}{'err %':>8}")
    for deg in (2.0, 5.0, 10.0):
        al = np.radians(deg)
        g_n, g_x = G0 * np.cos(al), G0 * np.sin(al)
        dt = args.cfl * dx / np.sqrt(G0 * h0)
        # strong damping so it settles rather than ringing: a linear drag on
        # momentum, applied outside the solver, purely to reach equilibrium
        U = jnp.stack([jnp.full((N,), h0), jnp.zeros((N,))], axis=-1)

        def body(U, _):
            U = swe.step_1d(U, dt, dx, g_n, g_x)
            return U.at[:, 1].multiply(0.98), None
        U, _ = jax.lax.scan(body, U, None, length=20000)
        x = (np.arange(N) + 0.5) * dx
        slope = np.polyfit(x, np.asarray(U[:, 0]), 1)[0]
        print(f"{deg:>10.1f}{slope:>16.5f}{np.tan(al):>12.5f}"
              f"{100*(slope/np.tan(al)-1):>8.2f}")

    # --- 5. the modelling error ---------------------------------------------
    print("\n=== 5. shallow water vs potential flow (the modelling error) ===")
    print(f"{'depth':>7}{'h/L':>7}{'SWE omega':>11}{'potential':>11}{'err %':>8}")
    k = np.pi / L
    for h0 in args.depths:
        w_swe = k * np.sqrt(G0 * h0)
        w_pot = np.sqrt(G0 * k * np.tanh(k * h0))
        print(f"{h0:>7.3f}{h0/L:>7.2f}{w_swe:>11.3f}{w_pot:>11.3f}"
              f"{100*(w_swe/w_pot-1):>8.2f}")
    print("\n  shallower is more accurate, which is the same direction the task "
          "wants for a\n  wide frequency spread -- the fill level moves omega "
          "as sqrt(h).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
