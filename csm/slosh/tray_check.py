"""Does the held tray behave as its equations say it should?

The tray is a rigid body pivoted at its centre, driven by a torque-limited hand
and loaded by the weight of the liquid it carries.  Checked here, before
anything is planned on it:

  1. torque formula   on a tilted equilibrium surface, the liquid's torque
                      about the pivot equals (rho A g L^2/12 + m g h/2) * theta.
  2. no liquid        with a negligible density the attitude loop is the bare
                      PD on the shell: second order at sqrt(KP/I).
  3. stable at rest   every tray in the prior, the heaviest full one included,
                      recovers from a disturbance with no command -- the hand's
                      stiffness beats the liquid's destabilising one.
  4. holding limit    commanded to the steepest lean, a heavy tray cannot be
                      held and tips over while a light one is held: the
                      constraint belongs to theta, through density as much as
                      through depth.
  5. righting         standing a leaning tray back up releases the liquid piled
                      on the low side; doing it abruptly sloshes more than doing
                      it gradually.
  6. survival         full-amplitude random attitude commands on both grids.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P
from csm.slosh import swe

G0 = 9.81


def fly(u_seq, theta, g=P.WORLD, tilt0=0.0):
    s = P.stage_init().at[5].set(tilt0)
    U = P.fluid_init(theta, g)

    def body(c, u):
        s, U = c
        U, s, lost = P.fluid_step(U, s, u, theta, g)
        h = U[..., 0]
        wall = jnp.max(jnp.concatenate([h[0], h[-1], h[:, 0], h[:, -1]]))
        return (s, U), (s, P.wave_energy(U, theta, g), lost, wall)
    (_, _), tr = jax.lax.scan(body, (s, U), u_seq)
    return tr


def calm(h0, rho):
    return jnp.array([h0, rho, 0.02, 0.0, 0.0, 0.0, 0.0])


def main() -> int:
    # ---- 1. the torque formula ----------------------------------------------
    print("=== 1. liquid torque on a tilted equilibrium surface ===")
    g = P.WORLD
    x = (jnp.arange(g.nx) + 0.5) * g.dx - P.LX / 2
    for h0, rho in ((0.02, 700.0), (0.05, 1000.0), (0.05, 1400.0)):
        for th in (0.05, 0.20):
            h = (h0 + x * jnp.tan(th))[:, None] * jnp.ones((1, g.ny))
            _, tp = P.liquid_torque(h, jnp.array([h0, rho, 0, 0, 0, 0, 0.0]),
                                    G0 * np.cos(th), G0 * np.sin(th), 0.0, g)
            m = rho * P.LX * P.LY * h0
            pred = (rho * P.LX * P.LY * G0 * P.LX ** 2 / 12 * np.tan(th) * np.cos(th)
                    + m * G0 * np.sin(th) * h0 / 2)
            print(f"  h0 {h0:.2f} rho {rho:5.0f} tilt {th:.2f}: torque {float(tp):.4f} "
                  f"predicted {pred:.4f}  err {100*(float(tp)/pred-1):+.2f}%")

    # ---- 2. no liquid ----------------------------------------------------------
    print("\n=== 2. negligible liquid: the bare PD on the shell ===")
    I0 = P.M_SHELL * P.LX ** 2 / 12
    tr = jax.jit(lambda u, t: fly(u, t, tilt0=0.1))(jnp.zeros((20, P.NU)), calm(0.035, 1e-3))
    pitch = np.asarray(tr[0][:, 5])
    wn = np.sqrt(P.KP / I0); zeta = P.KD / (2 * np.sqrt(P.KP * I0))
    print(f"  from 0.1 rad, zero command: pitch after 0.05 s {pitch[0]:+.4f}, "
          f"after 0.25 s {pitch[4]:+.5f}  (omega_n {wn:.0f} rad/s, zeta {zeta:.2f})")

    # ---- 3. stability at rest across the prior -----------------------------------
    print("\n=== 3. at rest, no command, 0.05 rad disturbance: does every tray recover? ===")
    worst = 0.0
    for h0 in (0.02, 0.035, 0.05):
        for rho in (700.0, 1400.0):
            tr = jax.jit(lambda u, t: fly(u, t, tilt0=0.05))(jnp.zeros((30, P.NU)), calm(h0, rho))
            p_end = float(tr[0][-1, 5]); p_max = float(jnp.max(jnp.abs(tr[0][:, 5])))
            worst = max(worst, abs(p_end))
            k_liq = rho * P.LX * P.LY * G0 * P.LX ** 2 / 12 + rho * P.LX * P.LY * h0 * G0 * h0 / 2
            print(f"  h0 {h0:.3f} rho {rho:6.0f}: liquid stiffness {k_liq:.2f} vs KP {P.KP}  "
                  f"|tilt| max {p_max:.3f} -> after 1.5 s {abs(p_end):.4f}")
    print(f"  {'all recover' if worst < 0.01 else 'SOME DO NOT RECOVER'}")

    # ---- 4. holding limit ---------------------------------------------------------
    print("\n=== 4. commanded to the steepest lean (0.35 rad) for 1 s ===")
    print(f"  {'tray':<22}{'peak tilt':>10}{'tipped':>8}{'torque load':>13}{'spilled':>9}")
    for h0, rho in ((0.02, 700.0), (0.035, 1000.0), (0.05, 1000.0), (0.035, 1400.0), (0.05, 1400.0)):
        tr = jax.jit(fly)(jnp.tile(jnp.array([1.0, 0.0]), (20, 1)), calm(h0, rho))
        pk = float(jnp.max(jnp.abs(tr[0][:, 5])))
        load = float(jnp.mean(tr[0][:, 9]))
        sp = float(jnp.sum(tr[2])) / (P.LX * P.LY * h0)
        print(f"  h0 {h0:.3f} rho {rho:6.0f}   {pk:>10.3f}{'YES' if pk >= P.TILT_STOP - 1e-6 else 'no':>8}"
              f"{load:>13.3f}{sp:>9.3f}")

    # ---- 5. righting ------------------------------------------------------------------
    print("\n=== 5. lean for 0.8 s at 0.6 of the range, then stand the tray back up ===")
    for h0, rho in ((0.035, 1000.0), (0.05, 1000.0)):
        lean = jnp.tile(jnp.array([0.6, 0.0]), (16, 1))
        abrupt = jnp.concatenate([lean, jnp.zeros((20, 2))])
        ramp = jnp.concatenate([lean, jnp.linspace(0.6, 0.0, 10)[:, None] * jnp.array([1.0, 0.0]),
                                jnp.zeros((10, 2))])
        for nm, u in (("abrupt", abrupt), ("gradual", ramp)):
            s, E, L, wall = jax.jit(fly)(u, calm(h0, rho))
            print(f"  h0 {h0:.3f} {nm:<8} energy before {float(E[15]):.3f}  peak after "
                  f"{float(jnp.max(E[16:])):.3f}  wall peak after {float(jnp.max(wall[16:]))/P.H_WALL:.2f} rim  "
                  f"spilled {float(jnp.sum(L))/(P.LX*P.LY*h0):.3f}")

    # ---- 6. survival ------------------------------------------------------------------
    print("\n=== 6. full-amplitude random attitude commands, 32 seeds, largest waves ===")
    for g_ in (P.WORLD, P.PLAN):
        for t_ in (jnp.array([0.050, 1400.0, 0.005, 0.25, 1.0, 0.10, 2.0]),
                   jnp.array([0.020, 700.0, 0.005, 0.25, 1.0, 0.10, 2.0])):
            def run(u):
                s, U = P.stage_init(), P.fluid_init(t_, g_)
                def body(c, uu):
                    s, U = c
                    U, s, _ = P.fluid_step(U, s, uu, t_, g_)
                    return (s, U), (swe.wave_speed(U, 9.81), jnp.abs(s[5]) + jnp.abs(s[4]))
                (_, U), (sp, tl) = jax.lax.scan(body, (s, U), u)
                return jnp.max(sp), jnp.any(~jnp.isfinite(U)), jnp.max(tl)
            us = jnp.stack([jnp.sign(jax.random.normal(jax.random.PRNGKey(i), (P.H, P.NU)))
                            for i in range(32)])
            sp, bad, tl = jax.jit(jax.vmap(run))(us)
            print(f"  grid {g_.nx}x{g_.ny}/{g_.substeps}, fill {float(t_[0]):.3f} rho {float(t_[1]):.0f}: "
                  f"peak CFL {float(jnp.max(sp))*g_.dt_sub/g_.dx:.2f}, non-finite "
                  f"{int(jnp.sum(bad))}/32, peak |roll|+|pitch| {float(jnp.max(tl)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
