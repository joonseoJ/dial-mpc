"""SUPERSEDED by tray_check.py (the tray is now a torque-held rigid body).

Does the drone-like carrier behave as its equations say it should?

Checked against closed forms before anything is planned on it:

  1. terminal speed    holding a tilt c, drag balances thrust at
                       v_t = sqrt(g tan(c) / K_DRAG).
  2. attitude          a step command is followed with time constant TAU_ATT,
                       inside the rate limit.
  3. coordination      the physics the task rests on.  With no drag, leaning
                       into an acceleration leaves the liquid exactly level --
                       thrust is along the tray normal -- so a hard, sustained
                       acceleration must not slosh it.  With drag, a cruise at
                       speed v stands the surface at slope K_DRAG v^2 / g.
                       If either fails, the in-plane forcing is wrong and so is
                       every trade-off built on it.
  4. speed limit       that slope reaches the rim sooner for a full tray: the
                       speed at which each fill starts to spill.
  5. survival          full-amplitude random attitude commands on both grids:
                       no non-finite state, CFL in range.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P
from csm.slosh import swe

G0 = 9.81


def fly(u_seq, theta, g=P.WORLD, k_drag=None):
    """Run a command sequence; return per-step stage state, wave energy, spill,
    and the surface slope along x (least squares over the tray)."""
    if k_drag is not None:
        P.K_DRAG = k_drag
    s, U = P.stage_init(), P.fluid_init(theta, g)
    x = (jnp.arange(g.nx) + 0.5) * g.dx - P.LX / 2

    def body(c, u):
        s, U = c
        U, s, lost = P.fluid_step(U, s, u, theta, g)
        hx = U[..., 0].mean(1)
        slope = jnp.sum(x * (hx - hx.mean())) / jnp.sum(x * x)
        return (s, U), (s, P.wave_energy(U, theta, g), lost, slope)
    (_, _), tr = jax.lax.scan(body, (s, U), u_seq)
    return tr


def main() -> int:
    K0 = P.K_DRAG
    calm = lambda h0: jnp.array([h0, 1000.0, 0.02, 0.0, 0.0, 0.0, 0.0])   # no wave

    # ---- 1. terminal speed ---------------------------------------------------
    print(f"=== 1. terminal speed, K_DRAG = {K0} ===")
    for c in (0.3, 0.6, 1.0):
        u = jnp.tile(jnp.array([c, 0.0]), (200, 1))              # 10 s
        s, E, L, _ = jax.jit(fly)(u, calm(0.02))
        v = float(s[-1, 2])
        vt = np.sqrt(G0 * np.tan(c * P.TILT_CMD_MAX) / K0)
        print(f"  command {c:.1f} (tilt {c*P.TILT_CMD_MAX:.3f} rad): v_end {v:.4f}  "
              f"analytic {vt:.4f}  err {100*(v/vt-1):+.2f}%")

    # ---- 2. attitude response ------------------------------------------------
    print("\n=== 2. attitude step response ===")
    u = jnp.tile(jnp.array([0.2, 0.0]), (10, 1))
    s, *_ = jax.jit(fly)(u, calm(0.02))
    pitch = np.asarray(s[:, 5]); target = 0.2 * P.TILT_CMD_MAX
    t63 = (np.argmax(pitch >= 0.632 * target) + 1) * P.DT
    print(f"  small step to {target:.3f} rad: reaches 63% by t = {t63:.2f} s "
          f"(TAU_ATT {P.TAU_ATT}; resolution {P.DT} s)")
    u = jnp.tile(jnp.array([1.0, 0.0]), (10, 1))
    s, *_ = jax.jit(fly)(u, calm(0.02))
    rate = np.max(np.abs(np.asarray(s[:, 7])))
    print(f"  full step: peak pitch rate {rate:.3f} rad/s (limit {P.W_MAX})")

    # ---- 3. coordination -------------------------------------------------------
    print("\n=== 3. coordination: does leaning into an acceleration slosh the liquid? ===")
    u = jnp.tile(jnp.array([0.8, 0.0]), (16, 1))                 # 0.8 s hard lean
    for k in (0.0, K0):
        s, E, L, sl = jax.jit(lambda u, t: fly(u, t, k_drag=k))(u, calm(0.035))
        print(f"  K_DRAG {k:.1f}: after 0.8 s at {float(s[-1,2]):.2f} m/s, "
              f"accel {float((s[-1,2]-s[-2,2])/P.DT):5.2f} m/s^2 -> wave energy "
              f"{float(E[-1]):.5f}, surface slope {float(sl[-1]):+.4f}")
    P.K_DRAG = K0
    print("  (no drag: energy and slope must stay ~0 however hard it accelerates)")
    print("\n  cruise slope vs K_DRAG v^2 / g, steady state:")
    for c in (0.3, 0.6):
        u = jnp.tile(jnp.array([c, 0.0]), (120, 1))              # 6 s, settles
        s, E, L, sl = jax.jit(fly)(u, calm(0.035))
        v = float(s[-1, 2]); tail = np.asarray(sl[-40:])
        print(f"    v {v:.3f} m/s: measured slope {tail.mean():+.4f} "
              f"(expected {K0*v*v/G0:.4f}; sign: surface rises toward -x, the back)")

    # ---- 4. depth-dependent speed limit ---------------------------------------
    print("\n=== 4. spill against cruise speed, by fill ===")
    print(f"  {'command':>8}{'v_t m/s':>9}" + "".join(f"{'h=' + format(h, '.3f'):>12}"
                                                   for h in (0.020, 0.035, 0.050)))
    for c in (0.3, 0.5, 0.7, 0.9, 1.0):
        vt = np.sqrt(G0 * np.tan(c * P.TILT_CMD_MAX) / K0)
        row = f"  {c:>8.1f}{vt:>9.2f}"
        for h in (0.020, 0.035, 0.050):
            u = jnp.tile(jnp.array([c, 0.0]), (60, 1))
            _, _, L, _ = jax.jit(fly)(u, calm(h))
            row += f"{float(jnp.sum(L))/(P.LX*P.LY*h):>12.3f}"
        print(row)
    for h in (0.020, 0.035, 0.050):
        f = P.H_WALL - h
        print(f"  static estimate v_max(h={h:.3f}) = sqrt(2 g f / (K L)) = "
              f"{np.sqrt(2*G0*f/(K0*P.LX)):.2f} m/s")

    # ---- 5. survival ------------------------------------------------------------
    print("\n=== 5. full-amplitude random attitude commands, 32 seeds, largest waves ===")
    th = jnp.array([0.050, 1400.0, 0.005, 0.25, 1.0, 0.10, 2.0])
    th_s = jnp.array([0.020, 1400.0, 0.005, 0.25, 1.0, 0.10, 2.0])
    for g in (P.WORLD, P.PLAN):
        for t_ in (th, th_s):
            def run(u):
                s, U = P.stage_init(), P.fluid_init(t_, g)
                def body(c, uu):
                    s, U = c
                    U, s, _ = P.fluid_step(U, s, uu, t_, g)
                    return (s, U), swe.wave_speed(U, 9.81)
                (_, U), sp = jax.lax.scan(body, (s, U), u)
                return jnp.max(sp), jnp.any(~jnp.isfinite(U))
            us = jnp.stack([jnp.sign(jax.random.normal(jax.random.PRNGKey(i), (P.H, P.NU)))
                            for i in range(32)])
            sp, bad = jax.jit(jax.vmap(run))(us)
            print(f"  grid {g.nx}x{g.ny}/{g.substeps}, fill {float(t_[0]):.3f}: "
                  f"peak CFL {float(jnp.max(sp))*g.dt_sub/g.dx:.2f}, non-finite "
                  f"{int(jnp.sum(bad))}/32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
