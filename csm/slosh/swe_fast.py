"""The same shallow-water scheme as `swe.py`, laid out for the GPU.

Same mathematics, term for term -- MUSCL reconstruction with a minmod limiter,
the positivity-preserving slope limit, desingularised velocities, a Rusanov
flux, reflective walls, SSP-RK2 -- so this is a pure performance change and is
checked against `swe.step_2d` to float32 round-off.

What changes is the shape of the computation.  Compiled, one `swe.step_2d`
substep was 9 fusions plus 42 gathers, 34 concatenates and 2 transposes -- on
the order of ninety kernels per substep, against the four per step that a
hand-written GPU shallow-water solver uses (Brodtkorb et al.: one nine-point
flux kernel carrying 87% of the runtime, plus time step, integration and
boundaries).  A rollout is 960 substeps, so a dispatch launched ~90k kernels,
each a round trip to memory.  The sources were all in the layout:

  gathers       the y pass reused the x pass by swapping the two momentum
                components with `U[..., jnp.array([0, 2, 1])]`; fancy indexing
                compiles to a gather, and it happened on every stage.
  transposes    the same reuse, turning the grid on its side to do it.
  concatenates  ghost cells built by concatenating slices on every call.
  minor axis 3  the state was `(..., NX, NY, 3)`, so the fastest-varying axis
                in memory was the three conserved variables -- strided access
                for every per-variable operation.

Here the three variables are separate arrays, each direction is written along
its own axis, and the ghost cells come from `jnp.pad`, which XLA fuses into
its consumers instead of materialising.
"""
from __future__ import annotations

import jax.numpy as jnp

from csm.slosh.swe import H_MIN, H_DRY


def _minmod(a, b):
    return jnp.where(a * b > 0.0, jnp.where(jnp.abs(a) < jnp.abs(b), a, b), 0.0)


def _vel(h, q):
    return 2.0 * h * q / (h ** 2 + jnp.maximum(h, H_DRY) ** 2)


def _pad(x, axis, flip):
    """Two reflective ghost cells per side along `axis`.

    'symmetric' padding repeats the edge, so the ghosts are [x1, x0 | x0, x1 ...]
    -- exactly the mirror the concatenating version built.  The momentum normal
    to the wall changes sign in the ghosts, which is the no-through-flow
    condition.
    """
    pw = [(0, 0)] * x.ndim
    pw[axis] = (2, 2)
    xp = jnp.pad(x, pw, mode="symmetric")
    if not flip:
        return xp
    n = x.shape[axis]
    sign = jnp.ones(n + 4, x.dtype).at[:2].set(-1.0).at[-2:].set(-1.0)
    shape = [1] * x.ndim
    shape[axis] = n + 4
    return xp * sign.reshape(shape)


def _sl(x, axis, start, stop):
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(start, stop)
    return x[tuple(idx)]


def _divergence(h, qn, qt, d, g_n, axis):
    """Flux divergence along `axis` for depth, normal and tangential momentum."""
    hp, qnp, qtp = _pad(h, axis, False), _pad(qn, axis, True), _pad(qt, axis, False)

    def slopes(xp):
        dd = _sl(xp, axis, 1, None) - _sl(xp, axis, 0, -1)          # N+3
        return _minmod(_sl(dd, axis, 0, -1), _sl(dd, axis, 1, None))  # N+2

    sh, sqn, sqt = slopes(hp), slopes(qnp), slopes(qtp)
    hc = _sl(hp, axis, 1, -1)                                        # N+2
    room = jnp.maximum(hc - H_MIN, 0.0)
    lim = jnp.minimum(1.0, 2.0 * room / (jnp.abs(sh) + 1e-30))
    sh, sqn, sqt = sh * lim, sqn * lim, sqt * lim
    qnc, qtc = _sl(qnp, axis, 1, -1), _sl(qtp, axis, 1, -1)

    def faces(c, s):
        left = _sl(c, axis, 0, -1) + 0.5 * _sl(s, axis, 0, -1)
        right = _sl(c, axis, 1, None) - 0.5 * _sl(s, axis, 1, None)
        return left, right                                           # N+1 each

    hL, hR = faces(hc, sh)
    qnL, qnR = faces(qnc, sqn)
    qtL, qtR = faces(qtc, sqt)
    hL, hR = jnp.maximum(hL, H_MIN), jnp.maximum(hR, H_MIN)
    uL, uR = _vel(hL, qnL), _vel(hR, qnR)
    a = jnp.maximum(jnp.abs(uL) + jnp.sqrt(g_n * hL),
                    jnp.abs(uR) + jnp.sqrt(g_n * hR))

    def rus(fL, fR, qL, qR):
        return 0.5 * (fL + fR) - 0.5 * a * (qR - qL)

    Fh = rus(hL * uL, hR * uR, hL, hR)
    Fn = rus(hL * uL * uL + 0.5 * g_n * hL ** 2,
             hR * uR * uR + 0.5 * g_n * hR ** 2, qnL, qnR)
    Ft = rus(qtL * uL, qtR * uR, qtL, qtR)
    div = lambda F: (_sl(F, axis, 1, None) - _sl(F, axis, 0, -1)) / d
    return div(Fh), div(Fn), div(Ft)


def rhs(h, hu, hv, dx, dy, g_n, g_x, g_y):
    """`d/dt (h, hu, hv)` on an `(NX, NY)` grid."""
    ax_h, ax_u, ax_v = _divergence(h, hu, hv, dx, g_n, axis=0)
    ay_h, ay_v, ay_u = _divergence(h, hv, hu, dy, g_n, axis=1)
    hp = jnp.maximum(h, 0.0)
    return (-(ax_h + ay_h),
            -(ax_u + ay_u) + hp * g_x,
            -(ax_v + ay_v) + hp * g_y)


def step(h, hu, hv, dt, dx, dy, g_n, g_x, g_y):
    """One SSP-RK2 step, identical in arithmetic to `swe.step_2d`."""
    r = rhs(h, hu, hv, dx, dy, g_n, g_x, g_y)
    h1 = jnp.maximum(h + dt * r[0], H_MIN)
    u1, v1 = hu + dt * r[1], hv + dt * r[2]
    r1 = rhs(h1, u1, v1, dx, dy, g_n, g_x, g_y)
    h2 = jnp.maximum(0.5 * (h + h1 + dt * r1[0]), H_MIN)
    return h2, 0.5 * (hu + u1 + dt * r1[1]), 0.5 * (hv + v1 + dt * r1[2])
