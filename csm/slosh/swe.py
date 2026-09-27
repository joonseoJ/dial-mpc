"""Shallow-water solver for liquid in a translating, tilting tray.

Why a fluid solver and not an equivalent pendulum
-------------------------------------------------
The usual engineering model for sloshing is a pendulum whose length sets the
first mode's frequency.  It is free to evaluate and it is the wrong model for
*this* study, for two reasons that both matter:

  it is least accurate exactly where the task lives.  The pendulum is a
  small-amplitude linearisation, and the interesting region is the one just
  short of spilling.

  it is trivially estimable.  A single pendulum is a second-order LTI system
  with four unknowns (omega, zeta, angle, rate) driven by a known input.
  Identifying that from the reaction force is a textbook exercise that
  converges in a couple of periods -- so a task built on it would have its
  belief collapsed by any competent estimator, which is precisely the failure
  the previous task design already ran into.

Shallow water fixes both.  Multiple modes and amplitude-dependent behaviour
come out of the equations rather than being added by hand, the wall run-up that
decides spilling is a direct output instead of a proxy, and the hidden state is
a whole grid observed through a single scalar reaction force -- an estimation
problem that is hard for a structural reason rather than by stipulation.

And it is affordable here for a reason that is not a coincidence: the task
wants a wide shallow tray (that is what makes the fill level move the slosh
frequency), shallow water is the regime where these equations are valid, and
shallow water is slow -- `c = sqrt(g h)` is 0.4-0.9 m/s -- so the CFL limit sits
near the physics substep the planner already uses.  Accuracy, validity and
speed all point the same way.

Equations
---------
In the tray frame, with a flat floor and a spatially uniform effective gravity,

    d_t h    + d_x (h u)             + d_y (h v)             = 0
    d_t (hu) + d_x (h u^2 + g_n h^2/2) + d_y (h u v)         = h g_x
    d_t (hv) + d_x (h u v)           + d_y (h v^2 + g_n h^2/2) = h g_y

`g_n` is the component of effective gravity normal to the floor and `(g_x, g_y)`
its in-plane part, both from

    g_eff = R^T (g_world - a_tray)

so translating the tray and tilting it enter through the same term.  That is
what makes tilt a real control authority over the slosh rather than a
decoration: a tilt of `alpha` produces an in-plane `g sin(alpha)` that can be
timed against the wave.

Neglected, deliberately: Coriolis (the tray has no yaw, so the in-plane
Coriolis term vanishes) and the spatial variation of acceleration across the
tray from roll/pitch rate, which for a 0.2 m tray is second order and enters
`g_n` rather than the driving terms.

Scheme
------
Finite volume, MUSCL reconstruction with a minmod limiter, Rusanov (local
Lax-Friedrichs) flux, SSP-RK2 in time.  Second order in space is not a
refinement here but a requirement: the physical damping ratio is one of the
unknown parameters and ranges over 0.005-0.05, so a first-order scheme's
numerical dissipation would swamp the very quantity the belief is over.
`validate.py` measures that numerical damping and is the check that this choice
was enough.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Wet-dry handling, which this task cannot avoid.
#
# The tray genuinely empties at one wall under hard lateral acceleration: the
# equilibrium surface slope is `g_horiz / g_n`, so at 5 m/s^2 on two axes the
# surface tilts by 0.6 across a 0.20 m tray -- 0.12 m of height difference on a
# 0.02-0.05 m fill.  And the acceleration that dries the shallow end of a
# shallow tray is 1.96 m/s^2, exactly the one that spills a full tray, so
# restricting the actuator to prevent drying would also remove the constraint
# the whole task is built on.  The scheme has to cope instead.
#
# Two standard measures, both needed (measured: with neither, every sampled
# full-amplitude plan diverged; the depth reached 1e15 within 7 control steps):
#   `H_MIN`  a hard positivity floor.
#   `H_DRY`  the desingularisation scale: below it `u = hu/h` is blended
#            smoothly to zero instead of dividing by a vanishing depth, which
#            is what produced the 1e20 wave speeds.
H_MIN = 1e-6
H_DRY = 1e-3                  # 1 mm, 20x below the shallowest fill used


def _minmod(a, b):
    return jnp.where(a * b > 0.0, jnp.where(jnp.abs(a) < jnp.abs(b), a, b), 0.0)


def _pad_reflect_x(U):
    """Two ghost cells per side on axis 0: solid wall.

    The wall mirrors depth and reverses the normal momentum, which is the
    no-through-flow condition; the tangential momentum is mirrored unchanged
    (free slip), since a viscous wall layer is far below what shallow water
    resolves anyway.
    """
    # component 1 is the momentum normal to this pair of walls; the y pass
    # reaches here with the components already swapped, so one flip serves both
    flip = jnp.ones(U.shape[-1]).at[1].set(-1.0)
    left = U[1::-1] * flip               # cells 1, 0 with normal momentum reversed
    right = U[-1:-3:-1] * flip           # cells N-1, N-2
    return jnp.concatenate([left, U, right], axis=0)


def _velocity(h, hq):
    """`hq / h`, desingularised: exact where the cell is wet, zero as it dries.

    `2 h hq / (h^2 + max(h, H_DRY)^2)` is the Kurganov-Petrova form.  For
    `h >> H_DRY` the denominator is `2 h^2` and this is exactly `hq / h`; as
    `h -> 0` it goes to zero linearly instead of to infinity.
    """
    return 2.0 * h * hq / (h ** 2 + jnp.maximum(h, H_DRY) ** 2)


def _flux_x(U, g_n):
    h = jnp.maximum(U[..., 0], H_MIN)
    u = _velocity(h, U[..., 1])
    f0 = h * u
    f1 = h * u * u + 0.5 * g_n * h ** 2
    if U.shape[-1] == 2:
        return jnp.stack([f0, f1], axis=-1)
    return jnp.stack([f0, f1, U[..., 2] * u], axis=-1)


def _rusanov_x(UL, UR, g_n):
    hL = jnp.maximum(UL[..., 0], H_MIN)
    hR = jnp.maximum(UR[..., 0], H_MIN)
    uL, uR = _velocity(hL, UL[..., 1]), _velocity(hR, UR[..., 1])
    a = jnp.maximum(jnp.abs(uL) + jnp.sqrt(g_n * hL),
                    jnp.abs(uR) + jnp.sqrt(g_n * hR))
    return 0.5 * (_flux_x(UL, g_n) + _flux_x(UR, g_n)) \
        - 0.5 * a[..., None] * (UR - UL)


def _divergence_x(U, dx, g_n):
    """Flux divergence along axis 0, with MUSCL reconstruction."""
    Up = _pad_reflect_x(U)                            # (N+4, ...)
    d = Up[1:] - Up[:-1]                              # (N+3, ...)
    slope = _minmod(d[:-1], d[1:])                    # (N+2, ...)
    Uc = Up[1:-1]                                     # (N+2, ...) cells with slopes
    # Positivity-preserving limit: shrink the slope so neither reconstructed
    # face depth can go negative.  Without this the limiter can hand the
    # Riemann solver a negative depth in a draining cell, and `sqrt(g_n h)` is
    # then NaN -- the same failure the desingularisation above is guarding,
    # arriving by the other route.  All components are scaled by the same
    # factor so the reconstruction stays a consistent state.
    room = jnp.maximum(Uc[..., 0] - H_MIN, 0.0)
    lim = jnp.minimum(1.0, 2.0 * room / (jnp.abs(slope[..., 0]) + 1e-30))
    slope = slope * lim[..., None]
    left = Uc[:-1] + 0.5 * slope[:-1]                 # right face of cell i
    right = Uc[1:] - 0.5 * slope[1:]                  # left face of cell i+1
    F = _rusanov_x(left, right, g_n)                  # (N+1, ...) interface fluxes
    return (F[1:] - F[:-1]) / dx


def rhs_1d(U, dx, g_n, g_x):
    """`dU/dt` for the 1-D channel.  `U` is `(N, 2)` of `[h, hu]`."""
    src = jnp.stack([jnp.zeros_like(U[..., 0]),
                     jnp.maximum(U[..., 0], 0.0) * g_x], axis=-1)
    return -_divergence_x(U, dx, g_n) + src


def rhs_2d(U, dx, dy, g_n, g_x, g_y):
    """`dU/dt` for the tray.  `U` is `(Nx, Ny, 3)` of `[h, hu, hv]`.

    Unsplit: the two flux divergences are formed independently and added.  The
    y pass reuses the x machinery by transposing and swapping the momentum
    components, so there is exactly one reconstruction and one Riemann solver
    in this file to get right.
    """
    Lx = _divergence_x(U, dx, g_n)
    # swap axes and the two momentum components, solve, swap back
    Uy = jnp.transpose(U, (1, 0, 2))[..., jnp.array([0, 2, 1])]
    Ly = _divergence_x(Uy, dy, g_n)
    Ly = jnp.transpose(Ly[..., jnp.array([0, 2, 1])], (1, 0, 2))
    h = jnp.maximum(U[..., 0], 0.0)
    src = jnp.stack([jnp.zeros_like(h), h * g_x, h * g_y], axis=-1)
    return -(Lx + Ly) + src


def step_1d(U, dt, dx, g_n, g_x):
    """One SSP-RK2 step."""
    U1 = U + dt * rhs_1d(U, dx, g_n, g_x)
    U1 = U1.at[..., 0].set(jnp.maximum(U1[..., 0], H_MIN))
    U2 = 0.5 * (U + U1 + dt * rhs_1d(U1, dx, g_n, g_x))
    return U2.at[..., 0].set(jnp.maximum(U2[..., 0], H_MIN))


def step_2d(U, dt, dx, dy, g_n, g_x, g_y):
    U1 = U + dt * rhs_2d(U, dx, dy, g_n, g_x, g_y)
    U1 = U1.at[..., 0].set(jnp.maximum(U1[..., 0], H_MIN))
    U2 = 0.5 * (U + U1 + dt * rhs_2d(U1, dx, dy, g_n, g_x, g_y))
    return U2.at[..., 0].set(jnp.maximum(U2[..., 0], H_MIN))


def wave_speed(U, g_n):
    """`max(|u| + sqrt(g h))`, for reporting the CFL number actually run.

    Uses the same desingularised velocity the fluxes use.  Dividing by a raw
    near-dry depth here instead reported 41 m/s and a CFL of 12 on runs that
    were in fact stable -- a number from the diagnostic, not from the scheme.
    """
    h = jnp.maximum(U[..., 0], H_MIN)
    c = jnp.sqrt(g_n * h)
    s = jnp.abs(_velocity(h, U[..., 1])) + c
    if U.shape[-1] == 3:
        s = jnp.maximum(s, jnp.abs(_velocity(h, U[..., 2])) + c)
    return jnp.max(s)


def effective_gravity(R, a_world, g0=9.81):
    """`(g_n, g_x, g_y)` in the tray frame.

    `R` maps tray coordinates to world, `a_world` is the tray's linear
    acceleration.  The fluid cannot tell gravity from acceleration, so both
    arrive through one vector; tilting the tray and accelerating it are the
    same control authority expressed two ways.
    """
    g_eff = R.T @ (jnp.array([0.0, 0.0, -g0]) - a_world)
    # Defensive floor on the normal component.  The caller is expected to keep
    # the tray upright (see plant.TILT_STOP); if it ever does not, a negative
    # `g_n` makes `sqrt(g_n h)` NaN and one rollout takes down the whole
    # vmapped batch, so the failure is bounded here rather than propagated.
    return jnp.maximum(-g_eff[2], 0.1 * g0), g_eff[0], g_eff[1]


def linear_modes(n, L, h, g0=9.81):
    """Analytic standing-wave frequencies of *these equations*, rad/s.

    Shallow water is non-dispersive, so `omega_n = (n pi / L) sqrt(g h)`.  This
    is the solver's own target, not the true free-surface answer: potential flow
    gives `omega^2 = g k tanh(k h)`, which the long-wave limit approximates to
    within 2% at `h/L = 0.1` and 10% at `h/L = 0.25`.  Validating against the
    shallow-water value checks the *code*; the gap to potential flow is the
    modelling error and is stated separately.
    """
    k = n * jnp.pi / L
    return k * jnp.sqrt(g0 * h)
