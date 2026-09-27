"""A tray of liquid on a stage: the plant, batched over hypotheses.

The task
--------
Carry a shallow open tray of liquid a fixed distance and stop, without
spilling and without leaving the liquid sloshing.  The stage is commanded in
acceleration; the liquid is simulated, not modelled by an oscillator.

    theta = (h0, rho, zeta, a_x, phi_x, a_y, phi_y)
      h0           fill depth, m          -- sets the slosh frequency AND the
                                             freeboard, so one unknown moves
                                             both the dynamics and the limit
      rho          density, kg/m^3        -- moves mass without moving frequency
      zeta         damping ratio          -- how fast slosh decays
      a_x, phi_x   first x-mode amplitude and phase  } the *hidden state*: what
      a_y, phi_y   first y-mode amplitude and phase  } the liquid is doing now

Why this shape
--------------
The previous task (a rigid payload with unknown mass and centre of mass) failed
the spec's E0 gate: success saturated at `M = 4`-`8` and 64 particles bought
nothing over 8.  Two measurements explain it and both are designed against
here.

**No constraint ever fired.**  Peak margins over 16 episodes were slip 0.44,
tilt 0.56, torque 0.47, speed 0.71 -- four of seven cost rows were identically
zero, so the objective was "reach a point cheaply", a problem whose answer
barely depends on theta.  Here the spill constraint has a threshold that
*theta itself sets*: freeboard is `H_WALL - h0`, so a full tray must be handled
gently and a shallow one need not be.  A plan good for one is wrong for the
other, which is the disagreement `M` is supposed to resolve.

**The uncertainty was estimable.**  A constant mass and centre of mass are
exactly what an estimator removes, and a single equivalent pendulum -- the usual
slosh model -- is no better, being a second-order LTI system with four unknowns
driven by a known input.  Here the hidden state is a 24x24x3 grid observed
through one scalar reaction force, and the wave is nonlinear at operating
amplitude: measured inside these equations, the first mode's apparent decay
runs 0.0004 at 0.1% of depth to 0.032 at 20%, because energy moves into
harmonics.  Whether an estimator can still collapse this belief is not assumed;
`estimate.py` measures it, and if it can, the task is wrong again.

Geometry, and why
-----------------
A wide shallow tray, not a deep pot.  `omega ~ sqrt(g h) * k`, so in shallow
water the fill level moves the frequency as `sqrt(h)` -- over the range here a
factor of 1.6 -- while in a deep container `tanh(k h)` saturates and the same
fill range moves it by a few percent.  The same choice keeps the shallow-water
equations valid: at `h/L = 0.25` they differ from potential flow by 9%, at
`h/L = 0.4` by 22%, which is why the fill range stops at 0.05 m on a 0.20 m
tray.  Frequency spread, model validity and CFL cost all improve together as
the tray gets shallower.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from csm.slosh import swe, swe_fast

# Full float32 matmuls, everywhere this plant is imported.  On this GPU JAX
# defaults float32 dots to TF32 (a 10-bit mantissa), and `restrict` -- two
# small matmuls -- lost 4.9e-4 of the liquid's volume and momentum on every
# call, where float32 round-off is 1e-7.  The same default silently applied to
# every MPPI update (`omega @ eps`) and every belief-weighted cost (`C @ bel`).
jax.config.update("jax_default_matmul_precision", "highest")

G0 = 9.81

# --- the tray ---------------------------------------------------------------
LX = LY = 0.20                # tray footprint, m
H_WALL = 0.07                 # rim height above the floor, m
# Grid and substeps, both set by measurement rather than by eye.
#   modes        (1,0), (1,1), (2,0) within 0.5% of analytic at this grid
#   peak height  within 0.9% of a 48x48 reference -- this is what the freeboard
#                row reads, and it is why the planner is steered by height
#   spilled vol  -4% at a heavy spill but -50% at a *marginal* one, because a
#                volume past a threshold is a difference of near-equal numbers.
#                Reported, not planned against.
#   robustness   no full-amplitude random-bang rollout goes non-finite at the
#                substep count chosen below; see SUBSTEPS, which is where the
#                CFL argument lives.
NX = NY = 24
DX, DY = LX / NX, LY / NY

# --- horizon and control ----------------------------------------------------
DT = 0.05                     # control period, s
H = 30                        # plan length -> 1.5 s, about two slosh periods
                              # of the *slowest* hypothesis (0.90 s at h=0.02)
# CFL is set by the *piled* depth, not the nominal one: under hard lateral
# acceleration the liquid stacks against a wall to 0.10 m on a 0.05 m fill, and
# `sqrt(g h)` rises with it.  Sizing the substeps on the nominal depth is what
# made every adversarial rollout diverge.
# 32, not 24.  The first robustness sweep chose the worst case wrongly: it used
# the *deepest* fill, which is the one most prone to spilling, when the failure
# mode is drying -- and a shallow tray dries far sooner, because the surface
# slope that exposes its floor is `h0 / (L/2)`, 0.2 at a 0.02 m fill against 0.5
# at 0.05 m.  Measured under full-amplitude commands, the peak CFL at the
# shallowest fill is 1.03 at 24 substeps and 0.61 at 32, and every belief
# contains that hypothesis.
SUBSTEPS = 32
DT_SUB = DT / SUBSTEPS


class Grid(NamedTuple):
    """A discretisation of the tray: cells per side and substeps per control step.

    The world and the planner need not share one.  `WORLD` is the validated
    grid every reported number is measured on; `PLAN` is the coarser one the
    planner's thousands of rollouts use.  A NamedTuple of ints so it is a
    static, hashable constant under `jit`.
    """
    nx: int
    ny: int
    substeps: int

    @property
    def dx(self):
        return LX / self.nx

    @property
    def dy(self):
        return LY / self.ny

    @property
    def dt_sub(self):
        return DT / self.substeps


WORLD = Grid(NX, NY, SUBSTEPS)
# The planner's grid, set by `verify_plan.py` on the drone carrier: modes within
# 0.8% of analytic; no non-finite rollout under full-amplitude random attitude
# commands with the largest initial waves, peak CFL 0.47 at 10 substeps (8
# reaches 0.58 and 6 diverges on every deep-fill rollout, so 10 keeps a margin
# off that cliff); and it ranks proposal clouds like the world from real
# mid-carry states -- Spearman 0.998 worst case, top-10% overlap 0.92 worst
# case, median cost error 0.8%.  The gimbal model needed 20: the drone can only
# force the liquid through drag, so the waves it raises are gentler.
PLAN = Grid(16, 16, 10)

# Twelve nodes, not the five the rigid-payload task used.  Controlling slosh is
# a question of *timing*: the residual after a move depends on where the
# acceleration pulses fall relative to the wave.  The slosh period here runs
# 0.57-0.90 s, so a quarter period is 0.14-0.22 s; twelve nodes over the 1.5 s
# horizon put one every 0.125 s, which is the coarsest spacing that can still
# place a pulse inside a quarter period of the fastest hypothesis.
N_NODE = 12
NU = 2                        # two attitude setpoints: pitch (drives x), roll (drives y)
DIM_U = N_NODE * NU
DIM_THETA = 7

# --- the carrier: a tray held at its centre --------------------------------------
# To move, the tray must lean: the carrier's horizontal acceleration comes from
# the tilt, like a multirotor's,
#
#     a = g * n_xy / n_z  -  K_DRAG |v| v,        n = R(roll, pitch) e_z .
#
# K_DRAG does not act on the liquid.  Its only job is to make cruising need a
# steeper lean the faster it goes -- at steady speed tan(tilt) = K_DRAG v^2 / g.
# What the liquid feels is the tray's tilt, less the part of it that the
# acceleration uses: leaning exactly into an acceleration leaves the surface
# level, as a waiter's tray does, while the lean that holds a cruise against
# drag is uncancelled and runs the liquid downhill.
#
# The tilt is not a commanded kinematic variable.  The tray is a rigid body
# pivoted at its centre, held by a hand whose torque is limited:
#
#     I theta'' = tau_hand + tau_liquid,
#     tau_hand  = clip(KP (theta_cmd - theta) - KD theta', +-TAU_MAX),
#
# and tau_liquid is the weight of the liquid about the pivot.  When the liquid
# runs to the low side its centre of mass follows, and that torque tips the tray
# *further* -- a destabilising load the hand must hold.  Quasi-statically it is
#
#     tau_liquid ~ (rho A g L^2 / 12 + m g h / 2) * theta,
#
# whose first term does not depend on the fill at all (a shallow tray has less
# liquid but it shifts further), so the load the hand must resist is set by the
# density as much as by the depth.  Leaning to fly fast therefore costs holding
# torque, and past TAU_MAX the tray cannot be held and tips over; standing it
# back up to relieve the torque releases the liquid piled on the low side, which
# is what sloshes.  That is the trade-off the task is built around.
TILT_CMD_MAX = 0.35           # rad, attitude setpoint range
K_DRAG = 2.0                  # 1/m, virtual drag: steeper lean at speed only
M_SHELL = 0.3                 # kg, the empty tray (its centre of mass at the pivot)
KP = 20.0                     # N m / rad.  A stiff hand: at equilibrium
                              # KP (cmd - tilt) = k_liquid tilt, so the tray settles
                              # at cmd * KP / (KP - k_liquid).  At KP = 4 even the
                              # lightest tray was pulled past the 0.40 rad stop by a
                              # 0.35 rad command -- every tray "tipped", which is a
                              # soft hand, not a heavy load.  At 20 the overshoot is
                              # 5-14%, and tipping happens only where the torque the
                              # hold needs exceeds TAU_MAX.
KD = 0.6                      # N m s / rad; damping ratio ~0.8 at the nominal tray
TAU_MAX = 0.7                 # N m.  Holding a steady lean theta needs about
                              # k_liquid * theta: the heaviest tray can be held
                              # to ~0.28 rad, the lightest far past 0.35
TILT_MAX = 0.30               # rad; cost hinge
TILT_STOP = 0.40              # rad; the tray has tipped over -- hard stop
V_MAX = 1.5                   # m/s, speed hinge
W_MAX = 2.0                   # rad/s, attitude-rate hinge

# --- the annealing ladder ---------------------------------------------------
# Measured, not inherited (`calibrate.py`).  Lambda is bisected online at every
# control step and every level to hit a target effective sample size, and the
# geometric mean over episodes and steps is taken; the target itself is swept
# against the task rather than assumed.  Selection is on the *weighted total
# cost*, because this objective is a trade -- ranking by distance alone picks
# the schedule that covers the most ground while spilling three times as much
# liquid.
#
#   ladder                ESS   total   spilled
#   (0.6, 0.25)          0.25   51.01    0.0156
#   (0.6, 0.25)          0.50   51.00    0.0101
#   (0.6, 0.25, 0.10)    0.25   50.50    0.0097
#   (0.6, 0.25, 0.10)    0.50   50.33    0.0055   <- this one
#
# The totals differ by 1.4% while the spill differs threefold, which is the
# useful reading: the ladder barely moves the objective's value and strongly
# moves *which* row pays.
SIGMA_SCHEDULE_3 = (0.6, 0.25, 0.10)            # the ladder above, kept for reference
LAM_SCHEDULE_3 = (105.73, 55.56, 38.02)

# What the closed-loop studies now use (`closed_loop.py`), measured on the
# oracle after the liquid-reset fix, 8 thetas x 2 seeds:
#
#                      cost    spilled
#   plain,  ESS 50%    72.33   0.024
#   plain,  ESS 25%    64.57   0.028
#   elite,  ESS 50%    55.55   0.000
#   elite,  ESS 25%    50.55   0.000     -21.78 +- 7.34 vs the first row
#
# Plain MPPI does not descend its own objective here: 40% of weighted-average
# updates made the plan worse under the planner's own model, by 18-25% on
# average, because averaging two well-timed wave cancellations gives a
# mistimed one.  "Elite" keeps the weighted average only when it beats every
# sample it was built from and otherwise keeps the best sample, so an update
# can never be worse than the plan it started from.  Temperatures are solved
# online to the ESS target at every step for every condition, so no planner is
# run at a different sharpness from another -- an oracle calibrated at another
# condition's temperature was one of the confounds in the earlier comparison.
SIGMA_SCHEDULE = (0.6, 0.25)
ESS_TARGET = 0.25
ELITE = True
LAM_SCHEDULE = None                             # solved online; see ESS_TARGET

# --- the task ---------------------------------------------------------------
GOAL = jnp.array([0.60, 0.0])  # m; a straight carry, long enough that it has
                               # to be hurried and short enough to settle after

# A delivery deadline.  Without one, caution looked free: planning for a fuller
# tray than the truth cost 0.10 against 6.37 for a shallower one.  That
# measurement was taken with the liquid-reset bug described in `make_rollout`,
# so the ratio itself needs re-measuring; the deadline stays because a waiter
# who never arrives has not done the task either.
T_ARRIVE = 1.25               # s; be within R_ARRIVE of the goal by now.
                              # With per-step slosh and late=50 the oracle meets it and
                              # spills nothing in 2.4 s, so it is a deadline that can
                              # be kept, not one that forces a trade.
R_ARRIVE = 0.05               # m
TIME_SOFT = 0.004             # m; width of the not-yet-delivered edge
# Slosh is scored at EVERY step, not only at the end of the horizon.
#
# A terminal-only slosh row was tried and is a receding-horizon trap.  The end
# of the horizon is always 1.5 s ahead, so the row asks only that the wave be
# calm at a moment that never arrives; once the tray reaches the goal and the
# goal rows stop pulling, all remaining authority goes into setting up a
# cancellation timed for that receding instant, and what actually executes is
# the aggressive first part of that set-up.  Measured with the true liquid
# state in the planner, 8 thetas x 2 seeds, same evaluation throughout:
#
#                         oracle     mean-theta    oracle - mean
#   terminal slosh x15     86.51       70.81       +15.70 +- 8.00
#   per-step slosh x2.05   43.61       59.26       -15.65 +- 6.45
#
# Under the terminal row the oracle spilled 20% of the liquid, almost all of it
# after arriving; per-step, it spilled nothing in 2.4 s and became the floor,
# which is the ordering a planner that knows the truth must have.
SLOSH_TERMINAL = False

# --- the unknown parameters -------------------------------------------------
# The tray is handed over already sloshing: a first x-mode wave of 10-25% of the
# depth with unknown phase, and a smaller cross-wave.  Two hypotheses half a
# cycle apart want speed changes at opposite moments -- a disagreement in
# *direction*, which no single quantile of the belief can stand in for.  Largest
# amplitudes keep the corner of a full tray below the rim at t=0:
# 0.05 * (1 + 0.25 + 0.10) = 0.0675 < 0.07.
THETA_LO = jnp.array([0.020, 700.0, 0.005, 0.10, 0.0, 0.00, 0.0])
THETA_HI = jnp.array([0.050, 1400.0, 0.050, 0.25, 2 * jnp.pi, 0.10, 2 * jnp.pi])
#                     h0     rho     zeta   a_x   phi_x      a_y   phi_y
# amplitudes are fractions of the fill depth, so `a_x = 0.06` is a wave 6% of
# the depth -- small enough to be plausible as "what is left over from picking
# the tray up", large enough that its phase matters over the horizon


@dataclass(frozen=True)
class Costs:
    """Fixed problem weights -- not the belief.

    Measured on this plant with `rows.py`, not inherited.  Three different
    rules set three groups, because the rows are three different kinds of
    thing:

      goal vs slosh   equalised on their *spread* at the finest annealing
                      level -- goal std 1.03, slosh std 5.03, so slosh takes
                      2.05 against goal's 10 and both pull 10.3.  Spread, not
                      mean: a softmax is blind to a constant offset, so what
                      ranks two plans is how much a row varies, and a row with
                      a large mean and no spread looks important in a table
                      while being invisible to the planner.

      goal's own level  from the distance-versus-spill frontier, which is the
                      one trade no spread can set.  At 1 the planner will not
                      move a full tray at all (0.367 m short of a 0.60 m
                      carry); at 50 it spills a fifth of the liquid; at 10 it
                      arrives 0.079 m short having spilled nothing.

      constraints     spill, freeboard, tilt, vel keep large weights.  They sit
                      at zero much of the time -- at the finest level spill is
                      zero on 27% of sampled plans and tilt on 48% -- and a
                      term that is zero until it is violated is *supposed* to
                      be expensive when it is.  The barrier weight was swept by
                      outcome rather than spread: at 0.5 the full tray stops
                      0.186 m short, at 15 it stops 0.309 m short, at 5 it gets
                      to 0.042 m with nothing spilled.

      effort          a regulariser, not an objective, held at 5% of the goal
                      row's pull.  Equalising its spread like the others gave
                      it a weight of 29.8 against the goal's 10, which simply
                      makes standing still optimal.
    """
    goal: float = 10.0
    spill: float = 50.0
    freeboard: float = 5.0
    slosh: float = 2.046
    late: float = 50.0
    # Set by `caution.py` on the held-tray carrier, paired cell by cell with
    # common random numbers (planner told depth a, tray at depth b, rho 1000):
    #   w_time    cautious       bold          matched spill  arrival
    #     2        4.06+-1.14   -1.05+-0.44       0.0000       1.27 s
    #     4        4.60+-2.34    2.69+-1.59       0.0000       1.22 s   <- this
    #     8        2.09+-4.06    1.18+-3.28       0.0000       1.13 s
    #    12       17.01+-6.09    5.32+-5.53       0.0000       1.08 s
    # On this carrier caution is already the dearer error at any weight: the
    # binding limit is the holding torque, set by density, so a wrong *depth*
    # barely hurts a bold plan.  12 (the drone carrier's balance) only adds
    # noise -- two steps of arrival are 24 cost units -- so the weight is the
    # lowest at which caution still costs more than boldness.
    time: float = 4.0
    # holding torque: provisional, not yet measured against the other rows on
    # this carrier; see `tray_check.py` for its magnitude
    torque: float = 5.0
    tilt: float = 20.0
    vel: float = 10.0
    effort: float = 0.267


ROW_NAMES = ("goal", "spill", "freeboard", "slosh", "late", "tilt", "vel",
             "effort", "time", "torque")

# How close to the rim the smooth barrier starts, m.  A cost that is exactly
# zero until the liquid goes over gives the sampler nothing to descend: MPPI
# could only avoid spilling by proposing plans that spill and down-weighting
# them, which wastes the whole proposal cloud on the failure it is trying to
# avoid.  The barrier also happens to be the better-converged quantity -- peak
# wall height is within 0.9% at this grid while the spilled *volume* at a
# marginal fill is 50% off, because volume past a threshold is a difference of
# two nearly equal numbers.  So the planner is steered by the margin and scored
# by the spill.
FREEBOARD_BAND = 0.012


# --- theta -> initial fluid state -------------------------------------------
def _mode_shapes(g: Grid = None):
    g = WORLD if g is None else g
    x = (jnp.arange(g.nx) + 0.5) * g.dx
    y = (jnp.arange(g.ny) + 0.5) * g.dy
    sx = jnp.cos(jnp.pi * x / LX)[:, None] * jnp.ones((1, g.ny))
    sy = jnp.ones((g.nx, 1)) * jnp.cos(jnp.pi * y / LY)[None, :]
    return sx, sy


SHAPE_X, SHAPE_Y = _mode_shapes()


def omega1(h0, k):
    """First-mode frequency along one axis: `(pi/L) sqrt(g h)`."""
    return k * jnp.sqrt(G0 * h0)


K1X, K1Y = jnp.pi / LX, jnp.pi / LY


def fluid_init(theta, g: Grid = None):
    """The liquid state this hypothesis says is in the tray right now.

    A modal amplitude and phase become a surface *and* a velocity field: a
    phase of zero is the wave at its extreme with the fluid at rest, a phase of
    pi/2 is a flat surface with the fluid moving.  Setting only the surface
    would silently restrict every hypothesis to the same quarter of the cycle.
    """
    g = WORLD if g is None else g
    sx_, sy_ = _mode_shapes(g)
    h0, ax, px, ay, py = theta[0], theta[3], theta[4], theta[5], theta[6]
    wx, wy = omega1(h0, K1X), omega1(h0, K1Y)
    eta = h0 * (ax * jnp.cos(px) * sx_ + ay * jnp.cos(py) * sy_)
    h = h0 + eta
    # continuity for a standing wave: d_t eta = -h0 d_x u  =>  u from the
    # quadrature component, with the sin() shape that d_x cos() produces
    x = (jnp.arange(g.nx) + 0.5) * g.dx
    y = (jnp.arange(g.ny) + 0.5) * g.dy
    ux = (h0 * ax * wx * jnp.sin(px) / (h0 * K1X)) * \
        (jnp.sin(jnp.pi * x / LX)[:, None] * jnp.ones((1, g.ny)))
    uy = (h0 * ay * wy * jnp.sin(py) / (h0 * K1Y)) * \
        (jnp.ones((g.nx, 1)) * jnp.sin(jnp.pi * y / LY)[None, :])
    return jnp.stack([h, h * ux, h * uy], axis=-1)


def sample_theta(key, n):
    """Uniform over the box, with the phases uniform on the circle."""
    return jax.random.uniform(key, (n, DIM_THETA),
                              minval=THETA_LO, maxval=THETA_HI)


# --- the stage --------------------------------------------------------------
# state: [x, y, vx, vy, roll, pitch, roll_rate, pitch_rate, t]
# The elapsed time rides in the state rather than being threaded through every
# signature, so the deadline can be absolute -- a receding horizon slides, and a
# deadline measured from the start of each plan would be a speed limit rather
# than a delivery time.
# s[9] is the holding torque the hand spent over the last control step,
# sum over both axes of (tau_hand / TAU_MAX)^2, averaged over substeps.
DIM_STAGE = 10


def stage_init():
    return jnp.zeros(DIM_STAGE)


def _rotation(roll, pitch):
    """Tray-to-world rotation for small roll about x then pitch about y."""
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    Rx = jnp.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = jnp.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx


def liquid_torque(h, theta, g_n, g_x, g_y, g: Grid):
    """Torque of the liquid's weight about the pivot, `(roll, pitch)`, N m.

    Each column's mass acts at its centroid, `(x, y, h/2)` in the tray frame,
    under the tray-frame effective gravity.  Signed so that a positive value
    *increases* the corresponding tilt: liquid on the low side, or an in-plane
    gravity pushing it there, tips the tray further.
    """
    rho = theta[1]
    x = (jnp.arange(g.nx) + 0.5) * g.dx - LX / 2
    y = (jnp.arange(g.ny) + 0.5) * g.dy - LY / 2
    m = rho * h * g.dx * g.dy
    t_pitch = jnp.sum(m * (x[:, None] * g_n + 0.5 * h * g_x))
    t_roll = -jnp.sum(m * (y[None, :] * g_n + 0.5 * h * g_y))
    return t_roll, t_pitch


def carrier_substep(s, u, h, theta, dt, g: Grid):
    """One substep of the held tray.  Returns `(s', (g_n, g_x, g_y), load)`.

    The acceleration and the effective gravity are computed from the same
    (pre-update) attitude, so the thrust is exactly along the tray normal and
    the liquid feels only the uncancelled part of the lean.
    """
    x, y, vx, vy, r, p, rd, pd, t, _ = s
    n = _rotation(r, p) @ jnp.array([0.0, 0.0, 1.0])
    spd = jnp.sqrt(vx ** 2 + vy ** 2)
    ax = G0 * n[0] / n[2] - K_DRAG * spd * vx
    ay = G0 * n[1] / n[2] - K_DRAG * spd * vy
    g_n, g_x, g_y = swe.effective_gravity(_rotation(r, p), jnp.array([ax, ay, 0.0]), G0)

    tl_r, tl_p = liquid_torque(h, theta, g_n, g_x, g_y, g)
    m_liq = theta[1] * jnp.sum(h) * g.dx * g.dy
    inertia = (M_SHELL + m_liq) * LX ** 2 / 12.0
    p_cmd = TILT_CMD_MAX * jnp.clip(u[0], -1.0, 1.0)
    r_cmd = -TILT_CMD_MAX * jnp.clip(u[1], -1.0, 1.0)      # +u[1] -> +y
    th_p = jnp.clip(KP * (p_cmd - p) - KD * pd, -TAU_MAX, TAU_MAX)
    th_r = jnp.clip(KP * (r_cmd - r) - KD * rd, -TAU_MAX, TAU_MAX)
    pd = pd + (th_p + tl_p) / inertia * dt
    rd = rd + (th_r + tl_r) / inertia * dt
    p, r = p + pd * dt, r + rd * dt
    # at the stop the tray has tipped over; the rate into the stop is lost
    pd = jnp.where(jnp.abs(p) >= TILT_STOP, 0.0, pd)
    rd = jnp.where(jnp.abs(r) >= TILT_STOP, 0.0, rd)
    p, r = jnp.clip(p, -TILT_STOP, TILT_STOP), jnp.clip(r, -TILT_STOP, TILT_STOP)

    vx, vy = vx + ax * dt, vy + ay * dt                      # semi-implicit Euler
    load = (th_p / TAU_MAX) ** 2 + (th_r / TAU_MAX) ** 2
    ns = jnp.array([x + vx * dt, y + vy * dt, vx, vy, r, p, rd, pd, t + dt, 0.0])
    return ns, (g_n, g_x, g_y), load


def tray_gravity(s, a):
    """`(g_n, g_x, g_y)` in the tray frame -- see `swe.effective_gravity`."""
    R = _rotation(s[4], s[5])
    return swe.effective_gravity(R, jnp.array([a[0], a[1], 0.0]), G0)


# --- fluid advance ----------------------------------------------------------
_EDGE = jnp.zeros((NX, NY)).at[0, :].set(1.0).at[-1, :].set(1.0) \
    .at[:, 0].set(1.0).at[:, -1].set(1.0)


def _spill(U):
    """Remove whatever stands above the rim at a wall cell, and report it.

    Liquid that goes over the rim leaves the tray.  Not modelling that was
    wrong twice over: the depth then piles to 0.17 m on a 0.07 m rim, which is
    fiction, and the wave speed `sqrt(g h)` rises with it until the CFL
    condition breaks and the rollout diverges -- the failure that took every
    sampled full-amplitude plan to NaN.  Draining the excess bounds the depth
    by construction and turns the spill cost into the physical quantity, the
    volume lost, rather than a height that stands in for it.

    Only the boundary ring drains: a crest in the middle of the tray is not
    over a wall.
    """
    h = U[..., 0]
    edge = jnp.zeros_like(h).at[0, :].set(1.0).at[-1, :].set(1.0) \
        .at[:, 0].set(1.0).at[:, -1].set(1.0)
    over = jnp.maximum(h - H_WALL, 0.0) * edge
    keep = jnp.where(h > 0, (h - over) / jnp.maximum(h, swe.H_MIN), 1.0)
    # the departing liquid carries its momentum with it
    return U * keep[..., None], jnp.sum(over) * DX * DY


def _edge(g: Grid):
    return jnp.zeros((g.nx, g.ny)).at[0, :].set(1.0).at[-1, :].set(1.0) \
        .at[:, 0].set(1.0).at[:, -1].set(1.0)


def fluid_step(U, s0, u, theta, g: Grid = None):
    """Advance carrier and liquid together across one control step.

    Returns `(U', s', lost)`.  The carrier is integrated at the liquid's
    substep resolution because its acceleration is no longer a constant of the
    control step: it follows the attitude as it lags towards the command and
    the drag as the speed changes.
    """
    g = WORLD if g is None else g
    h0, zeta = theta[0], theta[2]
    gamma = 2.0 * zeta * omega1(h0, K1X)     # modal damping -> momentum drag
    decay = jnp.exp(-gamma * g.dt_sub)
    edge = _edge(g)

    # The substep loop runs on the three variables as separate arrays
    # (`swe_fast`): bit-identical to `swe.step_2d` on the stacked state, and
    # 1.5-1.7x faster, because the stacked layout compiled to gathers,
    # transposes and a size-3 minor axis on every substep.
    def sub(carry, _):
        h, hu, hv, s, lost, load = carry
        s, (g_n, g_x, g_y), ld = carrier_substep(s, u, h, theta, g.dt_sub, g)
        h, hu, hv = swe_fast.step(h, hu, hv, g.dt_sub, g.dx, g.dy, g_n, g_x, g_y)
        hu, hv = hu * decay, hv * decay
        over = jnp.maximum(h - H_WALL, 0.0) * edge
        keep = jnp.where(h > 0, (h - over) / jnp.maximum(h, swe.H_MIN), 1.0)
        return (h * keep, hu * keep, hv * keep, s,
                lost + jnp.sum(over) * g.dx * g.dy, load + ld), None

    (h, hu, hv, s, lost, load), _ = jax.lax.scan(
        sub, (U[..., 0], U[..., 1], U[..., 2], s0, 0.0, 0.0), None, length=g.substeps)
    return jnp.stack([h, hu, hv], axis=-1), s.at[9].set(load / g.substeps), lost


def fluid_step_reference(U, s0, u, theta):
    """The stacked-layout solver on WORLD, kept only to check `fluid_step` against."""
    h0, zeta = theta[0], theta[2]
    decay = jnp.exp(-2.0 * zeta * omega1(h0, K1X) * DT_SUB)

    def sub(carry, _):
        U, s, lost, load = carry
        s, (g_n, g_x, g_y), ld = carrier_substep(s, u, U[..., 0], theta, DT_SUB, WORLD)
        U = swe.step_2d(U, DT_SUB, DX, DY, g_n, g_x, g_y)
        U = U.at[..., 1].multiply(decay).at[..., 2].multiply(decay)
        U, dv = _spill(U)
        return (U, s, lost + dv, load + ld), None

    (U, s, lost, load), _ = jax.lax.scan(sub, (U, s0, 0.0, 0.0), None, length=SUBSTEPS)
    return U, s.at[9].set(load / SUBSTEPS), lost


def reaction(U, theta, g_n):
    """Horizontal force the liquid exerts on the tray, N.

    From the hydrostatic pressure imbalance on the two pairs of walls -- the
    only quantity an estimator watching the stage could see, and therefore the
    observation channel `estimate.py` is allowed to use.
    """
    rho = theta[1]
    h = U[..., 0]
    fx = 0.5 * rho * g_n * (jnp.sum(h[0, :] ** 2) - jnp.sum(h[-1, :] ** 2)) * DY
    fy = 0.5 * rho * g_n * (jnp.sum(h[:, 0] ** 2) - jnp.sum(h[:, -1] ** 2)) * DX
    return jnp.array([fx, fy])


# --- moving a liquid state between grids ---------------------------------------
def _overlap(n_dst, n_src, length):
    """`(n_dst, n_src)` averaging matrix: overlap of each destination cell with
    each source cell, divided by the destination cell's width.  Rows sum to 1,
    so a constant stays constant; columns weighted by cell width sum to the
    source width, so the total is conserved exactly."""
    ed = jnp.linspace(0.0, length, n_dst + 1)
    es = jnp.linspace(0.0, length, n_src + 1)
    lo = jnp.maximum(ed[:-1, None], es[None, :-1])
    hi = jnp.minimum(ed[1:, None], es[None, 1:])
    return jnp.maximum(hi - lo, 0.0) / (length / n_dst)


def restrict(U, src: Grid, dst: Grid):
    """A liquid state on `src` re-expressed on `dst`, conserving mass and momentum.

    This is how the planner, which runs on a coarser grid than the world, is
    handed the world's current liquid: cell averages of depth and momentum,
    area-weighted, so the planner starts from the same volume of liquid moving
    with the same total momentum -- only resolved more coarsely.
    """
    if src == dst:
        return U
    Ax = _overlap(dst.nx, src.nx, LX)
    Ay = _overlap(dst.ny, src.ny, LY)
    return jnp.einsum("ai,ijc,bj->abc", Ax, U, Ay)


# --- cost rows --------------------------------------------------------------
def wave_energy(U, theta, g: Grid = None):
    """Slosh energy, normalised so that 1.0 is a wave filling the freeboard."""
    h0, rho = theta[0], theta[1]
    h = U[..., 0]
    hu, hv = U[..., 1], U[..., 2]
    ke = 0.5 * (hu ** 2 + hv ** 2) / jnp.maximum(h, swe.H_MIN)
    pe = 0.5 * G0 * (h - h0) ** 2
    g = WORLD if g is None else g
    e = rho * g.dx * g.dy * jnp.sum(ke + pe)
    ref = 0.5 * rho * G0 * (H_WALL - h0) ** 2 * LX * LY
    return e / jnp.maximum(ref, 1e-9)


def stage_cost(U, s, u, theta, goal, lost, last=1.0, g: Grid = None):
    """`(7,)` of per-step cost rows, before the fixed weights.

    Rows rather than a scalar because every diagnostic in this project reads
    the violation profile, and one number cannot say which constraint the
    hypotheses disagree about.
    """
    h0 = theta[0]
    r_goal = jnp.sum((s[:2] - goal) ** 2)

    # spill: the fraction of the liquid that went over the rim during this
    # control step.  A physical output of the solver, not a proxy for one --
    # which is the whole reason for simulating the liquid instead of hanging a
    # pendulum in it.
    r_spill = lost / (LX * LY * h0)

    # smooth barrier: how far into the last centimetre below the rim the
    # liquid has reached at any wall cell
    h = U[..., 0]
    wall = jnp.concatenate([h[0, :], h[-1, :], h[:, 0], h[:, -1]])
    enc = jnp.maximum(0.0, jnp.max(wall) - (H_WALL - FREEBOARD_BAND))
    r_free = (enc / FREEBOARD_BAND) ** 2

    # `last` is 1.0 on the final step of the horizon and 0.0 before it when
    # SLOSH_TERMINAL is set, so the wave is scored on arrival rather than
    # rewarded for never being raised.
    r_slosh = wave_energy(U, theta, g) * (last if SLOSH_TERMINAL else 1.0)

    # late: past the deadline, distance beyond the arrival radius is expensive.
    # Before it this row is exactly zero, so it does not compete with the goal
    # row during the carry -- it only prices taking too long.
    dist = jnp.linalg.norm(s[:2] - goal)
    r_late = jnp.where(s[8] >= T_ARRIVE,
                       jnp.maximum(0.0, dist - R_ARRIVE) ** 2, 0.0)

    tilt = jnp.sqrt(s[4] ** 2 + s[5] ** 2)
    r_tilt = jnp.maximum(0.0, tilt - TILT_MAX) ** 2
    spd = jnp.sqrt(s[2] ** 2 + s[3] ** 2)
    r_vel = (jnp.maximum(0.0, spd - V_MAX) / V_MAX) ** 2 + \
        (jnp.maximum(0.0, jnp.sqrt(s[6] ** 2 + s[7] ** 2) - W_MAX) / W_MAX) ** 2
    r_u = jnp.sum(jnp.clip(u, -1.0, 1.0) ** 2)

    # time: one unit for every control step the tray is not yet delivered.
    # This is what makes over-caution cost something.  The goal row is a
    # *squared* distance, so lagging a few centimetres behind near the goal is
    # almost free -- 0.05 m for ten steps is 0.25 -- and the deadline was easy
    # for a cautious plan to keep; planning for the fullest tray therefore cost
    # only +2.18 over the oracle while planning too boldly cost +8.8.  A cost
    # can never see what a planner assumed, only what the assumption did, and
    # what caution does is arrive late.  Charged every step, so unlike the
    # terminal slosh row there is no receding instant to game; softened over a
    # few millimetres so that nearby samples are ranked rather than tied.
    r_time = jax.nn.sigmoid((dist - R_ARRIVE) / TIME_SOFT)
    # torque: what the hand spent holding the tray against the liquid's weight
    # and against its own commands over this control step
    r_torque = s[9]
    return jnp.array([r_goal, r_spill, r_free, r_slosh, r_late, r_tilt,
                      r_vel, r_u, r_time, r_torque])


def cost_weights(w: Costs):
    return jnp.array([w.goal, w.spill, w.freeboard, w.slosh, w.late, w.tilt,
                      w.vel, w.effort, w.time, w.torque])


# --- plans ------------------------------------------------------------------
def node2u(U_nodes):
    """`(N_NODE * NU,) -> (H, NU)` by linear interpolation in time."""
    nodes = U_nodes.reshape(N_NODE, NU)
    tn = jnp.linspace(0.0, 1.0, N_NODE)
    ts = jnp.linspace(0.0, 1.0, H)
    return jax.vmap(lambda col: jnp.interp(ts, tn, col),
                    in_axes=1, out_axes=1)(nodes)


def shift_nodes(U_nodes):
    """Warm start: the same profile read one control step later."""
    nodes = U_nodes.reshape(N_NODE, NU)
    tn = jnp.linspace(0.0, 1.0, N_NODE)
    return jax.vmap(lambda col: jnp.interp(jnp.clip(tn + 1.0 / H, 0.0, 1.0),
                                           tn, col),
                    in_axes=1, out_axes=1)(nodes).reshape(-1)


# --- rollout ----------------------------------------------------------------
def make_rollout_state(w: Costs = Costs(), g: Grid = None):
    """`(stage0, fluid0, U_nodes, theta, goal) -> rows`, from a *given* liquid state.

    This is the rollout closed-loop planning must use.  The liquid is part of
    the state, not a parameter: once the carry starts, what the tray holds is
    whatever the previous commands did to it, and no function of theta alone
    can reconstruct that.
    """
    g = WORLD if g is None else g
    wv = cost_weights(w)

    def rollout_rows(s0, fluid, U_nodes, theta, goal):
        seq = node2u(U_nodes)

        # the scan carries the "is this the last step" flag alongside the
        # command, which is what lets the slosh row be terminal
        last = jnp.zeros(H).at[-1].set(1.0)

        def body(carry, xs):
            u, lt = xs
            s, U = carry
            nU, ns, lost = fluid_step(U, s, u, theta, g)
            return (ns, nU), stage_cost(nU, ns, u, theta, goal, lost, lt, g)

        (_, _), rows = jax.lax.scan(body, (s0, fluid), (seq, last))
        return rows.sum(0)

    def rollout_cost(s0, fluid, U_nodes, theta, goal):
        return jnp.dot(wv, rollout_rows(s0, fluid, U_nodes, theta, goal))

    return rollout_rows, rollout_cost


def make_rollout(w: Costs = Costs()):
    """`(stage0, U_nodes, theta, goal) -> rows`, liquid starting at `fluid_init`.

    **Valid only where the liquid really is in its initial state** -- an
    open-loop plan from rest, as in `task_check.py` and `rows.py`.  In closed
    loop it is wrong, and it was used there: every planning call reset the
    liquid to the t=0 wave while the world held what the carry had done to it.
    Measured on an oracle run, the planner assumed a wave energy of 0.003 and a
    wall level at 54% of the rim while the tray actually held 0.09-0.54 and
    75-83% -- 30-180x the energy and 1.8 cm less freeboard than it planned
    against.  Giving the oracle the true liquid state instead cut its cost
    87.15 -> 74.79 (+12.36 +- 3.97 for the bug).  This bug, together with the
    terminal slosh row (see SLOSH_TERMINAL), is why the planner that knew theta
    exactly lost to one averaging eight wrong guesses: the averaged plan was
    more conservative and so accidentally covered the wave neither could see.
    """
    rows_s, _ = make_rollout_state(w)
    wv = cost_weights(w)

    def rollout_rows(s0, U_nodes, theta, goal):
        return rows_s(s0, fluid_init(theta), U_nodes, theta, goal)

    def rollout_cost(s0, U_nodes, theta, goal):
        return jnp.dot(wv, rollout_rows(s0, U_nodes, theta, goal))

    return rollout_rows, rollout_cost


def make_cost_matrix(w: Costs = Costs()):
    """`(stage0, V (K,DIM_U), theta (N,DIM_THETA), goal) -> c (K, N)`."""
    rows_fn, cost_fn = make_rollout(w)
    over_samples = jax.vmap(cost_fn, in_axes=(None, 0, None, None))
    over_theta = jax.vmap(over_samples, in_axes=(None, None, 0, None), out_axes=1)
    return jax.jit(over_theta), jax.jit(rows_fn)
