"""7-DOF arm with an unknown payload: the plant, batched over hypotheses.

    theta = (m, c_x, c_y, c_z, mu)
      m            payload mass, kg
      c_x,c_y,c_z  payload centre of mass in the grasp frame, m
      mu           grasp friction coefficient

**Why the payload sits 0.20 m from the grasp.**  At the 0.05 m first tried, the
offset did not reach the cost at all -- the torque row was exactly zero for
every hypothesis -- and with only `(m, mu)` left the hypotheses stopped
disagreeing: fifteen of sixteen picked the same best plan out of 1024 and their
cost rankings correlated at 0.963.  The reason is visible in the slip term,
`(max(0, F_t - mu F_n) / m g)^2` with `F = m(a - g)`: the mass cancels, so `m`
barely enters and `mu` only moves a threshold.  Every hypothesis then gives the
same advice at a different strength, and averaging advice of the same sign is
something a first-order mixture already does perfectly.

A lateral centre-of-mass offset is the axis that is missing from that: it tips
the payload one way or the other, so two hypotheses ask for *opposite* wrist
corrections rather than the same one harder.  That is the case the method is
for, and it needs a lever to exist.

The shape this has to have
--------------------------
Belief-conditioned data needs `c(U + eps, theta)` as a `(K, N_theta)` matrix:
`K` perturbed plans against `N_theta` hypotheses, all from one batch.  A Brax
environment class is the wrong object for that -- it carries one system and one
state -- so the plant is functional and vmapped twice, over samples and over
hypotheses, with the *model* batched on the hypothesis axis.  `theta` enters
through `body_mass` and `body_ipos`, which are ordinary leaves of the MJX
`Model` pytree, so `jax.vmap` over a stacked model is all it takes.

`mu` never reaches the simulator.  The payload is rigidly attached, so there is
no contact to hang a friction coefficient on; slip is scored from the wrench
the attachment has to carry, which is available analytically and costs nothing:

    F_grasp = m (a_payload - g)        in world
    F_n     = component along the grasp frame's approach axis
    F_t     = the perpendicular part
    slip    = max(0, |F_t| - mu F_n)

Putting a real contact there would add a solver iteration per step and a
stochastic failure mode to a study whose subject is parameter uncertainty.

Control
-------
`u` is a joint *velocity* command in `[-1, 1]`, scaled by the Franka velocity
limits and realised by a PD whose torque is clipped at the real per-joint
torque limits.  A payload too heavy for a commanded motion therefore shows up
as saturation and tracking error, which is what a torque-limit cost should
measure, rather than as a command the model silently executes.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dial_mpc.utils.io_utils import get_model_path

# --- horizon and control ----------------------------------------------------
DT = 0.05                     # control period, s
SUBSTEPS = 25                 # physics substeps per control step (timestep 0.002)
H = 16                        # plan length -> 0.8 s
NU = 7

# --- what a physics step is allowed to cost ---------------------------------
# The whole cost of this method is `K * M * H * substeps * levels` physics
# steps, so every factor is linear and worth measuring.  Measured at
# 1024 plans x 8 hypotheses, one annealing level, against the stock settings:
#
#   iterations 100 -> 4, ls 50 -> 8     3.7x    cost err 3.6e-06   (identical)
#   iterations 100 -> 2, ls 50 -> 4     6.9x    cost err 1.1e-04
#   the above + planner substeps 10    17.1x    cost err 4.1e-03
#
# The stock 100 Newton iterations with 50 line-search steps exist for scenes
# with real contact.  This one has none -- every geom is `contype=0` -- and its
# only constraint rows are the seven dof-friction rows from `frictionloss`,
# which converge to nine decimal places in **one** iteration.  CPU MuJoCo hides
# the waste by exiting early on the convergence test; MJX runs the trip count
# literally on the GPU, so the stock value was paying 50x for nothing.  This
# was the entire bottleneck: 115x slower than real time in the live viewer.
#
# Not a knob, measured and rejected: computing the gravity-compensation bias
# through the minimal `kinematics -> com_pos -> crb -> com_vel -> rne` chain
# instead of a full `mjx.forward` on the payload-free model.  It is bit-exact
# (max abs diff 0.0) and exactly 0% faster -- XLA already dead-codes the
# constraint solve whose result is never read.
SOLVER_ITERS = 2
SOLVER_LS = 4

# The planner integrates coarser than the world does.  This is a statement
# about the planner's internal model, not about the plant: `make_world` keeps
# `SUBSTEPS`, so every number this project reports is still measured on the
# 0.002 s plant and the only question is whether a 0.005 s internal model still
# *ranks* plans correctly -- which is the kind of model error robust MPPI is
# supposed to absorb, and which is checked closed-loop rather than asserted.
#
# 10 and not 5.  At 0.01 s the implicitfast integrator on this arm falls off a
# cliff: the relative cost error jumps 4.1e-03 -> 2.98e-01 and the worst-case
# final joint angle moves 0.70 rad.  Below about 12 substeps there is also no
# throughput left to win -- measured 30.1M physics steps/s at 10 substeps
# against 27.7M at 5, i.e. the dispatch stops being GPU-bound and the step
# count no longer buys anything.
PLAN_SUBSTEPS = 10
# The plan is a few spline **nodes**, not a value per step.  Perturbing all
# 16 x 7 = 112 step values with iid noise essentially never proposes a
# coordinated motion: "hold joint 1 at 0.6 for the whole horizon" is one
# direction out of 112 and the sampler does not find it.  Measured -- a
# constant joint-1 command took the open-loop cost from 1.70 to 0.55 while
# closed-loop MPPI over the raw 112 dimensions covered 15% of the distance.
# Nodes fix both halves: 5 x 7 = 35 dimensions, and every sample is smooth by
# construction.  This is what DIAL does everywhere else in this repository.
N_NODE = 5
DIM_U = N_NODE * NU
DIM_THETA = 5

# Franka Panda limits, used as cost thresholds rather than as hard clamps
QDOT_MAX = jnp.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
TAU_MAX = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
Q_MAX = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
Q_MIN = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
KV = jnp.array([40.0, 40.0, 40.0, 40.0, 8.0, 8.0, 8.0])     # velocity-PD gain

# --- the task ----------------------------------------------------------------
# Found by search over joint space for a pose that holds the payload level with
# margin on the wrist joints.  The model's own keyframe put the grasp frame's
# z axis nearly horizontal, so the payload started 82 degrees from level, the
# tilt row sat at 18.6 of a total near 280, and no plan could move it -- a
# constant that swamped every row `theta` does affect.  Here the tilt is 7.7
# degrees and the row falls to 1.16.
Q_CARRY = jnp.array([-1.037, 0.877, 2.262, -1.138, 0.551, 1.847, 0.532])
P_START = jnp.array([0.05, -0.477, 0.721])
# 0.35 m, not the 0.95 m first tried.  The planner's horizon is 0.8 s and the
# payload sits at a 0.48 m radius, so 0.95 m of lateral transport is about pi
# radians of joint 1 -- 1.44 s at the Franka's speed limit, comfortably outside
# anything a plan can see.  MPPI then optimises a goal row that is nearly flat
# across every sampled plan, the effort row wins, and the arm covers 5% of the
# distance while slip and tilt stay at exactly zero.  A distance the horizon
# can actually cover is not a concession: a shorter, faster move raises the
# accelerations, which is what makes slip and the wrist torque -- the two rows
# `theta` moves -- bite at all.
# A swing at constant radius -- what joint 1 does -- not a translation toward
# the base.  The first goal was `start + 0.35 m in y`, which drops the payload's
# radius from 0.48 m to 0.14 m and asks the arm to retract; the oracle moved
# *away* from it.  0.7 rad of joint 1 covers 0.33 m in 0.32 s, well inside the
# 0.8 s horizon.
GOAL = jnp.array([0.346, -0.333, 0.721])    # 0.33 m, a 0.7 rad swing

# The annealing schedule and its temperatures, measured together.  A single
# temperature cannot serve both levels: the cost spread across the proposal
# cloud is 19.8 at sigma 0.5 and 3.6 at sigma 0.25, so one lambda leaves the
# coarse level lukewarm and the fine level at 99% effective sample size --
# pure noise.  This project measured the same thing on the walking task, where
# a fixed temperature spanned a factor of 84 across levels.
#
# Five levels, not two, and the reason is a floor the two-level schedule put
# under every experiment.  In command units `u = 1` is joint 1 at 2.175 rad/s
# and the payload sits at a 0.48 m radius, so the old finest level, `sigma =
# 0.25`, proposes about 0.2 m of displacement over the 0.8 s horizon.  Asked to
# correct a 5 cm error, every sampled plan overshoots in some direction, the
# softmax sees them as equally good, and the weighted average of symmetric
# noise is approximately no update -- the planner stops improving well short of
# the goal, and *more* optimisation passes make it worse (0.065 -> 0.107 at four
# passes with a planner that knows theta exactly), which is the signature of a
# proposal distribution that cannot express the correction rather than of an
# optimiser that has converged.
#
# Measured under one setting (16 episodes x 30 control steps, 1024 plans, M = 8
# drawn from the prior, identical thetas, fixed lambda):
#
#   (0.5, 0.25)                    gap 0.0953   worst 0.1661
#   (0.5, 0.25, 0.10, 0.03)        gap 0.0768   worst 0.1272
#   (0.5, 0.25, 0.12, 0.06, 0.03)  gap 0.0590   worst 0.0922   <- this one
#
# The temperatures are solved, not guessed: at each level and each control step
# lambda is bisected so the proposal cloud hits a target effective sample size,
# and the geometric mean over episodes and steps is taken.  The target is itself
# measured against the task rather than assumed -- the project's usual 25% gives
# 0.0631 here and 50% gives 0.0590, so 50% it is.  One temperature cannot serve
# the ladder: these span a factor of 3700.
SIGMA_SCHEDULE = (0.5, 0.25, 0.12, 0.06, 0.03)
LAM_SCHEDULE = (12.556, 0.5798, 0.0381, 0.0101, 0.0034)   # ~50% ESS per level

# --- the unknown parameters -------------------------------------------------
THETA_LO = jnp.array([0.2, -0.06, -0.06, -0.06, 0.2])
THETA_HI = jnp.array([2.5, 0.06, 0.06, 0.06, 0.8])

# --- cost weights (fixed problem weights, not the belief) -------------------
@dataclass(frozen=True)
class Costs:
    """Fixed problem weights -- not the belief.

    The spec's numbers assume rows of comparable scale; these rows are not, so
    the weights are set from the measured contributions instead.  The split is
    deliberate and follows what each row is *for*:

      always-active rows (goal, slip, effort) are balanced by their measured
        magnitude, so the objective is a genuine trade rather than one term
        with six decorations.  Left at the spec's numbers the slip row
        contributes 447 against the goal row's 9.8 and the arm simply refuses
        to move -- hypotheses disagree beautifully about a task nobody does.

      constraint rows (tau, jlim, vel, tilt) keep large weights.  They are
        hinges that sit at zero almost always, and a term that is zero until
        it is violated is *supposed* to be expensive when it is; scaling one
        by its own spread would make a rare violation cheap, which is the
        opposite of what a limit means.
    """
    goal: float = 1.0
    tau: float = 50.0
    slip: float = 2.0
    tilt: float = 20.0
    jlim: float = 50.0
    vel: float = 10.0
    effort: float = 0.01
    tilt_max: float = 0.35        # rad, payload tilt from vertical


ROW_NAMES = ("goal", "tau", "slip", "tilt", "jlim", "vel", "effort")


def load():
    """`(mjx.Model, ids)` -- the nominal model and the indices the costs need."""
    path = get_model_path("panda7", "panda7.xml")
    mj = mujoco.MjModel.from_xml_path(path.as_posix())
    mj.opt.timestep = DT / SUBSTEPS
    # See SOLVER_ITERS: MJX does not early-exit the solver, so the stock
    # contact-scene trip counts were 50x of this plant's actual need.
    mj.opt.iterations = SOLVER_ITERS
    mj.opt.ls_iterations = SOLVER_LS
    ids = {
        "payload": mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY.value, "payload"),
        "grasp": mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY.value, "grasp"),
        "grasp_site": mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE.value,
                                        "grasp_site"),
    }
    model = mjx.put_model(mj)
    # The same model with the payload weightless: what the arm's own inner loop
    # knows about gravity.  Everything `theta` does to the dynamics is then the
    # part of gravity nobody cancelled.
    bare = model.tree_replace(
        {"body_mass": model.body_mass.at[ids["payload"]].set(0.0)})
    return model, bare, ids, mj


def apply_theta(model, ids, theta):
    """The model this hypothesis implies.  `theta` may be batched on axis 0."""
    m, c = theta[..., 0], theta[..., 1:4]
    mass = model.body_mass.at[ids["payload"]].set(m)
    # The offset goes on `body_pos`, not `body_ipos`.  The payload has no joint
    # of its own, so MuJoCo welds it to link 7 and never reads the welded
    # body's `ipos` in kinematics -- writing there moves nothing, which cost an
    # hour to notice because the mass field right next to it does work.  The
    # payload's own `ipos` is zero, so its body origin *is* its centre of mass
    # and `body_pos` is exactly the quantity `theta` names.
    base = model.body_pos[ids["payload"]]
    pos = model.body_pos.at[ids["payload"]].set(base + c)
    # A heavier payload is also a bigger one: inertia scales with the mass at
    # fixed density, and leaving it fixed would make heavy hypotheses
    # unrealistically easy to rotate -- the wrist torque cost is one of the
    # terms that is supposed to separate them.
    inert = model.body_inertia.at[ids["payload"]].set(
        jnp.broadcast_to((0.004 * m / 1.0)[..., None], (3,) if m.ndim == 0
                         else m.shape + (3,)))
    return model.tree_replace({"body_mass": mass, "body_pos": pos,
                               "body_inertia": inert})


def act2tau(u, qd, bias):
    """Velocity command in `[-1, 1]` to joint torque.

    Gravity compensation plus a velocity PD, clipped at the real torque limits
    -- what the inner loop of a velocity-controlled arm actually does.

    **The compensation uses the arm's own weight and not the payload's**, which
    is both physically right (the robot knows its own links, not what it has
    picked up) and the cleanest place for `theta` to enter: the payload is
    exactly the part of gravity that is not cancelled, so a heavier one droops
    further and an offset one droops sideways.

    Without it the arm could not hold still at all.  Measured: commanding zero
    velocity, the payload sagged to 30 degrees within the 0.8 s horizon, the
    tilt row sat at 10.0 of a total cost of 12.3 and barely moved between
    plans, and the goal term -- the one that makes the arm do the task -- was
    competing against a constant it could not influence.
    """
    v_cmd = QDOT_MAX * jnp.clip(u, -1.0, 1.0)
    return jnp.clip(bias + KV * (v_cmd - qd), -TAU_MAX, TAU_MAX)


def node2u(U: Array) -> Array:
    """`(N_NODE * NU,) -> (H, NU)` by linear interpolation in time.

    The first interpolated command equals the first node, so `U[:NU]` is still
    the action to apply this step.
    """
    nodes = U.reshape(N_NODE, NU)
    tn = jnp.linspace(0.0, 1.0, N_NODE)
    ts = jnp.linspace(0.0, 1.0, H)
    return jax.vmap(lambda col: jnp.interp(ts, tn, col),
                    in_axes=1, out_axes=1)(nodes)


def shift_nodes(U: Array) -> Array:
    """Warm start: the same command profile read one control step later.

    Dropping the first node and appending a zero -- the right move when the
    plan is one value per step -- would throw away three quarters of the
    horizon here, since five nodes span sixteen steps.
    """
    nodes = U.reshape(N_NODE, NU)
    tn = jnp.linspace(0.0, 1.0, N_NODE)
    adv = 1.0 / H
    return jax.vmap(lambda col: jnp.interp(jnp.clip(tn + adv, 0.0, 1.0), tn, col),
                    in_axes=1, out_axes=1)(nodes).reshape(-1)


def _grasp_frame(data, ids):
    """Approach axis of the grasp frame, in world."""
    R = data.xmat[ids["grasp"]].reshape(3, 3)
    return R[:, 2]                      # the frame's z axis


def plan_model(model):
    """The same plant integrated at the planner's coarser timestep.

    Only the integration step changes -- same bodies, same masses, same limits
    -- so `apply_theta` and the cost rows are unaffected and a hypothesis means
    the same thing to the planner as to the world.
    """
    return model.tree_replace({"opt.timestep": DT / PLAN_SUBSTEPS})


def step(model, bare, ids, data, u, substeps: int = SUBSTEPS):
    """One control step: gravity-compensated velocity PD across the substeps.

    `substeps` must match `model.opt.timestep`: the caller owns that pairing,
    because the planner and the world deliberately differ (`plan_model` and
    `PLAN_SUBSTEPS`) and silently defaulting one of the two would change the
    control period rather than the integration accuracy.

    The compensation is computed **once per control step** and held across the
    substeps.  Recomputing it every substep is the more faithful model of a
    1 kHz inner loop, and it was unaffordable: an extra `mjx.forward` inside a
    25-deep scan, itself inside a 16-deep scan, vmapped over thousands of
    rollouts, put XLA compilation into the tens of minutes with the GPU idle --
    and since compilation holds the interpreter lock, it froze the viewer's
    render thread too.  Over 50 ms the arm's configuration barely moves, so the
    held bias is a good approximation of the recomputed one.
    """
    bias = mjx.forward(bare, data).qfrc_bias

    def one(d, _):
        return mjx.step(model, d.replace(ctrl=act2tau(u, d.qvel, bias))), None

    data, _ = jax.lax.scan(one, data, None, length=substeps)
    return data, bias


def stage_cost(model, ids, data, prev, u, theta, goal, bias, w: Costs):
    """`(7,)` of per-step cost rows, before the fixed weights are applied.

    Rows rather than a scalar because every diagnostic in this project reads
    the violation profile, and a single number cannot say which constraint the
    hypotheses disagree about.
    """
    m, mu = theta[0], theta[4]
    p = data.xpos[ids["payload"]]
    q, qd = data.qpos, data.qvel
    tau = act2tau(u, qd, bias)

    # goal: payload position, not the flange -- the payload is what is being
    # transported and its offset from the flange is part of theta
    r_goal = jnp.sum((p - goal) ** 2)

    r_tau = jnp.sum((jnp.maximum(0.0, jnp.abs(tau) - TAU_MAX) / TAU_MAX) ** 2)

    # the wrench the rigid attachment has to carry
    a = (data.cvel[ids["payload"], 3:] - prev) / DT          # linear accel
    F = m * (a - jnp.array([0.0, 0.0, -9.81]))
    n = _grasp_frame(data, ids)
    Fn = jnp.dot(F, n)
    Ft = jnp.linalg.norm(F - Fn * n)
    # Normalised by the payload's own weight, so the row is dimensionless and
    # means "how many payload-weights of tangential force the grasp cannot
    # hold".  Unnormalised it is newtons squared, which for a 2.5 kg payload
    # reaches 4000 against a goal row of 10, and the fixed weight of 200 then
    # puts slip at 788,000 -- the softmax sees one term and the other six are
    # decoration.  Row scales have to be equalised before a temperature can be
    # chosen; that is a standing rule on this project and it applies per row
    # here exactly as it does per basis weight elsewhere.
    r_slip = (jnp.maximum(0.0, Ft - mu * jnp.maximum(Fn, 0.0))
              / (m * 9.81)) ** 2

    # payload tilt away from vertical
    tilt = jnp.arccos(jnp.clip(jnp.dot(n, jnp.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    r_tilt = jnp.maximum(0.0, jnp.abs(tilt) - w.tilt_max) ** 2

    r_jlim = jnp.sum(jnp.maximum(0.0, q - Q_MAX) ** 2
                     + jnp.maximum(0.0, Q_MIN - q) ** 2)
    r_vel = jnp.sum((jnp.maximum(0.0, jnp.abs(qd) - QDOT_MAX) / QDOT_MAX) ** 2)
    r_u = jnp.sum(jnp.clip(u, -1.0, 1.0) ** 2)
    return jnp.array([r_goal, r_tau, r_slip, r_tilt, r_jlim, r_vel, r_u])


def cost_weights(w: Costs):
    return jnp.array([w.goal, w.tau, w.slip, w.tilt, w.jlim, w.vel, w.effort])


def make_rollout(model, bare, ids, w: Costs = Costs(),
                 substeps: int = PLAN_SUBSTEPS):
    """`(x0, U, theta, goal) -> (7,)` cost rows summed over the horizon.

    Integrates at the **planner's** timestep, not the world's.  Pass
    `substeps=SUBSTEPS` to roll at world fidelity, which is what the equivalence
    check does.
    """
    wv = cost_weights(w)
    plan = plan_model(model) if substeps != SUBSTEPS else model

    def rollout_rows(data0, U, theta, goal):
        mdl = apply_theta(plan, ids, theta)
        d0 = mjx.forward(mdl, data0)
        v0 = d0.cvel[ids["payload"], 3:]

        def body(carry, u):
            d, prev = carry
            nd, bias = step(mdl, bare, ids, d, u, substeps)
            rows = stage_cost(mdl, ids, nd, prev, u, theta, goal, bias, w)
            return (nd, nd.cvel[ids["payload"], 3:]), rows

        (_, _), rows = jax.lax.scan(body, (d0, v0), node2u(U))
        return rows.sum(0)

    def rollout_cost(data0, U, theta, goal):
        return jnp.dot(wv, rollout_rows(data0, U, theta, goal))

    return rollout_rows, rollout_cost


def make_cost_matrix(model, bare, ids, w: Costs = Costs(),
                     substeps: int = PLAN_SUBSTEPS):
    """`(x0, V (K,DIM_U), theta (N,5), goal) -> c (K, N)`, the spec's `c`.

    Vmapped over hypotheses on the outside and samples on the inside, so the
    model is rebuilt once per hypothesis rather than once per rollout.
    """
    rows_fn, cost_fn = make_rollout(model, bare, ids, w, substeps)
    over_samples = jax.vmap(cost_fn, in_axes=(None, 0, None, None))
    over_theta = jax.vmap(over_samples, in_axes=(None, None, 0, None),
                          out_axes=1)
    return jax.jit(over_theta), jax.jit(rows_fn)


def sample_theta(key, n):
    """Uniform over the box.  `m` log-uniform, which the spec allows and which
    matters because a factor of twelve in mass is not uniform in difficulty."""
    k1, k2 = jax.random.split(key)
    lm = jax.random.uniform(k1, (n,), minval=jnp.log(THETA_LO[0]),
                            maxval=jnp.log(THETA_HI[0]))
    rest = jax.random.uniform(k2, (n, 4), minval=THETA_LO[1:],
                              maxval=THETA_HI[1:])
    return jnp.concatenate([jnp.exp(lm)[:, None], rest], axis=-1)
