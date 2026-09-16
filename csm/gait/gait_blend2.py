"""Composition across every pair of gaits, on the policy that holds all four.

The fields were fitted at the pure rows only.  For each pair (i, j) the weight
`a*e_i + (1-a)*e_j` is solved to a coefficient per omega and the two networks
are combined linearly -- one matrix solve, no planner, no retraining.  DIAL is
run at the same mixed weight as the reference for what the objective actually
asks for.

Three contact correlations span the gait space: diagonal pairs (trot's
signature), lateral pairs (pace's), front/hind pairs (bound's).  Each pure gait
is high on one and negative on the other two; walk, four-beat, is negative on
all three.  So a blend is "in between" when its three-vector moves
continuously from one endpoint's to the other's.

A rollout counts as collapsed only if `done` holds for 60 consecutive steps or
the torso ends below 0.15 m -- `done` is a per-step dip flag, not a death.
"""
import sys, functools, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy
from csm.compose_walk_eval import make_student
from csm.dial_lean import make_dial_step
from dial_mpc.core.dial_core import make_controller
from brax import math
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

flush = functools.partial(print, flush=True)
POLICY = sys.argv[1]
T = 0.15
CMD = (0.8, 0.0, 0.0)
WIN, STEPS = 60, 170
RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0]
PAIRS = [(1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3)]
DIAL_SEEDS = jnp.asarray([0, 1, 2])

dc0, ec = _load_config("unitree_go2_gait", None)
dc = dataclasses.replace(dc0, temp_sample=T)
env = brax_envs.get_environment(dc.env_name, config=ec)
policy = ComposedDialScorePolicy.load(POLICY)
reset = jax.jit(env.reset)


def triple(contact):
    w = contact[WIN:]
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return float(np.corrcoef(w[:, i], w[:, j])[0, 1])
    return (np.nanmean([cor(0, 3), cor(1, 2)]),    # diagonal  (trot)
            np.nanmean([cor(0, 2), cor(1, 3)]),    # lateral   (pace)
            np.nanmean([cor(0, 1), cor(2, 3)]))    # front/hind(bound)


def collapsed(done, z):
    d = done > 0.5; run = best = 0
    for v in d:
        run = run + 1 if v else 0; best = max(best, run)
    return best >= 60 or z[-20:].mean() < 0.15


def record(st):
    ps = st.pipeline_state; ti = env._torso_idx - 1
    zf = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
    return ((zf < 0.02).astype(jnp.float32), jnp.linalg.norm(ps.xd.vel[ti][:2]),
            st.done, ps.x.pos[ti, 2],
            math.quat_to_euler(ps.x.rot[ti])[2])


student = make_student(env, policy, dc, init_passes=5, n_steps=STEPS,
                       record=record, step_passes=6)
mbdpi = make_controller(dc, env)
ctrl = make_dial_step(env, mbdpi, dc, std_normalize=False,
                      level_scales=(1.0,) * dc.Ndiffuse)


def teacher(seed, omega):
    st = env.reset(jax.random.PRNGKey(seed))
    st = set_command(env, st, jnp.asarray(CMD, jnp.float32))
    info = dict(st.info); info["reward_weights"] = omega
    st = st.replace(info=info)
    plan = jnp.zeros((dc.Hnode + 1, int(env.action_size)))
    def body(carry, _):
        s, k, p = carry
        s, k, p = ctrl(s, k, p)
        return (s, k, p), record(s)
    return jax.lax.scan(body, (st, jax.random.PRNGKey(seed + 100), plan),
                        None, length=STEPS)[1]


teacher_batch = jax.jit(jax.vmap(teacher, in_axes=(0, None)))

flush(f"blend on {POLICY}  command {CMD}  T={T}  {STEPS} steps (read after {WIN})")
flush("columns: diag / lat / f-h contact correlations, v/cmd, X = collapsed")
for i, j in PAIRS:
    gi, gj = GAIT_NAMES[i], GAIT_NAMES[j]
    flush(f"\n=== {gi}(e{i}) -> {gj}(e{j}) ===")
    flush(f"{'ratio':>10}  {'CSM diag':>8}{'lat':>6}{'f-h':>6}{'v':>6}{'drift':>7}{'':>3}"
          f"   {'DIAL diag':>9}{'lat':>6}{'f-h':>6}{'v':>6}{'drift':>7}{'':>3}   coeff")
    for a in RATIOS:
        omega = np.zeros(4, np.float32); omega[i] = a; omega[j] = 1 - a
        omega /= np.linalg.norm(omega); om = jnp.asarray(omega)
        coeff = np.asarray(policy.coefficients(om))

        deg = lambda y: np.degrees(np.unwrap(y)[-1] - np.unwrap(y)[WIN])
        st = set_command(env, reset(jax.random.PRNGKey(0)), CMD)
        c, spd, done, z, yaw = [np.asarray(v) for v in student(st, om, T)]
        sd, sl, sf = triple(c); sv = spd[WIN:].mean() / CMD[0]
        sdr = deg(yaw)
        sx = "X" if collapsed(done, z) else ""

        c, spd, done, z, yaw = [np.asarray(v) for v in teacher_batch(DIAL_SEEDS, om)]
        tt = np.array([triple(c[k]) for k in range(len(DIAL_SEEDS))])
        td, tl, tf = np.nanmean(tt, axis=0); tv = spd[:, WIN:].mean() / CMD[0]
        tdr = np.mean([deg(yaw[k]) for k in range(len(DIAL_SEEDS))])
        tx = "X" if any(collapsed(done[k], z[k]) for k in range(len(DIAL_SEEDS))) else ""

        flush(f"{f'{a:.2f}:{1-a:.2f}':>10}  {sd:8.2f}{sl:6.2f}{sf:6.2f}{sv:6.2f}{sdr:+7.0f}{sx:>3}"
              f"   {td:9.2f}{tl:6.2f}{tf:6.2f}{tv:6.2f}{tdr:+7.0f}{tx:>3}   "
              f"{np.round(coeff[[i, j]], 2)}")
flush("BLEND2DONE")
