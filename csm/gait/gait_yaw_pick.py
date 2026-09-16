"""Pick the heading weight, with DIAL, on the objective that now has one.

The gait objective never mentioned which way the robot pointed.  DIAL stayed
roughly straight anyway -- the symmetry of a 2049-sample cloud -- but the
distilled fields had no such symmetry and each turned its own way, up to
0.34 rad/s on a command of exactly zero yaw.  `reward_yaw` is now in the
tracking floor at weight `yaw_weight`, and `yaw_tar` advances at the commanded
rate, so the same term holds a line and follows a turn.

This sweeps that weight under raw Gibbs -- the convention the labels use --
and reports what the old screens never did: heading drift on the straight
commands and achieved-vs-commanded yaw rate on the turning ones.  A weight too
small leaves the drift; too large and the robot buys heading with its gait.

yaw_weight 0 is included as the control: it is the objective as it was, so the
drift column there is the bug being measured rather than assumed.
"""
import sys, functools, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from brax import math
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step
from dial_mpc.core.dial_core import make_controller
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

flush = functools.partial(print, flush=True)
PAIR = {"walk": None, "trot": (0, 3, 1, 2), "pace": (0, 2, 1, 3), "bound": (0, 1, 2, 3)}
# straight, lateral, and both turns: heading has to hold a line *and* follow a rate
COMMANDS = [(0.8, 0.0, 0.0), (0.6, 0.12, 0.0), (0.6, 0.0, 0.3), (0.8, 0.0, -0.3)]
WEIGHTS = [float(v) for v in (sys.argv[1:] or ["0.0", "0.3", "0.6", "1.0"])]
T = 0.15
STEPS, WIN = 220, 60          # read after the 50-step command ramp
dc0, ec0 = _load_config("unitree_go2_gait", None)
dc = dataclasses.replace(dc0, temp_sample=T)


def strength(contact, gait):
    idx = PAIR[gait]; w = contact[WIN:]
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return np.corrcoef(w[:, i], w[:, j])[0, 1]
    if idx is None:
        return -np.nanmean([cor(i, j) for i in range(4) for j in range(i + 1, 4)])
    a, b, c, d = idx
    return np.nanmean([cor(a, b), cor(c, d)])


for yw in WEIGHTS:
    ec = dataclasses.replace(ec0, yaw_weight=yw)
    env = brax_envs.get_environment(dc.env_name, config=ec)
    mbdpi = make_controller(dc, env)
    ctrl = make_dial_step(env, mbdpi, dc, std_normalize=False,
                          level_scales=(1.0,) * dc.Ndiffuse)
    ti = env._torso_idx - 1

    def one(cmd, w):
        st = env.reset(jax.random.PRNGKey(0))
        st = set_command(env, st, cmd)
        info = dict(st.info); info["reward_weights"] = w
        st = st.replace(info=info)
        plan = jnp.zeros((dc.Hnode + 1, int(env.action_size)))
        def body(carry, _):
            s, k, p = carry
            s, k, p = ctrl(s, k, p)
            ps = s.pipeline_state
            zf = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
            vb = math.rotate(ps.xd.vel[ti], math.quat_inv(ps.x.rot[ti]))
            ab = math.rotate(ps.xd.ang[ti], math.quat_inv(ps.x.rot[ti]))
            return (s, k, p), ((zf < 0.02).astype(jnp.float32), vb[:2], ab[2],
                               math.quat_to_euler(ps.x.rot[ti])[2], s.done)
        return jax.lax.scan(body, (st, jax.random.PRNGKey(1), plan), None,
                            length=STEPS)[1]

    batch = jax.jit(jax.vmap(one, in_axes=(0, None)))
    cmds = jnp.asarray(COMMANDS, dtype=jnp.float32)
    flush(f"\n=== yaw_weight {yw}  (gait_scale {ec.gait_scale}, track_floor "
          f"{ec.track_floor}, T={T}) ===")
    flush(f"{'gait':<7}{'cmd':>16}{'pattern':>9}{'vx/cmd':>8}{'vy':>7}"
          f"{'wz':>8}{'wz cmd':>8}{'drift':>8}{'falls':>7}")
    for gi, name in enumerate(GAIT_NAMES):
        w = np.zeros(4, np.float32); w[gi] = 1.0
        contact, vb, wz, yaw, done = [
            np.asarray(v) for v in batch(cmds, jnp.asarray(w, jnp.float32))]
        for ci, cmd in enumerate(COMMANDS):
            pat = strength(contact[ci], name)
            vx = vb[ci, WIN:, 0].mean() / cmd[0]
            vy = vb[ci, WIN:, 1].mean()
            wzm = wz[ci, WIN:].mean()
            hd = np.degrees(np.unwrap(yaw[ci])[-1] - np.unwrap(yaw[ci])[WIN])
            span = (STEPS - WIN) * 0.02
            drift = hd - np.degrees(cmd[2]) * span      # heading error vs commanded turn
            flush(f"{name:<7}{str(tuple(cmd)):>16}{pat:9.2f}{vx:8.2f}{vy:7.2f}"
                  f"{wzm:8.2f}{cmd[2]:8.2f}{drift:+8.0f}{int(done[ci].sum()):7d}")
flush("YAWPICKDONE")
