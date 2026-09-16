"""Judge a fitted gait policy the way the objective was chosen: on evidence.

The old gait_verify judged from one seed at one command and hard-coded T=0.10,
which is how a non-monotone falls count (8, then 0, then 0) got treated as a
real effect when it was spread.  This mirrors csm/gait/gait_pick.py instead:
three seeds crossed with three commands, measured after the 50-step command
ramp finishes, reported as mean +/- spread.

The teacher it is being compared against, at gait_scale 0.2 / track_floor 2.5 /
T=0.15, walks every gait at pattern 0.84-0.89 and 0.88-0.95 of the command with
no falls.  That is the bar; the student is not expected to match it, but a
field that has actually learned the gait should hold a pattern well clear of
the 0.3 that merely means "the feet are not moving together at random".
"""
import sys, functools, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy
from csm.compose_walk_eval import make_student
from brax import math
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

flush = functools.partial(print, flush=True)
POLICY = sys.argv[1]
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
PAIR = {"walk": None, "trot": (0, 3, 1, 2), "pace": (0, 2, 1, 3), "bound": (0, 1, 2, 3)}
SEEDS = [0, 1, 2]
# Heading is part of the objective now, so the screen has to command a turn as
# well as a straight line -- a policy can hold a line and still ignore vyaw.
COMMANDS = [(0.6, 0.0, 0.0), (0.8, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.6, 0.0, 0.3), (0.8, 0.0, -0.3)]
WIN, STEPS = 60, 170

dc, ec = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc.env_name, config=ec)
policy = ComposedDialScorePolicy.load(POLICY)
reset = jax.jit(env.reset)


def record(st):
    ps = st.pipeline_state; ti = env._torso_idx - 1
    zf = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
    ab = math.rotate(ps.xd.ang[ti], math.quat_inv(ps.x.rot[ti]))
    return ((zf < 0.02).astype(jnp.float32), jnp.linalg.norm(ps.xd.vel[ti][:2]),
            st.done, ps.x.pos[ti, 2], ab[2],
            math.quat_to_euler(ps.x.rot[ti])[2])


student = make_student(env, policy, dc, init_passes=5, n_steps=STEPS,
                       record=record, step_passes=6)


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


flush(f"verify {POLICY}  (T={T}, step_passes=6, {len(SEEDS)} seeds x {len(COMMANDS)} commands, "
      f"measured after the ramp)")
flush(f"{'gait':<7}{'pattern':>16}{'v/cmd':>8}{'|drift|':>9}{'turn err':>10}{'falls':>7}")
pats = {}
for gi, name in enumerate(GAIT_NAMES):
    omega = np.zeros(4, np.float32); omega[gi] = 1.0
    ps, ratios, drifts, turns, falls = [], [], [], [], 0
    for seed in SEEDS:
        for cmd in COMMANDS:
            st = reset(jax.random.PRNGKey(seed))
            st = set_command(env, st, cmd)
            c, spd, done, z, wz, yaw = [np.asarray(v) for v in
                                        student(st, jnp.asarray(omega), T)]
            ps.append(strength(c, name))
            ratios.append(spd[WIN:].mean() / cmd[0])
            span = (STEPS - WIN) * float(env.dt)
            hd = np.degrees(np.unwrap(yaw)[-1] - np.unwrap(yaw)[WIN])
            drifts.append(abs(hd - np.degrees(cmd[2]) * span))
            turns.append(abs(wz[WIN:].mean() - cmd[2]))
            falls += int(done.sum())
    pats[name] = (float(np.nanmean(ps)), falls)
    flush(f"{name:<7}{np.nanmean(ps):8.2f}+/-{np.nanstd(ps):<5.2f}"
          f"{np.mean(ratios):8.2f}{max(drifts):9.0f}{max(turns):10.2f}{falls:7d}")

gaits = {n: p for n, (p, _) in pats.items()}
falls = sum(f for _, f in pats.values())
ok = all((not np.isnan(v)) and v > 0.5 for n, v in gaits.items() if n != "walk") and falls < 20
flush(f"VERDICT: {'sufficient' if ok else 'insufficient'} "
      f"(trot {gaits['trot']:.2f} pace {gaits['pace']:.2f} bound {gaits['bound']:.2f} falls {falls})")
