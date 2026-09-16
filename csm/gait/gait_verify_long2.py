"""1500-step gait verification that scores collapses, not dips.

`done` in this environment is recomputed every step -- torso below 0.18 m or
flipped -- and the student loop never resets, so a single `done` is a dip that
may recover, and the first `done` is not a death.  The previous long verify
truncated every window at the first `done` and reported "0/3 survived" for
policies that had dipped once during the ramp and then walked for thirty
seconds.

Here the raw traces (contact, speed, done, torso height) are saved, and a
rollout is scored collapsed only when `done` holds for 100 consecutive steps
(2 s) or the torso ends below 0.15 m.  Pattern is read in five windows over
everything before a collapse; dips inside a window are just contacts.  Dip
runs are counted and reported so a policy that walks but grazes the line is
visible as exactly that.
"""
import sys, functools, numpy as np, jax, jax.numpy as jnp
from pathlib import Path
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy
from csm.compose_walk_eval import make_student
from brax import math
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

flush = functools.partial(print, flush=True)
POLICY = sys.argv[1]
TAG = sys.argv[2]
T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
PAIR = {"walk": None, "trot": (0, 3, 1, 2), "pace": (0, 2, 1, 3), "bound": (0, 1, 2, 3)}
# (vx, vyaw): heading is in the objective, so a turn has to be held for 30 s too
COMMANDS = [(0.6, 0.0), (0.8, 0.0), (1.0, 0.0), (0.8, 0.3)]
STEPS = 1500
WINDOWS = [(60, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]
COLLAPSE_RUN, COLLAPSE_Z = 100, 0.15

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


def strength(w, gait):
    idx = PAIR[gait]
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return np.corrcoef(w[:, i], w[:, j])[0, 1]
    if idx is None:
        return -np.nanmean([cor(i, j) for i in range(4) for j in range(i + 1, 4)])
    a, b, c, d = idx
    return np.nanmean([cor(a, b), cor(c, d)])


def runs_of(mask):
    """Lengths and start indices of consecutive True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - start)); start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


out_dir = Path("csm_runs/gait_traces"); out_dir.mkdir(exist_ok=True)
flush(f"long verify2 {TAG}  ({POLICY})  T={T}  {STEPS} steps, commands {COMMANDS}")
flush(f"{'gait':<6}{'cmd':>10}  " + "".join(f"{f'{a}-{b}':>9}" for a, b in WINDOWS)
      + f"{'v/cmd':>7}{'wz':>6}{'drift':>7}{'dips':>6}{'z end':>7}  verdict")
summary = {}
for name in GAIT_NAMES:
    gi = GAIT_NAMES.index(name)
    omega = np.zeros(4, np.float32); omega[gi] = 1.0
    for cmd, wcmd in COMMANDS:
        st = set_command(env, reset(jax.random.PRNGKey(0)), (cmd, 0.0, wcmd))
        c, spd, done, z, wz, yaw = [np.asarray(v) for v in student(st, jnp.asarray(omega), T)]
        np.savez(out_dir / f"{TAG}_{name}_{cmd:.1f}_{wcmd:+.1f}.npz",
                 contact=c, speed=spd, done=done, z=z, wz=wz, yaw=yaw)
        d = done > 0.5
        rr = runs_of(d)
        longest = max((n for _, n in rr), default=0)
        collapse_at = next((s for s, n in rr if n >= COLLAPSE_RUN), None)
        if collapse_at is None and z[-30:].mean() < COLLAPSE_Z:
            collapse_at = int(np.argmax(z < COLLAPSE_Z))
        usable = STEPS if collapse_at is None else collapse_at
        pats = [strength(c[a:min(b, usable)], name) if min(b, usable) - a > 40 else np.nan
                for a, b in WINDOWS]
        ratio = spd[60:usable].mean() / cmd if usable > 100 else np.nan
        wzm = wz[60:usable].mean() if usable > 100 else np.nan
        span = (usable - 60) * float(env.dt)
        drift = (np.degrees(np.unwrap(yaw)[usable - 1] - np.unwrap(yaw)[60])
                 - np.degrees(wcmd) * span) if usable > 100 else np.nan
        verdict = "collapsed@%d" % collapse_at if collapse_at is not None else (
            "clean" if not rr else f"dips only")
        summary[(name, cmd, wcmd)] = (pats, collapse_at is None)
        flush(f"{name:<6}{f'{cmd:.1f},{wcmd:+.1f}':>10}  " + "".join(f"{p:9.2f}" for p in pats)
              + f"{ratio:7.2f}{wzm:6.2f}{drift:+7.0f}{len(rr):6d}{z[-30:].mean():7.3f}  {verdict}")
flush("--- per gait: mean pattern over the last three windows, commands that did not collapse ---")
for name in GAIT_NAMES:
    ok = [np.nanmean(p[2:]) for (g, _, _), (p, alive) in summary.items() if g == name and alive]
    n_alive = sum(1 for (g, _, _), (_, alive) in summary.items() if g == name and alive)
    flush(f"  {name:<6} alive {n_alive}/{len(COMMANDS)}   late pattern "
          f"{np.nanmean(ok) if ok else float('nan'):.2f}")
flush("LONG2DONE")
