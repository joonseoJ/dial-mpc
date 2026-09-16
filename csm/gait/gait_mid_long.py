"""Do the six 50:50 gait blends hold for 1500 steps, or only for the 110 they were read on?

Each midpoint is a weight no field was ever fitted at, produced by one matrix
solve over two networks.  Same protocol as the long verify of the pure gaits:
three commands, 1500 steps, the three contact correlations (diag=trot,
lat=pace, f-h=bound) read in five windows so drift shows as a trend, and a
rollout counted collapsed only when `done` holds 100 consecutive steps or the
torso ends under 0.15 m.
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
T = 0.15
COMMANDS = [0.6, 0.8, 1.0]
STEPS = 1500
WINDOWS = [(60, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]
PAIRS = [(1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3)]
COLLAPSE_RUN, COLLAPSE_Z = 100, 0.15

dc, ec = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc.env_name, config=ec)
policy = ComposedDialScorePolicy.load(POLICY)
reset = jax.jit(env.reset)


def record(st):
    ps = st.pipeline_state; ti = env._torso_idx - 1
    zf = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
    return ((zf < 0.02).astype(jnp.float32), jnp.linalg.norm(ps.xd.vel[ti][:2]),
            st.done, ps.x.pos[ti, 2],
            math.quat_to_euler(ps.x.rot[ti])[2])


student = make_student(env, policy, dc, init_passes=5, n_steps=STEPS,
                       record=record, step_passes=6)


def triple(w):
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return float(np.corrcoef(w[:, i], w[:, j])[0, 1])
    return (np.nanmean([cor(0, 3), cor(1, 2)]), np.nanmean([cor(0, 2), cor(1, 3)]),
            np.nanmean([cor(0, 1), cor(2, 3)]))


def runs_of(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        if not v and start is not None: out.append((start, i - start)); start = None
    if start is not None: out.append((start, len(mask) - start))
    return out


out_dir = Path("csm_runs/gait_traces"); out_dir.mkdir(exist_ok=True)
flush(f"midpoint long verify on {POLICY}  T={T}  {STEPS} steps, commands {COMMANDS}")
flush("each cell: diag/lat/f-h over the window")
for i, j in PAIRS:
    name = f"{GAIT_NAMES[i]}+{GAIT_NAMES[j]}"
    omega = np.zeros(4, np.float32); omega[i] = omega[j] = 1.0
    omega /= np.linalg.norm(omega); om = jnp.asarray(omega)
    flush(f"\n=== {name} 50:50  coeff {np.round(np.asarray(policy.coefficients(om))[[i, j]], 2)} ===")
    flush(f"{'cmd':>4}  " + "".join(f"{f'{a}-{b}':>18}" for a, b in WINDOWS)
          + f"{'v/cmd':>7}{'drift':>7}{'dips':>6}{'z end':>7}  verdict")
    for cmd in COMMANDS:
        st = set_command(env, reset(jax.random.PRNGKey(0)), (cmd, 0.0, 0.0))
        c, spd, done, z, yaw = [np.asarray(v) for v in student(st, om, T)]
        np.savez(out_dir / f"mid_{GAIT_NAMES[i]}_{GAIT_NAMES[j]}_{cmd:.1f}.npz",
                 contact=c, speed=spd, done=done, z=z, yaw=yaw)
        d = done > 0.5; rr = runs_of(d)
        longest = max((n for _, n in rr), default=0)
        collapse_at = next((s for s, n in rr if n >= COLLAPSE_RUN), None)
        if collapse_at is None and z[-30:].mean() < COLLAPSE_Z:
            collapse_at = int(np.argmax(z < COLLAPSE_Z))
        usable = STEPS if collapse_at is None else collapse_at
        cells = []
        for a, b in WINDOWS:
            b2 = min(b, usable)
            cells.append("%5.2f/%5.2f/%5.2f" % triple(c[a:b2]) if b2 - a > 40 else f"{'--':>17}")
        ratio = spd[60:usable].mean() / cmd if usable > 100 else np.nan
        drift = (np.degrees(np.unwrap(yaw)[usable - 1] - np.unwrap(yaw)[60])
                 if usable > 100 else np.nan)
        verdict = (f"collapsed@{collapse_at}" if collapse_at is not None
                   else "clean" if not rr else "dips only")
        flush(f"{cmd:>4.1f}  " + "".join(f"{x:>18}" for x in cells)
              + f"{ratio:7.2f}{drift:+7.0f}{len(rr):6d}{z[-30:].mean():7.3f}  {verdict}")
flush("MIDLONGDONE")
