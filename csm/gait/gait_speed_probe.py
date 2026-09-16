"""Do walk and bound fail because the field is wrong, or because they cannot
hold the commanded speed?

All four fields came out of the same fit with the same error (val_rel_rms
0.856-0.864) yet trot and pace walk cleanly while walk and bound fall -- and
both fall right as the 50-step command ramp completes, which points at the
speed rather than the field.  If they walk at a low command and fail at a high
one, the gait was learned and the margin is what ran out.
"""
import sys, functools, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy
from csm.compose_walk_eval import make_student
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES
flush = functools.partial(print, flush=True)
POLICY, T = sys.argv[1], 0.15
PAIR = {"walk": None, "trot": (0,3,1,2), "pace": (0,2,1,3), "bound": (0,1,2,3)}
SEEDS, WIN, STEPS = [0,1,2], 60, 170
CMDS = [0.2, 0.4, 0.6, 0.8]
dc, ec = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc.env_name, config=ec)
policy = ComposedDialScorePolicy.load(POLICY)
reset = jax.jit(env.reset)
def record(st):
    zf = st.pipeline_state.site_xpos[env._feet_site_id][:,2] - env._foot_radius
    vb = st.pipeline_state.xd.vel[env._torso_idx-1]
    return (zf<0.02).astype(jnp.float32), jnp.linalg.norm(vb[:2]), st.done
student = make_student(env, policy, dc, init_passes=5, n_steps=STEPS, record=record, step_passes=6)
def strength(c, g):
    idx = PAIR[g]; w = c[WIN:]
    def cor(i,j):
        if w[:,i].std()<1e-6 or w[:,j].std()<1e-6: return np.nan
        return np.corrcoef(w[:,i],w[:,j])[0,1]
    if idx is None: return -np.nanmean([cor(i,j) for i in range(4) for j in range(i+1,4)])
    a,b,c2,d = idx; return np.nanmean([cor(a,b), cor(c2,d)])
flush(f"{'gait':<7}" + "".join(f"{'cmd '+str(c):>18}" for c in CMDS))
flush(f"{'':7}" + "".join(f"{'pattern':>10}{'falls':>8}" for _ in CMDS))
for gi, name in enumerate(GAIT_NAMES):
    om = np.zeros(4, np.float32); om[gi] = 1.0
    row = ""
    for cmd in CMDS:
        ps, falls = [], 0
        for s in SEEDS:
            st = set_command(env, reset(jax.random.PRNGKey(s)), (cmd,0.0,0.0))
            c, spd, done = [np.asarray(v) for v in student(st, jnp.asarray(om), T)]
            ps.append(strength(c, name)); falls += int(done.sum())
        row += f"{np.nanmean(ps):10.2f}{falls:8d}"
    flush(f"{name:<7}{row}")
flush("SPEEDDONE")
