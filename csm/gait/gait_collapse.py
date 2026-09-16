"""What happens in the ten steps before the torso goes through the floor?

Trot stands at +0.28 m; walk and bound settle at -0.22 m, twenty centimetres
under the ground plane.  Only the feet collide with the floor in this model,
so the one way down is all four feet off the ground at once.  Print the foot
contacts and torso height step by step around the collapse to see whether
that is what the student does.
"""
import sys, functools, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy
from csm.compose_walk_eval import make_student
from csm.screen import set_command
flush = functools.partial(print, flush=True)
POLICY, T = sys.argv[1], 0.15
dc, ec = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc.env_name, config=ec)
policy = ComposedDialScorePolicy.load(POLICY)
reset = jax.jit(env.reset)
def record(st):
    zf = st.pipeline_state.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
    z = st.pipeline_state.x.pos[env._torso_idx - 1, 2]
    return zf, z, st.done
student = make_student(env, policy, dc, init_passes=5, n_steps=120, record=record, step_passes=6)
for name, gi in (("bound", 3), ("walk", 0), ("trot", 1)):
    om = np.zeros(4, np.float32); om[gi] = 1.0
    st = set_command(env, reset(jax.random.PRNGKey(0)), (0.6, 0., 0.))
    zf, z, done = [np.asarray(v) for v in student(st, jnp.asarray(om), T)]
    first = int(np.argmax(done > 0.5)) if (done > 0.5).any() else None
    flush(f"\n=== {name}  first done step: {first} ===")
    flush(f"{'step':>5}{'torso z':>9}  FL     FR     RL     RR   (foot clearance, m)   feet up")
    lo = max(0, (first or 60) - 12); hi = min(120, (first or 60) + 6)
    for s in range(lo, hi):
        up = int((zf[s] > 0.02).sum())
        flush(f"{s:>5}{z[s]:9.3f}  " + " ".join(f"{v:6.3f}" for v in zf[s]) + f"   {up}/4" + ("  <-- done" if done[s] > 0.5 else ""))
flush("COLLAPSEDONE")
