"""Is a `done` step a fall, or a bound's crouch dipping under the 0.18 m line?

`done` is recomputed every step from the state (torso flipped, or lower than
0.18 m) and the student loop never resets, so a summed `done` counts steps
spent low, not falls.  Read the torso height trace itself: a fall is a height
that goes down and stays down; a deep crouch is a dip that comes back.
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
    z = st.pipeline_state.x.pos[env._torso_idx - 1, 2]
    up = jnp.dot(jax.numpy.array([0.,0.,1.]), st.pipeline_state.x.rot[env._torso_idx-1][:3]) # placeholder
    return z, st.done
student = make_student(env, policy, dc, init_passes=5, n_steps=600, record=record, step_passes=6)
G = {"walk": 0, "trot": 1, "pace": 2, "bound": 3}
flush(f"{'gait':<6}{'cmd':>4}{'seed':>5}{'done%':>7}{'longest':>8}{'min z':>7}{'z@end':>7}{'z mean':>7}  verdict")
for name in ["bound", "walk", "trot"]:
    om = np.zeros(4, np.float32); om[G[name]] = 1.0
    for cmd in (0.6, 0.8):
        for seed in (0, 1, 2):
            st = set_command(env, reset(jax.random.PRNGKey(seed)), (cmd, 0., 0.))
            z, done = [np.asarray(v) for v in student(st, jnp.asarray(om), T)]
            d = done > 0.5
            # longest run of consecutive done steps
            runs, cur = [], 0
            for v in d:
                cur = cur + 1 if v else 0; runs.append(cur)
            longest = max(runs)
            end_up = z[-50:].mean() > 0.22
            verdict = ("FALL (stays down)" if (longest > 150 and not end_up)
                       else "recovers" if d.any() else "clean")
            flush(f"{name:<6}{cmd:>4.1f}{seed:>5d}{100*d.mean():7.1f}{longest:8d}"
                  f"{z.min():7.3f}{z[-50:].mean():7.3f}{z[60:].mean():7.3f}  {verdict}")
flush("FALLCHECKDONE")
