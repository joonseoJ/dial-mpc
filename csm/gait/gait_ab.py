"""Same harness, same objective, one difference: the Gibbs convention.

The gaits were verified with std_normalize=True, but the collection driver and
the labels use raw Gibbs (std_normalize=False, csm/collect_cli.py:113).  If the
gaits only exist under the convention that was never collected, every field was
distilled from a teacher that does not walk them.
"""
import functools, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step
from dial_mpc.core.dial_core import make_controller
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES
flush = functools.partial(print, flush=True)
PAIR = {"walk": None, "trot": (0,3,1,2), "pace": (0,2,1,3), "bound": (0,1,2,3)}
CMD = (0.6, 0.0, 0.0)
dc0, ec0 = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc0.env_name, config=ec0)

def strength(contact, gait):
    idx = PAIR[gait]; w = contact[25:]
    def cor(i,j):
        if w[:,i].std()<1e-6 or w[:,j].std()<1e-6: return np.nan
        return np.corrcoef(w[:,i],w[:,j])[0,1]
    if idx is None: return -np.nanmean([cor(i,j) for i in range(4) for j in range(i+1,4)])
    a,b,c,d = idx; return np.nanmean([cor(a,b), cor(c,d)])

for sn in [True, False]:
    for temp in [0.10, 0.25]:
        dc = dataclasses.replace(dc0, temp_sample=temp)
        m = make_controller(dc, env)
        ctrl = make_dial_step(env, m, dc, std_normalize=sn, level_scales=(1.0,)*dc.Ndiffuse)
        reset = jax.jit(env.reset)
        @jax.jit
        def go(state, key, plan, w):
            info = dict(state.info); info["reward_weights"] = w
            state = state.replace(info=info)
            def body(c,_):
                s,k,p = c; s,k,p = ctrl(s,k,p)
                zf = s.pipeline_state.site_xpos[env._feet_site_id][:,2] - env._foot_radius
                vb = s.pipeline_state.xd.vel[env._torso_idx-1]
                return (s,k,p), ((zf<0.02).astype(jnp.float32), jnp.linalg.norm(vb[:2]),
                                 s.done, zf.max())
            return jax.lax.scan(body,(state,key,plan),None,length=120)[1]
        flush(f"\n=== std_normalize={sn}  T={temp}  (gait_scale 1.0, the collected objective) ===")
        flush(f"{'gait':<7}{'pattern':>10}{'speed':>8}{'falls':>7}{'max lift':>10}")
        for gi, name in enumerate(GAIT_NAMES):
            w = np.zeros(4); w[gi] = 1.0
            st = reset(jax.random.PRNGKey(0)); st = set_command(env, st, CMD)
            plan = jnp.zeros((dc.Hnode+1, int(env.action_size)))
            c, spd, done, lift = [np.asarray(v) for v in go(st, jax.random.PRNGKey(1), plan, jnp.asarray(w,jnp.float32))]
            flush(f"{name:<7}{strength(c,name):10.2f}{spd[25:].mean():8.2f}{int(done.sum()):7d}{lift[25:].mean():10.3f}")
flush("ABDONE")
