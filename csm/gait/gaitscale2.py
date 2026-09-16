"""How strong can the gait term get before DIAL stops walking?

The fields all learned the shared floor because the gait term (x0.1 under a
0.3+0.3 floor) moves a sample's return by only ~7% of its spread, so the softmax
picks on posture, not pattern.  Raising the gait term is the one fix that
attacks both obstacles (fit error and label noise) at once -- but the floor is
what keeps the robot standing, so there is a ceiling.  This finds it.

Measured with std_normalize=False, the convention the *labels* use, and at the
collection temperature, with the current scale included as the baseline so
everything is compared like for like.  Raising the gait term also raises the
return spread, which makes a fixed temperature sharper, so each scale is also
run at a larger temperature to separate "the objective is unreachable" from
"the temperature is now wrong".
"""
import functools, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step
from dial_mpc.core.dial_core import make_controller
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES
flush = functools.partial(print, flush=True)

PAIR = {"walk": None, "trot": (0, 3, 1, 2), "pace": (0, 2, 1, 3), "bound": (0, 1, 2, 3)}
# gait_scale divides the 0.1 pattern term: 1.0 -> x0.1 (current), 0.05 -> x2.0
SCALES = [1.0, 0.2, 0.1, 0.05]
TEMPS = [0.10, 0.30]
CMD = (0.6, 0.0, 0.0)
dc0, ec0 = _load_config("unitree_go2_gait", None)


def strength(contact, gait):
    idx = PAIR[gait]; w = contact[25:]
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return np.corrcoef(w[:, i], w[:, j])[0, 1]
    if idx is None:
        return -np.nanmean([cor(i, j) for i in range(4) for j in range(i + 1, 4)])
    a, b, c, d = idx
    return np.nanmean([cor(a, b), cor(c, d)])


for gs in SCALES:
    ec = dataclasses.replace(ec0, gait_scale=gs)
    env = brax_envs.get_environment(dc0.env_name, config=ec)
    for temp in TEMPS:
        dc = dataclasses.replace(dc0, temp_sample=temp)
        m = make_controller(dc, env)
        ctrl = make_dial_step(env, m, dc, std_normalize=False,
                              level_scales=(1.0,) * dc.Ndiffuse)
        reset = jax.jit(env.reset)

        @jax.jit
        def go(state, key, plan, w):
            info = dict(state.info); info["reward_weights"] = w
            state = state.replace(info=info)
            def body(c, _):
                s, k, p = c
                s, k, p = ctrl(s, k, p)
                zf = s.pipeline_state.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
                vb = s.pipeline_state.xd.vel[env._torso_idx - 1]
                return (s, k, p), ((zf < 0.02).astype(jnp.float32),
                                   jnp.linalg.norm(vb[:2]), s.done,
                                   s.info["reward_terms"])
            return jax.lax.scan(body, (state, key, plan), None, length=100)[1]

        flush(f"\n=== gait x{0.1 / gs:.2f} (gait_scale {gs})  T={temp} ===")
        flush(f"{'gait':<7}{'pattern':>10}{'speed':>8}{'falls':>7}{'gait/common':>13}")
        for gi, name in enumerate(GAIT_NAMES):
            w = np.zeros(4); w[gi] = 1.0
            st = reset(jax.random.PRNGKey(0)); st = set_command(env, st, CMD)
            plan = jnp.zeros((dc.Hnode + 1, int(env.action_size)))
            contact, spd, done, terms = [
                np.asarray(v) for v in go(st, jax.random.PRNGKey(1), plan,
                                          jnp.asarray(w, jnp.float32))]
            # how much of the objective's magnitude is the gait-differentiating
            # part, as opposed to the floor every row shares
            common = terms.mean(-1, keepdims=True)
            ratio = np.abs(terms - common).mean() / max(np.abs(common).mean(), 1e-9)
            flush(f"{name:<7}{strength(contact, name):10.2f}{spd[25:].mean():8.2f}"
                  f"{int(done.sum()):7d}{ratio:13.3f}")
flush("SCALEDONE")
