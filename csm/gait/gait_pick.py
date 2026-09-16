"""Pick the gait objective on evidence, not on one rollout.

The sweep that found the gait term was too weak judged each setting from a
single seed at a single command, which cannot tell a real difference from
noise -- it reported 8 falls for bound at x1.0 and none at x0.5 or x2.0, a
pattern no monotone mechanism explains.  Before spending a night collecting at
a chosen objective, measure the candidates properly: three seeds crossed with
three commands, run in parallel, reported as mean +/- spread.

Two things are being chosen at once.  The gait term has to be strong enough
that the softmax selects on the foot pattern (below x0.5 it does not, and DIAL
drags its feet), and `track_floor` has to be strong enough that the robot still
follows the command -- at x0.5-x2.0 the measured speed fell to 0.21-0.44
against a 0.6-1.0 command, which is the gait term outcompeting the floor.  So
the grid varies both.

Everything runs under raw Gibbs, the convention the collection and the labels
actually use; std_normalize walks the gaits at the old weight but breaks the
nu-linearity composition rests on, so it is not an option here.
"""
import sys, functools, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step
from dial_mpc.core.dial_core import make_controller
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

flush = functools.partial(print, flush=True)
PAIR = {"walk": None, "trot": (0, 3, 1, 2), "pace": (0, 2, 1, 3), "bound": (0, 1, 2, 3)}
SEEDS = [0, 1, 2]
COMMANDS = [(0.6, 0.0, 0.0), (0.8, 0.0, 0.0), (1.0, 0.0, 0.0)]
# (gait_scale, track_floor): gait_scale *divides* the 0.1 term, so smaller is
# stronger.  track_floor 0.3 is what was collected; 0.6 buys back the command.
GRID = [(0.2, 0.3), (0.15, 0.3), (0.1, 0.3), (0.15, 0.6), (0.1, 0.6)]
if len(sys.argv) > 1:  # "gait_scale:track_floor ..." to probe a different corner
    GRID = [tuple(float(v) for v in a.split(":")) for a in sys.argv[1:]]
T = 0.10
STEPS = 120
dc0, ec0 = _load_config("unitree_go2_gait", None)
dc = dataclasses.replace(dc0, temp_sample=T)


def strength(contact, gait):
    """Contact correlation of the pairs that define this gait."""
    idx = PAIR[gait]; w = contact[25:]
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return np.corrcoef(w[:, i], w[:, j])[0, 1]
    if idx is None:  # walk is four-beat: every pair should be anti-correlated
        return -np.nanmean([cor(i, j) for i in range(4) for j in range(i + 1, 4)])
    a, b, c, d = idx
    return np.nanmean([cor(a, b), cor(c, d)])


for gait_scale, track in GRID:
    ec = dataclasses.replace(ec0, gait_scale=gait_scale, track_floor=track)
    env = brax_envs.get_environment(dc.env_name, config=ec)
    mbdpi = make_controller(dc, env)
    ctrl = make_dial_step(env, mbdpi, dc, std_normalize=False,
                          level_scales=(1.0,) * dc.Ndiffuse)

    def one(seed, cmd, w):
        st = env.reset(jax.random.PRNGKey(seed))
        st = set_command(env, st, cmd)
        info = dict(st.info); info["reward_weights"] = w
        st = st.replace(info=info)
        plan = jnp.zeros((dc.Hnode + 1, int(env.action_size)))
        def body(carry, _):
            s, k, p = carry
            s, k, p = ctrl(s, k, p)
            zf = s.pipeline_state.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
            vb = s.pipeline_state.xd.vel[env._torso_idx - 1]
            return (s, k, p), ((zf < 0.02).astype(jnp.float32),
                               jnp.linalg.norm(vb[:2]), s.done, zf.max())
        return jax.lax.scan(body, (st, jax.random.PRNGKey(seed + 100), plan),
                            None, length=STEPS)[1]

    batch = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))
    seeds = jnp.asarray([s for s in SEEDS for _ in COMMANDS])
    cmds = jnp.asarray([c for _ in SEEDS for c in COMMANDS], dtype=jnp.float32)
    cmd_speed = np.asarray(cmds)[:, 0]

    flush(f"\n=== gait x{0.1 / gait_scale:.2f} (gait_scale {gait_scale})  "
          f"track_floor {track}  T={T}  [{len(SEEDS)} seeds x {len(COMMANDS)} commands] ===")
    flush(f"{'gait':<7}{'pattern':>16}{'speed/cmd':>12}{'lift':>8}{'falls':>8}")
    for gi, name in enumerate(GAIT_NAMES):
        w = np.zeros(4); w[gi] = 1.0
        contact, spd, done, lift = [
            np.asarray(v) for v in batch(seeds, cmds, jnp.asarray(w, jnp.float32))]
        pats = np.array([strength(contact[b], name) for b in range(contact.shape[0])])
        ratio = spd[:, 25:].mean(1) / cmd_speed
        flush(f"{name:<7}{np.nanmean(pats):8.2f}+/-{np.nanstd(pats):<5.2f}"
              f"{ratio.mean():12.2f}{lift[:, 25:].mean():8.3f}"
              f"{int(done.sum()):8d}")
flush("PICKDONE")
