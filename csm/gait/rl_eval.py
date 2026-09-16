"""Score RL gait policies on exactly the measurements the CSM ones were scored on.

Three arms, all read with the same contact correlations, heading drift and
collapse rule as `gait_verify2.py` / `gait_blend2.py`, so the comparison is
between controllers rather than between screens:

  `specialists`  each pure gait's own PPO policy, at the gait it was trained on
  `blend`        two specialists mixed as `a*pi_i(s) + (1-a)*pi_j(s)`
  `conditioned`  one PPO policy that takes omega as input, asked for the same
                 pure and mixed weights

Averaging two policies' actions is the only linear operation RL offers, and it
has no justification behind it -- that is the point of running it.  CSM mixes
*scores*, and the Gibbs score is linear in nu by construction; the conditioned
policy is the honest strong baseline, since it was actually trained on mixed
weights and could in principle interpolate.

The specialists are trained with termination, so they see an implicit
stay-alive bonus the distilled fields never got.  That favours the baseline,
which is the direction an honest comparison should lean.
"""
import sys, functools, argparse, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from brax import math
from csm.basis_screen import _load_config
from csm.rl_baseline import load_policy
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

flush = functools.partial(print, flush=True)
PAIR = {"walk": None, "trot": (0, 3, 1, 2), "pace": (0, 2, 1, 3), "bound": (0, 1, 2, 3)}
PAIRS = [(1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3)]
RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0]
WIN = 60

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--mode", choices=("specialists", "blend", "conditioned"), required=True)
p.add_argument("--policies", nargs="+", required=True,
               help="four specialist policy.pkl in gait order, or one conditioned policy")
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--commands", type=float, nargs="+", default=[0.6, 0.8, 1.0],
               help="forward speeds; a turn is appended at the middle speed")
args = p.parse_args()

dc, ec = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc.env_name, config=ec)
reset = jax.jit(env.reset)
ti = env._torso_idx - 1
STEPS = args.steps
COMMANDS = [(c, 0.0) for c in args.commands] + [(args.commands[len(args.commands) // 2], 0.3)]

infer = [load_policy(q)[0] for q in args.policies]
conditioned = args.mode == "conditioned"


def rollout(act_fn, cmd, wcmd, omega):
    """Run one episode; `act_fn(obs, key) -> action`.  No resets, as the CSM loop."""
    st = set_command(env, reset(jax.random.PRNGKey(0)), (cmd, 0.0, wcmd))
    st = st.replace(info={**st.info, "reward_weights": jnp.asarray(omega, jnp.float32)})

    def body(carry, _):
        s, k = carry
        k, sub = jax.random.split(k)
        obs = jnp.concatenate([s.obs, jnp.asarray(omega, jnp.float32)]) if conditioned else s.obs
        s = env.step(s, act_fn(obs, sub))
        ps = s.pipeline_state
        zf = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
        return (s, k), ((zf < 0.02).astype(jnp.float32),
                        jnp.linalg.norm(ps.xd.vel[ti][:2]), s.done,
                        ps.x.pos[ti, 2], math.quat_to_euler(ps.x.rot[ti])[2])

    return jax.lax.scan(body, (st, jax.random.PRNGKey(1)), None, length=STEPS)[1]


def triple(contact):
    w = contact[WIN:]
    def cor(i, j):
        if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6:
            return np.nan
        return float(np.corrcoef(w[:, i], w[:, j])[0, 1])
    return (np.nanmean([cor(0, 3), cor(1, 2)]), np.nanmean([cor(0, 2), cor(1, 3)]),
            np.nanmean([cor(0, 1), cor(2, 3)]))


def strength(contact, gait):
    d, l, f = triple(contact)
    idx = {"trot": d, "pace": l, "bound": f}
    if gait == "walk":
        return -np.nanmean([d, l, f])
    return idx[gait]


def collapsed(done, z):
    d = np.asarray(done) > 0.5; run = best = 0
    for v in d:
        run = run + 1 if v else 0; best = max(best, run)
    return best >= 100 or float(np.asarray(z)[-30:].mean()) < 0.15


def measure(act_fn, cmd, wcmd, omega):
    c, spd, done, z, yaw = [np.asarray(v) for v in rollout(act_fn, cmd, wcmd, omega)]
    span = (STEPS - WIN) * float(env.dt)
    drift = np.degrees(np.unwrap(yaw)[-1] - np.unwrap(yaw)[WIN]) - np.degrees(wcmd) * span
    return (triple(c), spd[WIN:].mean() / cmd, drift, collapsed(done, z), c)


if args.mode == "specialists":
    flush(f"RL specialists, {STEPS} steps, commands {COMMANDS}")
    flush(f"{'gait':<7}{'cmd':>10}{'pattern':>9}{'diag':>7}{'lat':>7}{'f-h':>7}"
          f"{'v/cmd':>7}{'drift':>7}  verdict")
    for gi, name in enumerate(GAIT_NAMES):
        omega = np.zeros(4, np.float32); omega[gi] = 1.0
        fn = functools.partial(infer[gi])
        for cmd, wcmd in COMMANDS:
            (d, l, f), v, dr, bad, c = measure(lambda o, k: fn(o, k)[0], cmd, wcmd, omega)
            flush(f"{name:<7}{f'{cmd:.1f},{wcmd:+.1f}':>10}{strength(c, name):9.2f}"
                  f"{d:7.2f}{l:7.2f}{f:7.2f}{v:7.2f}{dr:+7.0f}  "
                  f"{'COLLAPSED' if bad else 'ok'}")

elif args.mode == "blend":
    flush(f"RL specialists blended as a*pi_i + (1-a)*pi_j, {STEPS} steps")
    for i, j in PAIRS:
        flush(f"\n=== {GAIT_NAMES[i]} -> {GAIT_NAMES[j]} ===")
        flush(f"{'ratio':>10}{'diag':>7}{'lat':>7}{'f-h':>7}{'v/cmd':>7}{'drift':>7}  verdict")
        for a in RATIOS:
            omega = np.zeros(4, np.float32); omega[i] = a; omega[j] = 1 - a
            omega /= np.linalg.norm(omega)
            fi, fj = infer[i], infer[j]
            def act(o, k, a=a, fi=fi, fj=fj):
                k1, k2 = jax.random.split(k)
                return a * fi(o, k1)[0] + (1 - a) * fj(o, k2)[0]
            (d, l, f), v, dr, bad, c = measure(act, 0.8, 0.0, omega)
            flush(f"{f'{a:.2f}:{1-a:.2f}':>10}{d:7.2f}{l:7.2f}{f:7.2f}{v:7.2f}{dr:+7.0f}  "
                  f"{'COLLAPSED' if bad else 'ok'}")

else:
    fn = infer[0]
    act = lambda o, k: fn(o, k)[0]
    flush(f"omega-conditioned PPO, {STEPS} steps")
    flush(f"{'target':<14}{'pattern':>9}{'diag':>7}{'lat':>7}{'f-h':>7}{'v/cmd':>7}"
          f"{'drift':>7}  verdict")
    for gi, name in enumerate(GAIT_NAMES):
        omega = np.zeros(4, np.float32); omega[gi] = 1.0
        (d, l, f), v, dr, bad, c = measure(act, 0.8, 0.0, omega)
        flush(f"{name:<14}{strength(c, name):9.2f}{d:7.2f}{l:7.2f}{f:7.2f}{v:7.2f}"
              f"{dr:+7.0f}  {'COLLAPSED' if bad else 'ok'}")
    for i, j in PAIRS:
        omega = np.zeros(4, np.float32); omega[i] = omega[j] = 1.0
        omega /= np.linalg.norm(omega)
        (d, l, f), v, dr, bad, c = measure(act, 0.8, 0.0, omega)
        flush(f"{GAIT_NAMES[i] + '+' + GAIT_NAMES[j]:<14}{'':>9}{d:7.2f}{l:7.2f}{f:7.2f}"
              f"{v:7.2f}{dr:+7.0f}  {'COLLAPSED' if bad else 'ok'}")
flush("RLEVALDONE")
