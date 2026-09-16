"""CSM against RL on the same weights, the same commands, the same resets.

Every gait comparison up to here rested on one trajectory per command: the
student is deterministic and `reset` did not randomise, so the three "seeds"
were three identical rollouts.  The midpoint gap this project rests on --
0.57 composed against 0.16 blended -- was a single sample against a single
sample.  Here both arms are run from a batch of randomised initial states
(height, orientation and joint noise) crossed with a command grid, and the
same batch is handed to every arm.

Arms, all producing an action from the same observation:

  `csm`       the composed score fields, refined `step_passes` times per step
  `rl_blend`  two specialists averaged, `a*pi_i(s) + (1-a)*pi_j(s)`
  `rl_cond`   one PPO policy that takes omega as an input
  `rl_spec`   the specialist for a pure row (pure weights only)

Weights come in suites so the interesting question -- how far from a trained
weight can you ask -- can be turned up: `pure`, `pairs` (0.25/0.5/0.75 of every
pair), `deep` (three- and four-way mixtures, where averaging more actions
should wash out further than mixing more scores).  Commands likewise include a
band outside the training box.

Reported per (arm, weight, command): the three contact correlations averaged
over the surviving seeds, speed over commanded, heading error against the
commanded turn, and how many of the seeds collapsed.
"""
from __future__ import annotations

import argparse, functools, itertools, json, dataclasses
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import brax.envs as brax_envs
from brax import math

import dial_mpc.envs  # noqa: F401
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import mixture_from_pinv
from csm.rl_baseline import load_policy
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

GAITS = list(GAIT_NAMES)
PAIRS = [(1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3)]
WIN = 60
COLLAPSE_RUN, COLLAPSE_Z = 100, 0.15


def weight_suite(name: str):
    """(label, omega) pairs.  Omegas are unit vectors; the env normalises anyway."""
    out = []
    if name in ("pure", "all"):
        for i, g in enumerate(GAITS):
            w = np.zeros(4, np.float32); w[i] = 1.0
            out.append((g, w))
    if name in ("pairs", "all"):
        for i, j in PAIRS:
            for a in (0.75, 0.5, 0.25):
                w = np.zeros(4, np.float32); w[i] = a; w[j] = 1 - a
                out.append((f"{GAITS[i]}{a:.2f}+{GAITS[j]}{1-a:.2f}", w / np.linalg.norm(w)))
    if name in ("deep", "all"):
        for trio in itertools.combinations(range(4), 3):
            w = np.zeros(4, np.float32)
            for i in trio: w[i] = 1.0
            out.append(("+".join(GAITS[i] for i in trio), w / np.linalg.norm(w)))
        out.append(("all four", np.full(4, 0.5, np.float32)))
    return out


def command_suite(name: str):
    """(label, vx, vy, vyaw).  The training box is vx 0.6-1.0, vy +-0.15, vyaw +-0.3."""
    inbox = [("vx 0.6", 0.6, 0.0, 0.0), ("vx 0.8", 0.8, 0.0, 0.0),
             ("vx 1.0", 1.0, 0.0, 0.0), ("turn +0.3", 0.8, 0.0, 0.3),
             ("turn -0.3", 0.8, 0.0, -0.3), ("lateral", 0.8, 0.12, 0.0)]
    outbox = [("vx 1.3 !", 1.3, 0.0, 0.0), ("vx 1.6 !", 1.6, 0.0, 0.0),
              ("turn +0.6 !", 0.8, 0.0, 0.6), ("vx 0.3 !", 0.3, 0.0, 0.0)]
    return {"inbox": inbox, "outbox": outbox, "all": inbox + outbox}[name]


def build(args):
    dial_config, env_config = _load_config("unitree_go2_gait", None)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env_config = dataclasses.replace(
        env_config, randomize_start_state=True,
        start_height_noise=0.02, start_rpy_noise=0.05,
        start_joint_position_noise=0.05)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    return env, dial_config


def make_reset(env, n_seeds):
    """A batch of randomised resets, pinned to one command and weight."""
    def one(key, cmd, omega):
        st = env.reset(key)
        vel = jnp.array([cmd[0], cmd[1], 0.0])
        ang = jnp.array([0.0, 0.0, cmd[2]])
        scale = env._command_ramp_scale(0)
        info = {**st.info, "vel_cmd": vel, "ang_vel_cmd": ang,
                "vel_tar": vel * scale, "ang_vel_tar": ang * scale,
                "reward_weights": omega}
        st = st.replace(info=info)
        return st.replace(obs=env._get_obs(st.pipeline_state, st.info))
    return jax.vmap(one, in_axes=(0, None, None))


def csm_arm(env, dial_config, policy, step_passes, init_passes):
    """Composed fields: carry the plan, refine, act on its first node."""
    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    lo, hi = float(jnp.min(factors)), float(jnp.max(factors))
    shift = jnp.asarray(fields[0].shift_matrix)
    pinv_nu = getattr(policy, "pinv_nu_weights", None)
    pinv_mode = policy.pinv_mode_weights
    horizon = int(dial_config.Hnode) + 1

    def refine(plan, obs, mixture, passes):
        def level(carry, factor):
            t = factor_to_t(factor, lo, hi).reshape(1)
            parts = jnp.stack([f.delta(carry, obs, t) for f in fields])
            return jnp.clip(carry + jnp.einsum("k,kij->ij", mixture, parts), -1.0, 1.0), None
        return jax.lax.scan(level, plan, jnp.tile(factors, passes))[0]

    def init(state, omega, temperature):
        mixture = mixture_from_pinv(omega, temperature, pinv_nu, pinv_mode)
        plan = jnp.zeros((horizon, int(env.action_size)))
        return refine(plan, state.obs, mixture, init_passes), mixture

    def act(carry, state, omega, temperature):
        plan, mixture = carry
        plan = refine(plan, state.obs, mixture, step_passes)
        return (jnp.einsum("ij,ja->ia", shift, plan), mixture), plan[0]

    return init, act


def rl_mix_arm(infers):
    """Every specialist, averaged with coefficients read off omega.

    A pure weight collapses this to that gait's own policy, a pair to the
    two-policy average, a three-way to a three-policy one -- so the same
    program covers `rl_spec`, `rl_blend` and the deeper mixtures, and the
    coefficients are exactly the ones the composed fields are given.
    """
    def init(state, omega, temperature):
        return jnp.zeros(())

    def act(carry, state, omega, temperature):
        c = omega / jnp.maximum(omega.sum(), 1e-8)
        key = jax.random.PRNGKey(0)
        actions = jnp.stack([f(state.obs, key)[0] for f in infers])
        return carry, jnp.einsum("k,ka->a", c, actions)

    return init, act


def rl_cond_arm(infer):
    def init(state, omega, temperature):
        return jnp.zeros(())

    def act(carry, state, omega, temperature):
        return carry, infer(jnp.concatenate([state.obs, omega]), jax.random.PRNGKey(0))[0]

    return init, act


def make_rollout(env, reset_fn, init, act, steps):
    """`(keys, cmd, omega, temperature) -> traces`, compiled once for this arm."""
    ti = env._torso_idx - 1

    @jax.jit
    def run(keys, cmd, omega, temperature):
        states = reset_fn(keys, cmd, omega)
        carry = jax.vmap(init, in_axes=(0, None, None))(states, omega, temperature)

        def body(state_carry, _):
            st, cr = state_carry
            cr, action = jax.vmap(act, in_axes=(0, 0, None, None))(cr, st, omega, temperature)
            st = jax.vmap(env.step)(st, action)
            ps = st.pipeline_state
            zf = ps.site_xpos[:, env._feet_site_id, 2] - env._foot_radius
            vb = jax.vmap(lambda v, q: math.rotate(v, math.quat_inv(q)))(
                ps.xd.vel[:, ti], ps.x.rot[:, ti])
            yaw = jax.vmap(lambda q: math.quat_to_euler(q)[2])(ps.x.rot[:, ti])
            return (st, cr), ((zf < 0.02).astype(jnp.float32),
                              jnp.linalg.norm(vb[:, :2], axis=-1), st.done,
                              ps.x.pos[:, ti, 2], yaw)

        return jax.lax.scan(body, (states, carry), None, length=steps)[1]

    return run


def score(contact, speed, done, z, yaw, cmd, dt, steps):
    """Per-seed metrics from the batched traces (time-major)."""
    out = []
    for s in range(contact.shape[1]):
        c, sp, d, zz, yy = (contact[:, s], speed[:, s], done[:, s] > 0.5,
                            z[:, s], np.unwrap(yaw[:, s]))
        run = best = 0
        for v in d:
            run = run + 1 if v else 0; best = max(best, run)
        collapsed = best >= COLLAPSE_RUN or zz[-30:].mean() < COLLAPSE_Z
        w = c[WIN:]
        def cor(i, j):
            if w[:, i].std() < 1e-6 or w[:, j].std() < 1e-6: return np.nan
            return float(np.corrcoef(w[:, i], w[:, j])[0, 1])
        trip = (np.nanmean([cor(0, 3), cor(1, 2)]), np.nanmean([cor(0, 2), cor(1, 3)]),
                np.nanmean([cor(0, 1), cor(2, 3)]))
        span = (steps - WIN) * dt
        drift = np.degrees(yy[-1] - yy[WIN]) - np.degrees(cmd[2]) * span
        out.append((trip, sp[WIN:].mean() / max(cmd[0], 1e-6), drift, collapsed))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csm-policy", type=Path, required=True)
    p.add_argument("--rl-specialists", type=Path, nargs=4, required=True)
    p.add_argument("--rl-conditioned", type=Path, required=True)
    p.add_argument("--weights", default="pure", choices=("pure", "pairs", "deep", "all"))
    p.add_argument("--commands", default="inbox", choices=("inbox", "outbox", "all"))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--step-passes", type=int, default=6)
    p.add_argument("--init-passes", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("csm_runs/head2head.json"))
    args = p.parse_args()

    env, dial_config = build(args)
    keys = jax.random.split(jax.random.PRNGKey(20260916), args.seeds)
    dt = float(env.dt)

    policy = ComposedDialScorePolicy.load(args.csm_policy)
    spec = [load_policy(q)[0] for q in args.rl_specialists]
    cond = load_policy(args.rl_conditioned)[0]

    weights = weight_suite(args.weights)
    commands = command_suite(args.commands)
    flush = functools.partial(print, flush=True)

    # One compiled program per arm; weights and commands are traced arguments,
    # so the 30-odd combinations below re-use three compilations rather than
    # forcing one each.
    reset_j = make_reset(env, args.seeds)
    runners = {
        "csm": make_rollout(env, reset_j, *csm_arm(env, dial_config, policy,
                                                   args.step_passes, args.init_passes),
                            args.steps),
        "rl_mix": make_rollout(env, reset_j, *rl_mix_arm(spec), args.steps),
        "rl_cond": make_rollout(env, reset_j, *rl_cond_arm(cond), args.steps),
    }

    flush(f"head to head: {len(weights)} weights x {len(commands)} commands x "
          f"{args.seeds} randomised resets, {args.steps} steps")
    flush(f"{'weight':<22}{'command':<12}{'arm':<9}{'diag':>7}{'lat':>7}{'f-h':>7}"
          f"{'v/cmd':>7}{'drift':>8}{'down':>6}")

    rows = []
    temp = jnp.asarray(args.temperature, jnp.float32)
    for label, omega in weights:
        om = jnp.asarray(omega, jnp.float32)
        for cl, vx, vy, vyaw in commands:
            cmd = (vx, vy, vyaw)
            cm = jnp.asarray(cmd, jnp.float32)
            for arm, run in runners.items():
                tr = [np.asarray(t) for t in run(keys, cm, om, temp)]
                per = score(*tr, cmd, dt, args.steps)
                alive = [m for m in per if not m[3]]
                down = sum(1 for m in per if m[3])
                if alive:
                    d, l, f = np.nanmean([m[0] for m in alive], axis=0)
                    v = float(np.mean([m[1] for m in alive]))
                    dr = float(np.mean([m[2] for m in alive]))
                else:
                    d = l = f = v = dr = float("nan")
                rows.append(dict(weight=label, omega=[float(x) for x in omega],
                                 command=cl, cmd=list(cmd), arm=arm,
                                 diag=float(d), lat=float(l), fh=float(f), v=v,
                                 drift=dr, down=down, seeds=args.seeds,
                                 per_seed=[[list(map(float, m[0])), float(m[1]),
                                            float(m[2]), bool(m[3])] for m in per]))
                flush(f"{label:<22}{cl:<12}{arm:<9}{d:7.2f}{l:7.2f}{f:7.2f}"
                      f"{v:7.2f}{dr:+8.0f}{down:>4}/{args.seeds}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    flush(f"wrote {args.out}")
    flush("HEAD2HEAD DONE")


if __name__ == "__main__":
    main()
