"""What each controller costs per control step, against the 20 ms period.

Quality is only half the comparison; a controller that cannot produce an action
inside the control period is not a controller.  DIAL plans from scratch every
step over a cloud of `Nsample` rollouts, the composed fields run `k` networks
`step_passes` times, and a PPO policy is a single forward pass.  Those are
three different orders of magnitude and they decide what can actually be
deployed.

Timed on device with the inputs a real step gets, after a warm-up so the
measurement is the step and not the compilation, and reported as a multiple of
the environment's own `dt`.
"""
from __future__ import annotations

import argparse, dataclasses, statistics, time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import brax.envs as brax_envs

import dial_mpc.envs  # noqa: F401
from dial_mpc.core.dial_core import make_controller
from csm.basis_screen import _load_config
from csm.dial_lean import make_dial_step
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import mixture_from_pinv
from csm.rl_baseline import load_policy


def timeit(fn, args, repeats, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        xs.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(xs), min(xs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csm-policy", type=Path, required=True)
    p.add_argument("--rl-policy", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--step-passes", type=int, nargs="+", default=[1, 6])
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    dial_config, env_config = _load_config("unitree_go2_gait", None)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    dt = float(env.dt)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    omega = jnp.asarray([0.0, 1.0, 0.0, 0.0], jnp.float32)
    state = state.replace(info={**state.info, "reward_weights": omega})
    temp = jnp.asarray(args.temperature, jnp.float32)

    print(f"control period dt = {dt * 1e3:.0f} ms   (Nsample {dial_config.Nsample}, "
          f"Hsample {dial_config.Hsample}, Ndiffuse {dial_config.Ndiffuse})")
    print(f"{'controller':<34}{'ms/step':>9}{'best':>8}{'x dt':>8}{'Hz':>8}")
    out = []

    # --- DIAL, the teacher: a fresh sample cloud every step -----------------
    mbdpi = make_controller(dial_config, env)
    ctrl = jax.jit(make_dial_step(env, mbdpi, dial_config, std_normalize=False,
                                  level_scales=(1.0,) * dial_config.Ndiffuse))
    plan = jnp.zeros((dial_config.Hnode + 1, int(env.action_size)))
    med, best = timeit(lambda s, k, p_: ctrl(s, k, p_),
                       (state, jax.random.PRNGKey(0), plan), args.repeats)
    out.append(("DIAL-MPC (teacher)", med, best))

    # --- the composed fields ------------------------------------------------
    policy = ComposedDialScorePolicy.load(args.csm_policy)
    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    lo, hi = float(jnp.min(factors)), float(jnp.max(factors))
    shift = jnp.asarray(fields[0].shift_matrix)
    pinv_nu = getattr(policy, "pinv_nu_weights", None)

    for passes in args.step_passes:
        def step(st, pl, om, tp, passes=passes):
            mixture = mixture_from_pinv(om, tp, pinv_nu, policy.pinv_mode_weights)
            def level(carry, factor):
                t = factor_to_t(factor, lo, hi).reshape(1)
                parts = jnp.stack([f.delta(carry, st.obs, t) for f in fields])
                return jnp.clip(carry + jnp.einsum("k,kij->ij", mixture, parts), -1.0, 1.0), None
            pl = jax.lax.scan(level, pl, jnp.tile(factors, passes))[0]
            return jnp.einsum("ij,ja->ia", shift, pl), pl[0]
        j = jax.jit(step)
        med, best = timeit(j, (state, plan, omega, temp), args.repeats)
        out.append((f"CSM composed ({len(fields)} fields x {passes} passes)", med, best))

    # --- one PPO forward pass ----------------------------------------------
    infer = load_policy(args.rl_policy)[0]
    j = jax.jit(lambda o, k: infer(o, k)[0])
    med, best = timeit(j, (state.obs, jax.random.PRNGKey(0)), args.repeats)
    out.append(("PPO policy (1 forward pass)", med, best))

    for name, med, best in out:
        print(f"{name:<34}{med:9.2f}{best:8.2f}{med / (dt * 1e3):8.1f}{1e3 / med:8.0f}")
    print("\nx dt above 1.0 cannot run in real time at this control period.")
    print("DIAL / CSM ratio:", round(out[0][1] / out[-2][1], 1),
          "  DIAL / PPO ratio:", round(out[0][1] / out[-1][1], 1))


if __name__ == "__main__":
    main()
