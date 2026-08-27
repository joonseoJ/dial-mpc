"""Score a composed student against DIAL on the disturbance-rejection task.

Two things are being asked, and they are not the same question.

*Does the student reproduce DIAL?*  Answered per weight vector by driving both
from identical initial states with identical pushes and comparing the reward
and the behaviour.  The reward ratio alone is weak here -- DIAL's own
seed-to-seed noise swallows small differences -- so the recovery strategy is
measured directly.

*Does composition work?*  Answered by the spread between weights.  A composed
policy that tracks DIAL's mean reward but produces the same recovery at every
omega has learned an average, not a family.  The comparison that matters is
whether the student's P(step)-versus-push curve moves with omega the way DIAL's
does, since that midpoint is the "how hard before I stop standing my ground"
threshold the weights are supposed to control.

Basis rows are included as targets on purpose: composition reduces to a single
field there, so any gap at a basis row is the field's own fitting error and
bounds what the interior targets can achieve.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import brax.envs as brax_envs
from brax import math as brax_math

from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step, make_lean_update
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import normalize_omega_np
from csm.push_recover_eval import make_push_reset, summarise


def make_observer(env):
    nominal_feet = env._nominal_foot_pos
    nominal_xy = env._nominal_base_xy
    torso = env._torso_idx - 1
    up = jnp.array([0.0, 0.0, 1.0])

    def observe(state):
        ps = state.pipeline_state
        feet = ps.site_xpos[env._feet_site_id]
        return {
            "foot_shift": jnp.linalg.norm(
                feet[:, :2] - nominal_feet[:, :2], axis=-1
            ).max(),
            "lifted": state.info["feet_lifted"],
            "tilt": jnp.linalg.norm(brax_math.rotate(up, ps.x.rot[torso]) - up),
            "base_drift": jnp.linalg.norm(ps.x.pos[torso][:2] - nominal_xy),
            "height": ps.x.pos[torso][2],
            "joint_dev": jnp.linalg.norm(ps.q[7:] - env._default_pose),
            "done": state.done,
            "reward": state.reward,
            "reward_terms": state.info["reward_terms"],
        }

    return observe


def make_student_episode(env, policy, dial_config, n_steps, init_passes):
    """Closed loop on the composed score fields, one coefficient set per omega."""

    observe = make_observer(env)
    factors = jnp.asarray(policy.policies[0].factors)
    factor_min = float(jnp.min(factors))
    factor_max = float(jnp.max(factors))
    shift = jnp.asarray(policy.policies[0].shift_matrix)

    def refine(plan, obs, coefficients, passes):
        def one_level(carry, factor):
            current = carry
            t = factor_to_t(factor, factor_min, factor_max).reshape(1)
            update = None
            for index, field in enumerate(policy.policies):
                term = coefficients[index] * field.delta(current, obs, t)
                update = term if update is None else update + term
            return jnp.clip(current + update, -1.0, 1.0), None

        schedule = jnp.tile(factors, passes)
        plan, _ = jax.lax.scan(one_level, plan, schedule)
        return plan

    def episode(state, coefficients):
        plan = jnp.zeros(
            (dial_config.Hnode + 1, int(env.action_size)), dtype=jnp.float32
        )
        plan = refine(plan, state.obs, coefficients, init_passes)

        def body(carry, _):
            st, pl = carry
            st = env.step(st, pl[0])
            pl = jnp.einsum("ij,ja->ia", shift, pl)
            pl = refine(pl, st.obs, coefficients, 1)
            return (st, pl), observe(st)

        _, traj = jax.lax.scan(body, (state, plan), None, length=n_steps)
        return traj

    return episode


def make_dial_episode(env, mbdpi, dial_config, n_steps, level_scales,
                      init_passes: int = 1):
    """DIAL, given the same head start on its first plan as the student.

    The student refines from zero with `init_passes` tiles of the annealing
    schedule before it acts; DIAL's control step anneals once.  Leaving that
    asymmetry in place hands the student a better opening plan on an episode
    that begins with a shove, and the first control step is exactly where a
    recovery is decided -- so DIAL gets the same treatment here.
    """

    observe = make_observer(env)
    control = make_dial_step(
        env, mbdpi, dial_config, std_normalize=False, level_scales=level_scales
    )
    update = make_lean_update(env, mbdpi, dial_config, std_normalize=False)
    sigma = mbdpi.sigma_control
    factors = dial_config.traj_diffuse_factor ** jnp.arange(dial_config.Ndiffuse)
    scales = jnp.asarray(level_scales, dtype=factors.dtype)
    temps = scales * dial_config.temp_sample

    def episode(state, rng):
        plan = jnp.zeros(
            (dial_config.Hnode + 1, int(env.action_size)), dtype=jnp.float32
        )

        def warm(carry, level):
            key, current = carry
            factor, temp = level
            key, current = update(state, key, current, sigma * factor, temp)
            return (key, current), None

        schedule = (jnp.tile(factors, init_passes), jnp.tile(temps, init_passes))
        (rng, plan), _ = jax.lax.scan(warm, (rng, plan), schedule)

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control(st, key, pl)
            return (st, key, pl), observe(st)

        _, traj = jax.lax.scan(body, (state, rng, plan), None, length=n_steps)
        return traj

    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str, default=None)
    source.add_argument("--config", type=str, default=None)
    parser.add_argument("--targets", nargs="+",
                        default=["boost0", "boost1", "boost2", "boost3",
                                 "uniform", "2,1,1,1", "1,2,2,1", "1,1,2,2"])
    parser.add_argument("--speeds", type=float, nargs="+",
                        default=[0.25, 0.45, 0.70, 0.90])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--init-passes", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--level-scales", type=float, nargs="+",
                        default=[3.175, 1.0])
    parser.add_argument("--step-threshold", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, args.config)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    policy = ComposedDialScorePolicy.load(args.policy)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    targets, omegas = [], []
    for name in args.targets:
        if name in catalogue:
            omegas.append(np.asarray(catalogue[name]))
        else:
            omegas.append(normalize_omega_np(
                np.array([float(v) for v in name.split(",")])
            ))
        targets.append(name)
    omega_array = jnp.asarray(np.stack(omegas))
    coefficients = jnp.stack([
        policy.coefficients(omega_array[i]) for i in range(len(targets))
    ])
    print("[eval] composition coefficients")
    for name, row in zip(targets, np.asarray(coefficients)):
        print(f"  {name:<10} {np.round(row, 4)}")

    speeds = jnp.asarray(args.speeds)
    n_target, n_speed, n_seed = len(targets), len(speeds), args.seeds
    rng = jax.random.PRNGKey(args.seed)
    rng, head_rng, reset_rng, roll_rng = jax.random.split(rng, 4)
    headings = jax.random.uniform(head_rng, (n_seed,), minval=-jnp.pi, maxval=jnp.pi)
    reset_keys = jax.random.split(reset_rng, n_seed)
    roll_keys = jax.random.split(roll_rng, n_seed)

    push_reset = make_push_reset(env)
    student = make_student_episode(
        env, policy, dial_config, args.steps, args.init_passes
    )
    teacher = make_dial_episode(
        env, mbdpi, dial_config, args.steps, tuple(args.level_scales),
        args.init_passes,
    )

    grid = jnp.meshgrid(
        jnp.arange(n_target), jnp.arange(n_speed), jnp.arange(n_seed),
        indexing="ij",
    )
    flat = [g.ravel() for g in grid]
    total = flat[0].size

    def run_student(it, is_, id_):
        state = push_reset(reset_keys[id_], speeds[is_], headings[id_],
                           omega_array[it])
        return summarise(student(state, coefficients[it]), args.step_threshold)

    def run_teacher(it, is_, id_):
        state = push_reset(reset_keys[id_], speeds[is_], headings[id_],
                           omega_array[it])
        return summarise(teacher(state, roll_keys[id_]), args.step_threshold)

    def sweep(fn, chunk, label):
        pad = (-total) % chunk
        padded = [jnp.pad(f, (0, pad)).reshape(-1, chunk) for f in flat]

        @jax.jit
        def run():
            out = jax.lax.map(lambda c: jax.vmap(fn)(*c), tuple(padded))
            return jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:])[:total], out
            )

        started = time.time()
        result = run()
        jax.block_until_ready(result)
        print(f"[eval] {label}: {total} episodes in {time.time() - started:.1f}s")
        return {k: np.asarray(v) for k, v in result.items()}

    student_out = sweep(run_student, min(args.chunk * 4, total), "student")
    teacher_out = sweep(run_teacher, min(args.chunk, total), "DIAL")

    idx_t, idx_s = np.asarray(flat[0]), np.asarray(flat[1])

    def grid_of(source, key):
        out = np.zeros((n_target, n_speed))
        for i in range(n_target):
            for j in range(n_speed):
                out[i, j] = source[key][(idx_t == i) & (idx_s == j)].mean()
        return out

    width = max(len(n) for n in targets) + 2
    header = " ".join(f"{float(v):>5.2f}" for v in speeds)
    report = {}
    for key, title in (("stepped", "P(step)"), ("mean_tilt", "mean torso tilt"),
                       ("mean_drift", "mean base drift"),
                       ("lift_time", "foot air-time"), ("fell", "fall rate")):
        a, b = grid_of(student_out, key), grid_of(teacher_out, key)
        report[key] = {"student": a.tolist(), "dial": b.tolist()}
        print(f"\n=== {title} ===")
        print(f"{'':<{width}}{'student':^{len(header)}} | {'DIAL':^{len(header)}}")
        print(f"{'target':<{width}}{header} | {header}")
        for i, name in enumerate(targets):
            print(f"{name:<{width}}" + " ".join(f"{v:5.2f}" for v in a[i])
                  + " | " + " ".join(f"{v:5.2f}" for v in b[i]))

    # Cost, and the ratio the student is judged on.
    cost_s = -grid_of(student_out, "terms").mean(axis=1) if False else None
    reward_s = np.zeros(n_target)
    reward_d = np.zeros(n_target)
    for i in range(n_target):
        sel = idx_t == i
        weights = np.asarray(omegas[i])
        reward_s[i] = float((student_out["terms"][sel] @ weights).mean())
        reward_d[i] = float((teacher_out["terms"][sel] @ weights).mean())
    print(f"\n=== mean reward under each target's own weight ===")
    print(f"{'target':<{width}}{'student':>10}{'DIAL':>10}{'ratio':>9}")
    ratios = {}
    for i, name in enumerate(targets):
        ratio = reward_s[i] / reward_d[i]
        ratios[name] = ratio
        print(f"{name:<{width}}{reward_s[i]:10.4f}{reward_d[i]:10.4f}{ratio:9.3f}")

    def spread(source, key):
        g = grid_of(source, key)
        return float((g.max(0) - g.min(0)).mean())

    print(f"\n=== target-spread of behaviour (student vs DIAL) ===")
    print(f"{'metric':<20}{'student':>10}{'DIAL':>10}{'kept':>8}")
    spreads = {}
    for key in ("stepped", "mean_tilt", "mean_drift", "lift_time"):
        a, b = spread(student_out, key), spread(teacher_out, key)
        spreads[key] = {"student": a, "dial": b}
        print(f"{key:<20}{a:10.4f}{b:10.4f}{a / max(b, 1e-9):8.2f}x")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "policy": str(args.policy), "targets": targets,
            "speeds": [float(v) for v in speeds], "seeds": n_seed,
            "coefficients": np.asarray(coefficients).tolist(),
            "reward_ratio": ratios, "behaviour": report, "spreads": spreads,
        }, indent=2))
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
