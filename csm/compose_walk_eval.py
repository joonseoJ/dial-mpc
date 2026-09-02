"""Composition quality on a locomotion basis: student against its own teacher.

`compose_eval` measures the push-recovery basis, and everything it reports is
about a push -- reset with an impulse, count the steps taken, plot P(step)
against push magnitude.  None of that exists here.  What a walking basis has
instead is the question the walking report answered badly: at a target weight
that is *not* one of the trained fields, does the composed policy walk as well
as DIAL does at that same weight?

So each target weight is run twice from the same state with the same command --
once by DIAL, once by the composed fields -- and the reported number is the
ratio of mean cost.  One means the student matches its teacher; the report's
earlier numbers on this axis were 1.11 for a single field and 1.49 to 2.18 for
compositions, which is what "composing made it worse" meant.

The row normalizers are arguments because two policies fitted under different
ones cannot be compared by cost at all; each has to be scored against the DIAL
that shares its objective.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step, make_lean_update
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import mixture_from_pinv, normalize_omega_np
from csm.screen import COMMANDS, set_command, set_omega


def _reward_done(state):
    return state.reward, state.done


def make_student(env, policy, dial_config, init_passes, n_steps,
                 record=_reward_done):
    """The composed policy's own control loop, mixing solved per target.

    `record` picks what each step contributes to the returned trajectory.  The
    default is what scoring needs; the report's renderer passes one that keeps
    `pipeline_state` instead, so the pictures come out of this loop rather than
    a second copy of it that could drift away from the measured one.
    """

    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    lo, hi = float(jnp.min(factors)), float(jnp.max(factors))
    shift = jnp.asarray(fields[0].shift_matrix)
    pinv_nu = getattr(policy, "pinv_nu_weights", None)
    pinv_mode = policy.pinv_mode_weights

    def refine(plan, obs, mixture, passes):
        def level(carry, factor):
            t = factor_to_t(factor, lo, hi).reshape(1)
            parts = jnp.stack([f.delta(carry, obs, t) for f in fields])
            return jnp.clip(
                carry + jnp.einsum("k,kij->ij", mixture, parts), -1.0, 1.0
            ), None

        plan, _ = jax.lax.scan(level, plan, jnp.tile(factors, passes))
        return plan

    @jax.jit
    def run(state, omega, temperature):
        mixture = mixture_from_pinv(omega, temperature, pinv_nu, pinv_mode)
        plan = refine(
            jnp.zeros((dial_config.Hnode + 1, int(env.action_size))),
            state.obs, mixture, init_passes,
        )

        def body(carry, _):
            st, pl = carry
            st = env.step(st, pl[0])
            pl = refine(jnp.einsum("ij,ja->ia", shift, pl), st.obs, mixture, 1)
            return (st, pl), record(st)

        _, out = jax.lax.scan(body, (state, plan), None, length=n_steps)
        return out

    return run


def make_teacher(env, mbdpi, dial_config, init_passes, std_normalize,
                 n_steps, level_scales, record=_reward_done):
    """DIAL at the same weight, from the same state.

    `level_scales` is not optional.  Raw Gibbs means the temperature is the only
    thing setting the softmax's sharpness, and the coarse annealing level's
    returns spread about four times as wide as the fine level's -- run both at
    one temperature and the coarse update is nearly the mean of a cloud in which
    a tenth of the samples fall over.  Omitting it here made the teacher fall in
    every single episode while the student walked, which is not a result about
    composition.
    """

    control = make_dial_step(env, mbdpi, dial_config,
                             std_normalize=std_normalize,
                             level_scales=level_scales)
    update = make_lean_update(env, mbdpi, dial_config, std_normalize)
    sigma = mbdpi.sigma_control
    factors = dial_config.traj_diffuse_factor ** jnp.arange(dial_config.Ndiffuse)

    @jax.jit
    def run(state, rng):
        plan = jnp.zeros((dial_config.Hnode + 1, int(env.action_size)))

        def warm(carry, factor):
            key, cur = carry
            key, cur = update(state, key, cur, sigma * factor)
            return (key, cur), None

        (rng, plan), _ = jax.lax.scan(
            warm, (rng, plan), jnp.tile(factors, init_passes)
        )

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control(st, key, pl)
            return (st, key, pl), record(st)

        _, out = jax.lax.scan(body, (state, rng, plan), None, length=n_steps)
        return out

    return run


def summarise(reward, done, n_steps):
    """Mean cost over the whole episode, and how long it stayed up.

    Truncating at the first fall scores a policy that goes down at step ten on
    the ten steps before it did, which is exactly when the cost has not yet
    accumulated -- an earlier version of this did that and reported the fallen
    policies as *better* than DIAL.  The environment keeps stepping after
    `done`, so the full-episode mean already prices lying on the ground.
    """

    done = np.asarray(done)
    alive = int(np.argmax(done > 0.5)) if np.any(done > 0.5) else n_steps
    return -float(np.asarray(reward).mean()), bool(np.any(done > 0.5)), alive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--example", default="unitree_go2_trot_csm")
    parser.add_argument("--targets", nargs="+",
                        default=["uniform", "boost0", "boost1", "boost2",
                                 "2,1,1", "1,2,1", "1,1,2", "3,1,2"])
    parser.add_argument("--commands", default="train",
                        help="`train` is the command the data was collected "
                             "at, read from the config; anything else names an "
                             "entry in csm.screen.COMMANDS and is off the "
                             "training distribution unless the collection "
                             "randomised its command")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--init-passes", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--row-scales", type=float, nargs="+", default=None,
                        help="track/stability/gait normalizers the policy was "
                             "fitted under; required to score it against the "
                             "DIAL that shares its objective")
    # The collection runs DIAL as a raw Gibbs distribution, which is what makes
    # the score linear in the weights; scoring the student against a teacher
    # that keeps the spread normalisation compares it to a controller it was
    # never shown.
    parser.add_argument("--std-normalize", action="store_true")
    parser.add_argument("--level-scales", type=float, nargs="+",
                        default=[4.074, 1.0],
                        help="per-level temperature profile the data was "
                             "collected under; the teacher has to share it")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    if args.row_scales:
        track, stability, gait = args.row_scales
        env_config = dataclasses.replace(
            env_config, track_scale=track, stability_scale=stability,
            gait_scale=gait,
        )
    policy = ComposedDialScorePolicy.load(args.policy)
    temperature = args.temperature or float(policy.temperature or
                                            dial_config.temp_sample)
    dial_config = dataclasses.replace(dial_config, temp_sample=temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    reset = jax.jit(env.reset)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    targets = {}
    for name in args.targets:
        targets[name] = (catalogue[name] if name in catalogue
                         else normalize_omega_np(
                             np.array([float(v) for v in name.split(",")])))

    student = make_student(env, policy, dial_config, args.init_passes,
                           args.steps)
    teacher = make_teacher(env, mbdpi, dial_config, args.init_passes,
                           args.std_normalize, args.steps,
                           tuple(args.level_scales))

    print(f"policy {args.policy}")
    print(f"env {dial_config.env_name}  T={temperature}  "
          f"levels={args.level_scales}  "
          f"row_scales={args.row_scales or 'from config'}  "
          f"fields={len(policy.policies)}  nu={policy.pinv_nu_weights is not None}")
    report = {}
    train_command = (float(env_config.default_vx), float(env_config.default_vy),
                     float(env_config.default_vyaw))
    for cname in args.commands.split(","):
        cname = cname.strip()
        command = (train_command if cname == "train" else COMMANDS[cname])
        print(f"\n=== command {cname} = {command} ===")
        print(f"{'target':<9}{'DIAL cost':>11}{'student':>10}{'ratio':>8}"
              f"{'D.fall':>8}{'S.fall':>8}{'S.alive':>9}")
        for name, omega in targets.items():
            teach, stud, tf, sf, alive = [], [], 0, 0, []
            for seed in range(args.seeds):
                state = reset(jax.random.PRNGKey(11 + seed))
                state = set_command(env, state, command)
                state = set_omega(state, omega)
                r, d = teacher(state, jax.random.PRNGKey(seed))
                c, fell, _ = summarise(r, d, args.steps)
                teach.append(c); tf += fell
                r, d = student(state, jnp.asarray(omega), temperature)
                c, fell, a = summarise(r, d, args.steps)
                stud.append(c); sf += fell; alive.append(a)
            t, s = float(np.mean(teach)), float(np.mean(stud))
            report[f"{cname}/{name}"] = {
                "dial_cost": t, "student_cost": s, "ratio": s / max(t, 1e-9),
                "dial_falls": tf, "student_falls": sf,
                "student_alive": float(np.mean(alive)),
            }
            print(f"{name:<9}{t:11.4f}{s:10.4f}{s / max(t, 1e-9):8.3f}"
                  f"{tf:8d}{sf:8d}{np.mean(alive):9.0f}", flush=True)

    ratios = [v["ratio"] for v in report.values()]
    print(f"\nmean ratio {np.mean(ratios):.3f}   "
          f"student falls {sum(v['student_falls'] for v in report.values())}   "
          f"DIAL falls {sum(v['dial_falls'] for v in report.values())}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
