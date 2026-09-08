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
from dial_mpc.core.dial_core import MBDPI, make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step, make_lean_update, mppi_logits
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import mixture_from_pinv, normalize_omega_np
from csm.rl_baseline import load_policy as load_rl_policy
from csm.screen import COMMANDS, set_command, set_omega
from csm.teacher_cache import (DEFAULT_ROOT, TeacherCache, episode_key,
                               fingerprint)


def _reward_done(state):
    return state.reward, state.done


def _identity(state):
    return state


def make_student(env, policy, dial_config, init_passes, n_steps,
                 record=_reward_done, transform=_identity, absolute=False):
    """The composed policy's own control loop, mixing solved per target.

    `record` picks what each step contributes to the returned trajectory.  The
    default is what scoring needs; the report's renderer passes one that keeps
    `pipeline_state` instead, so the pictures come out of this loop rather than
    a second copy of it that could drift away from the measured one.

    `transform` is applied to the state after every step, for diagnostics that
    need to intervene on it -- the observation-symmetry probe re-centres the
    base through here.  Anything it does is invisible to the objective only if
    the objective is invariant to it; that is the caller's problem, not this
    loop's.

    `absolute` switches the refinement from `plan + mix(deltas)` to `mix(plan
    predictions)`, for a field fitted with `fit_from_clouds --absolute`.  That
    is behaviour cloning of DIAL, and it is only meaningful for a single field
    at the weight it was fitted at: mixing absolute plans across weights is not
    a composition of anything.  The loop is otherwise identical, so the two
    arms are compared through the same code.
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
            mixed = jnp.einsum("k,kij->ij", mixture, parts)
            return jnp.clip(mixed if absolute else carry + mixed,
                            -1.0, 1.0), None

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
            st = transform(st)
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


def make_rl_student(env, inference, n_steps, record=_reward_done,
                    append_omega=False):
    """A trained-at-one-weight RL policy, run through the same loop.

    Signature-compatible with `make_student` so the scoring path, the cached
    teacher and the summary are literally the same code.  `omega` and
    `temperature` are accepted and ignored: an RL policy has no weight input,
    which is the whole point of the comparison -- it answers for the one weight
    it was trained at and the caller is responsible for asking only that one.

    `append_omega` is the exception, for a policy trained with
    `--condition-omega`.  That one does have a weight input, so the target is
    appended to the observation exactly as its training wrapper did, and it can
    be scored across the whole target list like the composed student.
    """

    @jax.jit
    def run(state, omega, temperature):
        def body(carry, _):
            st, key = carry
            key, sub = jax.random.split(key)
            obs = jnp.concatenate([st.obs, omega]) if append_omega else st.obs
            action, _ = inference(obs, sub)
            st = env.step(st, action)
            return (st, key), record(st)

        _, out = jax.lax.scan(
            body, (state, jax.random.PRNGKey(0)), None, length=n_steps)
        return out

    return run


def make_dial_student(env, dial_config, init_passes, std_normalize, n_steps,
                      level_scales, overrides):
    """A DIAL variant scored as a student against the configured DIAL.

    The planner arms -- the published algorithm with its spread normalisation
    on, and a DIAL cut down until it fits the control period -- are controllers
    like any other, so they belong in the student slot rather than in a second
    script that would drift away from this one.  `overrides` is applied to the
    planner's own config, and `make_controller` is rebuilt from it because
    `Nsample` and the horizon are baked into the sampler.

    The target weight needs no argument: `set_omega` has already written it
    into `state.info`, and the environment's reward reads it from there, so a
    DIAL variant plans for whatever weight the row is about.

    One limitation worth stating. The sampling key is fixed, so the two seeds
    of a row differ by their initial state but not by the planner's noise.
    That understates this arm's seed-to-seed spread; it does not bias its mean,
    and the initial state is where the variation in this grid actually comes
    from.
    """

    cfg = dataclasses.replace(dial_config, **overrides)
    mbdpi = make_controller(cfg, env)
    run = make_teacher(env, mbdpi, cfg, init_passes, std_normalize, n_steps,
                       level_scales)

    def student(state, omega, temperature):
        return run(state, jax.random.PRNGKey(0))

    return student


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy", type=Path,
                        help="a composed score-field policy")
    # The RL baseline is trained at one weight and has no weight input, so
    # it can only be scored at that weight -- the target list comes from
    # the policy rather than the command line, and asking for others would
    # quietly report a policy answering a question it was never asked.
    source.add_argument("--rl-policy", type=Path,
                        help="a fixed-weight policy from csm.rl_baseline")
    source.add_argument("--dial-student", action="store_true",
                        help="score a DIAL variant against the configured "
                             "DIAL; combine with --student-samples, "
                             "--student-diffuse, --student-horizon and "
                             "--student-std-normalize")
    parser.add_argument("--student-samples", type=int, default=None)
    parser.add_argument("--student-diffuse", type=int, default=None)
    parser.add_argument("--student-horizon", type=int, default=None,
                        help="Hsample: the rollout length, which is where "
                             "DIAL's cost actually lives -- the sample count "
                             "is parallel on a GPU and cutting it 512x buys 6%")
    parser.add_argument("--student-std-normalize", action="store_true",
                        help="the published DIAL, which divides sample returns "
                             "by their own spread; the collection does not, so "
                             "this is a different controller from the teacher")
    parser.add_argument("--absolute", action="store_true",
                        help="the score policy was fitted with "
                             "fit_from_clouds --absolute, so its fields "
                             "predict the plan rather than the update")
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
    parser.add_argument("--also-report", type=int, nargs="*", default=None,
                        help="extra horizons to score from the same rollouts. "
                             "A short evaluation is a prefix of a long one -- "
                             "same policy, same seed, same command -- so "
                             "`--steps 1500 --also-report 150` replaces a "
                             "second pass that cost as much as the first, and "
                             "the two numbers then come from one trajectory "
                             "rather than two")
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
    # DIAL is ~99.9% of an evaluation and does not depend on the student,
    # so scoring a second controller on the same grid is nearly free.  The
    # key covers the configs, the environment class's source and the
    # planner functions, so an objective change lands in a new directory
    # rather than silently reusing the old numbers.
    parser.add_argument("--teacher-cache", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-teacher-cache", action="store_true")
    parser.add_argument("--refresh-teacher-cache", action="store_true",
                        help="recompute and overwrite every episode; for "
                             "checking that a cached grid still reproduces")
    args = parser.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    if args.row_scales:
        track, stability, gait = args.row_scales
        env_config = dataclasses.replace(
            env_config, track_scale=track, stability_scale=stability,
            gait_scale=gait,
        )
    if args.dial_student:
        inference = blob = policy = None
        temperature = args.temperature or float(dial_config.temp_sample)
    elif args.rl_policy is not None:
        inference, blob = load_rl_policy(args.rl_policy)
        policy = None
        temperature = args.temperature or float(dial_config.temp_sample)
    else:
        inference = blob = None
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
    if args.absolute and len(policy.policies) > 1:
        raise ValueError(
            f"--absolute with {len(policy.policies)} fields: an absolute plan "
            "does not compose, so only a single field fitted at the target "
            "weight can be scored this way"
        )
    pinned = blob is not None and not blob.get("omega_conditioned")
    if blob is not None and blob.get("omega_conditioned"):
        # Conditioned on the weight, so it answers for the whole target list
        # exactly as the composed student does; nothing to pin, and the target
        # list is built below like any other arm's.
        print("rl policy is omega-conditioned; scoring the full target list")
    if pinned:
        name = blob.get("omega_name") or "rl"
        targets[name] = normalize_omega_np(
            np.asarray(blob["omega"], dtype=float))
        args.targets = [name]
        print(f"rl policy trained at {name} = "
              f"{np.round(targets[name], 4).tolist()}; scoring that weight only")
    if not pinned:
        for name in args.targets:
            targets[name] = (catalogue[name] if name in catalogue
                             else normalize_omega_np(
                                 np.array([float(v) for v in name.split(",")])))

    if args.absolute and policy is not None:
        # The mixture solve still runs, and away from the fitted weight it
        # returns a coefficient that scales an absolute plan -- a number with
        # no meaning.  Nothing would crash; the row would just be wrong, so
        # refuse instead.
        fitted = np.asarray(policy.mode_weights[0], dtype=float)
        for name, omega in targets.items():
            if not np.allclose(np.asarray(omega, dtype=float), fitted, atol=1e-6):
                raise ValueError(
                    f"--absolute policy was fitted at "
                    f"{np.round(fitted, 4).tolist()}; target {name} = "
                    f"{np.round(omega, 4).tolist()} is a different weight and "
                    "an absolute plan cannot be composed to reach it"
                )

    if args.dial_student:
        overrides = {}
        if args.student_samples is not None:
            overrides["Nsample"] = args.student_samples
        if args.student_diffuse is not None:
            overrides["Ndiffuse"] = args.student_diffuse
        if args.student_horizon is not None:
            overrides["Hsample"] = args.student_horizon
        student = make_dial_student(
            env, dial_config, args.init_passes, args.student_std_normalize,
            args.steps, tuple(args.level_scales), overrides)
        print(f"dial student: overrides={overrides or 'none'}  "
              f"std_normalize={args.student_std_normalize}")
    elif inference is not None:
        student = make_rl_student(
            env, inference, args.steps,
            append_omega=bool(blob.get("omega_conditioned")))
    else:
        student = make_student(env, policy, dial_config, args.init_passes,
                               args.steps, absolute=args.absolute)
    teacher = make_teacher(env, mbdpi, dial_config, args.init_passes,
                           args.std_normalize, args.steps,
                           tuple(args.level_scales))

    cache = None
    if not args.no_teacher_cache:
        digest, manifest = fingerprint(
            dial_config=dial_config, env_config=env_config, env=env,
            functions=(make_teacher, make_dial_step, make_lean_update,
                       mppi_logits, MBDPI, type(mbdpi)),
            # Everything the teacher closure is built with that is not already
            # in the two configs.
            extra={"init_passes": int(args.init_passes),
                   "std_normalize": bool(args.std_normalize),
                   "level_scales": [float(v) for v in args.level_scales],
                   "temperature": float(temperature)},
        )
        cache = TeacherCache(args.teacher_cache, digest, manifest,
                             refresh=args.refresh_teacher_cache)
        print(f"teacher cache {cache.dir}")

    print(f"policy {args.rl_policy or args.policy or 'dial-student'}")
    print(f"env {dial_config.env_name}  T={temperature}  "
          f"levels={args.level_scales}  "
          f"row_scales={args.row_scales or 'from config'}  "
          + ("  arm=dial" if args.dial_student else
             "  arm=rl" if policy is None else
             f"  fields={len(policy.policies)}  "
             f"nu={policy.pinv_nu_weights is not None}"))
    horizons = sorted({args.steps, *(args.also_report or [])})
    for h in horizons:
        if h > args.steps:
            raise ValueError(
                f"--also-report {h} exceeds --steps {args.steps}; a horizon can "
                "only be scored from a rollout at least that long"
            )
    report: dict[int, dict] = {h: {} for h in horizons}
    train_command = (float(env_config.default_vx), float(env_config.default_vy),
                     float(env_config.default_vyaw))
    for cname in args.commands.split(","):
        cname = cname.strip()
        command = (train_command if cname == "train" else COMMANDS[cname])
        print(f"\n=== command {cname} = {command} ===")
        print(f"{'target':<9}{'DIAL cost':>11}{'student':>10}{'ratio':>8}"
              f"{'D.fall':>8}{'S.fall':>8}{'S.alive':>9}")
        for name, omega in targets.items():
            acc = {h: {"teach": [], "stud": [], "tf": 0, "sf": 0, "alive": []}
                   for h in horizons}
            for seed in range(args.seeds):
                state = reset(jax.random.PRNGKey(11 + seed))
                state = set_command(env, state, command)
                state = set_omega(state, omega)
                ekey = episode_key(omega=omega, command=command, seed=seed)
                cached = cache.load(ekey, args.steps) if cache else None
                if cached is None:
                    tr, td = teacher(state, jax.random.PRNGKey(seed))
                    tr, td = np.asarray(tr), np.asarray(td)
                    if cache is not None:
                        cache.store(ekey, tr, td)
                else:
                    tr, td = cached
                sr, sd = student(state, jnp.asarray(omega), temperature)
                for h in horizons:
                    c, fell, _ = summarise(tr[:h], td[:h], h)
                    acc[h]["teach"].append(c); acc[h]["tf"] += fell
                    c, fell, a = summarise(sr[:h], sd[:h], h)
                    acc[h]["stud"].append(c); acc[h]["sf"] += fell
                    acc[h]["alive"].append(a)
            for h in horizons:
                a = acc[h]
                t, s = float(np.mean(a["teach"])), float(np.mean(a["stud"]))
                entry = {
                    "dial_cost": t, "student_cost": s, "ratio": s / max(t, 1e-9),
                    "dial_falls": a["tf"], "student_falls": a["sf"],
                    "student_alive": float(np.mean(a["alive"])), "steps": h,
                }
                report.setdefault(h, {})[f"{cname}/{name}"] = entry
                if h == args.steps:
                    print(f"{name:<9}{t:11.4f}{s:10.4f}{s / max(t, 1e-9):8.3f}"
                          f"{a['tf']:8d}{a['sf']:8d}{np.mean(a['alive']):9.0f}",
                          flush=True)

    if cache is not None:
        print(f"\n{cache.summary()}")
    for h in horizons:
        rows = report[h]
        ratios = [v["ratio"] for v in rows.values()]
        print(f"\n{h} steps ({h * float(env.dt):.1f} s): "
              f"mean ratio {np.mean(ratios):.3f}   "
              f"median {np.median(ratios):.3f}   "
              f"student falls {sum(v['student_falls'] for v in rows.values())}"
              f"/{len(rows) * args.seeds}   "
              f"DIAL falls {sum(v['dial_falls'] for v in rows.values())}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for h in horizons:
            # One file per horizon, so everything that already reads these
            # keeps working.
            path = out if h == args.steps else out.with_name(
                f"{out.stem}_{h}{out.suffix}")
            path.write_text(json.dumps(report[h], indent=2, default=float))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
