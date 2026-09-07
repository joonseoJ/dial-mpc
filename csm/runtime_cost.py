"""What one control step costs, per arm.

The distillation argument rests on a number the baseline table never measured:
DIAL and the composed student are compared on quality, and the reason to
prefer the student is that it is supposed to be orders of magnitude cheaper to
run.  "2049 samples against one forward pass" is an argument from the source
code, not a measurement -- on a GPU those 2049 rollouts are parallel, so the
gap is an empirical question and could be far smaller than the sample count
suggests.

Two things make the timing honest.

**The warm-up is differenced out.**  Every arm pays a one-off cost that has
nothing to do with steady-state control -- DIAL's `init_passes` opening
diffusion, the student's first refine, and in every case the jit compile.
Timing one rollout folds all of that into the per-step number and flatters
whichever arm has the cheapest opening.  So each arm is run at two horizons
and the *marginal* cost is reported: `(t(N2) - t(N1)) / (N2 - N1)`.  The same
trick `--also-report` uses, for the same reason.

**The physics is measured too.**  Each arm's loop steps the environment, and
on this plant that step is not free.  A controller that looks 3x cheaper than
another may be entirely explained by a shared cost that neither of them
controls, so the environment step is timed on its own and reported alongside,
leaving the planner's own share visible.

What matters at the end is not the ratio between arms but whether each clears
the control period: `env.dt` is 20 ms, so anything above 50 Hz runs in real
time and anything below does not, however good its cost ratio looks.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.compose_walk_eval import make_rl_student, make_student, make_teacher
from csm.dial_score import ComposedDialScorePolicy
from csm.screen import COMMANDS, set_command, set_omega


def time_call(fn, args, repeats: int) -> float:
    """Best of `repeats`, with the compile paid first and the result awaited.

    Best rather than mean: the machine is shared and every source of noise
    here only adds time, so the minimum is the closest estimate of the cost
    that is actually intrinsic.
    """

    out = fn(*args)
    jax.block_until_ready(out)
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - started)
    return best


def marginal_ms(build, args, n1: int, n2: int, repeats: int) -> float:
    """Per-step cost with everything that does not scale with the horizon removed."""

    t1 = time_call(build(n1), args, repeats)
    t2 = time_call(build(n2), args, repeats)
    return (t2 - t1) / (n2 - n1) * 1e3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example", default="unitree_go2_trot_csm")
    p.add_argument("--policy", type=Path,
                   default=Path("csm_runs/clouds-fit-20260905-083613/policy.pkl"))
    p.add_argument("--rl-policy", type=Path,
                   default=Path("csm_runs/rl-sweep/uniform/policy.pkl"))
    p.add_argument("--row-scales", type=float, nargs="+",
                   default=[2.479, 0.261, 1.551])
    p.add_argument("--level-scales", type=float, nargs="+", default=[2.625, 1.0])
    p.add_argument("--dial-samples", type=int, nargs="*", default=[2048, 64, 8, 4],
                   help="sample counts to time DIAL at; the first is the "
                        "configured planner, the rest are the compute-matched "
                        "arms")
    p.add_argument("--dial-variants", nargs="*", default=[],
                   help="`Nsample:Ndiffuse:Hsample` triples to time as well. "
                        "The horizon is the axis that matters: cost is the "
                        "sequential rollout, Hsample x Ndiffuse physics steps "
                        "per control step, so this is where a compute-matched "
                        "DIAL has to give ground")
    p.add_argument("--target", default="uniform")
    p.add_argument("--command", default="box_fast")
    p.add_argument("--n1", type=int, default=20)
    p.add_argument("--n2", type=int, default=120)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--init-passes", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    track, stability, gait = args.row_scales
    env_config = dataclasses.replace(
        env_config, track_scale=track, stability_scale=stability, gait_scale=gait)
    policy = ComposedDialScorePolicy.load(args.policy)
    temperature = float(policy.temperature or dial_config.temp_sample)
    dial_config = dataclasses.replace(dial_config, temp_sample=temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)

    omega = build_omegas(int(np.asarray(env_config.reward_weights).shape[0]))[args.target]
    state = jax.jit(env.reset)(jax.random.PRNGKey(11))
    state = set_command(env, state, COMMANDS[args.command])
    state = set_omega(state, omega)

    dt = float(env.dt)
    rows: list[dict] = []

    def record(name, ms, note=""):
        hz = 1e3 / ms if ms > 0 else float("inf")
        rows.append({"arm": name, "ms_per_step": ms, "hz": hz,
                     "realtime": hz >= 1.0 / dt, "note": note})
        print(f"{name:<22}{ms:9.3f} ms{hz:10.1f} Hz   "
              f"{'real time' if hz >= 1.0 / dt else 'TOO SLOW':>10}  {note}",
              flush=True)

    print(f"env {dial_config.env_name}  dt {dt * 1e3:.0f} ms "
          f"({1.0 / dt:.0f} Hz needed)  target {args.target}  "
          f"command {args.command}")
    print(f"marginal cost from {args.n1} vs {args.n2} steps, "
          f"best of {args.repeats}\n")
    print(f"{'arm':<22}{'per step':>12}{'':>10}{'':>10}")

    # Physics alone: the floor every arm pays.
    def build_physics(n):
        action = jnp.zeros(int(env.action_size))

        @jax.jit
        def run(st):
            def body(carry, _):
                return env.step(carry, action), None
            out, _ = jax.lax.scan(body, st, None, length=n)
            return out
        return run

    physics = marginal_ms(build_physics, (state,), args.n1, args.n2, args.repeats)
    record("environment step", physics, "the floor, shared by every arm")

    for nsample in args.dial_samples:
        cfg = dataclasses.replace(dial_config, Nsample=nsample)
        mbdpi = make_controller(cfg, env)
        ms = marginal_ms(
            lambda n, c=cfg, m=mbdpi: make_teacher(
                env, m, c, args.init_passes, False, n, tuple(args.level_scales)),
            (state, jax.random.PRNGKey(0)), args.n1, args.n2, args.repeats)
        tag = "the planner as configured" if nsample == args.dial_samples[0] else ""
        record(f"DIAL Nsample={nsample}", ms, tag)

    for spec in args.dial_variants:
        ns, nd, hs = (int(v) for v in spec.split(":"))
        cfg = dataclasses.replace(dial_config, Nsample=ns, Ndiffuse=nd,
                                  Hsample=hs)
        m = make_controller(cfg, env)
        ms = marginal_ms(
            lambda n, c=cfg, mm=m: make_teacher(
                env, mm, c, args.init_passes, False, n, tuple(args.level_scales)),
            (state, jax.random.PRNGKey(0)), args.n1, args.n2, args.repeats)
        record(f"DIAL {ns}/{nd}/{hs}", ms, f"{nd * hs} sequential steps")

    ms = marginal_ms(
        lambda n: make_student(env, policy, dial_config, args.init_passes, n),
        (state, jnp.asarray(omega), temperature), args.n1, args.n2, args.repeats)
    record("CSM composed", ms, f"{len(policy.policies)} fields")

    if args.rl_policy and args.rl_policy.exists():
        from csm.rl_baseline import load_policy as load_rl_policy
        inference, _ = load_rl_policy(args.rl_policy)
        ms = marginal_ms(lambda n: make_rl_student(env, inference, n),
                         (state, jnp.asarray(omega), temperature),
                         args.n1, args.n2, args.repeats)
        record("PPO policy", ms, "one MLP forward")

    base = next(r for r in rows if r["arm"].startswith("DIAL"))["ms_per_step"]
    print(f"\n{'arm':<22}{'vs DIAL':>10}{'planner only':>15}")
    for r in rows[1:]:
        own = r["ms_per_step"] - physics
        print(f"{r['arm']:<22}{base / r['ms_per_step']:9.1f}x"
              f"{own:12.3f} ms")
    print("\n`planner only` subtracts the environment step, which no arm controls.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"dt": dt, "physics_ms": physics, "rows": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
