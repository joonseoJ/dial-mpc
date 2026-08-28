"""Screen the walk-recovery basis: does the weight pick the recovery strategy?

The robot walks on a velocity command and an impulse hits the floating base
between control steps, so the planner sees a disturbed state but never predicts
the disturbance.  Two measurements, kept apart for the same reason as in the
gait-choice screen.

`cloud` is the discriminativeness test.  One shared warm-up, one push, one
proposal cloud drawn at the final annealing level, and every omega re-weights
those same samples.  Shared samples mean the comparison is of the objective and
not of the states each omega happened to reach.

`recover` is the physical evidence.  Each omega drives a full DIAL loop through
the same impulse and the response is measured directly: how far the feet
departed from the stride the command asked for, how much trunk attitude and
angular rate were spent, how much joint torque, and how much course was lost.
Those four are the four terms of the centre-of-mass identity, so a basis that
works has to move them against each other rather than together.
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
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step, make_rollout, make_sampler
from csm.gait_choice_eval import (
    COMMANDS, analyse_cloud, make_warm_start, set_command, set_omega,
)

ROWS = ["stepping", "trunk", "effort", "velocity"]


def push_state(state, speed, heading):
    """Add a horizontal velocity impulse to the floating base."""

    ps = state.pipeline_state
    impulse = jnp.stack(
        [speed * jnp.cos(heading), speed * jnp.sin(heading), jnp.asarray(0.0)]
    )
    return state.replace(pipeline_state=ps.replace(qvel=ps.qvel.at[:3].add(impulse)))


def make_recovery_loop(env, mbdpi, dial_config, n_steps, push_at, std_normalize):
    control = make_dial_step(env, mbdpi, dial_config, std_normalize=std_normalize)

    def diagnostics(state):
        ps = state.pipeline_state
        torso = env._torso_idx - 1
        foot_z = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
        up = jnp.array([0.0, 0.0, 1.0])
        from brax import math as bmath
        return {
            "pos": ps.x.pos[torso],
            "vel": ps.xd.vel[torso],
            "ang": ps.xd.ang[torso],
            "tilt": 1.0 - jnp.dot(bmath.rotate(up, ps.x.rot[torso]), up),
            "tau_sq": jnp.sum(jnp.square(ps.ctrl)),
            "contact": (foot_z < 1e-3).astype(jnp.float32),
            "terms": state.info["reward_terms"],
            "done": state.done,
            "vel_tar": state.info["vel_tar"],
        }

    @jax.jit
    def run(state, rng, plan, speed, heading):
        def body(carry, index):
            st, key, pl = carry
            # The impulse lands between control steps.  Injecting it inside
            # env.step would put it in the planner's own rollout, and a push
            # the planner predicts is not a disturbance.
            st = jax.lax.cond(
                index == push_at,
                lambda s: push_state(s, speed, heading),
                lambda s: s,
                st,
            )
            st, key, pl = control(st, key, pl)
            return (st, key, pl), diagnostics(st)

        _, trace = jax.lax.scan(
            body, (state, rng, plan), jnp.arange(n_steps)
        )
        return trace

    return run


def recovery_statistics(trace, env, push_at, command):
    """What the impulse cost, split across the four channels it can go into."""

    dt = float(env.dt)
    done = np.asarray(trace["done"])
    alive = int(np.argmax(done > 0.5)) if np.any(done > 0.5) else len(done)
    fell = bool(np.any(done > 0.5))

    pos = np.asarray(trace["pos"])
    vel = np.asarray(trace["vel"])
    ang = np.asarray(trace["ang"])
    tilt = np.asarray(trace["tilt"])
    tau_sq = np.asarray(trace["tau_sq"])
    contact = np.asarray(trace["contact"])
    terms = np.asarray(trace["terms"])

    pre = slice(max(push_at - 25, 0), push_at)
    win = slice(push_at, min(push_at + 60, alive if alive > push_at else push_at + 1))
    n_win = max(win.stop - win.start, 1)

    # Course loss: where the base ended up against where the pre-push heading
    # and the command would have taken it.
    if push_at < alive:
        base0 = pos[push_at, :2]
        v_ref = np.array([command[0], command[1]])
        drift = pos[min(push_at + 60, alive) - 1, :2] - base0
        expected = v_ref * (n_win * dt)
        course_loss = float(np.linalg.norm(drift - expected))
        speed_err = float(np.mean(np.linalg.norm(
            vel[win, :2] - v_ref[None, :], axis=1)))
        peak_tilt = float(np.max(tilt[win]))
        trunk_rate = float(np.sqrt(np.mean(np.sum(ang[win, :2] ** 2, axis=1))))
        peak_tau = float(np.max(tau_sq[win]))
        mean_tau = float(np.mean(tau_sq[win]))
        duty = float(np.mean(contact[win]))
        # Extra footfalls relative to the pre-push cadence: the direct
        # signature of a recovery step.
        td_pre = ((contact[pre][1:] > 0.5) & (contact[pre][:-1] < 0.5)).sum()
        td_win = ((contact[win][1:] > 0.5) & (contact[win][:-1] < 0.5)).sum()
        rate_pre = td_pre / max((pre.stop - pre.start) * dt, dt)
        rate_win = td_win / max(n_win * dt, dt)
        step_row = float(-np.sum(terms[win, 0]))
    else:
        course_loss = speed_err = peak_tilt = trunk_rate = np.nan
        peak_tau = mean_tau = duty = rate_pre = rate_win = step_row = np.nan

    return {
        "fell": fell,
        "alive_steps": alive,
        "course_loss": course_loss,
        "speed_err": speed_err,
        "peak_tilt": peak_tilt,
        "trunk_rate": trunk_rate,
        "peak_tau": peak_tau,
        "mean_tau": mean_tau,
        "duty_in_window": duty,
        "cadence_pre": float(rate_pre),
        "cadence_post": float(rate_win),
        "step_cost": step_row,
        "row_means_window": [float(v) for v in terms[win].mean(axis=0)]
        if push_at < alive else [float("nan")] * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", default="unitree_go2_walk_recover")
    parser.add_argument("--mode", default="both", choices=["cloud", "recover", "both"])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--push-at", type=int, default=50)
    parser.add_argument("--push-speed", type=float, default=None)
    parser.add_argument("--headings", default="0,90,180,270")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--elite-frac", type=float, default=0.02)
    parser.add_argument("--no-std-normalize", action="store_true")
    parser.add_argument("--commands", default="straight")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    std_normalize = not args.no_std_normalize
    dial_config, env_config = _load_config(args.example, None)
    if args.nsample is not None:
        dial_config = dataclasses.replace(dial_config, Nsample=args.nsample)
    speed = (args.push_speed if args.push_speed is not None
             else float(env_config.push_linear_velocity))

    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    boot_env = brax_envs.get_environment(
        dial_config.env_name,
        config=dataclasses.replace(
            env_config,
            bootstrap_gait_weight=max(
                float(env_config.bootstrap_gait_weight), 1.0),
        ),
    )
    mbdpi = make_controller(dial_config, env)
    boot_reset = jax.jit(boot_env.reset)

    catalogue = build_omegas(4)
    names = (args.weights.split(",") if args.weights
             else ["uniform"] + [f"boost{i}" for i in range(4)]
                  + [f"e{i}" for i in range(4)])
    omegas = {n: catalogue[n] for n in names}
    omega_mat = np.stack([omegas[n] for n in names])
    headings = [np.deg2rad(float(h)) for h in args.headings.split(",")]

    print(f"env {dial_config.env_name}  Hsample {dial_config.Hsample} "
          f"({dial_config.Hsample*env.dt:.2f} s)  Nsample {dial_config.Nsample}  "
          f"temp {dial_config.temp_sample}  push {speed:.2f} m/s  "
          f"at step {args.push_at}", flush=True)
    print("rows: " + ", ".join(ROWS))

    # The bootstrap drives the robot into a trot; every omega and every cloud
    # starts from the same trotting state and the same plan.
    warm = make_warm_start(boot_env, mbdpi, dial_config, std_normalize)
    boot_control = make_dial_step(boot_env, mbdpi, dial_config,
                                  std_normalize=std_normalize)

    @jax.jit
    def bootstrap(state, rng):
        rng, plan = warm(state, rng)

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = boot_control(st, key, pl)
            return (st, key, pl), None

        (state, rng, plan), _ = jax.lax.scan(
            body, (state, rng, plan), None, length=args.warmup
        )
        return state, plan

    report = {"example": args.example, "rows": ROWS, "push_speed": speed,
              "weights": {n: omegas[n].tolist() for n in names}}
    commands = [c.strip() for c in args.commands.split(",")]

    # ------------------------------------------------------------------ #
    if args.mode in ("cloud", "both"):
        sample = make_sampler(mbdpi, dial_config)
        final_noise = mbdpi.sigma_control * (
            dial_config.traj_diffuse_factor ** (dial_config.Ndiffuse - 1)
        )
        rollout = make_rollout(env, lambda s: (s.info["reward_terms"], s.done))
        rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))

        @jax.jit
        def cloud(state, plan, rng, heading):
            # The cloud is drawn from a just-pushed state: that is the moment
            # the four strategies are actually in competition.
            state = push_state(state, speed, heading)
            plan = mbdpi.shift(plan)
            rng, cloud_rng = jax.random.split(rng)
            nodes = sample(cloud_rng, plan, final_noise)
            terms, done = rollout_vmap(state, mbdpi.node2u_vvmap(nodes))
            return terms.mean(axis=1), done.max(axis=1)

        all_terms, t0 = [], time.time()
        for k in range(args.states):
            cname = commands[k % len(commands)]
            st = boot_reset(jax.random.PRNGKey(400 + k))
            st = set_command(boot_env, st, COMMANDS[cname])
            st = set_omega(st, catalogue["uniform"])
            st, plan = bootstrap(st, jax.random.PRNGKey(k))
            heading = headings[k % len(headings)]
            terms, done = cloud(st, plan, jax.random.PRNGKey(1000 + k), heading)
            all_terms.append(np.asarray(terms))
            print(f"  cloud {k+1}/{args.states} cmd={cname} "
                  f"heading={np.rad2deg(heading):.0f} deg  "
                  f"fall-rate {float(np.mean(np.asarray(done))):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        row_std = np.mean([t.std(axis=0) for t in all_terms], axis=0)
        target = float(np.mean(row_std))
        mult = target / np.maximum(row_std, 1e-9)
        # Equalise at constant total magnitude: raising the weak rows alone is
        # what broke the bootstrap in the gait-choice screen.
        mult = mult / float(np.exp(np.mean(np.log(mult))))
        print("\n=== row scale calibration ===")
        print(f"{'row':<11}{'std':>10}{'x to equalise':>15}{'new scale':>12}")
        current = [env_config.step_scale, env_config.trunk_scale,
                   env_config.effort_scale, env_config.velocity_scale]
        for i, r in enumerate(ROWS):
            print(f"{r:<11}{row_std[i]:10.4f}{mult[i]:15.3f}"
                  f"{current[i]/mult[i]:12.4g}")

        per = [analyse_cloud(t, omega_mat, dial_config.temp_sample,
                             args.elite_frac, std_normalize) for t in all_terms]
        cross = np.mean([p["cross"] for p in per], axis=0)
        jac = np.mean([p["jaccard"] for p in per], axis=0)
        head = "".join(f"{n:>11}" for n in names)
        print("\n=== cross-evaluation on the shared cloud ===")
        print(f"{'update':<9}{head}")
        for i, n in enumerate(names):
            print(f"{n:<9}" + "".join(f"{cross[i,j]:11.4f}"
                                      for j in range(len(names))))
        wins = sum(int(np.argmax(cross[:, j]) == j) for j in range(len(names)))
        margins = []
        for j in range(len(names)):
            other = np.max(np.delete(cross[:, j], j))
            margins.append(100.0 * (cross[j, j] - other) / max(abs(other), 1e-9))
        print(f"\ndiagonal wins: {wins}/{len(names)}")
        print("margins %:  " + "  ".join(f"{n}={m:+.2f}"
                                         for n, m in zip(names, margins)))
        print("vs temperature: " + "  ".join(
            f"{t:.3f}:{sum(int(np.argmax(c[:, j]) == j) for j in range(len(names)))}"
            f"/{len(names)}"
            for t, c in ((t, np.mean([analyse_cloud(x, omega_mat, t,
                                                    args.elite_frac,
                                                    std_normalize)["cross"]
                                      for x in all_terms], axis=0))
                         for t in [dial_config.temp_sample * f
                                   for f in (0.2, 0.4, 1, 2, 4)])))
        print(f"\n=== elite overlap (chance ~ {args.elite_frac:.3f}) ===")
        print(f"{'':<9}{head}")
        for i, n in enumerate(names):
            print(f"{n:<9}" + "".join(f"{jac[i,j]:11.3f}"
                                      for j in range(len(names))))
        print(f"\n=== ESS vs temperature (of {dial_config.Nsample+1}) ===")
        print(f"{'temp':<8}" + "".join(f"{n:>10}" for n in names))
        for temp in [dial_config.temp_sample * f for f in (0.2, 0.4, 1, 2, 4)]:
            e = np.mean([analyse_cloud(t, omega_mat, temp, args.elite_frac,
                                       std_normalize)["ess"]
                         for t in all_terms], axis=0)
            print(f"{temp:<8.3f}" + "".join(
                f"{100*v/(dial_config.Nsample+1):9.1f}%" for v in e))
        report["cloud"] = {"row_std": row_std.tolist(), "cross": cross.tolist(),
                           "jaccard": jac.tolist(), "diagonal_wins": wins,
                           "margins_pct": margins}

    # ------------------------------------------------------------------ #
    if args.mode in ("recover", "both"):
        run = make_recovery_loop(env, mbdpi, dial_config, args.steps,
                                 args.push_at, std_normalize)
        stats, t0 = {}, time.time()
        for cname in commands:
            for heading in headings:
                print(f"\n=== command {cname}, push {speed:.2f} m/s at "
                      f"{np.rad2deg(heading):.0f} deg ===", flush=True)
                print(f"{'weight':<9}{'course':>8}{'v.err':>8}{'tilt':>8}"
                      f"{'w_xy':>8}{'tau^2':>9}{'duty':>7}{'cad-':>7}"
                      f"{'cad+':>7}{'step':>8}{'fall':>6}")
                starts = []
                for seed in range(args.seeds):
                    st = boot_reset(jax.random.PRNGKey(7 + seed))
                    st = set_command(boot_env, st, COMMANDS[cname])
                    st = set_omega(st, catalogue["uniform"])
                    starts.append(bootstrap(st, jax.random.PRNGKey(seed)))
                for n in names:
                    runs = []
                    for seed, (st, plan) in enumerate(starts):
                        trace = run(set_omega(st, omegas[n]),
                                    jax.random.PRNGKey(seed), plan,
                                    speed, heading)
                        runs.append(recovery_statistics(
                            trace, env, args.push_at, COMMANDS[cname]))
                    key = f"{cname}/{np.rad2deg(heading):.0f}/{n}"
                    stats[key] = runs
                    m = lambda k: float(np.nanmean([r[k] for r in runs]))
                    falls = sum(r["fell"] for r in runs)
                    print(f"{n:<9}{m('course_loss'):8.3f}{m('speed_err'):8.3f}"
                          f"{m('peak_tilt'):8.4f}{m('trunk_rate'):8.3f}"
                          f"{m('peak_tau'):9.0f}{m('duty_in_window'):7.3f}"
                          f"{m('cadence_pre'):7.2f}{m('cadence_post'):7.2f}"
                          f"{m('step_cost'):8.2f}"
                          f"{(f'{falls}/{len(runs)}' if falls else '-'):>6}",
                          flush=True)
                print(f"  ({time.time()-t0:.0f}s elapsed)")
        report["recover"] = stats

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
