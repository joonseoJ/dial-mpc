"""Screen a reward basis, and measure what the weight does to the gait.

Two measurements, kept separate on purpose.

`cloud` is the honest discriminativeness test.  DIAL is warmed up once on
shared control weights, one proposal cloud is drawn at the final annealing
level, and every omega re-weights that same cloud.  Because the samples are
shared, comparing omegas measures the objective and not the difficulty of the
states each omega happened to visit -- the confound that made the walking
task's first closed-loop numbers unreadable.  It also reports each row's spread
across the cloud, which is what the row normalizers have to equalise: a row
whose weight cannot move the argmax is not part of the basis however large you
make it.

`walk` is the physical evidence.  Each omega drives a full DIAL loop against a
velocity command and the resulting gait is measured directly: duty factor,
cadence, stride, torso bounce, cost of transport, touchdown speed.  Cost scores
can differ for uninteresting reasons; these cannot.  If the gait statistics do
not separate, the basis does not work, whatever the scores say.

Nothing here is task-specific: the row count comes from the environment's
weight vector and the per-step diagnostics are read defensively, so the same
screen runs on any basis registered in `dial_mpc.envs`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import (
    make_dial_step, make_lean_update, make_rollout, mppi_weights,
)
from csm.omega import normalize_omega_np

# Row labels per task; the count itself comes from the environment's weight
# vector so the screen runs on any basis.
ROW_NAMES = {
    "unitree_go2_walk": ["tracking", "stability", "gait"],
    "unitree_go2_push_recover": ["tilt", "base", "feet", "shape"],
}
ROWS: list[str] = []
GRAVITY = 9.81

# The joystick the one field has to cover: straight, turning, strafing,
# spinning in place and a slow crawl.  A basis that only separates on a
# straight line is the walking task again.
COMMANDS = {
    "random": None,          # the environment samples its own
    # Inside the narrowed training box (vx 0.6-1.0, vy +-0.15, vyaw +-0.3):
    # its two extremes and its two corners.  A screen run on the presets below
    # measures a regime the fields are never trained on.
    "box_fast": (1.0, 0.0, 0.0),
    "box_slow": (0.6, 0.0, 0.0),
    "box_turn": (0.8, 0.0, 0.3),
    "box_strafe": (0.8, 0.15, 0.0),
    # Outside it, kept as off-distribution probes.
    "straight": (1.2, 0.0, 0.0),
    "turn": (1.0, 0.0, 0.8),
    "strafe": (0.8, 0.4, 0.0),
    "spin": (0.0, 0.0, 1.2),
    "crawl": (0.5, 0.0, 0.0),
}


def set_command(env, state, command):
    """Pin one velocity command for the whole episode, ramp included.

    `command=None` leaves the environment's own randomisation in place, which
    is what a screen has to do once the collection samples its commands: pinning
    one would measure a single point of the distribution the fields are trained
    on.
    """

    if command is None:
        return state

    vel = jnp.array([command[0], command[1], 0.0])
    ang = jnp.array([0.0, 0.0, command[2]])
    info = dict(state.info)
    info["vel_cmd"] = vel
    info["ang_vel_cmd"] = ang
    info["vel_tar"] = vel * env._command_ramp_scale(0)
    info["ang_vel_tar"] = ang * env._command_ramp_scale(0)
    info["randomize_target"] = jnp.asarray(False)
    state = state.replace(info=info)
    return state.replace(obs=env._get_obs(state.pipeline_state, state.info))


def set_omega(state, omega):
    info = dict(state.info)
    info["reward_weights"] = jnp.asarray(omega, dtype=jnp.float32)
    return state.replace(info=info)


def make_warm_start(env, mbdpi, dial_config, std_normalize):
    """DIAL's Ndiffuse_init opening pass, from a zero plan."""

    update = make_lean_update(env, mbdpi, dial_config, std_normalize)
    sigma = mbdpi.sigma_control
    factors = dial_config.traj_diffuse_factor ** jnp.arange(
        dial_config.Ndiffuse_init
    )

    def warm(state, rng):
        plan = jnp.zeros((dial_config.Hnode + 1, mbdpi.nu))

        def level(carry, factor):
            key, cur = carry
            key, cur = update(state, key, cur, sigma * factor)
            return (key, cur), None

        (rng, plan), _ = jax.lax.scan(level, (rng, plan), factors)
        return rng, plan

    return warm


# --------------------------------------------------------------------- #
# closed loop
# --------------------------------------------------------------------- #
def make_bootstrap(env, mbdpi, dial_config, steps, std_normalize):
    """Put the planner into a trotting basin, using an objective we then drop.

    Sampling MPC does not discover a gait from a standing plan: standing is a
    local optimum of any of these rows, and a sampled step pays support,
    impulse and swing work at once while the progress it buys arrives later.
    So a separate environment instance -- same physics, same action space, one
    extra foot-height term in its reward -- drives the robot into a steady trot
    and hands over its pipeline state and its plan.  The measured instance
    never sees that term.  Every omega then starts from the identical trotting
    state and the identical plan, which is the same fairness argument the
    shared cloud makes: comparing omegas from their own starting points would
    measure the starting points.
    """

    control = make_dial_step(env, mbdpi, dial_config, std_normalize=std_normalize)
    warm = make_warm_start(env, mbdpi, dial_config, std_normalize)

    @jax.jit
    def boot(state, rng):
        rng, plan = warm(state, rng)

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control(st, key, pl)
            return (st, key, pl), None

        (state, rng, plan), _ = jax.lax.scan(
            body, (state, rng, plan), None, length=steps
        )
        return state, plan

    return boot


def make_closed_loop(env, mbdpi, dial_config, n_steps, std_normalize,
                     level_scales=None):
    control = make_dial_step(env, mbdpi, dial_config,
                             std_normalize=std_normalize,
                             level_scales=level_scales)
    warm = make_warm_start(env, mbdpi, dial_config, std_normalize)

    def diagnostics(state):
        ps = state.pipeline_state
        torso = env._torso_idx - 1
        foot_z = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
        return {
            "base_pos": ps.x.pos[torso],
            "base_vel": ps.xd.vel[torso],
            "base_ang": ps.xd.ang[torso],
            "foot_z": foot_z,
            "contact": (foot_z < 1e-3).astype(jnp.float32),
            "terms": state.info["reward_terms"],
            # Mechanical power is published by some environments and not
            # others; the screen should not require it.
            "power": state.info.get("power", jnp.zeros(())),
            "done": state.done,
            "vel_tar": state.info["vel_tar"],
            "ang_vel_tar": state.info["ang_vel_tar"],
        }

    @jax.jit
    def run(state, rng, plan=None):
        if plan is None:
            rng, plan = warm(state, rng)

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control(st, key, pl)
            return (st, key, pl), diagnostics(st)

        _, trace = jax.lax.scan(body, (state, rng, plan), None, length=n_steps)
        return trace

    return run


def gait_statistics(trace, env, mass, command):
    """Turn a trajectory into the numbers a gait is actually described by."""

    dt = float(env.dt)
    contact = np.asarray(trace["contact"])            # (T, 4)
    foot_z = np.asarray(trace["foot_z"])
    pos = np.asarray(trace["base_pos"])
    vel = np.asarray(trace["base_vel"])
    ang = np.asarray(trace["base_ang"])
    done = np.asarray(trace["done"])
    terms = np.asarray(trace["terms"])
    power = np.asarray(trace["power"])

    # Everything after the first termination is a corpse sliding around, so
    # measure only the live part of the episode.
    alive = int(np.argmax(done > 0.5)) if np.any(done > 0.5) else len(done)
    alive = max(alive, 2)
    sl = slice(0, alive)
    duration = alive * dt

    c = contact[sl]
    touchdown = (c[1:] > 0.5) & (c[:-1] < 0.5)
    n_td = float(touchdown.sum())
    cadence = n_td / 4.0 / max(duration, dt)          # cycles per second per foot
    duty = float(c.mean())
    flight = float((c.sum(axis=1) == 0).mean())

    zdot = np.diff(foot_z[sl], axis=0) / dt
    td_speed = -zdot[touchdown]
    td_speed = td_speed[td_speed > 0]

    az = np.diff(vel[sl, 2]) / dt
    speed = float(np.linalg.norm(vel[sl, :2], axis=1).mean())
    mean_power = float(power[sl].mean())
    cot = mean_power / max(mass * GRAVITY * speed, 1e-3)

    # Realized command tracking, in the body frame the command lives in.
    yaw_rate = float(ang[sl, 2].mean())

    return {
        "alive_steps": alive,
        "fell": bool(np.any(done > 0.5)),
        "distance": float(np.linalg.norm(pos[alive - 1, :2] - pos[0, :2])),
        "speed": speed,
        "yaw_rate": yaw_rate,
        "cmd": list(command),
        "duty_factor": duty,
        "cadence_hz": cadence,
        "stride_m": speed / max(cadence, 1e-3),
        "flight_frac": flight,
        "torso_az_rms": float(np.sqrt(np.mean(az**2))),
        "torso_pitchroll_rms": float(
            np.sqrt(np.mean(np.sum(ang[sl, :2] ** 2, axis=1)))
        ),
        "height_mean": float(pos[sl, 2].mean()),
        "height_std": float(pos[sl, 2].std()),
        "td_speed_rms": float(np.sqrt(np.mean(td_speed**2))) if td_speed.size else 0.0,
        "mean_power_w": mean_power,
        "cost_of_transport": cot,
        "row_means": [float(v) for v in terms[sl].mean(axis=0)],
    }


# --------------------------------------------------------------------- #
# shared cloud
# --------------------------------------------------------------------- #
def make_cloud(env, mbdpi, dial_config, warmup_steps, std_normalize,
               drive_env=None, level_scales=None):
    """Warm up on one shared objective, then score that cloud with every omega.

    `drive_env` is what does the warming.  It defaults to the measured
    environment, but when the measured objective's basin is standing, a cloud
    drawn around a standing plan says nothing about how the rows rank gaits.
    Pointing `drive_env` at the bootstrap instance draws the cloud around a
    trotting plan instead; the samples are still scored by the measured rows.
    """

    from csm.dial_lean import make_sampler

    drive_env = env if drive_env is None else drive_env
    control = make_dial_step(drive_env, mbdpi, dial_config,
                             std_normalize=std_normalize,
                             level_scales=level_scales)
    warm = make_warm_start(drive_env, mbdpi, dial_config, std_normalize)
    sample = make_sampler(mbdpi, dial_config)
    final_noise = mbdpi.sigma_control * (
        dial_config.traj_diffuse_factor ** (dial_config.Ndiffuse - 1)
    )
    rollout = make_rollout(env, lambda s: (s.info["reward_terms"], s.done))
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))

    @jax.jit
    def cloud(state, rng):
        rng, plan = warm(state, rng)

        def body(carry, _):
            st, key, pl = carry
            st, key, pl = control(st, key, pl)
            return (st, key, pl), None

        (state, rng, plan), _ = jax.lax.scan(
            body, (state, rng, plan), None, length=warmup_steps
        )
        plan = mbdpi.shift(plan)
        rng, cloud_rng = jax.random.split(rng)
        nodes = sample(cloud_rng, plan, final_noise)
        terms, done = rollout_vmap(state, mbdpi.node2u_vvmap(nodes))
        return terms.mean(axis=1), done.max(axis=1)

    return cloud


def analyse_cloud(terms, omega_mat, temp, elite_frac, std_normalize):
    """One cloud, every omega.  Same samples, so the exam is identical."""

    returns = terms @ omega_mat.T                       # (n_sample, n_omega)
    # Same softmax the planner runs, not a re-derivation of it: a screen that
    # scored the cloud by a slightly different rule would not be measuring the
    # controller it is meant to describe.
    w = np.asarray(mppi_weights(jnp.asarray(returns), temp, std_normalize,
                                axis=0))

    ess = 1.0 / np.sum(w**2, axis=0)
    # What omega i's update actually achieves on every row.
    row_scores = w.T @ terms                            # (n_omega, n_rows)
    # ... and on every omega's own objective: entry (i, j) is what i's update
    # scores when j does the marking.
    cross = row_scores @ omega_mat.T                    # (n_omega, n_omega)

    n_elite = max(int(round(elite_frac * terms.shape[0])), 1)
    order = np.argsort(-returns, axis=0)[:n_elite]
    sets = [set(order[:, k].tolist()) for k in range(order.shape[1])]
    n = len(sets)
    jac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jac[i, j] = len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
    return {"ess": ess, "cross": cross, "jaccard": jac,
            "row_scores": row_scores, "returns_std": returns.std(axis=0)}


# --------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", default="unitree_go2_trot_csm")
    parser.add_argument("--mode", default="both",
                        choices=["cloud", "walk", "both"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--nsample", type=int, default=None,
                        help="override Nsample (smoke tests)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="rollouts per (command, omega).  Same-seed reruns "
                             "of this plant diverge -- float32 rounding is "
                             "amplified by the contact dynamics -- so a single "
                             "rollout cannot separate a weight effect from run "
                             "noise")
    parser.add_argument("--speed-floor", type=float, default=None,
                        help="override the env's speed_floor: 0 prices rows "
                             "1-3 per second, >0 prices them per metre")
    parser.add_argument("--elite-frac", type=float, default=0.02)
    parser.add_argument("--no-std-normalize", action="store_true")
    # The cloud is drawn around a plan the warm-up produced, so the warm-up has
    # to be the controller the data will be collected with -- profile included.
    parser.add_argument("--level-scales", type=float, nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="control steps of gait-referenced bootstrap before "
                             "handing the state and plan to every omega")
    parser.add_argument("--commands", default="straight,turn,strafe,spin,crawl")
    parser.add_argument("--weights", default=None,
                        help="comma-separated names from build_omegas, or "
                             "semicolon-separated raw vectors")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    std_normalize = not args.no_std_normalize
    dial_config, env_config = _load_config(args.example, None)
    if args.temp is not None or args.nsample is not None:
        import dataclasses
        overrides = {}
        if args.temp is not None:
            overrides["temp_sample"] = args.temp
        if args.nsample is not None:
            overrides["Nsample"] = args.nsample
        dial_config = dataclasses.replace(dial_config, **overrides)
    if args.speed_floor is not None:
        import dataclasses
        env_config = dataclasses.replace(env_config, speed_floor=args.speed_floor)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    mbdpi = make_controller(dial_config, env)
    mass = float(jnp.sum(env.sys.mj_model.body_mass))
    reset = jax.jit(env.reset)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    rows = ROW_NAMES.get(dial_config.env_name,
                         [f"row{i}" for i in range(n_rows)])[:n_rows]
    globals()["ROWS"] = rows
    catalogue = build_omegas(n_rows)
    if args.weights is None:
        names = ["uniform"] + [f"boost{i}" for i in range(n_rows)] + \
                [f"e{i}" for i in range(n_rows)]
        omegas = {n: catalogue[n] for n in names}
    elif ";" in args.weights or "," in args.weights and args.weights[0].isdigit():
        omegas = {}
        for spec in args.weights.split(";"):
            vec = normalize_omega_np(np.array([float(v) for v in spec.split(",")]))
            omegas[spec] = vec
    else:
        omegas = {n: catalogue[n] for n in args.weights.split(",")}

    names = list(omegas)
    omega_mat = np.stack([omegas[n] for n in names])
    report = {"example": args.example, "weights": {n: omegas[n].tolist()
                                                   for n in names},
              "std_normalize": std_normalize,
              "temp": float(dial_config.temp_sample),
              "rows": rows}

    print(f"env {dial_config.env_name}  mass {mass:.2f} kg  "
          f"Hsample {dial_config.Hsample} ({dial_config.Hsample*env.dt:.2f} s)  "
          f"Nsample {dial_config.Nsample}  temp {dial_config.temp_sample}  "
          f"std_normalize {std_normalize}", flush=True)
    print("weights: " + "  ".join(
        f"{n}=({','.join(f'{v:.2f}' for v in omegas[n])})" for n in names))

    # ---------------------------------------------------------------- #
    if args.mode in ("cloud", "both"):
        drive_env = None
        if args.bootstrap > 0:
            import dataclasses
            drive_env = brax_envs.get_environment(
                dial_config.env_name,
                config=dataclasses.replace(
                    env_config, gait="trot",
                    bootstrap_gait_weight=max(
                        float(getattr(env_config, "bootstrap_gait_weight", 0.0)),
                        1.0,
                    ),
                ),
            )
        cloud = make_cloud(env, mbdpi, dial_config, args.warmup, std_normalize,
                           drive_env=drive_env,
                           level_scales=args.level_scales)
        all_terms, all_done = [], []
        t0 = time.time()
        for k in range(args.states):
            state = reset(jax.random.PRNGKey(1000 + k))
            command = list(COMMANDS.values())[k % len(COMMANDS)]
            state = set_command(env, state, command)
            state = set_omega(state, catalogue["uniform"])
            terms, done = cloud(state, jax.random.PRNGKey(k))
            all_terms.append(np.asarray(terms))
            all_done.append(np.asarray(done))
            print(f"  cloud {k+1}/{args.states} cmd={command} "
                  f"fall-rate {float(np.mean(all_done[-1])):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        per_state = [analyse_cloud(t, omega_mat, dial_config.temp_sample,
                                   args.elite_frac, std_normalize)
                     for t in all_terms]
        cross = np.mean([p["cross"] for p in per_state], axis=0)
        jac = np.mean([p["jaccard"] for p in per_state], axis=0)
        ess = np.mean([p["ess"] for p in per_state], axis=0)
        row_scores = np.mean([p["row_scores"] for p in per_state], axis=0)
        row_std = np.mean([t.std(axis=0) for t in all_terms], axis=0)

        print("\n=== row scale calibration (std across the cloud) ===")
        print(f"{'row':<10}{'std':>12}{'x to equalise':>16}")
        multiplier = float(np.mean(row_std)) / np.maximum(row_std, 1e-12)
        # Equalise at constant total magnitude: raising the weak rows alone is
        # what put the walking robot on the ground in an earlier pass.
        multiplier = multiplier / float(np.exp(np.mean(np.log(multiplier))))
        for i, r in enumerate(rows):
            print(f"{r:<10}{row_std[i]:12.4g}{multiplier[i]:16.3f}")
        print(f"  spread max/min: {row_std.max()/max(row_std.min(),1e-12):.1f}x")

        print("\n=== cross-evaluation on the shared cloud "
              "(row: whose update, column: who marks) ===")
        head = "".join(f"{n:>12}" for n in names)
        print(f"{'update':<10}{head}")
        for i, n in enumerate(names):
            row = "".join(f"{cross[i, j]:12.4f}" for j in range(len(names)))
            print(f"{n:<10}{row}")
        diag_wins = sum(int(np.argmax(cross[:, j]) == j) for j in range(len(names)))
        margins = []
        for j in range(len(names)):
            col = cross[:, j].copy()
            best_other = np.max(np.delete(col, j))
            margins.append(100.0 * (col[j] - best_other) / max(abs(best_other), 1e-9))
        print(f"\ndiagonal wins: {diag_wins}/{len(names)}")
        # The cross matrix is only meaningful where the softmax is an average.
        # Reporting it at one temperature hides whether the ranking survives
        # the sharpness the planner actually runs at.
        print("diagonal wins vs temperature: " + "  ".join(
            f"{t:.3f}:{sum(int(np.argmax(c[:, j]) == j) for j in range(len(names)))}"
            f"/{len(names)}"
            for t, c in (
                (t, np.mean([analyse_cloud(x, omega_mat, t, args.elite_frac,
                                           std_normalize)["cross"]
                             for x in all_terms], axis=0))
                for t in [dial_config.temp_sample * f for f in (0.2, 0.4, 1, 2, 4)]
            )
        ))
        print("margin over the best other update, %:  " +
              "  ".join(f"{n}={m:+.2f}" for n, m in zip(names, margins)))

        chance = args.elite_frac
        print(f"\n=== elite-set overlap (chance ~ {chance/(2-chance):.3f}) ===")
        print(f"{'':<10}{head}")
        for i, n in enumerate(names):
            print(f"{n:<10}" + "".join(f"{jac[i, j]:12.3f}" for j in range(len(names))))

        # The same stored cloud answers every temperature for free, and the
        # temperature is what decides whether the softmax is an average or an
        # argmax -- at 0.05 it was selecting two samples out of 2049.
        print("\n=== effective sample size vs temperature "
              f"(of {dial_config.Nsample + 1}) ===")
        grid = [dial_config.temp_sample * f for f in (0.2, 0.4, 1, 2, 4)]
        print(f"{'temp':<8}" + "".join(f"{n:>10}" for n in names))
        for temp in grid:
            per = [analyse_cloud(t, omega_mat, temp, args.elite_frac,
                                 std_normalize) for t in all_terms]
            e = np.mean([q["ess"] for q in per], axis=0)
            print(f"{temp:<8.3f}" + "".join(
                f"{100*v/(dial_config.Nsample+1):9.1f}%" for v in e))

        report["cloud"] = {
            "row_std": row_std.tolist(),
            "cross": cross.tolist(),
            "jaccard": jac.tolist(),
            "ess": ess.tolist(),
            "row_scores": row_scores.tolist(),
            "diagonal_wins": diag_wins,
            "margins_pct": margins,
        }

    # ---------------------------------------------------------------- #
    if args.mode in ("walk", "both"):
        run = make_closed_loop(env, mbdpi, dial_config, args.steps,
                               std_normalize, args.level_scales)
        boot = None
        if args.bootstrap > 0:
            import dataclasses
            boot_config = dataclasses.replace(
                env_config, gait="trot",
                bootstrap_gait_weight=max(
                    float(getattr(env_config, "bootstrap_gait_weight", 0.0)), 1.0
                ),
            )
            boot_env = brax_envs.get_environment(
                dial_config.env_name, config=boot_config
            )
            boot = make_bootstrap(boot_env, mbdpi, dial_config,
                                  args.bootstrap, std_normalize)
            boot_reset = jax.jit(boot_env.reset)
            boot_stats = make_closed_loop(boot_env, mbdpi, dial_config,
                                          args.steps, std_normalize)
        commands = [c.strip() for c in args.commands.split(",")]
        stats, collected = {}, {}
        t0 = time.time()
        for cname in commands:
            command = COMMANDS[cname]
            print(f"\n=== command {cname} = "
                  f"(vx {command[0]}, vy {command[1]}, vyaw {command[2]}) ===",
                  flush=True)
            print(f"{'weight':<10}{'dist':>7}{'speed':>7}{'yaw':>7}{'duty':>7}"
                  f"{'cad':>7}{'stride':>8}{'fly':>6}{'az':>7}{'wxy':>7}"
                  f"{'h.std':>7}{'td':>7}{'P(W)':>8}{'CoT':>7}{'fall':>6}")
            starts = []
            for seed in range(args.seeds):
                if boot is None:
                    st = reset(jax.random.PRNGKey(7 + seed))
                    st = set_command(env, st, command)
                    starts.append((st, None))
                    continue
                bst = boot_reset(jax.random.PRNGKey(7 + seed))
                bst = set_command(boot_env, bst, command)
                bst = set_omega(bst, catalogue["uniform"])
                bst, bplan = boot(bst, jax.random.PRNGKey(seed))
                starts.append((bst, bplan))
                bs = gait_statistics(
                    boot_stats(bst, jax.random.PRNGKey(100 + seed), bplan),
                    boot_env, mass, command,
                )
                stats[f"{cname}/bootstrap/{seed}"] = bs
                collected.setdefault(f"{cname}/bootstrap", []).append(bs)

            def summarise(label, runs):
                def m(key):
                    return float(np.mean([r[key] for r in runs]))
                def sd(key):
                    return float(np.std([r[key] for r in runs]))
                falls = sum(r["fell"] for r in runs)
                tag = f"{falls}/{len(runs)}" if falls else "-"
                spread = f"+-{sd('speed'):.2f}" if len(runs) > 1 else ""
                print(f"{label:<10}{m('distance'):7.2f}{m('speed'):7.2f}"
                      f"{m('yaw_rate'):7.2f}{m('duty_factor'):7.3f}"
                      f"{m('cadence_hz'):7.2f}{m('stride_m'):8.3f}"
                      f"{m('flight_frac'):6.2f}{m('torso_az_rms'):7.2f}"
                      f"{m('torso_pitchroll_rms'):7.2f}{m('height_std'):7.3f}"
                      f"{m('td_speed_rms'):7.2f}{m('mean_power_w'):8.1f}"
                      f"{m('cost_of_transport'):7.2f}{tag:>6}  {spread}",
                      flush=True)

            if boot is not None:
                summarise("(boot)", collected[f"{cname}/bootstrap"])
            for n in names:
                runs = []
                for seed, (st, bplan) in enumerate(starts):
                    st_n = set_omega(st, omegas[n])
                    trace = run(st_n, jax.random.PRNGKey(seed), bplan)
                    r = gait_statistics(trace, env, mass, command)
                    stats[f"{cname}/{n}/{seed}"] = r
                    runs.append(r)
                collected[f"{cname}/{n}"] = runs
                summarise(n, runs)
            print(f"  ({time.time()-t0:.0f}s elapsed)")
        report["walk"] = stats

        # The verdict the whole task turns on: how much of the between-weight
        # variation survives once run-to-run noise is accounted for?
        print("\n=== weight effect against run noise (per command) ===")
        print(f"{'command':<10}{'metric':<12}{'between-w sd':>14}"
              f"{'within-w sd':>13}{'ratio':>8}")
        for cname in commands:
            for key in ("speed", "duty_factor", "cadence_hz", "torso_az_rms",
                        "td_speed_rms", "cost_of_transport"):
                per = [[r[key] for r in collected[f"{cname}/{n}"]] for n in names]
                between = float(np.std([np.mean(p) for p in per]))
                within = float(np.mean([np.std(p) for p in per]))
                ratio = between / max(within, 1e-9)
                print(f"{cname:<10}{key:<12}{between:14.3f}{within:13.3f}"
                      f"{ratio:8.2f}")
        report["noise"] = {
            f"{cname}/{key}": {
                "between": float(np.std([
                    np.mean([r[key] for r in collected[f"{cname}/{n}"]])
                    for n in names])),
                "within": float(np.mean([
                    np.std([r[key] for r in collected[f"{cname}/{n}"]])
                    for n in names])),
            }
            for cname in commands
            for key in ("speed", "duty_factor", "cadence_hz", "torso_az_rms",
                        "td_speed_rms", "cost_of_transport")
        }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
