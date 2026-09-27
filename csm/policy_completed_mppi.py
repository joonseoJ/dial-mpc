"""Runtime customisation of an RL locomotion policy: plan a short chunk, let the policy finish.

The measurements of 2026-09-26 settled who should optimise: PPO with the gait
clock is 7-10x cheaper than DIAL on every walking weight, so a controller
distilled from DIAL cannot win on quality.  What an RL policy cannot do is
honour an objective or a constraint it was never trained on.  This module is
the layer that adds that, without retraining:

  every control step
    1. roll the policy K steps in a planner model -> the nominal chunk
    2. sample N chunks around it (clipped to any runtime joint band)
    3. score each: K steps of the chunk, then H - K steps of the policy in
       closed loop, under the *runtime* objective = the task reward minus
       whatever new cost terms were asked for at run time
    4. execute the first action of the softmax-weighted chunk

It is a sampling version of Bertsekas' rollout algorithm with the RL policy as
the base policy: the nominal chunk is always a candidate, so in the planner's
model the chosen chunk is never worse than the policy under the runtime
objective (exactly so as the softmax sharpens to an argmin).  Why so few
samples suffice is `csm.smoothness_probe`: completing the horizon in closed
loop makes the cost far better conditioned than the open-loop plan cost MPPI
normally scores (non-smooth residual 0.08 vs 1.13 logits, cloud ESS 0.87 vs
0.19 on the walking plant).

The planner model may use a coarser physics substep than the world
(`--plan-timestep 0.02` against the world's 0.01), which halves the rollout's
sequential depth -- the only thing that costs time below ~512 samples.

Runtime objectives (none of them in the policy's training):

  band:<name>      joint-target band from `csm.constraint_compose`
                   (knee, front, crouch, lock)
  height:<z>:<b>   walk with the torso at z metres        cost b*(z_torso - z)^2
  step:<a>:<b>     foot-height schedule amplitude a (trained: 0.08), in the
                   gait row's own form                     cost b*0.1*sum(((zt-zf)/0.05)^2)
  energy:<b>       joint torque effort                     cost b*sum(tau^2)/1e3

Scored on the task reward alone (the objective the policy was trained on), the
runtime term's own metric, collapses (torso below 0.15 m for 100 steps or at
the end) and the control period.
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

from csm.basis_screen import _load_config, build_omegas
from csm.constraint_compose import ROW_SCALES, collapsed, test_band
from csm.rl_baseline import gait_clock, load_policy
from csm.screen import COMMANDS, set_command, set_omega
from dial_mpc.utils.function_utils import get_foot_step

NO_BAND = (jnp.full(12, -1.0), jnp.ones(12))


def make_env(timestep: float = 0.01):
    dc, ec = _load_config("unitree_go2_trot_csm", None)
    ec = dataclasses.replace(ec, **ROW_SCALES, timestep=timestep)
    return brax_envs.get_environment(dc.env_name, config=ec)


def parse_objective(spec: str):
    """-> (band, cost(state) -> scalar, metric(state) -> scalar, label)."""

    kind, *args = spec.split(":")
    zero = lambda st: jnp.zeros(())
    if kind == "none":
        return NO_BAND, zero, lambda st: st.pipeline_state.x.pos[0, 2], "torso z"
    if kind == "band":
        return test_band(args[0]), zero, lambda st: st.pipeline_state.x.pos[0, 2], "torso z"
    if kind == "height":
        z, b = float(args[0]), float(args[1])
        torso = lambda st: st.pipeline_state.x.pos[0, 2]
        return NO_BAND, lambda st: b * (torso(st) - z) ** 2, torso, "torso z"
    if kind == "step":
        amp, b = float(args[0]), float(args[1])

        def feet(st, env):
            return st.pipeline_state.site_xpos[env._feet_site_id][:, 2]

        def cost_for(env):
            ratio, cadence, _ = env._gait_params[env._gait]
            phases = env._gait_phase[env._gait]

            def cost(st):
                tar = get_foot_step(ratio, cadence, amp, phases, st.info["step"] * env.dt)
                return b * 0.1 * jnp.sum(((tar - feet(st, env)) / 0.05) ** 2)
            return cost

        return NO_BAND, cost_for, lambda st, env=None: None, "foot apex"
    if kind == "energy":
        b = float(args[0])
        effort = lambda st: jnp.sum(st.pipeline_state.qfrc_actuator[6:] ** 2)
        return NO_BAND, lambda st: b * effort(st) / 1e3, effort, "sum tau^2"
    raise ValueError(spec)


def _bind_one(spec, env):
    band, cost, metric, label = parse_objective(spec)
    if spec.startswith("step"):
        cost = cost(env)
        feet_id = env._feet_site_id
        metric = lambda st: jnp.max(st.pipeline_state.site_xpos[feet_id][:, 2])
    return band, cost, metric, label


def bind(spec, env):
    """Objective pieces bound to an environment (the step schedule needs its gait).

    `a+b` composes runtime terms: bands intersect, costs add, and the metric
    is the first soft term's (torso height if there is none).
    """

    parts = [_bind_one(p, env) for p in spec.split("+")]
    lo = jnp.max(jnp.stack([p[0][0] for p in parts]), 0)
    hi = jnp.min(jnp.stack([p[0][1] for p in parts]), 0)
    costs = [p[1] for p in parts]
    soft = [p for p, name in zip(parts, spec.split("+")) if not name.startswith(("band", "none"))]
    metric, label = (soft[0][2], soft[0][3]) if soft else (parts[0][2], parts[0][3])
    return (lo, hi), (lambda st: sum(c(st) for c in costs)), metric, label


def make_controller(world, planner, path, spec, n_steps, *, n_samples=256, chunk=4,
                    horizon=16, sigma=0.15, temp=0.1, plan=True, record=None,
                    replan_every=1, residual_sigma=0.0):
    """`plan=False` is the policy alone (actions clipped to the band).

    `record(state)` picks what each step returns; the default is what scoring
    needs, and the clip renderer passes one that keeps the pose instead.

    `replan_every=R` plans once every R control steps and executes the next R
    actions of the weighted chunk in between -- the asynchronous deployment,
    where a plan may take up to R control periods to compute.

    `residual_sigma > 0` also samples a constant action offset b that the
    policy carries through the completion, `a = clip(pi(s) + b)`, warm-started
    from the previous plan's weighted offset.  A chunk alone can only buy K
    steps of deviation before the completion returns the robot to the
    policy's own style, so an objective that asks for a *sustained* change
    (walk lower) sees little credit; the offset gives the completion that
    authority while keeping it closed loop.
    """

    inference, blob = load_policy(path)
    cadence = blob["clock_cadence"]
    (lo, hi), cost_w, metric, _ = bind(spec, world)
    _, cost_p, _, _ = bind(spec, planner)

    def act(env, st, b=0.0):
        obs = jnp.concatenate([st.obs, gait_clock(st.info["step"], env.dt, cadence)])
        a, _ = inference(obs, jax.random.PRNGKey(0))
        return jnp.clip(a + b, lo, hi)

    def nominal(st, b):
        def step(s, _):
            a = act(planner, s, b)
            return planner.step(s, a), a
        return jax.lax.scan(step, st, None, length=chunk)[1]

    def score(st, a_chunk, b):
        def step(s, t):
            a = jnp.where(t < chunk, a_chunk[jnp.minimum(t, chunk - 1)], act(planner, s, b))
            s = planner.step(s, a)
            return s, s.reward - cost_p(s)
        return jax.lax.scan(step, st, jnp.arange(horizon))[1].mean()

    def control(st, key, b_prev):
        k1, k2 = jax.random.split(key)
        base = nominal(st, b_prev)
        eps = jax.random.normal(k1, (n_samples, chunk, base.shape[-1]))
        cand = jnp.concatenate([jnp.clip(base[None] + sigma * eps, lo, hi), base[None]], 0)
        offs = b_prev + residual_sigma * jax.random.normal(k2, (n_samples + 1, base.shape[-1]))
        offs = offs.at[-1].set(b_prev)
        r = jax.vmap(lambda c, b: score(st, c, b))(cand, offs)
        w = jax.nn.softmax((r - r[-1]) / jnp.maximum(r.std(), 1e-6) / temp)
        return (jnp.clip(jnp.einsum("n,nka->ka", w, cand), lo, hi),
                jnp.einsum("n,na->a", w, offs))

    @jax.jit
    def run(state, rng):
        def body(c, t):
            st, k, held, b = c
            k, sub = jax.random.split(k)
            # The planner reads the world state under its own physics substep:
            # same qpos/qvel/info, possibly a coarser step.
            if plan:
                held, b = jax.lax.cond(t % replan_every == 0,
                                       lambda: control(st, sub, b), lambda: (held, b))
                a = held[t % replan_every]
            else:
                a = act(world, st)
            st = world.step(st, a)
            out = ((st.reward, st.pipeline_state.x.pos[0, 2], metric(st)) if record is None
                   else record(st))
            return (st, k, held, b), out
        held0 = jnp.zeros((chunk, world.action_size))
        b0 = jnp.zeros(world.action_size)
        return jax.lax.scan(body, (state, rng, held0, b0), jnp.arange(n_steps))[1]

    return run


def evaluate(arms, specs, commands, seeds, steps, out, plan_timestep, **kw):
    world = make_env(0.01)
    planner = make_env(plan_timestep)
    reset = jax.jit(world.reset)
    omega = build_omegas(3)["uniform"]
    res = json.loads(out.read_text()) if out is not None and out.exists() else {}
    for arm, path in arms:
        for spec in specs:
            key = f"{arm}/{spec}"
            if key in res:
                continue
            run = make_controller(world, planner, path, spec, steps,
                                  plan=not arm.startswith("policy"), **kw)
            costs, cols, mets, t0 = [], [], [], time.time()
            for cname in commands:
                for seed in range(seeds):
                    st = set_omega(set_command(world, reset(jax.random.PRNGKey(11 + seed)),
                                               COMMANDS[cname]), omega)
                    r, z, m = run(st, jax.random.PRNGKey(seed))
                    m = np.asarray(m)[50:]
                    costs.append(-float(np.asarray(r).mean())); cols.append(collapsed(z))
                    first_soft = [p for p in spec.split("+")
                                  if not p.startswith(("band", "none"))]
                    mets.append(float(np.percentile(m, 95)
                                      if first_soft and first_soft[0].startswith("step")
                                      else m.mean()))
            res[key] = {"task_cost": costs, "collapsed": cols, "metric": mets}
            print(f"{arm:<10}{spec:<18} task {np.mean(costs):.4f} +- "
                  f"{np.std(costs) / np.sqrt(len(costs)):.4f}  metric {np.mean(mets):.4f}  "
                  f"collapsed {sum(cols)}/{len(cols)}  ({time.time() - t0:.0f}s)", flush=True)
            if out is not None:
                out.write_text(json.dumps(res, indent=1))
    return res


def train_with_objective(spec: str, out: Path, steps: int) -> None:
    """The retraining baseline: PPO + gait clock with the runtime term in its reward.

    Same recipe as the clock sweep (1024 envs, episode 1000, discount 0.97,
    fixed horizon), so the only difference from the base policy is the term --
    this is what the planner's zero-shot result is measured against.
    """

    import cloudpickle
    from brax.envs.base import Wrapper
    from csm import rl_baseline as RB

    dc, ec = _load_config("unitree_go2_trot_csm", None)
    omega = build_omegas(3)["uniform"]
    ec = dataclasses.replace(ec, **ROW_SCALES, reward_weights=jnp.asarray(omega, jnp.float32))
    raw = brax_envs.get_environment(dc.env_name, config=ec)
    (lo, hi), cost, _, _ = bind(spec, raw)

    class WithTerm(Wrapper):
        def step(self, state, action):
            nxt = self.env.step(state, jnp.clip(action, lo, hi))
            return nxt.replace(reward=nxt.reward - cost(nxt))

    cadence = RB.gait_cadence(raw)
    env = RB.GaitClockWrapper(RB.FixedHorizonWrapper(WithTerm(raw)), cadence)
    args = RB._parser().parse_args([
        "--output", str(out), "--num-timesteps", str(steps), "--num-envs", "1024",
        "--episode-length", "1000", "--num-evals", "6", "--hidden", "512,256,128",
        "--discounting", "0.97"])
    started = time.time()
    _, params, _, _ = RB.train(args, env)
    RB.save_policy(out / "policy.pkl", algo="ppo", hidden=(512, 256, 128), params=params,
                   obs_size=env.observation_size, act_size=env.action_size, omega=omega,
                   omega_name="uniform", clock_cadence=cadence)
    blob = cloudpickle.load(open(out / "policy.pkl", "rb"))
    blob["runtime_objective"] = spec
    cloudpickle.dump(blob, open(out / "policy.pkl", "wb"))
    print(f"retrained on {spec} in {(time.time() - started) / 60:.1f} min -> {out}", flush=True)


def time_step(path, spec, plan_timestep, **kw):
    world, planner = make_env(0.01), make_env(plan_timestep)
    omega = build_omegas(3)["uniform"]
    st = set_omega(set_command(world, world.reset(jax.random.PRNGKey(0)),
                               COMMANDS["box_fast"]), omega)
    run = make_controller(world, planner, path, spec, 50, **kw)
    jax.block_until_ready(run(st, jax.random.PRNGKey(0)))
    t0 = time.time()
    jax.block_until_ready(run(st, jax.random.PRNGKey(1)))
    return (time.time() - t0) / 50 * 1e3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "POLICY"),
                    help="NAME starting with 'policy' runs the policy alone; anything "
                         "else plans on top of it")
    ap.add_argument("--objectives", nargs="+", default=["none", "band:lock"])
    ap.add_argument("--commands", default="box_fast,box_turn")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--plan-timestep", type=float, default=0.01)
    ap.add_argument("--replan-every", type=int, default=1)
    ap.add_argument("--residual-sigma", type=float, default=0.0,
                    help="also plan a constant offset the policy carries through the "
                         "completion (sustained objectives)")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--train", nargs=2, metavar=("OBJECTIVE", "OUT"),
                    help="train the retraining baseline for one runtime objective")
    ap.add_argument("--train-steps", type=int, default=50_000_000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.train:
        train_with_objective(a.train[0], Path(a.train[1]), a.train_steps)
        return 0
    kw = dict(n_samples=a.samples, chunk=a.chunk, horizon=a.horizon,
              replan_every=a.replan_every, residual_sigma=a.residual_sigma)
    policy = "csm_runs/rl-sweep-clock/uniform/policy.pkl"
    ms = time_step(policy, a.objectives[0], a.plan_timestep, **kw)
    print(f"control step {ms:.1f} ms amortised incl. the world step (N={a.samples}, "
          f"K={a.chunk}, H={a.horizon}, planner substep {a.plan_timestep}, replan every "
          f"{a.replan_every}; period 20 ms, so a plan may take {20 * a.replan_every} ms)",
          flush=True)
    if a.time_only:
        return 0
    arms = a.arm or [("policy", policy), ("pc", policy)]
    evaluate(arms, a.objectives, a.commands.split(","), a.seeds, a.steps, a.out,
             a.plan_timestep, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
