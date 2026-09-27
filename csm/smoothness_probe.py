"""Is the open-loop plan cost rough where a feedback-completed one is smooth?

Every expensive thing measured on this project has one candidate cause.  MPPI
scores a plan by rolling it out *open loop*: a fixed action sequence, so a
slightly different plan lands a foot a few milliseconds earlier or later and the
16-step cost jumps.  Such a landscape needs many samples and a sharp softmax
(the label estimator's 1-10% ESS), and a smooth regressor cannot represent its
extremes (the plan-cost surrogate's bias at sharp temperatures).  A closed-loop
value -- the same first actions, then a feedback policy for the rest of the
horizon -- absorbs those timing slips, which would be why RL's objects are cheap
to learn.

This probe measures it directly.  At states the shipped walking student
reaches, along random directions through its own plan (in units of the fine
level's per-node sigma), it compares the 16-step cost of

  open loop    the perturbed plan executed for all 16 steps (what MPPI scores)
  chunk + pi   the perturbed plan for the first K steps, then a feedback
               policy (PPO + gait clock) for the remaining 16 - K

and reports the share of each 1-D profile's variance a quadratic cannot
explain, the same residual in softmax-logit units at the collection
temperature, and the effective sample size a 2049-sample MPPI cloud would get
under each cost.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np
import jax
import jax.numpy as jnp

import brax.envs as brax_envs

import dial_mpc.envs as dial_envs  # noqa: F401
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.compose_walk_eval import make_student
from csm.dial_lean import make_sampler
from csm.dial_score import ComposedDialScorePolicy
from csm.rl_baseline import gait_clock, load_policy
from csm.screen import COMMANDS, set_command, set_omega

ROW_SCALES = dict(track_scale=2.479, stability_scale=0.261, gait_scale=1.551)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="csm_runs/clouds-fit-20260905-083613/policy.pkl")
    ap.add_argument("--feedback", default="csm_runs/rl-sweep-clock/uniform/policy.pkl")
    ap.add_argument("--chunk", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--directions", type=int, default=24)
    ap.add_argument("--grid", type=int, default=81)
    ap.add_argument("--span", type=float, default=2.0, help="+- span in sigma units")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    dc, ec = _load_config("unitree_go2_trot_csm", None)
    ec = dataclasses.replace(ec, **ROW_SCALES)
    dc = dataclasses.replace(dc, temp_sample=0.02)
    env = brax_envs.get_environment(dc.env_name, config=ec)
    mbdpi = make_controller(dc, env)
    omega = jnp.asarray(build_omegas(3)["uniform"], jnp.float32)
    policy = ComposedDialScorePolicy.load(a.policy)
    inference, blob = load_policy(a.feedback)
    cadence = blob["clock_cadence"]
    H = dc.Hsample
    sigma = mbdpi.sigma_control * 0.5          # the fine level
    temp = 0.02                                 # its temperature

    # states and plans the student actually holds
    def keep(st):
        return st
    student = make_student(env, policy, dc, 5, 300,
                           record=lambda st: st)
    states, plans = [], []
    for cname in ("box_fast", "box_turn"):
        for seed in range(2):
            st0 = set_omega(set_command(env, env.reset(jax.random.PRNGKey(11 + seed)),
                                        COMMANDS[cname]), omega)
            traj = student(st0, omega, 0.02)
            for t in (100, 160, 220, 280):
                states.append(jax.tree.map(lambda x: x[t], traj))
    # the plan at those states: re-derive by running the student's refinement
    # from its own warm start is what make_student does internally; here the
    # probe centres on a fresh 5-pass refinement at the state, which is what
    # the student would hold after re-planning.
    from csm.dial_score import factor_to_t
    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    lo, hi = float(factors.min()), float(factors.max())
    from csm.omega import mixture_from_pinv
    mix = mixture_from_pinv(omega, 0.02, policy.pinv_nu_weights, policy.pinv_mode_weights)

    @jax.jit
    def refine(obs):
        def level(pl, f):
            t = factor_to_t(f, lo, hi).reshape(1)
            d = sum(m * fl.delta(pl, obs, t) for m, fl in zip(mix, fields))
            return jnp.clip(pl + d, -1.0, 1.0), None
        return jax.lax.scan(level, jnp.zeros((dc.Hnode + 1, 12)), jnp.tile(factors, 5))[0]

    plans = [refine(s.obs) for s in states]

    def cost_open(state, nodes):
        us = mbdpi.node2u_vmap(nodes)
        def step(st, u):
            st = env.step(st, u)
            return st, st.reward
        return -jax.lax.scan(step, state, us)[1].mean()

    def cost_chunk(state, nodes, k):
        us = mbdpi.node2u_vmap(nodes)
        def step(carry, t):
            st = carry
            obs = jnp.concatenate([st.obs, gait_clock(st.info["step"], env.dt, cadence)])
            a_pi, _ = inference(obs, jax.random.PRNGKey(0))
            st = env.step(st, jnp.where(t < k, us[t], a_pi))
            return st, st.reward
        return -jax.lax.scan(step, state, jnp.arange(H))[1].mean()

    ts = jnp.linspace(-a.span, a.span, a.grid)

    @jax.jit
    def profiles(state, plan, dirs):
        def line(d):
            nodes = jnp.clip(plan[None] + ts[:, None, None] * sigma[None, :, None] * d[None],
                             -1.0, 1.0)
            nodes = nodes.at[:, 0].set(plan[0])
            op = jax.vmap(lambda n: cost_open(state, n))(nodes)
            ch = [jax.vmap(lambda n, k=k: cost_chunk(state, n, k))(nodes) for k in a.chunk]
            return jnp.stack([op] + ch)
        return jax.vmap(line)(dirs)                       # (D, 1+len(chunk), G)

    sample = make_sampler(mbdpi, dc)

    @jax.jit
    def cloud_ess(state, plan, key):
        nodes = sample(key, plan, sigma)
        op = jax.vmap(lambda n: cost_open(state, n))(nodes)
        ch = [jax.vmap(lambda n, k=k: cost_chunk(state, n, k))(nodes) for k in a.chunk]
        out = []
        for c in [op] + ch:
            w = jax.nn.softmax(-(c - c[-1]) / temp)
            out.append(1.0 / jnp.sum(w ** 2) / c.shape[0])
        return jnp.stack(out)

    names = ["open"] + [f"chunk{k}+pi" for k in a.chunk]
    rough, rough_logit, ess = [[] for _ in names], [[] for _ in names], [[] for _ in names]
    X = np.stack([np.ones(a.grid), np.asarray(ts), np.asarray(ts) ** 2], 1)
    for i, (st, pl) in enumerate(zip(states, plans)):
        key = jax.random.PRNGKey(100 + i)
        dirs = jax.random.normal(key, (a.directions, dc.Hnode + 1, 12)).at[:, 0].set(0.0)
        dirs = dirs / jnp.linalg.norm(dirs.reshape(a.directions, -1), axis=1)[:, None, None] \
            * np.sqrt((dc.Hnode) * 12)
        P = np.asarray(profiles(st, pl, dirs))
        for j in range(len(names)):
            for prof in P[:, j]:
                beta, *_ = np.linalg.lstsq(X, prof, rcond=None)
                res = prof - X @ beta
                rough[j].append(res.var() / max(prof.var(), 1e-12))
                rough_logit[j].append(res.std() / temp)
        e = np.asarray(cloud_ess(st, pl, jax.random.fold_in(key, 7)))
        for j in range(len(names)):
            ess[j].append(float(e[j]))
        print(f"state {i + 1}/{len(states)}  " + "  ".join(
            f"{n}: unexplained {np.median(rough[j][-a.directions:]):.3f} "
            f"resid {np.median(rough_logit[j][-a.directions:]):.2f} logit ESS {e[j]:.3f}"
            for j, n in enumerate(names)), flush=True)
    summary = {n: {"unexplained_median": float(np.median(rough[j])),
                   "unexplained_mean": float(np.mean(rough[j])),
                   "resid_logit_median": float(np.median(rough_logit[j])),
                   "ess_mean": float(np.mean(ess[j]))} for j, n in enumerate(names)}
    print("\nsummary (quadratic-unexplained variance share along 1-D lines, "
          "residual in logits at T=0.02, cloud ESS share):")
    for n, s in summary.items():
        print(f"  {n:<12} unexplained median {s['unexplained_median']:.3f} "
              f"mean {s['unexplained_mean']:.3f}  resid {s['resid_logit_median']:.2f} logits  "
              f"ESS {s['ess_mean']:.3f}")
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
