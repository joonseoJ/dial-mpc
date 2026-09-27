"""Gait mixtures scored on the objective itself: composed fields against one conditioned PPO.

The gait head-to-head (`csm/gait/head2head.py`) scored contact patterns.  The
question here is the objective's own number: at a weight omega, which
controller achieves the lower cost under omega?  This is where conditioned RL
is most likely to be weak -- the four rows ask for mutually exclusive stepping
patterns, one network has to hold all of them, and its training reward was 4x
the specialists' -- while the composed fields are trained one pattern at a time.

Arms (same randomised resets, commands and weights; the gait observation
already carries the phase clock, so the RL arms see everything the fields see):

  csm       composed score fields (gait-fit-d1), 6 refinement passes per step
  rl_cond   one PPO conditioned on omega
  rl_mix    the four PPO specialists averaged with omega's coefficients
            (at a pure weight: that gait's own specialist)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from csm.dial_score import ComposedDialScorePolicy
from csm.gait.head2head import (build, csm_arm, make_reset, rl_cond_arm, rl_mix_arm,
                                weight_suite, COLLAPSE_RUN, COLLAPSE_Z)
from csm.rl_baseline import load_policy

COMMANDS = [("vx 0.6", (0.6, 0.0, 0.0)), ("vx 0.8", (0.8, 0.0, 0.0)), ("vx 1.0", (1.0, 0.0, 0.0))]


def make_rollout(env, reset_fn, init, act, steps):
    ti = env._torso_idx - 1

    @jax.jit
    def run(keys, cmd, omega, temperature):
        states = reset_fn(keys, cmd, omega)
        carry = jax.vmap(init, in_axes=(0, None, None))(states, omega, temperature)

        def body(sc, _):
            st, cr = sc
            cr, a = jax.vmap(act, in_axes=(0, 0, None, None))(cr, st, omega, temperature)
            st = jax.vmap(env.step)(st, a)
            return (st, cr), (st.reward, st.done, st.pipeline_state.x.pos[:, ti, 2])

        return jax.lax.scan(body, (states, carry), None, length=steps)[1]

    return run


def collapsed(done, z):
    run = best = 0
    for v in done > 0.5:
        run = run + 1 if v else 0
        best = max(best, run)
    return bool(best >= COLLAPSE_RUN or z[-30:].mean() < COLLAPSE_Z)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csm-policy", default="csm_runs/gait-fit-d1/clouds-fit-20260915-115906/policy.pkl")
    p.add_argument("--rl-conditioned", default="csm_runs/rl-gait-cond/policy.pkl")
    p.add_argument("--rl-specialists", nargs=4,
                   default=[f"csm_runs/rl-gait-e{i}/policy.pkl" for i in range(4)])
    p.add_argument("--weights", default="all")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--step-passes", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("csm_runs/niche_gait_cost.json"))
    a = p.parse_args(argv)
    env, dial_config = build(a)
    keys = jax.random.split(jax.random.PRNGKey(20260927), a.seeds)
    reset_j = make_reset(env, a.seeds)
    policy = ComposedDialScorePolicy.load(a.csm_policy)
    runners = {
        "csm": make_rollout(env, reset_j, *csm_arm(env, dial_config, policy, a.step_passes, 5), a.steps),
        "rl_cond": make_rollout(env, reset_j, *rl_cond_arm(load_policy(a.rl_conditioned)[0]), a.steps),
        "rl_mix": make_rollout(env, reset_j, *rl_mix_arm([load_policy(q)[0] for q in a.rl_specialists]), a.steps),
    }
    res = {}
    print(f"{'weight':<24}{'command':<9}" + "".join(f"{n:>16}" for n in runners), flush=True)
    for wname, omega in weight_suite(a.weights):
        for cname, cmd in COMMANDS:
            line = f"{wname:<24}{cname:<9}"
            for arm, run in runners.items():
                r, d, z = (np.asarray(x) for x in run(keys, jnp.asarray(cmd), jnp.asarray(omega),
                                                       a.temperature))
                costs = (-r.mean(0)).tolist()
                cols = [collapsed(d[:, s], z[:, s]) for s in range(a.seeds)]
                res[f"{arm}/{wname}/{cname}"] = {"costs": costs, "collapsed": cols}
                line += f"{np.mean(costs):>12.3f} c{sum(cols)}"
            print(line, flush=True)
            a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
