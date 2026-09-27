"""Is the CSM student's win over clock-less PPO a memory effect a history window also buys?

Without a gait clock, PPO scores 2.6-2.8x DIAL at the gait-heavy weight boost2
while the CSM student scores 0.98-1.26: the student carries the schedule's
phase in its warm-started plan.  With the clock PPO scores 0.10.  The question
for a "no clock available" niche is whether a reactive policy given a window
of its own past observations recovers the phase too.  If it does, the niche is
closed by frame stacking.

PPO at boost2, same recipe as the sweeps (50M steps, fixed horizon), with the
last K observations concatenated (no clock), scored like every other arm on
box_fast/slow/turn/strafe x 2 seeds x 1500 steps against the cached DIAL.
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import jax
import jax.numpy as jnp
import brax.envs as brax_envs
from brax.envs.base import Wrapper

from csm import rl_baseline as RB
from csm.basis_screen import _load_config, build_omegas
from csm.compose_walk_eval import make_teacher, summarise
from csm.dial_lean import make_dial_step, make_lean_update, mppi_logits
from csm.screen import COMMANDS, set_command, set_omega
from csm.teacher_cache import DEFAULT_ROOT, TeacherCache, episode_key, fingerprint
from dial_mpc.core.dial_core import MBDPI, make_controller


class History(Wrapper):
    """Observation = the last `k` raw observations, newest first."""

    def __init__(self, env, k: int):
        super().__init__(env)
        self._k = int(k)

    @property
    def observation_size(self) -> int:
        return int(self.env.observation_size) * self._k

    def reset(self, rng):
        st = self.env.reset(rng)
        hist = jnp.tile(st.obs[None], (self._k, 1))
        return st.replace(obs=hist.reshape(-1), info={**st.info, "obs_hist": hist})

    def step(self, st, action):
        nxt = self.env.step(st, action)
        hist = jnp.concatenate([nxt.obs[None], st.info["obs_hist"][:-1]], 0)
        info = dict(nxt.info)
        info["obs_hist"] = hist
        return nxt.replace(obs=hist.reshape(-1), info=info)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--omega", default="boost2")
    ap.add_argument("--steps", type=float, default=5e7)
    a = ap.parse_args(argv)
    omega = RB.resolve_omega(a.omega, 3)
    rows = (2.479, 0.261, 1.551)
    env, _, _ = RB.build_env("unitree_go2_trot_csm", omega, terminate=False,
                             randomize_start=False, row_scales=rows)
    env = History(env, a.k)
    args = RB._parser().parse_args([
        "--output", "/tmp/unused", "--num-timesteps", str(int(a.steps)), "--num-envs", "1024",
        "--episode-length", "1000", "--num-evals", "6", "--hidden", "512,256,128",
        "--discounting", "0.97"])
    t0 = time.time()
    make_inf, params, _, _ = RB.train(args, env)
    print(f"trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    inf = make_inf(params, deterministic=True)

    dc, ec = _load_config("unitree_go2_trot_csm", None)
    ec = dataclasses.replace(ec, track_scale=rows[0], stability_scale=rows[1], gait_scale=rows[2])
    dc = dataclasses.replace(dc, temp_sample=0.02)
    raw = brax_envs.get_environment(dc.env_name, config=ec)
    mbdpi = make_controller(dc, raw)
    digest, manifest = fingerprint(
        dial_config=dc, env_config=ec, env=raw,
        functions=(make_teacher, make_dial_step, make_lean_update, mppi_logits, MBDPI, type(mbdpi)),
        extra={"init_passes": 5, "std_normalize": False, "level_scales": [2.625, 1.0],
               "temperature": 0.02})
    cache = TeacherCache(DEFAULT_ROOT, digest, manifest)
    hist_env = History(raw, a.k)
    w = build_omegas(3)[a.omega]

    @jax.jit
    def run(state):
        def body(c, _):
            st, k = c
            k, sub = jax.random.split(k)
            act, _ = inf(st.obs, sub)
            st = hist_env.step(st, act)
            return (st, k), (st.reward, st.done)
        return jax.lax.scan(body, (state, jax.random.PRNGKey(0)), None, length=1500)[1]

    ratios = []
    for cname in ("box_fast", "box_slow", "box_turn", "box_strafe"):
        cs, ts = [], []
        for seed in range(2):
            st = set_omega(set_command(raw, raw.reset(jax.random.PRNGKey(11 + seed)),
                                       COMMANDS[cname]), w)
            tr, td = cache.load(episode_key(omega=w, command=COMMANDS[cname], seed=seed), 1500)
            ts.append(summarise(tr, td, 1500)[0])
            hist = jnp.tile(st.obs[None], (a.k, 1))
            st = st.replace(obs=hist.reshape(-1), info={**st.info, "obs_hist": hist})
            r, d = run(st)
            cs.append(summarise(np.asarray(r), np.asarray(d), 1500)[0])
        ratios.append(np.mean(cs) / np.mean(ts))
        print(f"{cname:<11} PPO + {a.k}-step history (no clock) ratio {ratios[-1]:.3f}", flush=True)
    print(f"{a.omega}: history-PPO mean ratio {np.mean(ratios):.3f}  "
          f"(no clock 2.71, clock 0.10, CSM 1.10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
