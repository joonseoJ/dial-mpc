"""Stepping stones with a known layout: does PPO match the planner that sees the terrain?

The terrain gate's oracle -- DIAL planning under the true stone layout --
crossed 12 x 2 stones (a quarter of them holes 0.3 m deep) with 1 fall in 16
strips.  Precise foothold choice with a known map is the textbook case for
look-ahead planning.  Here PPO gets the same information: the true layout as a
window of the stones around its body (solid/hole per stone, 10 rows x 2 lanes,
starting just behind the rear feet), where it sits within the stone grid, and
the gait clock; a fresh random layout every episode; the stock walking reward;
fixed horizon.  Scored on the gate's own 16 strips and reset keys.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper

from csm import rl_baseline as RB
from csm.niche.crate_rl import Guard
from csm.terrain import stones as S
from csm.terrain.check import make_env

WIN = 10


def window(tops, x):
    """(WIN*NY + 1,): solid(1)/hole(0) for the stones from just behind the rear
    feet forward, and the fractional position within the current stone."""
    s = (x - 0.25 - S.X0) / S.CX
    ix0 = jnp.floor(s).astype(jnp.int32)
    ix = ix0 + jnp.arange(WIN)
    grid = (tops.reshape(S.NX, S.NY) > -0.01).astype(jnp.float32)
    inside = (ix >= 0) & (ix < S.NX)
    rows = jnp.where(inside[:, None], grid[jnp.clip(ix, 0, S.NX - 1)], 1.0)
    return jnp.concatenate([rows.reshape(-1), (s - ix0)[None]])


class StonesRL(Wrapper):
    def __init__(self, env: S.StonesEnv, lay: S.Layout, cadence: float):
        super().__init__(env)
        self._lay, self._cad = lay, cadence

    @property
    def observation_size(self):
        return int(self.env.observation_size) + WIN * S.NY + 1 + 2

    def _obs(self, st):
        x = st.pipeline_state.x.pos[0, 0]
        return st.replace(obs=jnp.concatenate([
            st.obs, window(st.info["tops"], x), RB.gait_clock(st.info["step"], self.dt, self._cad)]))

    def reset_with(self, rng, tops):
        st = self.env.reset(rng)
        return self._obs(st.replace(info={**st.info, "tops": tops}))

    def reset(self, rng):
        k1, k2 = jax.random.split(rng)
        tops, _ = S.sample_episode(k2, self._lay)
        return self.reset_with(k1, tops)

    def step(self, st, action):
        nxt = self.env.step_in(st.info["tops"], st, action)
        return self._obs(nxt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppo-steps", type=float, default=5e8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--curriculum", action="store_true",
                    help="the usual RL remedy: train on flat strips, then 10%% holes, then "
                         "the gate's layout, resuming the parameters each stage")
    ap.add_argument("--train-holes", type=float, default=None,
                    help="train on strips where every stone is seen and a hole with "
                         "this probability (0 = flat); default: the gate's layout")
    a = ap.parse_args(argv)
    base = make_env(0.6)
    lay = S.Layout()
    train_lay = lay if a.train_holes is None else S.Layout(p_seen=1.0, p_seen_hole=a.train_holes)
    cadence = float(np.asarray(base._gait_params[base._gait])[1])
    from brax.training.agents.ppo import train as ppo
    stages = ([(S.Layout(p_seen=1.0, p_seen_hole=0.0), 0.2), (S.Layout(p_seen=1.0, p_seen_hole=0.1), 0.3),
               (lay, 0.5)] if a.curriculum else [(train_lay, 1.0)])
    t0 = time.time()
    params = None
    for stage_lay, share in stages:
        train_env = Guard(RB.FixedHorizonWrapper(StonesRL(base, stage_lay, cadence)))
        make_inf, params, _ = ppo.train(
            environment=train_env, num_timesteps=int(a.ppo_steps * share), episode_length=a.steps,
            num_envs=1024, batch_size=1024, unroll_length=20, num_minibatches=32,
            num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
            num_evals=4, seed=0, normalize_observations=True, restore_params=params,
            progress_fn=lambda n, m: print(f"  {n:>13,} steps  eval reward "
                                           f"{m['eval/episode_reward']:.2f}", flush=True))
        print(f"stage done (holes {stage_lay.p_seen_hole if stage_lay.p_seen == 1.0 else 'gate'}) "
              f"at {(time.time() - t0) / 60:.1f} min", flush=True)
    env = StonesRL(base, lay, cadence)
    inf = make_inf(params, deterministic=True)

    E = a.episodes
    tt, _ = jax.vmap(lambda k: S.sample_episode(k, lay))(jax.random.split(jax.random.PRNGKey(123), E))
    run_keys = jax.random.split(jax.random.PRNGKey(456), E)

    @jax.jit
    def run(tops, key):
        _, kr = jax.random.split(key)
        st = env.reset_with(kr, tops)

        def body(s, _):
            act, _ = inf(s.obs, jax.random.PRNGKey(0))
            s = env.step(s, act)
            ps = s.pipeline_state
            return s, (s.reward, s.done, ps.x.pos[0, 0], ps.x.pos[0, 1],
                       jnp.min(ps.site_xpos[base._feet_site_id][:, 2]))
        return jax.lax.scan(body, st, None, length=a.steps)[1]

    for label, strips in (("the gate's strips", tt),
                          ("all-solid strips", jnp.zeros_like(tt))):
        costs, fells, xs, why = [], [], [], []
        for i in range(E):
            r, d, x, y, fz = (np.asarray(v) for v in run(strips[i], run_keys[i]))
            costs.append(-r.sum()); fells.append(bool((d > 0).any())); xs.append(x[-1])
            if fells[-1]:
                t = int(np.argmax(d > 0))
                why.append(f"t={t} x={x[t]:.2f} y={y[t]:+.2f} lowest foot z={fz[:t + 1].min():+.2f}")
        print(f"PPO on {label}: cost {np.mean(costs):.2f}  fell {sum(fells)}/{E}  "
              f"x_end {np.mean(xs):.2f} (min {np.min(xs):.2f})   "
              f"[oracle DIAL, gate strips, 300 steps: 6.97, 0/16 fell, x_end 3.23]", flush=True)
        for w in why[:4]:
            print(f"    first fall {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
