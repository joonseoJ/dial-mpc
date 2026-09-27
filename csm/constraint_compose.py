"""Can a learned controller keep the planner's runtime flexibility?

An online sampling planner accepts a constraint it has never seen -- clip its
samples to the allowed set and it re-plans around it.  A distilled plan-space
student can do the same in principle: project its plan after every annealing
level, and the fields, which see the whole plan, re-optimise the free
coordinates around the clipped ones.  A reactive policy has no plan to project;
all it can do is have its action filtered, or be trained on a constraint family
it is told about.

Measured on the shipped walking student (2026-09-26): a single-knee band costs
the student +111% with a safety filter and +26% composed into the plan (DIAL
itself: +38%), but crouching -- all four calves capped -- flips the student and
the PPO arms 6/6 while DIAL walks.  Composition works near the data and nowhere
else.  The hypothesis tested here is that *coverage* is the missing piece:

  H  Collect DAgger clouds with the student driving under a random family of
     single-joint bands (composed into its plan; labels stay the unconstrained
     rows), refit, and the student composes constraints outside that family --
     conjunctions of bands, a lock -- while a PPO policy trained on the same
     family with the band as an input does not.

Pre-registered test (target `uniform`, box_fast and box_turn, 3 seeds, 1500
steps):

  in family     knee    FR calf flexion bound  (lo[5] = -0.2)
  out of family front   both front calves      (lo[2] = lo[5] = -0.2)
                crouch  all four calves capped (hi[2,5,8,11] = 0.1)
                lock    FR calf fixed          (lo[5] = hi[5] = 0.24)

  arms: dial (in-planner, ground truth), csm-prod (shipped fields, projected),
        csm-cov (coverage fields, projected), ppo (fixed-weight PPO + gait
        clock, safety filter), bppo (PPO + clock trained on the band family
        with the band observed, env clips its action)

  H holds only if, over the three out-of-family constraints (18 episodes),
  csm-cov collapses at least 4 fewer times than bppo, or both never collapse
  and csm-cov's relative degradation (constrained / own unconstrained cost) is
  lower by more than 2 SE -- and csm-cov beats csm-prod on the same set.
  A collapse is the torso below 0.15 m for 100 consecutive steps or at the end;
  the environment's `done` also fires on a deliberately low torso, which a
  crouch is, so it is not used.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import cloudpickle
import numpy as np
import jax
import jax.numpy as jnp

import brax.envs as brax_envs
from brax.envs.base import State, Wrapper

import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_rollout, make_sampler, mppi_weights
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.omega import mixture_from_pinv
from csm.screen import COMMANDS, set_command, set_omega

# Executed-action percentiles (p5, p50, p95) per joint of the shipped walking
# student at `uniform`, box_fast and box_turn -- where a bound actually bites.
PCT = np.array([
    [-0.429, -0.051, 0.226], [-0.016, 0.175, 0.485], [-0.546, 0.345, 0.542],
    [-0.257, -0.011, 0.422], [-0.020, 0.212, 0.450], [-0.558, 0.238, 0.474],
    [-0.276, 0.031, 0.190], [-0.026, 0.303, 0.440], [-0.433, 0.336, 0.537],
    [-0.197, -0.009, 0.302], [0.018, 0.274, 0.431], [-0.439, 0.294, 0.562],
], dtype=np.float32)
P_NONE = 0.25       # share of draws with no constraint at all
RESAMPLE = 250      # control steps between draws inside an episode
CALVES = (2, 5, 8, 11)
ROW_SCALES = dict(track_scale=2.479, stability_scale=0.261, gait_scale=1.551)
T0, LEVEL_SCALES, FACTORS = 0.02, (2.625, 1.0), (1.0, 0.5)
COLLAPSE_Z, COLLAPSE_RUN = 0.15, 100


def sample_band(key):
    """One draw from the training family: a one-sided bound on one joint.

    The bound lands between the joint's median and its 5th/95th percentile of
    normal use (with 0.15 of slack past the median), so it always bites and
    never asks for a pose the robot cannot stand in.
    """

    kj, ks, kv, kn = jax.random.split(key, 4)
    pct = jnp.asarray(PCT)
    j = jax.random.randint(kj, (), 0, 12)
    upper = jax.random.bernoulli(ks)
    u = jax.random.uniform(kv)
    p5, p50, p95 = pct[j, 0], pct[j, 1], pct[j, 2]
    v_hi = (p50 - 0.15) + u * (p95 - p50 + 0.15)
    v_lo = p5 + u * (p50 + 0.15 - p5)
    onehot = jnp.arange(12) == j
    lo = jnp.where(onehot & ~upper, v_lo, -1.0)
    hi = jnp.where(onehot & upper, v_hi, 1.0)
    none = jax.random.bernoulli(kn, P_NONE)
    return jnp.where(none, -1.0, lo), jnp.where(none, 1.0, hi)


def test_band(name: str):
    lo, hi = np.full(12, -1.0, np.float32), np.ones(12, np.float32)
    if name == "knee":
        lo[5] = -0.2
    elif name == "front":
        lo[[2, 5]] = -0.2
    elif name == "crouch":
        hi[list(CALVES)] = 0.1
    elif name == "lock":
        lo[5] = hi[5] = 0.24
    elif name != "none":
        raise ValueError(name)
    return jnp.asarray(lo), jnp.asarray(hi)


TESTS = ("none", "knee", "front", "crouch", "lock")
OUT_OF_FAMILY = ("front", "crouch", "lock")


def project_to_band(plan, state):
    """Compose the state's band into a plan: every node, action coordinates."""

    return jnp.clip(plan, state.info["band_lo"], state.info["band_hi"])


class BandEnv(Wrapper):
    """Carry a joint-target band in `info`, redrawn at reset and every RESAMPLE steps.

    `clip_actions` makes the band physical (the executed action is clipped), for
    training a policy that lives under it.  `observe` appends `(lo, hi)` to the
    observation.  `fixed` pins one band instead of drawing from the family.
    Without either flag the wrapper is invisible to the dynamics, which is what
    the collector needs: its rollouts are the rows' own, and only the driver
    reads the band.
    """

    def __init__(self, env, clip_actions: bool = False, observe: bool = False,
                 fixed=None):
        super().__init__(env)
        self._clip = bool(clip_actions)
        self._observe = bool(observe)
        self._fixed = fixed

    @property
    def observation_size(self) -> int:
        return int(self.env.observation_size) + (24 if self._observe else 0)

    def _draw(self, key):
        return self._fixed if self._fixed is not None else sample_band(key)

    def _obs(self, state: State) -> State:
        if not self._observe:
            return state
        return state.replace(obs=jnp.concatenate(
            [state.obs, state.info["band_lo"], state.info["band_hi"]]))

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        state = self.env.reset(rng)
        lo, hi = self._draw(key)
        info = {**state.info, "band_lo": lo, "band_hi": hi, "band_key": key}
        return self._obs(state.replace(info=info))

    def step(self, state: State, action: jax.Array) -> State:
        lo, hi, key = state.info["band_lo"], state.info["band_hi"], state.info["band_key"]
        if self._clip:
            action = jnp.clip(action, lo, hi)
        nxt = self.env.step(state, action)
        step = nxt.info["step"]
        new_lo, new_hi = self._draw(jax.random.fold_in(key, step))
        redraw = (step % RESAMPLE == 0) & (self._fixed is None)
        info = dict(nxt.info)
        info["band_lo"] = jnp.where(redraw, new_lo, lo)
        info["band_hi"] = jnp.where(redraw, new_hi, hi)
        info["band_key"] = key
        return self._obs(nxt.replace(info=info))


# --------------------------------------------------------------------------- #
# controllers, each `run(state, lo, hi) -> (reward, torso_z)` over n_steps
# --------------------------------------------------------------------------- #


def setup():
    dial_config, env_config = _load_config("unitree_go2_trot_csm", None)
    env_config = dataclasses.replace(env_config, **ROW_SCALES)
    dial_config = dataclasses.replace(dial_config, temp_sample=T0)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    return dial_config, env_config, env


def _record(st):
    return st.reward, st.pipeline_state.x.pos[0, 2]


def make_dial(env, dial_config, n_steps, init_passes=5):
    """DIAL with the band inside the planner: samples and plan are clipped."""

    mbdpi = make_controller(dial_config, env)
    sample = make_sampler(mbdpi, dial_config)
    rollout_vmap = jax.vmap(make_rollout(env), in_axes=(None, 0))
    sigma = mbdpi.sigma_control
    factors, scales = jnp.asarray(FACTORS), jnp.asarray(LEVEL_SCALES)

    def update(state, rng, plan, level, lo, hi):
        rng, sr = jax.random.split(rng)
        nodes = jnp.clip(sample(sr, plan, sigma * factors[level]), lo, hi)
        returns = rollout_vmap(state, mbdpi.node2u_vvmap(nodes)).mean(-1)
        w = mppi_weights(returns, scales[level] * T0, False)
        return rng, jnp.clip(jnp.einsum("n,nij->ij", w, nodes), lo, hi)

    def anneal(state, rng, plan, passes, lo, hi):
        def body(c, i):
            r, p = c
            return update(state, r, p, i % 2, lo, hi), None
        (rng, plan), _ = jax.lax.scan(body, (rng, plan), jnp.arange(2 * passes))
        return rng, plan

    @jax.jit
    def run(state, lo, hi, rng):
        rng, plan = anneal(state, rng, jnp.zeros((dial_config.Hnode + 1, mbdpi.nu)),
                           init_passes, lo, hi)

        def body(c, _):
            st, r, pl = c
            r, pl = anneal(st, r, jnp.clip(mbdpi.shift(pl), lo, hi), 1, lo, hi)
            st = env.step(st, pl[0])
            return (st, r, pl), _record(st)

        return jax.lax.scan(body, (state, rng, plan), None, length=n_steps)[1]

    return run


def make_csm(env, policy, n_steps, init_passes=5):
    """The composed student with the band projected after every level."""

    fields = policy.policies
    shift = jnp.asarray(fields[0].shift_matrix)
    factors = jnp.asarray(FACTORS)
    omega = jnp.asarray(build_omegas(3)["uniform"], jnp.float32)
    mixture = mixture_from_pinv(omega, T0, policy.pinv_nu_weights,
                                policy.pinv_mode_weights)

    def refine(plan, obs, passes, lo, hi):
        def level(carry, i):
            t = factor_to_t(factors[i % 2], 0.5, 1.0).reshape(1)
            parts = jnp.stack([f.delta(carry, obs, t) for f in fields])
            m = carry + jnp.einsum("k,kij->ij", mixture, parts)
            return jnp.clip(jnp.clip(m, -1.0, 1.0), lo, hi), None
        return jax.lax.scan(level, plan, jnp.arange(2 * passes))[0]

    @jax.jit
    def run(state, lo, hi, rng):
        plan = refine(jnp.zeros((5, int(env.action_size))), state.obs, init_passes, lo, hi)

        def body(c, _):
            st, pl = c
            st = env.step(st, pl[0])
            pl = refine(jnp.einsum("ij,ja->ia", shift, pl), st.obs, 1, lo, hi)
            return (st, pl), _record(st)

        return jax.lax.scan(body, (state, plan), None, length=n_steps)[1]

    return run


def make_ppo(env, path, n_steps):
    """A PPO policy with its action clipped to the band (a safety filter).

    Rebuilds whatever observation it was trained on: the gait clock if
    `clock_cadence` is recorded, and `(lo, hi)` before it if `band_observed`.
    """

    from csm.rl_baseline import gait_clock, load_policy

    inference, blob = load_policy(path)
    cadence = blob.get("clock_cadence")
    banded = bool(blob.get("band_observed"))

    @jax.jit
    def run(state, lo, hi, rng):
        def body(c, _):
            st, k = c
            k, sub = jax.random.split(k)
            obs = st.obs
            if banded:
                obs = jnp.concatenate([obs, lo, hi])
            if cadence is not None:
                obs = jnp.concatenate([obs, gait_clock(st.info["step"], env.dt, cadence)])
            a, _ = inference(obs, sub)
            st = env.step(st, jnp.clip(a, lo, hi))
            return (st, k), _record(st)

        return jax.lax.scan(body, (state, rng), None, length=n_steps)[1]

    return run


def collapsed(z):
    z = np.asarray(z)
    low = z < COLLAPSE_Z
    run = best = 0
    for v in low:
        run = run + 1 if v else 0
        best = max(best, run)
    return bool(best >= COLLAPSE_RUN or z[-1] < COLLAPSE_Z)


# --------------------------------------------------------------------------- #
# band-conditioned PPO
# --------------------------------------------------------------------------- #


def train_band_ppo(out: Path, steps: int) -> None:
    from csm import rl_baseline as RB

    dial_config, env_config = _load_config("unitree_go2_trot_csm", None)
    omega = build_omegas(3)["uniform"]
    env_config = dataclasses.replace(env_config, **ROW_SCALES,
                                     reward_weights=jnp.asarray(omega, jnp.float32))
    base = brax_envs.get_environment(dial_config.env_name, config=env_config)
    cadence = RB.gait_cadence(base)
    env = RB.GaitClockWrapper(RB.FixedHorizonWrapper(
        BandEnv(base, clip_actions=True, observe=True)), cadence)
    args = RB._parser().parse_args([
        "--output", str(out), "--num-timesteps", str(steps), "--num-envs", "1024",
        "--episode-length", "1000", "--num-evals", "6", "--hidden", "512,256,128",
        "--discounting", "0.97"])
    started = time.time()
    _, params, _, progress = RB.train(args, env)
    RB.save_policy(out / "policy.pkl", algo="ppo", hidden=(512, 256, 128),
                   params=params, obs_size=env.observation_size,
                   act_size=env.action_size, omega=omega, omega_name="uniform",
                   clock_cadence=cadence)
    blob = cloudpickle.load(open(out / "policy.pkl", "rb"))
    blob["band_observed"] = True
    cloudpickle.dump(blob, open(out / "policy.pkl", "wb"))
    (out / "report.json").write_text(json.dumps(
        {"steps": steps, "seconds": time.time() - started, "progress": progress}, indent=1))
    print(f"band PPO trained in {(time.time() - started) / 60:.1f} min -> {out}")


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def evaluate(arms: dict, tests, commands, seeds, steps, out: Path | None):
    dial_config, _, env = setup()
    reset = jax.jit(env.reset)
    omega = build_omegas(3)["uniform"]
    runners = {}
    for name, spec in arms.items():
        if spec == "dial":
            runners[name] = make_dial(env, dial_config, steps)
        elif spec.startswith("csm:"):
            runners[name] = make_csm(env, ComposedDialScorePolicy.load(spec[4:]), steps)
        elif spec.startswith("ppo:"):
            runners[name] = make_ppo(env, spec[4:], steps)
        else:
            raise ValueError(spec)
    res = json.loads(out.read_text()) if out is not None and out.exists() else {}
    for arm, run in runners.items():
        for test in tests:
            key = f"{arm}/{test}"
            if key in res:
                continue
            lo, hi = test_band(test)
            costs, cols, t0 = [], [], time.time()
            for cname in commands:
                for seed in range(seeds):
                    st = set_omega(set_command(env, reset(jax.random.PRNGKey(11 + seed)),
                                               COMMANDS[cname]), omega)
                    r, z = run(st, lo, hi, jax.random.PRNGKey(seed))
                    costs.append(-float(np.asarray(r).mean())); cols.append(collapsed(z))
            res[key] = {"costs": costs, "collapsed": cols}
            print(f"{arm:<9}{test:<8} cost {np.mean(costs):.4f} +- "
                  f"{np.std(costs) / np.sqrt(len(costs)):.4f}  collapsed "
                  f"{sum(cols)}/{len(cols)}  ({time.time() - t0:.0f}s)", flush=True)
            if out is not None:
                out.write_text(json.dumps(res, indent=1))
    return res


def verdict(res: dict) -> str:
    """Apply the pre-registered rule to a finished result file."""

    def col(arm, tests):
        return sum(sum(res[f"{arm}/{t}"]["collapsed"]) for t in tests)

    def degradation(arm, t):
        base = np.mean(res[f"{arm}/none"]["costs"])
        return np.asarray(res[f"{arm}/{t}"]["costs"]) / base

    lines = []
    for arm in ("dial", "csm-prod", "csm-cov", "ppo", "bppo"):
        if f"{arm}/none" not in res:
            continue
        cells = []
        for t in TESTS:
            if f"{arm}/{t}" in res:
                c = res[f"{arm}/{t}"]
                d = degradation(arm, t)
                cells.append(f"{t} {np.mean(c['costs']):.4f} x{d.mean():.2f} "
                             f"c{sum(c['collapsed'])}")
        lines.append(f"{arm:<9}" + " | ".join(cells))
    need = [f"{a}/{t}" for a in ("csm-prod", "csm-cov", "bppo") for t in ("none",) + OUT_OF_FAMILY]
    if not all(k in res for k in need):
        return "\n".join(lines + ["verdict: incomplete"])
    cc, cb, cp = col("csm-cov", OUT_OF_FAMILY), col("bppo", OUT_OF_FAMILY), col("csm-prod", OUT_OF_FAMILY)
    dc = np.concatenate([degradation("csm-cov", t) for t in OUT_OF_FAMILY])
    db = np.concatenate([degradation("bppo", t) for t in OUT_OF_FAMILY])
    dp = np.concatenate([degradation("csm-prod", t) for t in OUT_OF_FAMILY])
    diff = db.mean() - dc.mean()
    se = np.sqrt(db.var() / db.size + dc.var() / dc.size)
    beats_rl = (cb - cc >= 4) or (cc == 0 and cb == 0 and diff > 2 * se)
    beats_prod = cc < cp or (cc == cp and dc.mean() < dp.mean())
    lines.append(f"out of family: collapses csm-cov {cc}, bppo {cb}, csm-prod {cp}; "
                 f"degradation csm-cov x{dc.mean():.2f}, bppo x{db.mean():.2f} "
                 f"(diff {diff:+.2f} +- {se:.2f}), csm-prod x{dp.mean():.2f}")
    lines.append(f"verdict: H {'HOLDS' if beats_rl and beats_prod else 'REJECTED'} "
                 f"(beats band-PPO: {beats_rl}, beats shipped fields: {beats_prod})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-ppo")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=200_000_000)
    e = sub.add_parser("eval")
    e.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "SPEC"),
                   required=True, help="dial | csm:<policy.pkl> | ppo:<policy.pkl>")
    e.add_argument("--tests", nargs="+", default=list(TESTS))
    e.add_argument("--commands", default="box_fast,box_turn")
    e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--steps", type=int, default=1500)
    e.add_argument("--out", type=Path, default=None)
    v = sub.add_parser("verdict")
    v.add_argument("results", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "train-ppo":
        train_band_ppo(a.out, a.steps)
    elif a.cmd == "eval":
        evaluate(dict(a.arm), a.tests, a.commands.split(","), a.seeds, a.steps, a.out)
    else:
        res = {}
        for p in a.results:
            res.update(json.loads(p.read_text()))
        print(verdict(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
