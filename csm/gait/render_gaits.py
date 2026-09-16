"""Render gait clips: the pure rows, a weight sweep, or CSM against RL side by side.

A gait is a phase relation between four feet, and a table of correlations is a
poor way to see one.  These clips put the robot next to a live contact diagram
-- four rows, one per foot, scrolling right to left -- so the pairing that the
numbers summarise is visible directly: diagonal blocks aligned for trot,
lateral for pace, front-then-hind for bound, a four-beat ripple for walk.

Three modes:

  `pure`   one panel per gait, the four rows of the basis
  `sweep`  one pair of gaits at several mixing ratios, to see the interpolation
  `versus` the same weight driven by the composed fields and by the RL blend,
           which is where the two differ most

Both controllers are stepped through the same environment with the same reset,
so a difference on screen is a difference in the controller.
"""
from __future__ import annotations

# Before anything pulls in mujoco: the renderer needs a headless GL backend,
# and `mjr_makeContext` fails outright if the platform library is chosen after
# the module is loaded.  The live viewers do the same.
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse, dataclasses
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw
import brax.envs as brax_envs
from brax import math

import dial_mpc.envs  # noqa: F401
from csm.basis_screen import _load_config
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.dial_score_serve import FrameRenderer
from csm.omega import mixture_from_pinv
from csm.rl_baseline import load_policy
from csm.screen import set_command
from dial_mpc.envs.unitree_go2_gait import GAIT_NAMES

GAITS = list(GAIT_NAMES)
FEET = ["FL", "FR", "RL", "RR"]


class _Frame:
    """The two fields `FrameRenderer` reads off a pipeline state."""
    def __init__(self, q, qd):
        self.q, self.qd = q, qd


def csm_runner(env, dial_config, policy, step_passes, init_passes, steps):
    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    lo, hi = float(jnp.min(factors)), float(jnp.max(factors))
    shift = jnp.asarray(fields[0].shift_matrix)
    pinv_nu = getattr(policy, "pinv_nu_weights", None)
    horizon = int(dial_config.Hnode) + 1

    def refine(plan, obs, mixture, passes):
        def level(carry, factor):
            t = factor_to_t(factor, lo, hi).reshape(1)
            parts = jnp.stack([f.delta(carry, obs, t) for f in fields])
            return jnp.clip(carry + jnp.einsum("k,kij->ij", mixture, parts), -1.0, 1.0), None
        return jax.lax.scan(level, plan, jnp.tile(factors, passes))[0]

    @jax.jit
    def run(state, omega, temperature):
        mixture = mixture_from_pinv(omega, temperature, pinv_nu, policy.pinv_mode_weights)
        plan = refine(jnp.zeros((horizon, int(env.action_size))), state.obs,
                      mixture, init_passes)

        def body(carry, _):
            st, pl = carry
            pl = refine(pl, st.obs, mixture, step_passes)
            st = env.step(st, pl[0])
            return (st, jnp.einsum("ij,ja->ia", shift, pl)), record(env, st)

        return jax.lax.scan(body, (state, plan), None, length=steps)[1]

    return run


def rl_runner(env, infers, steps, conditioned=False):
    @jax.jit
    def run(state, omega, temperature):
        c = omega / jnp.maximum(omega.sum(), 1e-8)

        def body(st, _):
            if conditioned:
                action = infers[0](jnp.concatenate([st.obs, omega]), jax.random.PRNGKey(0))[0]
            else:
                acts = jnp.stack([f(st.obs, jax.random.PRNGKey(0))[0] for f in infers])
                action = jnp.einsum("k,ka->a", c, acts)
            st = env.step(st, action)
            return st, record(env, st)

        return jax.lax.scan(body, state, None, length=steps)[1]

    return run


def record(env, st):
    ps = st.pipeline_state
    ti = env._torso_idx - 1
    zf = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius
    vb = math.rotate(ps.xd.vel[ti], math.quat_inv(ps.x.rot[ti]))
    ab = math.rotate(ps.xd.ang[ti], math.quat_inv(ps.x.rot[ti]))
    return ps.q, ps.qd, (zf < 0.02).astype(jnp.float32), vb[:2], ab[2]


def contact_strip(contact, i, width, height, window=90):
    """A scrolling four-row contact diagram ending at step `i`."""
    img = Image.new("RGB", (width, height), (14, 17, 22))
    d = ImageDraw.Draw(img)
    lo = max(0, i - window)
    seg = contact[lo:i + 1]
    if len(seg) < 2:
        return img
    rh = height / 4.0
    cw = width / float(window)
    for r in range(4):
        d.rectangle([0, r * rh + 1, width, (r + 1) * rh - 2], fill=(26, 32, 42))
        for t in range(len(seg)):
            if seg[t, r] > 0.5:
                x = (window - len(seg) + t) * cw
                d.rectangle([x, r * rh + 2, x + max(cw, 1.2), (r + 1) * rh - 3],
                            fill=(91, 157, 217))
        d.text((3, r * rh + rh * 0.25 - 4), FEET[r], fill=(141, 153, 169))
    return img


def label_pairs(contact, i, window=90):
    lo = max(0, i - window)
    w = contact[lo:i + 1]
    def cor(a, b):
        if len(w) < 30 or w[:, a].std() < 1e-6 or w[:, b].std() < 1e-6:
            return float("nan")
        return float(np.corrcoef(w[:, a], w[:, b])[0, 1])
    return (np.nanmean([cor(0, 3), cor(1, 2)]), np.nanmean([cor(0, 2), cor(1, 3)]),
            np.nanmean([cor(0, 1), cor(2, 3)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("pure", "sweep", "versus"), default="pure")
    p.add_argument("--csm-policy", type=Path, required=True)
    p.add_argument("--rl-specialists", type=Path, nargs=4, default=None)
    p.add_argument("--pair", nargs=2, default=["trot", "pace"])
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    p.add_argument("--command", type=float, nargs=3, default=[0.8, 0.0, 0.0])
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--width", type=int, default=300)
    p.add_argument("--height", type=int, default=230)
    p.add_argument("--strip", type=int, default=54)
    p.add_argument("--camera", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--step-passes", type=int, default=6)
    p.add_argument("--init-passes", type=int, default=5)
    p.add_argument("--colors", type=int, default=96)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    dial_config, env_config = _load_config("unitree_go2_gait", None)
    dial_config = dataclasses.replace(dial_config, temp_sample=args.temperature)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset = jax.jit(env.reset)
    policy = ComposedDialScorePolicy.load(args.csm_policy)
    temp = jnp.asarray(args.temperature, jnp.float32)

    csm = csm_runner(env, dial_config, policy, args.step_passes, args.init_passes, args.steps)
    rl = None
    if args.rl_specialists:
        rl = rl_runner(env, [load_policy(q)[0] for q in args.rl_specialists], args.steps)

    # (panel label, runner, omega)
    panels = []
    if args.mode == "pure":
        for gi, g in enumerate(GAITS):
            w = np.zeros(4, np.float32); w[gi] = 1.0
            panels.append((g, csm, w))
    elif args.mode == "sweep":
        i, j = (GAITS.index(x) for x in args.pair)
        for a in args.ratios:
            w = np.zeros(4, np.float32); w[i] = a; w[j] = 1 - a
            w /= np.linalg.norm(w)
            panels.append((f"{a:.2f}:{1-a:.2f}", csm, w))
    else:
        i, j = (GAITS.index(x) for x in args.pair)
        w = np.zeros(4, np.float32); w[i] = w[j] = 1.0; w /= np.linalg.norm(w)
        panels.append(("CSM  (score mix)", csm, w))
        if rl is None:
            raise SystemExit("--versus needs --rl-specialists")
        panels.append(("RL  (action mix)", rl, w))

    renderer = FrameRenderer(env.sys, args.width, args.height, args.camera)
    ph = args.height + args.strip
    columns = []
    for label, run, omega in panels:
        st = set_command(env, reset(jax.random.PRNGKey(args.seed)), tuple(args.command))
        st = st.replace(info={**st.info, "reward_weights": jnp.asarray(omega, jnp.float32)})
        q, qd, contact, vb, wz = [np.asarray(v) for v in
                                  run(st, jnp.asarray(omega, jnp.float32), temp)]
        frames = []
        for i in range(0, args.steps, args.every):
            panel = Image.new("RGB", (args.width, ph), (16, 20, 26))
            panel.paste(Image.fromarray(renderer.render(_Frame(q[i], qd[i]))), (0, 0))
            panel.paste(contact_strip(contact, i, args.width, args.strip), (0, args.height))
            d = ImageDraw.Draw(panel)
            d.rectangle([0, 0, args.width, 24], fill=(0, 0, 0))
            d.text((5, 2), label, fill=(240, 245, 250))
            d.text((args.width - 52, 2), f"t={(i + 1) * env.dt:4.1f}s", fill=(150, 162, 178))
            dg, lt, fh = label_pairs(contact, i)
            d.text((5, 13), f"diag {dg:+.2f}  lat {lt:+.2f}  f/h {fh:+.2f}",
                   fill=(255, 214, 140))
            v = vb[max(0, i - 25):i + 1].mean(axis=0)
            d.text((args.width - 118, 13),
                   f"vx {v[0]:4.2f}  wz {wz[max(0,i-25):i+1].mean():+4.2f}",
                   fill=(150, 200, 255))
            frames.append(panel)
        columns.append(frames)
        dg, lt, fh = label_pairs(contact, args.steps - 1)
        print(f"  {label:<18} diag {dg:+.2f} lat {lt:+.2f} f/h {fh:+.2f}  "
              f"vx {vb[60:, 0].mean():.2f}", flush=True)

    gap, n = 3, len(columns)
    total = args.width * n + gap * (n - 1)
    sheets = []
    for k in range(len(columns[0])):
        sheet = Image.new("RGB", (total, ph), (16, 20, 26))
        for j, col in enumerate(columns):
            sheet.paste(col[k], (j * (args.width + gap), 0))
        sheets.append(sheet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    palette = sheets[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    qs = [f.quantize(palette=palette, dither=Image.NONE) for f in sheets]
    qs[0].save(args.out, save_all=True, append_images=qs[1:],
               duration=int(env.dt * args.every * 1000), loop=0, optimize=True)
    print(f"wrote {args.out}  {total}x{ph}  {len(sheets)} frames  "
          f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
