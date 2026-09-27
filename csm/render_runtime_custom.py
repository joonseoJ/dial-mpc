"""Side-by-side clips: the PPO policy alone against the same policy with the chunk planner.

Both panels start from the same reset and command and run the same runtime
objective (`csm.policy_completed_mppi`); the left panel can only clip its
action to a band, the right one plans for the objective.  A difference on
screen is the planner.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse

import numpy as np
import jax
from PIL import Image, ImageDraw

from csm.basis_screen import build_omegas
from csm.dial_score_serve import FrameRenderer
from csm.policy_completed_mppi import make_controller, make_env
from csm.screen import COMMANDS, set_command, set_omega


class _Pose:
    def __init__(self, q, qd):
        self.q, self.qd = q, qd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--policy", default="csm_runs/rl-sweep-clock/uniform/policy.pkl")
    ap.add_argument("--command", default="box_fast")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--labels", nargs=2, default=["PPO alone", "PPO + chunk planner"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--plan-timestep", type=float, default=0.01)
    ap.add_argument("--replan-every", type=int, default=2)
    ap.add_argument("--residual-sigma", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    world, planner = make_env(0.01), make_env(a.plan_timestep)
    st = set_omega(set_command(world, world.reset(jax.random.PRNGKey(11)),
                               COMMANDS[a.command]), build_omegas(3)["uniform"])
    rec = lambda s: (s.pipeline_state.q, s.pipeline_state.qd)
    trajs = []
    for plan in (False, True):
        run = make_controller(world, planner, a.policy, a.objective, a.steps, plan=plan,
                              record=rec, horizon=a.horizon,
                              replan_every=a.replan_every,
                              residual_sigma=a.residual_sigma)
        q, qd = run(st, jax.random.PRNGKey(0))
        trajs.append((np.asarray(q), np.asarray(qd)))
    renderer = FrameRenderer(world.sys, a.width, a.height, "track")
    frames = []
    for t in range(0, a.steps, a.every):
        sheet = Image.new("RGB", (2 * a.width, a.height + 24), (16, 20, 26))
        for i, (q, qd) in enumerate(trajs):
            img = Image.fromarray(renderer.render(_Pose(q[t], qd[t])))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 7 * len(a.labels[i]) + 10, 18], fill=(0, 0, 0))
            d.text((6, 4), a.labels[i], fill=(240, 245, 250))
            sheet.paste(img, (i * a.width, 24))
        d = ImageDraw.Draw(sheet)
        d.text((8, 6), f"{a.title or a.objective}   t = {t * world.dt:4.1f} s",
               fill=(220, 225, 230))
        frames.append(sheet)
    # One palette for the whole clip.  Quantising each frame on its own and
    # letting the GIF writer diff them dropped the robot's body from later
    # frames (only the feet survived); a shared palette built from frames
    # across the clip keeps every frame self-consistent.
    probe = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for k, f in enumerate(frames[:: max(len(frames) // 4, 1)][:4]):
        probe.paste(f, (0, k * frames[0].height))
    palette = probe.quantize(colors=96)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 * world.dt * a.every), loop=0, disposal=1)
    print(f"wrote {a.out} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
