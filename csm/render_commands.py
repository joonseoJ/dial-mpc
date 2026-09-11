"""Render the composed controller under each velocity command, with the
commanded and measured velocity written on the frame.

Two things this does that the first version of the clip did not.

**It runs long enough for the command to show.**  Three seconds of a 0.3 rad/s
turn is 52 degrees, which is not visibly a turn; the strafe moves 4 cm sideways
while walking 0.8 m forward, which reads as walking straight.  The command only
becomes legible once it has been integrated for a while, so the default here is
twelve seconds -- a turn of 206 degrees and 1.8 m of strafe.

**It shows velocity, not position.**  Position tells you where the robot ended
up; whether it *tracked the command* is a statement about velocity, and that is
what the objective's tracking row actually prices.  Both the ramped target and
the measured body-frame velocity are drawn, so the two can be read against each
other frame by frame.

The measured value is a trailing mean over `--smooth` steps.  A trot bounces
the base at 2 Hz and the instantaneous body velocity swings with every
footfall; the raw number is unreadable and says nothing about tracking, which
is a property of the average over a gait cycle.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)
from brax import math

from csm.basis_screen import _load_config, build_omegas
from csm.compose_walk_eval import make_student
from csm.dial_score import ComposedDialScorePolicy
from csm.dial_score_serve import FrameRenderer
from csm.omega import normalize_omega_np
from csm.screen import COMMANDS, set_command, set_omega
from dial_mpc.utils.function_utils import global_to_body_velocity


# The four commands the report's grid uses, with the labels it uses for them.
PANELS = [
    ("box_fast", "forward 1.0 m/s"),
    ("box_slow", "forward 0.6 m/s"),
    ("box_turn", "turn 0.3 rad/s"),
    ("box_strafe", "strafe 0.15 m/s"),
]


class _Frame:
    """The two fields `FrameRenderer` reads off a pipeline state."""

    def __init__(self, q, qd) -> None:
        self.q = q
        self.qd = qd


def _trailing_mean(series: np.ndarray, i: int, window: int) -> np.ndarray:
    return series[max(0, i - window + 1):i + 1].mean(axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--example", default="unitree_go2_trot_csm")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--omega", default="uniform",
                   help="target weight: a catalogue name or 'a,b,c'")
    p.add_argument("--steps", type=int, default=600, help="control steps (50 Hz)")
    p.add_argument("--every", type=int, default=5,
                   help="steps per rendered frame; 5 is 10 Hz, and sampling "
                        "much coarser than that aliases the 2 Hz trot into "
                        "legs that appear to walk backwards")
    p.add_argument("--smooth", type=int, default=25,
                   help="trailing window, in steps, for the measured velocity")
    p.add_argument("--width", type=int, default=300)
    p.add_argument("--height", type=int, default=200)
    p.add_argument("--colors", type=int, default=64,
                   help="GIF palette size.  The scene is a blue-grey room and "
                        "a white robot, so a tight shared palette costs "
                        "nothing visible and a third of the file")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--init-passes", type=int, default=5)
    p.add_argument("--camera", default="track")
    args = p.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset = jax.jit(env.reset)
    torso = env._torso_idx - 1

    policy = ComposedDialScorePolicy.load(args.policy)
    temperature = float(policy.temperature or dial_config.temp_sample)

    n_rows = int(np.asarray(env_config.reward_weights).shape[0])
    catalogue = build_omegas(n_rows)
    omega = (catalogue[args.omega] if args.omega in catalogue else
             normalize_omega_np(np.array([float(v) for v in args.omega.split(",")])))

    def record(st):
        ps = st.pipeline_state
        vb = global_to_body_velocity(ps.xd.vel[torso], ps.x.rot[torso])
        ab = global_to_body_velocity(ps.xd.ang[torso], ps.x.rot[torso])
        return (ps.q, ps.qd, st.info["vel_tar"], st.info["ang_vel_tar"], vb, ab)

    student = make_student(env, policy, dial_config, args.init_passes,
                           args.steps, record=record)

    print(f"policy {args.policy}  omega {np.round(omega, 4).tolist()}  "
          f"T={temperature}")
    print(f"{args.steps} steps ({args.steps * env.dt:.1f} s), frame every "
          f"{args.every} ({1.0 / (env.dt * args.every):.0f} Hz)")

    renderer = FrameRenderer(env.sys, args.width, args.height, args.camera)
    columns = []
    for key, label in PANELS:
        state = reset(jax.random.PRNGKey(args.seed))
        state = set_command(env, state, COMMANDS[key])
        state = set_omega(state, omega)
        q, qd, vel_tar, ang_tar, vb, ab = student(
            state, jnp.asarray(omega), temperature)
        q, qd = np.asarray(q), np.asarray(qd)
        vel_tar, ang_tar = np.asarray(vel_tar), np.asarray(ang_tar)
        vb, ab = np.asarray(vb), np.asarray(ab)

        frames = []
        for i in range(0, args.steps, args.every):
            image = Image.fromarray(renderer.render(_Frame(q[i], qd[i])))
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, args.width, 36], fill=(0, 0, 0))
            t = (i + 1) * env.dt
            draw.text((6, 2), f"command: {label}", fill=(240, 245, 250))
            draw.text((args.width - 62, 2), f"t={t:5.2f}s", fill=(160, 172, 186))
            mv = _trailing_mean(vb, i, args.smooth)
            mw = _trailing_mean(ab, i, args.smooth)
            draw.text((6, 13),
                      f"cmd   vx {vel_tar[i, 0]:5.2f}  vy {vel_tar[i, 1]:5.2f}"
                      f"  wz {ang_tar[i, 2]:5.2f}", fill=(150, 200, 255))
            draw.text((6, 24),
                      f"meas  vx {mv[0]:5.2f}  vy {mv[1]:5.2f}"
                      f"  wz {mw[2]:5.2f}", fill=(255, 214, 140))
            frames.append(image)
        columns.append(frames)
        print(f"  {label:<18} {len(frames)} frames  "
              f"final meas vx {_trailing_mean(vb, args.steps - 1, args.smooth)[0]:.3f}"
              f"  wz {_trailing_mean(ab, args.steps - 1, args.smooth)[2]:.3f}")

    gap = 2
    n = len(columns)
    total = args.width * n + gap * (n - 1)
    canvas = []
    for k in range(len(columns[0])):
        sheet = Image.new("RGB", (total, args.height), (16, 20, 26))
        for j, col in enumerate(columns):
            sheet.paste(col[k], (j * (args.width + gap), 0))
        canvas.append(sheet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One palette for the whole clip: a per-frame palette would make every
    # frame a full keyframe and defeat the inter-frame compression.
    palette = canvas[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.NONE) for f in canvas]
    quantized[0].save(args.out, save_all=True, append_images=quantized[1:],
                      duration=int(env.dt * args.every * 1000), loop=0,
                      optimize=True)
    print(f"\nwrote {args.out}  {total}x{args.height}  {len(canvas)} frames  "
          f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
