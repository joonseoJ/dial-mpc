"""Render the composed controller at several target weights, one command.

Two figures come out of here.

`panels` puts one weight per panel, each with its own tracking camera, and
writes the gait numbers on the frame -- speed, stride cadence, torso height.
The weight changes the *gait*, and a gait is a rate: at three seconds the four
panels differ by a few centimetres of travel and nothing legible, while the
stride frequencies differ by 14% and that only becomes visible once enough
strides have gone by.

`overlay` draws two weights into one frame, warm and opaque against cool and
translucent, so that the divergence is a single picture rather than a
comparison the reader has to make across panels.  The camera is a free camera
on the midpoint of the two, and it pulls back as they separate -- a camera
locked to either robot would walk the other one out of frame within a few
seconds, and a fixed camera would leave both of them a few pixels tall.

The command matters as much as the weight.  Measured across the training box,
the separation the weights produce is largest at the fast corner and is
*destroyed* by a yaw command: at (1.0, 0.15, 0) the four weights end 3.3 m
apart with stride frequencies spread 13.6%, while at (0.8, 0, 0.3) they end
0.8 m apart with 4% -- a turn makes every weight spend its effort on the same
thing.  Faster still separates more (4.6 m at 1.2 m/s) but that is outside the
box the fields were fitted in, so it is not what these figures show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw

import brax.envs as brax_envs
import dial_mpc.envs as dial_envs  # noqa: F401  (registers the environments)

from csm.basis_screen import _load_config, build_omegas
from csm.compose_walk_eval import make_student
from csm.dial_score import ComposedDialScorePolicy
from csm.screen import set_command, set_omega
from dial_mpc.utils.function_utils import global_to_body_velocity


LABELS = {
    "uniform": "uniform (1,1,1)",
    "boost0": "tracking-heavy (3,1,1)",
    "boost1": "stability-heavy (1,3,1)",
    "boost2": "gait-heavy (1,1,3)",
}

# The `track` camera's offset from the torso, in world axes.  The free camera
# used by the overlay reproduces this direction so the two figures look like
# the same shot.
# Derived from the model's `track` camera: pos="0.846 -1.465 0.916" with
# xyaxes giving a view direction of (-0.470, 0.814, -0.342), so the free
# camera reproduces the same shot.
TRACK_AZIMUTH = 120.0
TRACK_ELEVATION = -20.0
TRACK_DISTANCE = 1.924


class Renderer:
    """MuJoCo renderer that accepts a camera per frame."""

    def __init__(self, sys, width: int, height: int) -> None:
        self.model = sys.mj_model
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.data = mujoco.MjData(self.model)

    def render(self, q, qd, camera) -> np.ndarray:
        self.data.qpos = np.asarray(q)
        self.data.qvel = np.asarray(qd)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()


def free_camera(lookat, distance: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = TRACK_AZIMUTH
    cam.elevation = TRACK_ELEVATION
    return cam


def rollout(env, student, reset, omega, command, temperature, seed, torso):
    def _record(st):
        ps = st.pipeline_state
        vb = global_to_body_velocity(ps.xd.vel[torso], ps.x.rot[torso])
        contact = ps.site_xpos[env._feet_site_id][:, 2] - env._foot_radius < 1e-3
        return ps.q, ps.qd, ps.x.pos[torso], vb, contact

    state = reset(jax.random.PRNGKey(seed))
    state = set_command(env, state, command)
    state = set_omega(state, omega)
    out = student(state, jnp.asarray(omega), temperature)
    return [np.asarray(a) for a in out]


def _window(i: int, steps: int) -> slice:
    return slice(max(0, i - steps + 1), i + 1)


def _cadence(contact: np.ndarray, i: int, steps: int, dt: float) -> float:
    """Touchdowns per foot per second over the trailing window."""

    w = contact[_window(i, steps)]
    if len(w) < 2:
        return 0.0
    return float((w[1:] & ~w[:-1]).sum()) / 4.0 / ((len(w) - 1) * dt)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--example", default="unitree_go2_trot_csm")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["panels", "overlay"], default="panels")
    p.add_argument("--weights", nargs="+",
                   default=["uniform", "boost0", "boost1", "boost2"])
    p.add_argument("--command", type=float, nargs=3, default=[1.0, 0.15, 0.0],
                   help="vx vy vyaw, held for the whole episode")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--every", type=int, default=6)
    p.add_argument("--smooth", type=int, default=25)
    p.add_argument("--cadence-window", type=int, default=100)
    p.add_argument("--width", type=int, default=264)
    p.add_argument("--height", type=int, default=176)
    p.add_argument("--colors", type=int, default=48)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--init-passes", type=int, default=5)
    args = p.parse_args()

    dial_config, env_config = _load_config(args.example, None)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset = jax.jit(env.reset)
    torso = env._torso_idx - 1
    dt = float(env.dt)

    policy = ComposedDialScorePolicy.load(args.policy)
    temperature = float(policy.temperature or dial_config.temp_sample)
    catalogue = build_omegas(int(np.asarray(env_config.reward_weights).shape[0]))
    command = tuple(args.command)

    student = make_student(env, policy, dial_config, args.init_passes,
                           args.steps, record=lambda st: (
                               st.pipeline_state.q, st.pipeline_state.qd,
                               st.pipeline_state.x.pos[torso],
                               global_to_body_velocity(
                                   st.pipeline_state.xd.vel[torso],
                                   st.pipeline_state.x.rot[torso]),
                               st.pipeline_state.site_xpos[
                                   env._feet_site_id][:, 2] - env._foot_radius
                               < 1e-3))

    print(f"policy {args.policy}  command {command}  T={temperature}")
    print(f"{args.steps} steps ({args.steps * dt:.1f} s), frame every "
          f"{args.every} ({1.0 / (dt * args.every):.1f} Hz)")

    runs = {}
    for name in args.weights:
        omega = catalogue[name]
        state = reset(jax.random.PRNGKey(args.seed))
        state = set_command(env, state, command)
        state = set_omega(state, omega)
        runs[name] = [np.asarray(a) for a in
                      student(state, jnp.asarray(omega), temperature)]
        q, qd, pos, vb, contact = runs[name]
        print(f"  {LABELS.get(name, name):<24} travelled "
              f"{np.linalg.norm(pos[-1, :2] - pos[0, :2]):5.2f} m  "
              f"speed {vb[100:, 0].mean():.3f}  "
              f"cadence {_cadence(contact, args.steps - 1, args.steps - 100, dt):.2f} Hz")

    renderer = Renderer(env.sys, args.width, args.height)
    frames = range(0, args.steps, args.every)

    if args.mode == "panels":
        columns = []
        for name in args.weights:
            q, qd, pos, vb, contact = runs[name]
            panel = []
            for i in frames:
                cam = free_camera(
                    [pos[i, 0], pos[i, 1], pos[i, 2]], TRACK_DISTANCE)
                image = Image.fromarray(renderer.render(q[i], qd[i], cam))
                draw = ImageDraw.Draw(image)
                draw.rectangle([0, 0, args.width, 36], fill=(0, 0, 0))
                draw.text((6, 2), f"weight {LABELS.get(name, name)}",
                          fill=(240, 245, 250))
                draw.text((args.width - 62, 2), f"t={(i + 1) * dt:5.2f}s",
                          fill=(160, 172, 186))
                travelled = np.linalg.norm(pos[i, :2] - pos[0, :2])
                draw.text((6, 13),
                          f"dist {travelled:5.2f} m   speed "
                          f"{vb[_window(i, args.smooth), 0].mean():4.2f} m/s",
                          fill=(200, 210, 222))
                draw.text((6, 24),
                          f"cadence {_cadence(contact, i, args.cadence_window, dt):4.2f} Hz"
                          f"   z {pos[_window(i, args.smooth), 2].mean():5.3f} m",
                          fill=(255, 214, 140))
                panel.append(image)
            columns.append(panel)

        gap, n = 2, len(columns)
        total = args.width * n + gap * (n - 1)
        canvas = []
        for k in range(len(columns[0])):
            sheet = Image.new("RGB", (total, args.height), (16, 20, 26))
            for j, col in enumerate(columns):
                sheet.paste(col[k], (j * (args.width + gap), 0))
            canvas.append(sheet)
    else:
        if len(args.weights) != 2:
            raise ValueError("overlay takes exactly two weights")
        a, b = (runs[name] for name in args.weights)
        warm = np.array([1.00, 0.93, 0.82])
        cool = np.array([0.78, 0.88, 1.00])
        canvas = []
        for i in frames:
            mid = (a[2][i] + b[2][i]) / 2.0
            sep = float(np.linalg.norm(a[2][i, :2] - b[2][i, :2]))
            # Pull back as they separate: the pair has to stay in frame, and
            # the zoom itself reads as the divergence it is tracking.
            cam = free_camera([mid[0], mid[1], mid[2]],
                              TRACK_DISTANCE + 0.6 * sep)
            fa = renderer.render(a[0][i], a[1][i], cam).astype(np.float32)
            fb = renderer.render(b[0][i], b[1][i], cam).astype(np.float32)
            blend = 0.66 * (fa * warm) + 0.34 * (fb * cool)
            image = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, args.width, 47], fill=(0, 0, 0))
            draw.text((6, 2), f"{LABELS[args.weights[0]]}  (opaque / warm)",
                      fill=(255, 226, 176))
            draw.text((args.width - 62, 2), f"t={(i + 1) * dt:5.2f}s",
                      fill=(160, 172, 186))
            draw.text((6, 13), f"{LABELS[args.weights[1]]}  (translucent / cool)",
                      fill=(178, 206, 255))
            draw.text((6, 24), "same command, same start: the weight moves the gait",
                      fill=(200, 210, 222))
            draw.text((6, 35), f"gap {sep:4.2f} m", fill=(240, 245, 250))
            canvas.append(image)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    palette = canvas[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.NONE) for f in canvas]
    quantized[0].save(args.out, save_all=True, append_images=quantized[1:],
                      duration=int(dt * args.every * 1000), loop=0, optimize=True)
    print(f"\nwrote {args.out}  {canvas[0].size[0]}x{canvas[0].size[1]}  "
          f"{len(canvas)} frames  {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
