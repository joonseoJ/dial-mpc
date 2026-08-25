"""Live MJPEG viewer for a trained single-weight DIAL score policy.

The policy runs its denoising control loop in a background thread and every
rendered frame is pushed to an MJPEG stream, so a browser on the other end of an
SSH tunnel watches the robot in real time instead of downloading a finished
rollout.  The endpoint layout matches the viewer ``dial-mpc-sim`` already
exposes (``/`` for the page, ``/stream.mjpg`` for the stream), so the same
``ssh -L`` habit works for both.

``--with-dial`` runs real DIAL-MPC alongside the network from the same initial
state and composites both views into one frame.  Note that DIAL needs ~100 ms
per control step against the network's ~6 ms, so the pair advances at DIAL's
pace; the stats line reports the realtime factor for each.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw
from brax import envs as brax_envs
from flask import Flask, Response, jsonify

import dial_mpc.envs as dial_envs
from csm.dial_score import (
    DialScorePolicy,
    DialScoreTeacher,
    build_shift_matrix,
    dial_factors,
)
from csm.dial_score_eval import _load_config, _resolve_sources
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path

CONTROL_HZ = 50.0


class FrameRenderer:
    """Persistent MuJoCo renderer.

    ``env.render`` builds a fresh ``mujoco.Renderer`` on every call, which costs
    ~130 ms per frame — far too slow to stream.  Holding the renderer and the
    ``MjData`` open drops that to ~14 ms.
    """

    def __init__(self, sys, width: int, height: int, camera: str | None) -> None:
        self.model = sys.mj_model
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.data = mujoco.MjData(self.model)
        self.camera = camera if camera is not None else -1

    def render(self, pipeline_state) -> np.ndarray:
        self.data.qpos = np.asarray(pipeline_state.q)
        self.data.qvel = np.asarray(pipeline_state.qd)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()


def _label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 7 * len(text) + 10, 18], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(240, 245, 250))
    return image


class LiveRollout:
    """Runs the control loop and publishes the newest JPEG frame."""

    def __init__(self, args) -> None:
        self.args = args
        policy_path, report = _resolve_sources(args)
        trained = report.get("arguments", {})
        config = _load_config(args, report)
        self.policy_path = policy_path

        dial_config = _dataclass_from(DialConfig, config)
        overrides = {"time_correlated": False}
        temp = args.temp_sample
        if temp is None and trained.get("temp_sample") is not None:
            temp = float(trained["temp_sample"])
        if temp is not None:
            overrides["temp_sample"] = float(temp)
        samples = args.samples
        if samples is None and trained.get("samples") is not None:
            samples = int(trained["samples"])
        if samples is not None:
            overrides["Nsample"] = int(samples)
        self.dial_config = dataclasses.replace(dial_config, **overrides)

        env_config = _dataclass_from(
            dial_envs.get_config(self.dial_config.env_name),
            config,
            convert_list_to_array=True,
        )
        env_config = _randomize(env_config, args.start_noise, args.randomize_command)
        self.env = brax_envs.get_environment(
            self.dial_config.env_name, config=env_config
        )
        self.horizon = int(self.dial_config.Hnode) + 1
        self.policy = DialScorePolicy.load(policy_path)
        self.init_passes = args.init_passes or int(trained.get("init_passes", 5))

        omega = jnp.asarray(
            [float(v) for v in str(args.omega or trained.get("omega", "1,1,1")).split(",")],
            dtype=jnp.float32,
        )
        self.planner = MBDPI(self.dial_config, self.env)
        self.teacher = DialScoreTeacher(self.planner, omega, repeats=1)
        self.factors = dial_factors(self.dial_config.traj_diffuse_factor, 2)
        self.shift_matrix = build_shift_matrix(self.planner)

        self._reset_env = jax.jit(self.env.reset)
        self._step_env = jax.jit(self.env.step)
        self._apply_first = jax.jit(
            lambda plan, obs: self.policy.apply(plan, obs, self.init_passes)
        )
        self._apply_step = jax.jit(lambda plan, obs: self.policy.apply(plan, obs, 1))
        self._shift = jax.jit(self.policy.shift)
        if args.with_dial:
            sigma_control = jnp.asarray(
                self.planner.sigma_control, dtype=jnp.float32
            )

            @jax.jit
            def anneal(state, plan, rng, schedule):
                def body(carry, factor):
                    plan, rng = carry
                    rng, call = jax.random.split(rng)
                    _, plan, _ = self.planner.reverse_once(
                        state, call, plan, sigma_control * factor
                    )
                    return (plan, rng), None

                (plan, rng), _ = jax.lax.scan(body, (plan, rng), schedule)
                return plan, rng

            self._anneal = anneal

        # The renderer is built inside the rollout thread, not here: an EGL
        # context belongs to the thread that created it, and binding it from
        # another one fails with EGL_BAD_ACCESS.
        self.camera = args.camera or "track"
        self.render_every = max(1, int(round(CONTROL_HZ / max(args.fps, 1e-6))))

        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._reset_requested = False
        self.stats: dict[str, object] = {
            "policy": str(policy_path),
            "status": "starting",
            "step": 0,
            "sim_time": 0.0,
            "reward": 0.0,
            "mean_reward": 0.0,
            "falls": 0,
            "episode_step": 0,
            "control_hz": 0.0,
            "realtime": 0.0,
            "seed": int(args.seed),
            "with_dial": bool(args.with_dial),
            "dial_reward": 0.0,
            "dial_mean_reward": 0.0,
            "dial_falls": 0,
            "start_noise": float(args.start_noise),
            "randomize_command": bool(args.randomize_command),
        }

    # ---------------------------------------------------------------- frames
    def publish(self, images: list[tuple[str, np.ndarray]]) -> None:
        panels = [
            _label(Image.fromarray(np.asarray(pixels)), name)
            for name, pixels in images
        ]
        if len(panels) == 1:
            canvas = panels[0]
        else:
            width = sum(p.width for p in panels) + 2 * (len(panels) - 1)
            canvas = Image.new("RGB", (width, panels[0].height), (20, 23, 29))
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, 0))
                x += panel.width + 2
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=self.args.quality)
        with self._condition:
            self._frame = buffer.getvalue()
            self._condition.notify_all()

    def frames(self):
        previous = None
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame is not None and self._frame is not previous
                )
                previous = self._frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + previous + b"\r\n"

    def request_reset(self) -> None:
        with self._condition:
            self._reset_requested = True

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        args = self.args
        seed = int(args.seed)
        rng = jax.random.PRNGKey(seed + 500_000)
        state = self.teacher.with_reward_weights(
            self._reset_env(jax.random.PRNGKey(seed))
        )
        dial_state = state if args.with_dial else None
        plan = jnp.zeros((self.horizon, int(self.env.action_size)), dtype=jnp.float32)
        dial_plan = plan
        first_schedule = jnp.tile(self.factors, self.init_passes)

        renderer = FrameRenderer(
            self.env.sys, args.width, args.height, self.camera
        )
        self.stats["status"] = "compiling"
        rewards: list[float] = []
        dial_rewards: list[float] = []
        falls = 0
        dial_falls = 0
        episode_step = 0
        step_index = 0
        period = 1.0 / CONTROL_HZ
        next_deadline = time.perf_counter()
        rate_window = time.perf_counter()
        rate_steps = 0

        while True:
            with self._condition:
                do_reset = self._reset_requested
                self._reset_requested = False
            if do_reset:
                seed += 1
                state = self.teacher.with_reward_weights(
                    self._reset_env(jax.random.PRNGKey(seed))
                )
                dial_state = state if args.with_dial else None
                plan = jnp.zeros_like(plan)
                dial_plan = plan
                episode_step = 0
                self.stats["seed"] = seed

            apply_fn = self._apply_first if episode_step == 0 else self._apply_step
            plan = apply_fn(plan, state.obs)
            plan.block_until_ready()
            state = self._step_env(state, plan[0])
            reward = float(state.reward)
            rewards.append(reward)

            if args.with_dial:
                schedule = first_schedule if episode_step == 0 else self.factors
                dial_plan, rng = self._anneal(dial_state, dial_plan, rng, schedule)
                dial_plan.block_until_ready()
                dial_state = self._step_env(dial_state, dial_plan[0])
                dial_rewards.append(float(dial_state.reward))

            if step_index % self.render_every == 0:
                panels = [("score MLP", renderer.render(state.pipeline_state))]
                if args.with_dial:
                    panels.append(
                        ("DIAL-MPC", renderer.render(dial_state.pipeline_state))
                    )
                self.publish(panels)

            step_index += 1
            episode_step += 1
            rate_steps += 1
            now = time.perf_counter()
            if now - rate_window >= 1.0:
                hz = rate_steps / (now - rate_window)
                self.stats["control_hz"] = round(hz, 1)
                self.stats["realtime"] = round(hz / CONTROL_HZ, 2)
                rate_window, rate_steps = now, 0

            self.stats.update(
                status="running",
                step=step_index,
                sim_time=round(step_index / CONTROL_HZ, 2),
                reward=round(reward, 4),
                mean_reward=round(float(np.mean(rewards[-500:])), 4),
                falls=falls,
                episode_step=episode_step,
            )
            if args.with_dial:
                self.stats.update(
                    dial_reward=round(dial_rewards[-1], 4),
                    dial_mean_reward=round(float(np.mean(dial_rewards[-500:])), 4),
                    dial_falls=dial_falls,
                )

            fell = float(state.done) > 0.5
            dial_fell = args.with_dial and float(dial_state.done) > 0.5
            if fell:
                falls += 1
            if dial_fell:
                dial_falls += 1
            # Keep the two panels on a shared clock: restart both whenever
            # either one goes down, so the views never drift apart in time.
            if fell or dial_fell:
                seed += 1
                state = self.teacher.with_reward_weights(
                    self._reset_env(jax.random.PRNGKey(seed))
                )
                dial_state = state if args.with_dial else None
                plan = jnp.zeros_like(plan)
                dial_plan = plan
                episode_step = 0
                self.stats["seed"] = seed
            else:
                plan = self._shift(plan)
                if args.with_dial:
                    dial_plan = jnp.einsum(
                        "ij,ja->ia", self.shift_matrix, dial_plan
                    )

            if args.realtime and not args.with_dial:
                next_deadline += period
                sleep = next_deadline - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_deadline = time.perf_counter()


def _dataclass_from(kind, config, **kwargs):
    from dial_mpc.utils.io_utils import load_dataclass_from_dict

    return load_dataclass_from_dict(kind, config, **kwargs)


def _randomize(env_config, start_noise: float, randomize_command: bool):
    updates: dict[str, object] = {}
    if start_noise > 0.0:
        scale = float(start_noise)
        updates.update(
            randomize_start_state=True,
            start_height_noise=0.02 * scale,
            start_rpy_noise=0.10 * scale,
            start_joint_position_noise=0.10 * scale,
            start_body_linear_velocity_noise=0.20 * scale,
            start_body_angular_velocity_noise=0.20 * scale,
            start_joint_velocity_noise=0.50 * scale,
        )
    if randomize_command:
        updates["randomize_tasks"] = True
    if not updates:
        return env_config
    missing = [key for key in updates if not hasattr(env_config, key)]
    if missing:
        raise ValueError(f"environment has no randomization fields {missing}")
    return dataclasses.replace(env_config, **updates)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>DIAL score policy — live</title>
<style>
 body{margin:0;background:#14171d;color:#eceff4;
      font:14px ui-monospace,SFMono-Regular,Menlo,monospace}
 header{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;
        padding:12px 18px;border-bottom:1px solid #2a303a}
 h1{font-size:15px;margin:0;letter-spacing:.08em;text-transform:uppercase;color:#98a3b2}
 .path{color:#6d7887;font-size:12px;word-break:break-all}
 main{padding:18px;display:flex;flex-direction:column;gap:14px;align-items:flex-start}
 img{background:#000;max-width:100%;border:1px solid #2a303a}
 table{border-collapse:collapse}
 td{padding:4px 16px 4px 0}
 td:first-child{color:#6d7887;text-transform:uppercase;font-size:11px;letter-spacing:.1em}
 .v{font-variant-numeric:tabular-nums}
 button{font:inherit;background:#1b1f27;color:#eceff4;border:1px solid #3a424e;
        padding:7px 14px;border-radius:3px;cursor:pointer}
 button:hover{border-color:#6d7887}
</style></head><body>
<header><h1>DIAL score policy — live</h1><span class="path" id="policy"></span></header>
<main>
  <img id="view" src="/stream.mjpg" alt="live rollout">
  <div><button onclick="fetch('/reset',{method:'POST'})">새 초기상태로 리셋</button></div>
  <table><tbody id="stats"></tbody></table>
</main>
<script>
const ROWS=[["status","상태"],["sim_time","시뮬 시간 (s)"],["step","스텝"],
 ["mean_reward","평균 보상"],["dial_mean_reward","DIAL 평균 보상"],
 ["falls","낙상"],["dial_falls","DIAL 낙상"],
 ["control_hz","제어 Hz"],["realtime","실시간 배속"],["seed","시드"]];
async function tick(){
  const s=await (await fetch('/stats')).json();
  document.getElementById('policy').textContent=s.policy;
  document.getElementById('stats').innerHTML=ROWS
    .filter(([k])=>s.with_dial||!k.startsWith('dial_'))
    .map(([k,l])=>`<tr><td>${l}</td><td class="v">${s[k]}</td></tr>`).join('');
}
tick();setInterval(tick,500);
</script></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="training run directory")
    target.add_argument("--policy", type=Path, help="path to a policy.pkl")
    parser.add_argument("--example", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; keep the default and use ssh -L")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--start-noise", type=float, default=1.0,
                        help="randomized initial states, as in verification")
    parser.add_argument("--randomize-command", action="store_true")
    parser.add_argument("--init-passes", type=int, default=None)
    parser.add_argument("--omega", default=None)
    parser.add_argument("--temp-sample", type=float, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--with-dial", action="store_true",
                        help="run real DIAL-MPC beside the network")
    parser.add_argument("--fast", dest="realtime", action="store_false",
                        help="run as fast as possible instead of real time")
    parser.set_defaults(realtime=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    rollout = LiveRollout(args)

    app = Flask("dial_score_live_viewer")

    @app.get("/")
    def index():
        return PAGE

    @app.get("/stream.mjpg")
    def stream():
        return Response(
            rollout.frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stats")
    def stats():
        return jsonify(rollout.stats)

    @app.post("/reset")
    def reset():
        rollout.request_reset()
        return jsonify(ok=True)

    threading.Thread(target=rollout.run, daemon=True).start()
    print(f"policy={rollout.policy_path}")
    print(f"serving http://{args.host}:{args.port}  (stream at /stream.mjpg)")
    print("first frame appears after JAX finishes compiling")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
