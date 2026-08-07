"""Evaluate a saved CSM policy with a real-time browser viewer."""

from __future__ import annotations

import argparse
import io
import json
import logging
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from brax import envs as brax_envs
from brax import math
from flask import Flask, Response, jsonify
from PIL import Image

import dial_mpc.envs as dial_envs
from csm.energy_policy import CompositionalEnergyPolicy
from csm.gibbs_policy import AnchorGibbsPolicy
from csm.policy import CompositionalPolicy
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


class BrowserViewer:
    """Small MJPEG viewer driven by MJX/Brax pipeline states."""

    def __init__(self, model, host: str, port: int, width: int, height: int):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.camera.distance = 3.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -20.0
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._status = {
            "phase": "initializing",
            "episode": 0,
            "step": 0,
            "forward_speed_mps": 0.0,
            "distance_m": 0.0,
            "resets": 0,
        }
        self._start_server(host, port)

    def _start_server(self, host: str, port: int) -> None:
        app = Flask("csm_live_evaluation")
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.get("/")
        def index():
            return """<!doctype html><meta charset="utf-8">
<title>CSM policy evaluation</title>
<style>body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui}
.wrap{max-width:1000px;margin:auto;padding:18px}img{width:100%;aspect-ratio:16/9;
object-fit:contain;background:#000;border-radius:10px}.status{margin-top:12px;
padding:12px;background:#1f2937;border-radius:8px;white-space:pre-wrap}</style>
<div class="wrap"><h2>CSM Go2 policy — live evaluation</h2>
<img src="/stream.mjpg"><div id="status" class="status">initializing…</div></div>
<script>setInterval(async()=>{const s=await(await fetch('/api/status')).json();
document.getElementById('status').textContent=`phase: ${s.phase} | episode: ${s.episode} | step: ${s.step}\nforward speed: ${s.forward_speed_mps.toFixed(3)} m/s | distance: ${s.distance_m.toFixed(3)} m | resets: ${s.resets}`},250)</script>"""

        @app.get("/api/status")
        def status():
            with self._condition:
                return jsonify(dict(self._status))

        @app.get("/stream.mjpg")
        def stream():
            def frames():
                previous = None
                while True:
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._frame is not None
                            and self._frame is not previous
                        )
                        previous = self._frame
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + previous
                        + b"\r\n"
                    )

            return Response(
                frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        threading.Thread(
            target=lambda: app.run(
                host=host, port=port, threaded=True, use_reloader=False
            ),
            daemon=True,
        ).start()

    def update(self, pipeline_state, **status) -> None:
        self.data.qpos[:] = np.asarray(pipeline_state.qpos)
        self.data.qvel[:] = np.asarray(pipeline_state.qvel)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        output = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(
            output, format="JPEG", quality=85
        )
        with self._condition:
            self._frame = output.getvalue()
            self._status.update(status)
            self._condition.notify_all()

    def set_status(self, **status) -> None:
        with self._condition:
            self._status.update(status)

    def close(self) -> None:
        self.renderer.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--example", default="unitree_go2_trot")
    parser.add_argument(
        "--omega",
        default=None,
        help="tracking,stability,gait weights; defaults to training mode 1",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="number of episodes; 0 repeats until Ctrl-C",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm-start", type=float, default=1.0)
    parser.add_argument("--real-time-factor", type=float, default=1.0)
    parser.add_argument("--web-viewer-host", default="127.0.0.1")
    parser.add_argument("--web-viewer-port", type=int, default=8082)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--viewer-fps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 1 or args.episodes < 0:
        raise ValueError("--steps must be positive and --episodes non-negative")
    if args.real_time_factor <= 0 or args.viewer_fps <= 0:
        raise ValueError("real-time factor and viewer FPS must be positive")

    with open(get_example_path(args.example + ".yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(config["env_name"]),
        config,
        convert_list_to_array=True,
    )
    env = brax_envs.get_environment(config["env_name"], config=env_config)
    try:
        policy = AnchorGibbsPolicy.load(args.policy)
    except TypeError:
        try:
            policy = CompositionalEnergyPolicy.load(args.policy)
        except TypeError:
            policy = CompositionalPolicy.load(args.policy)
    if args.omega is None:
        omega = jnp.asarray(policy.mode_weights[0])
    else:
        omega = jnp.asarray(
            [float(value) for value in args.omega.split(",")], dtype=jnp.float32
        )
    objective_dim = int(policy.mode_weights.shape[1])
    if omega.shape != (objective_dim,):
        raise ValueError(f"--omega must contain {objective_dim} values")
    if int(policy.model.action_size) != env.action_size:
        raise ValueError("policy and environment action sizes do not match")

    viewer = BrowserViewer(
        env.sys.mj_model,
        args.web_viewer_host,
        args.web_viewer_port,
        args.width,
        args.height,
    )
    print(f"policy={args.policy}")
    print(f"omega={omega.tolist()}")
    print(
        f"viewer=http://{args.web_viewer_host}:{args.web_viewer_port} "
        "(use an SSH port forward from your local PC)"
    )

    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    apply_policy = jax.jit(
        lambda previous, observation, key: policy.apply(
            previous,
            observation,
            key,
            warm_start_level=args.warm_start,
            omega=omega,
        )
    )
    torso_idx = int(env._torso_idx) - 1
    render_period = 1.0 / args.viewer_fps
    episode_idx = 0
    reset_count = 0

    # Compile reset, policy inference, and environment stepping before the
    # real-time clock starts.  Otherwise the first episode would try to catch
    # up with time spent compiling and play much faster than real time.
    warmup_rng = jax.random.PRNGKey(args.seed)
    warmup_state = reset_env(warmup_rng)
    warmup_plan = jnp.zeros((int(policy.model.horizon), env.action_size))
    viewer.update(warmup_state.pipeline_state, phase="compiling")
    warmup_rng, warmup_policy_rng = jax.random.split(warmup_rng)
    warmup_plan = apply_policy(
        warmup_plan, warmup_state.obs, warmup_policy_rng
    )
    warmup_state = step_env(warmup_state, warmup_plan[0])
    warmup_state.reward.block_until_ready()
    viewer.set_status(phase="ready")

    try:
        while args.episodes == 0 or episode_idx < args.episodes:
            episode_idx += 1
            rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), episode_idx)
            state = reset_env(rng)
            plan = jnp.zeros((int(policy.model.horizon), env.action_size))
            viewer.update(
                state.pipeline_state,
                phase="running",
                episode=episode_idx,
                step=0,
                forward_speed_mps=0.0,
                distance_m=0.0,
                resets=reset_count,
            )
            initial_x = float(state.pipeline_state.x.pos[torso_idx, 0])
            episode_started = time.perf_counter()
            last_render_time = -np.inf
            speeds = []
            fell = False

            for step_idx in range(args.steps):
                rng, policy_rng = jax.random.split(rng)
                plan = apply_policy(plan, state.obs, policy_rng)
                state = step_env(state, plan[0])
                state.reward.block_until_ready()
                plan = policy.shift(plan)

                pipeline = state.pipeline_state
                speed = float(pipeline.xd.vel[torso_idx, 0])
                distance = float(pipeline.x.pos[torso_idx, 0]) - initial_x
                speeds.append(speed)
                sim_time = (step_idx + 1) * env.dt
                if sim_time - last_render_time >= render_period:
                    viewer.update(
                        pipeline,
                        phase="running",
                        episode=episode_idx,
                        step=step_idx + 1,
                        forward_speed_mps=speed,
                        distance_m=distance,
                        resets=reset_count,
                    )
                    last_render_time = sim_time

                euler = math.quat_to_euler(pipeline.x.rot[torso_idx])
                height = float(pipeline.x.pos[torso_idx, 2])
                tilt = float(jnp.linalg.norm(euler[:2]))
                if height < 0.18 or tilt > np.pi / 2.0:
                    fell = True
                    reset_count += 1
                    viewer.set_status(phase="fallen; resetting", resets=reset_count)
                    break

                target_wall_time = episode_started + sim_time / args.real_time_factor
                remaining = target_wall_time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

            print(
                json.dumps(
                    {
                        "episode": episode_idx,
                        "steps": len(speeds),
                        "mean_forward_speed_mps": float(np.mean(speeds)),
                        "fell": fell,
                    }
                )
            )
            if args.episodes != 0 and episode_idx >= args.episodes:
                viewer.set_status(phase="complete")
                print("evaluation complete; press Ctrl-C to close the viewer")
                while True:
                    time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
