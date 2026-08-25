"""Residual-MPPI: online policy customization with a learned prior.

Uses a pre-trained RL prior policy (SAC actor) as the base distribution
and optimizes an add-on velocity tracking cost via MPPI.  The augmented
per-step reward is:

    r_aug = r_addon(x, u) + omega_prime * log pi(u | x)

where r_addon is the velocity tracking reward and omega_prime balances
prior adherence vs add-on task performance.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
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
from brax.envs.base import State, Wrapper
from flask import Flask, Response, jsonify, request
from PIL import Image

import dial_mpc.envs as dial_envs
from csm.rl_prior import RLPriorPolicy, SafetyCritic
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import make_controller
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


class ResidualMPPIEnv(Wrapper):
    """Wraps a Go2 env to produce the Residual-MPPI augmented reward."""

    def __init__(self, env, prior, omega_prime=0.1, addon_weights=None,
                 critic=None):
        super().__init__(env)
        self._prior = prior
        self._critic = critic
        self._omega_prime = omega_prime
        self._addon_weights = (
            addon_weights if addon_weights is not None
            else jnp.array([1.0, 0.0, 0.0])
        )

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["reward_weights"] = self._addon_weights
        state.info["omega_prime"] = jnp.array(self._omega_prime)
        return state

    def step(self, state, action):
        if self._critic is not None:
            safety = self._critic.q_value(state.obs, action)
        else:
            safety = self._prior.log_prob(state.obs, action)
        next_state = self.env.step(state, action)
        omega = state.info["omega_prime"]
        augmented_reward = next_state.reward + omega * safety
        return next_state.replace(reward=augmented_reward)


def initialize_from_prior(env, prior, state, Hsample, u2node_vmap):
    def scan_fn(carry, _):
        s = carry
        action = prior.mode(s.obs)
        s = env.step(s, action)
        return s, action

    _, us = jax.lax.scan(scan_fn, state, None, length=Hsample + 1)
    Y0 = u2node_vmap(us)
    return jnp.clip(Y0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Thread-safe control channel between web UI and main loop
# ---------------------------------------------------------------------------

class SharedControl:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset_requested = False
        self._vel_cmd = None  # (vx, vy, vyaw) or None
        self._weights = None  # (omega_prime, tracking, stability, gait) or None

    def request_reset(self):
        with self._lock:
            self._reset_requested = True

    def consume_reset(self):
        with self._lock:
            r = self._reset_requested
            self._reset_requested = False
            return r

    def set_vel_cmd(self, vx, vy, vyaw):
        with self._lock:
            self._vel_cmd = (float(vx), float(vy), float(vyaw))

    def clear_vel_cmd(self):
        with self._lock:
            self._vel_cmd = None

    def get_vel_cmd(self):
        with self._lock:
            return self._vel_cmd

    def set_weights(self, omega_prime, tracking, stability, gait):
        with self._lock:
            self._weights = (
                float(omega_prime), float(tracking),
                float(stability), float(gait),
            )

    def get_weights(self):
        with self._lock:
            return self._weights


# ---------------------------------------------------------------------------
# MJPEG web viewer with GUI controls
# ---------------------------------------------------------------------------

_INDEX_HTML = """\
<!doctype html><meta charset="utf-8">
<title>Residual-MPPI</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#111827;color:#e5e7eb;font:15px/1.5 system-ui}
.wrap{max-width:1060px;margin:auto;padding:16px}
h2{margin:0 0 12px;font-size:22px}
img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;
    border-radius:10px}
.panel{display:flex;gap:16px;margin-top:14px;flex-wrap:wrap}
.card{flex:1;min-width:240px;background:#1f2937;border-radius:10px;
      padding:16px}
.card h3{margin:0 0 10px;font-size:15px;color:#9ca3af}
.status{white-space:pre-wrap;font:13px/1.6 monospace}
.slider-row{display:flex;align-items:center;gap:8px;margin:6px 0}
.slider-row label{width:52px;text-align:right;font-size:13px;
                   color:#9ca3af;flex-shrink:0}
.slider-row input[type=range]{flex:1;accent-color:#3b82f6}
.slider-row .val{width:68px;font:13px monospace;color:#60a5fa;
  background:#111827;border:1px solid #374151;border-radius:4px;
  padding:2px 4px;text-align:right;-moz-appearance:textfield}
.slider-row .val::-webkit-inner-spin-button,
.slider-row .val::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
.slider-row .val:focus{border-color:#3b82f6;outline:none}
button{padding:8px 20px;border:none;border-radius:8px;font:14px system-ui;
       cursor:pointer;transition:background .15s}
.btn-reset{background:#ef4444;color:#fff}
.btn-reset:hover{background:#dc2626}
.btn-random{background:#3b82f6;color:#fff;margin-left:8px}
.btn-random:hover{background:#2563eb}
.actions{margin-top:14px;display:flex;align-items:center}
</style>
<div class="wrap">
<h2>Residual-MPPI &mdash; live</h2>
<img src="/stream.mjpg">
<div class="panel">

<div class="card">
<h3>Velocity Command</h3>
<div class="slider-row">
  <label>vx</label>
  <input type="range" id="vx" min="-1.5" max="1.5" step="0.01" value="0.5">
  <input type="number" class="val" id="vx_val" min="-1.5" max="1.5" step="0.01" value="0.50">
</div>
<div class="slider-row">
  <label>vy</label>
  <input type="range" id="vy" min="-0.5" max="0.5" step="0.01" value="0.0">
  <input type="number" class="val" id="vy_val" min="-0.5" max="0.5" step="0.01" value="0.00">
</div>
<div class="slider-row">
  <label>vyaw</label>
  <input type="range" id="vyaw" min="-1.5" max="1.5" step="0.01" value="0.0">
  <input type="number" class="val" id="vyaw_val" min="-1.5" max="1.5" step="0.01" value="0.00">
</div>
<div class="actions">
  <button class="btn-reset" onclick="doReset()">Reset Episode</button>
  <button class="btn-random" onclick="randomCmd()">Random Cmd</button>
</div>
</div>

<div class="card">
<h3>Cost Weights</h3>
<div class="slider-row">
  <label>omega'</label>
  <input type="range" id="w_omega" min="0" max="0.1" step="0.0001" value="0.001">
  <input type="number" class="val" id="w_omega_val" min="0" max="1" step="0.0001" value="0.0010">
</div>
<div class="slider-row">
  <label>track</label>
  <input type="range" id="w_track" min="0" max="3" step="0.01" value="1.0">
  <input type="number" class="val" id="w_track_val" min="0" max="10" step="0.01" value="1.00">
</div>
<div class="slider-row">
  <label>stab</label>
  <input type="range" id="w_stab" min="0" max="3" step="0.01" value="1.0">
  <input type="number" class="val" id="w_stab_val" min="0" max="10" step="0.01" value="1.00">
</div>
<div class="slider-row">
  <label>gait</label>
  <input type="range" id="w_gait" min="0" max="3" step="0.01" value="1.0">
  <input type="number" class="val" id="w_gait_val" min="0" max="10" step="0.01" value="1.00">
</div>
</div>

<div class="card">
<h3>Status</h3>
<div class="status" id="status">initializing&hellip;</div>
</div>

</div></div>

<script>
const sl=id=>document.getElementById(id);
function link(sliderId,numId){
  const s=sl(sliderId),n=sl(numId);
  s.addEventListener('input',()=>{n.value=s.value});
  n.addEventListener('input',()=>{s.value=n.value});
  n.addEventListener('change',()=>{s.value=n.value});
}
function sendCmd(){
  const vx=+sl('vx').value, vy=+sl('vy').value, vyaw=+sl('vyaw').value;
  sl('vx_val').value=vx.toFixed(2);
  sl('vy_val').value=vy.toFixed(2);
  sl('vyaw_val').value=vyaw.toFixed(2);
  fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vx,vy,vyaw})});
}
function sendCmdFromNum(){
  const vx=+sl('vx_val').value, vy=+sl('vy_val').value, vyaw=+sl('vyaw_val').value;
  sl('vx').value=vx; sl('vy').value=vy; sl('vyaw').value=vyaw;
  fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vx,vy,vyaw})});
}
['vx','vy','vyaw'].forEach(id=>{
  sl(id).addEventListener('input',sendCmd);
  sl(id+'_val').addEventListener('input',sendCmdFromNum);
  sl(id+'_val').addEventListener('change',sendCmdFromNum);
});
function sendWeights(){
  const o=+sl('w_omega').value, t=+sl('w_track').value,
        s=+sl('w_stab').value, g=+sl('w_gait').value;
  sl('w_omega_val').value=o.toFixed(4);
  sl('w_track_val').value=t.toFixed(2);
  sl('w_stab_val').value=s.toFixed(2);
  sl('w_gait_val').value=g.toFixed(2);
  fetch('/api/weights',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({omega_prime:o,tracking:t,stability:s,gait:g})});
}
function sendWeightsFromNum(){
  const o=+sl('w_omega_val').value, t=+sl('w_track_val').value,
        s=+sl('w_stab_val').value, g=+sl('w_gait_val').value;
  sl('w_omega').value=o; sl('w_track').value=t;
  sl('w_stab').value=s; sl('w_gait').value=g;
  fetch('/api/weights',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({omega_prime:o,tracking:t,stability:s,gait:g})});
}
['w_omega','w_track','w_stab','w_gait'].forEach(id=>{
  sl(id).addEventListener('input',sendWeights);
  sl(id+'_val').addEventListener('input',sendWeightsFromNum);
  sl(id+'_val').addEventListener('change',sendWeightsFromNum);
});
function doReset(){fetch('/api/reset',{method:'POST'})}
function randomCmd(){
  sl('vx').value=(Math.random()*2-0.5).toFixed(2);
  sl('vy').value=(Math.random()*0.6-0.3).toFixed(2);
  sl('vyaw').value=(Math.random()*2-1).toFixed(2);
  sendCmd();
}
(async function initSliders(){
  try{
    const s=await(await fetch('/api/status')).json();
    if(s.omega_prime!==undefined){
      sl('w_omega').value=s.omega_prime; sl('w_omega_val').value=s.omega_prime.toFixed(4);
      sl('w_track').value=s.w_tracking; sl('w_track_val').value=s.w_tracking.toFixed(2);
      sl('w_stab').value=s.w_stability; sl('w_stab_val').value=s.w_stability.toFixed(2);
      sl('w_gait').value=s.w_gait; sl('w_gait_val').value=s.w_gait.toFixed(2);
    }
  }catch(e){}
})();
setInterval(async()=>{
  const s=await(await fetch('/api/status')).json();
  sl('status').textContent=
    `phase: ${s.phase}  |  step: ${s.step}\\n`+
    `vel target: [${s.vel_tar_x.toFixed(2)}, ${s.vel_tar_y.toFixed(2)}, ${s.vel_tar_yaw.toFixed(2)}]\\n`+
    `body vel:   [${s.body_vx.toFixed(2)}, ${s.body_vy.toFixed(2)}, ${s.body_vyaw.toFixed(2)}]\\n`+
    `r_aug: ${s.r_aug.toFixed(3)}  r_base: ${s.r_base.toFixed(3)}  ${s.safety_mode}: ${s.safety.toFixed(1)}\\n`+
    `omega': ${s.omega_prime.toFixed(4)}  w: [${s.w_tracking.toFixed(1)}, ${s.w_stability.toFixed(1)}, ${s.w_gait.toFixed(1)}]\\n`+
    `mppi: ${s.mppi_hz.toFixed(0)} Hz  |  height: ${s.height.toFixed(3)} m`;
},250);
sendCmd();
</script>
"""


class BrowserViewer:
    """MJPEG-over-HTTP streaming viewer with velocity command GUI."""

    def __init__(self, model, host, port, width, height, shared):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base",
        )
        self.camera.distance = 3.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -20.0
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._status: dict = {
            "phase": "initializing", "step": 0,
            "r_aug": 0.0, "r_base": 0.0, "safety": 0.0, "mppi_hz": 0.0,
            "safety_mode": "log-prob",
            "vel_tar_x": 0.0, "vel_tar_y": 0.0, "vel_tar_yaw": 0.0,
            "body_vx": 0.0, "body_vy": 0.0, "body_vyaw": 0.0,
            "height": 0.3,
            "omega_prime": 0.001, "w_tracking": 1.0,
            "w_stability": 1.0, "w_gait": 1.0,
        }
        self._shared = shared
        self._start_server(host, port)

    def _start_server(self, host, port):
        app = Flask("residual_mppi_viewer")
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        shared = self._shared

        @app.get("/")
        def index():
            return _INDEX_HTML

        @app.get("/api/status")
        def status():
            with self._condition:
                return jsonify(dict(self._status))

        @app.post("/api/reset")
        def reset():
            shared.request_reset()
            return jsonify({"ok": True})

        @app.post("/api/command")
        def command():
            data = request.get_json(force=True)
            shared.set_vel_cmd(data["vx"], data["vy"], data["vyaw"])
            return jsonify({"ok": True})

        @app.post("/api/weights")
        def weights():
            data = request.get_json(force=True)
            shared.set_weights(
                data["omega_prime"], data["tracking"],
                data["stability"], data["gait"],
            )
            return jsonify({"ok": True})

        @app.get("/stream.mjpg")
        def stream():
            def frames():
                previous = None
                while True:
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._frame is not None
                            and self._frame is not previous,
                        )
                        previous = self._frame
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + previous + b"\r\n"
                    )
            return Response(
                frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        threading.Thread(
            target=lambda: app.run(
                host=host, port=port, threaded=True, use_reloader=False,
            ),
            daemon=True,
        ).start()

    def update(self, pipeline_state, **status):
        self.data.qpos[:] = np.asarray(pipeline_state.qpos)
        self.data.qvel[:] = np.asarray(pipeline_state.qvel)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        output = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(
            output, format="JPEG", quality=85,
        )
        with self._condition:
            self._frame = output.getvalue()
            self._status.update(status)
            self._condition.notify_all()

    def set_status(self, **status):
        with self._condition:
            self._status.update(status)

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Residual-MPPI with an RL prior policy",
    )
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    parser.add_argument(
        "--prior", type=Path, required=True,
        help="path to prior_policy.pkl",
    )
    parser.add_argument(
        "--critic", type=Path, default=None,
        help="path to safety_critic.pkl (uses Q-value instead of log-prob)",
    )
    parser.add_argument(
        "--omega-prime", type=float, default=0.001,
        help="weight on safety signal (default: 0.001)",
    )
    parser.add_argument(
        "--addon-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="reward component weights [tracking, stability, gait] "
             "(default: 1 1 1 = full reward)",
    )
    parser.add_argument(
        "--ramp-up-time", type=float, default=None,
        help="override velocity command ramp (0 = no ramp)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--real-time-factor", type=float, default=1.0)
    parser.add_argument("--web-viewer-host", default="0.0.0.0")
    parser.add_argument("--web-viewer-port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--viewer-fps", type=float, default=20.0)
    args = parser.parse_args()

    prior = RLPriorPolicy.load(args.prior)
    print(f"Loaded prior: obs={prior.observation_size}, "
          f"act={prior.action_size}, hidden={prior.hidden_layer_sizes}")

    critic = None
    if args.critic is not None:
        critic = SafetyCritic.load(args.critic)
        print(f"Loaded safety critic: obs={critic.observation_size}, "
              f"act={critic.action_size}, hidden={critic.hidden_layer_sizes}")

    if args.example is not None:
        config_dict = yaml.safe_load(
            open(get_example_path(args.example + ".yaml")),
        )
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    if args.seed is not None:
        dial_config.seed = args.seed
    if args.ramp_up_time is not None:
        config_dict["ramp_up_time"] = args.ramp_up_time

    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True,
    )
    print(f"Creating environment: {dial_config.env_name}")
    base_env = brax_envs.get_environment(dial_config.env_name, config=env_config)

    test_state = jax.jit(base_env.reset)(jax.random.PRNGKey(0))
    env_obs_size = int(test_state.obs.shape[-1])
    if env_obs_size != prior.observation_size:
        raise RuntimeError(
            f"Observation size mismatch: env produces {env_obs_size} but "
            f"prior expects {prior.observation_size}. "
            f"Check include_foot_height_observation and other obs settings.",
        )

    wrapped_env = ResidualMPPIEnv(
        base_env, prior=prior,
        omega_prime=args.omega_prime,
        addon_weights=jnp.array(args.addon_weights),
        critic=critic,
    )
    safety_mode = "Q-value (critic)" if critic else "log-prob (prior)"
    print(f"Residual-MPPI: omega'={args.omega_prime}, "
          f"addon_weights={args.addon_weights}, safety={safety_mode}")

    controller = make_controller(dial_config, wrapped_env)
    torso_idx = int(base_env._torso_idx) - 1

    shared = SharedControl()
    viewer = BrowserViewer(
        base_env.sys.mj_model,
        args.web_viewer_host, args.web_viewer_port,
        args.width, args.height, shared,
    )
    print(
        f"viewer=http://{args.web_viewer_host}:{args.web_viewer_port} "
        "(use SSH port forward from local PC)",
    )

    reset_env = jax.jit(wrapped_env.reset)
    step_env = jax.jit(wrapped_env.step)
    if critic is not None:
        safety_jit = jax.jit(critic.q_value)
    else:
        safety_jit = jax.jit(prior.log_prob)
    render_period = 1.0 / args.viewer_fps

    init_Y0 = jax.jit(lambda state: initialize_from_prior(
        base_env, prior, state, dial_config.Hsample,
        controller.u2node_vmap,
    ))

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, st, history = rng_Y0_state
        rng, Y0, info = controller.reverse_once(st, rng, Y0, factor, history)
        return (rng, Y0, st, history), info

    traj_diffuse_factors = (
        controller.sigma_control
        * dial_config.traj_diffuse_factor
        ** jnp.arange(dial_config.Ndiffuse)[:, None]
    )

    # ---- warmup / JIT compilation ----
    rng = jax.random.PRNGKey(dial_config.seed)
    rng, rng_reset = jax.random.split(rng)
    state = reset_env(rng_reset)
    viewer.update(
        state.pipeline_state, phase="compiling",
        omega_prime=args.omega_prime,
        w_tracking=args.addon_weights[0],
        w_stability=args.addon_weights[1],
        w_gait=args.addon_weights[2],
        safety_mode="Q-value" if critic else "log-prob",
    )

    print("JIT-compiling (first call may be slow)...")
    Y0 = init_Y0(state)
    Y0.block_until_ready()
    action_history = jnp.zeros(
        (dial_config.tc_history_length, controller.nu), dtype=Y0.dtype,
    )
    _ = safety_jit(state.obs, Y0[0]).block_until_ready()
    state = step_env(state, Y0[0])
    state.reward.block_until_ready()
    Y0 = init_Y0(state)
    Y0.block_until_ready()
    (rng, Y0, _, _), _ = jax.lax.scan(
        reverse_scan,
        (rng, Y0, state, action_history),
        traj_diffuse_factors,
    )
    Y0.block_until_ready()
    viewer.set_status(phase="ready")
    print("Compilation done.")

    # ---- helpers ----
    def apply_vel_cmd(state, cmd):
        """Override velocity command in state.info."""
        vx, vy, vyaw = cmd
        state.info["vel_cmd"] = jnp.array([vx, vy, 0.0])
        state.info["ang_vel_cmd"] = jnp.array([0.0, 0.0, vyaw])
        state.info["vel_tar"] = jnp.array([vx, vy, 0.0])
        state.info["ang_vel_tar"] = jnp.array([0.0, 0.0, vyaw])
        state.info["randomize_target"] = False
        return state

    def get_body_vel(state):
        x, xd = state.pipeline_state.x, state.pipeline_state.xd
        vb = global_to_body_velocity(xd.vel[torso_idx], x.rot[torso_idx])
        ab = global_to_body_velocity(
            xd.ang[torso_idx] * jnp.pi / 180.0, x.rot[torso_idx],
        )
        return float(vb[0]), float(vb[1]), float(ab[2])

    # ---- main loop: single infinite episode ----
    episode_seed = 1
    cur_omega = args.omega_prime
    cur_weights = list(args.addon_weights)
    try:
        while True:
            rng = jax.random.fold_in(
                jax.random.PRNGKey(dial_config.seed), episode_seed,
            )
            state = reset_env(rng)
            Y0 = init_Y0(state)
            action_history = jnp.zeros(
                (dial_config.tc_history_length, controller.nu),
                dtype=Y0.dtype,
            )

            cmd = shared.get_vel_cmd()
            if cmd is not None:
                state = apply_vel_cmd(state, cmd)

            w = shared.get_weights()
            if w is not None:
                cur_omega, cur_weights[0], cur_weights[1], cur_weights[2] = w
                state.info["omega_prime"] = jnp.array(cur_omega)
                state.info["reward_weights"] = jnp.array(cur_weights)

            viewer.update(state.pipeline_state, phase="running", step=0)
            episode_started = time.perf_counter()
            last_render_time = -np.inf
            step_idx = 0

            while True:
                if shared.consume_reset():
                    print(f"Reset requested at step {step_idx}")
                    episode_seed += 1
                    break

                w = shared.get_weights()
                if w is not None:
                    cur_omega, cur_weights[0], cur_weights[1], cur_weights[2] = w
                    state.info["omega_prime"] = jnp.array(cur_omega)
                    state.info["reward_weights"] = jnp.array(cur_weights)

                action = Y0[0]
                lp = safety_jit(state.obs, action)
                state = step_env(state, action)
                state.reward.block_until_ready()
                step_idx += 1

                cmd = shared.get_vel_cmd()
                if cmd is not None:
                    state = apply_vel_cmd(state, cmd)

                r_aug = float(state.reward)
                r_base = r_aug - cur_omega * float(lp)

                action_history = jnp.concatenate(
                    [action_history[1:], action[None]], axis=0,
                )
                Y0 = controller.shift(Y0)

                t0 = time.perf_counter()
                (rng, Y0, _, _), _ = jax.lax.scan(
                    reverse_scan,
                    (rng, Y0, state, action_history),
                    traj_diffuse_factors,
                )
                Y0.block_until_ready()
                mppi_hz = 1.0 / max(time.perf_counter() - t0, 1e-9)

                height = float(state.pipeline_state.x.pos[torso_idx, 2])
                sim_time = step_idx * base_env.dt

                if sim_time - last_render_time >= render_period:
                    bvx, bvy, bvyaw = get_body_vel(state)
                    viewer.update(
                        state.pipeline_state,
                        phase="running",
                        step=step_idx,
                        r_aug=r_aug,
                        r_base=r_base,
                        safety=float(lp),
                        mppi_hz=mppi_hz,
                        vel_tar_x=float(state.info["vel_tar"][0]),
                        vel_tar_y=float(state.info["vel_tar"][1]),
                        vel_tar_yaw=float(state.info["ang_vel_tar"][2]),
                        body_vx=bvx, body_vy=bvy, body_vyaw=bvyaw,
                        height=height,
                        omega_prime=cur_omega,
                        w_tracking=cur_weights[0],
                        w_stability=cur_weights[1],
                        w_gait=cur_weights[2],
                    )
                    last_render_time = sim_time

                if height < 0.18:
                    print(f"Fell at step {step_idx}")
                    viewer.set_status(phase="fallen; resetting")
                    time.sleep(1.0)
                    episode_seed += 1
                    break

                target_wall = (
                    episode_started + sim_time / args.real_time_factor
                )
                remaining = target_wall - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
