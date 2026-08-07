import os
import time
import io
import threading
from multiprocessing import shared_memory
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scienceplots
import art

import mujoco
import mujoco.viewer
from flask import Flask, Response

from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.utils.io_utils import (
    load_dataclass_from_dict,
    get_model_path,
    get_example_path,
)
from dial_mpc.examples import deploy_examples

plt.style.use(["science"])


@dataclass
class DialSimConfig:
    robot_name: str
    scene_name: str
    sim_leg_control: str
    plot: bool
    record: bool
    real_time_factor: float
    sim_dt: float
    sync_mode: bool
    reward_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    web_viewer_port: int = 0
    web_viewer_host: str = "127.0.0.1"
    web_viewer_width: int = 640
    web_viewer_height: int = 360
    web_viewer_fps: float = 20.0


class DialSim:
    def __init__(
        self,
        sim_config: DialSimConfig,
        env_config: BaseEnvConfig,
        dial_config: DialConfig,
    ):
        # control related
        self.plot = sim_config.plot
        self.record = sim_config.record
        self.data = []
        self.ctrl_dt = env_config.dt
        self.real_time_factor = sim_config.real_time_factor
        self.sim_dt = sim_config.sim_dt
        self.n_acts = dial_config.Hsample + 1
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.t = 0.0
        self.sync_mode = sim_config.sync_mode
        self.leg_control = sim_config.sim_leg_control
        self.control_kp = np.asarray(env_config.kp, dtype=np.float64)
        self.control_kd = np.asarray(env_config.kd, dtype=np.float64)
        self.web_viewer_port = sim_config.web_viewer_port
        self.web_viewer_host = sim_config.web_viewer_host
        self.web_viewer_width = sim_config.web_viewer_width
        self.web_viewer_height = sim_config.web_viewer_height
        self.web_viewer_period = 1.0 / sim_config.web_viewer_fps
        self._last_web_frame_time = -np.inf
        self._web_frame = None
        self._web_frame_condition = threading.Condition()
        self.mj_model = mujoco.MjModel.from_xml_path(
            get_model_path(sim_config.robot_name, sim_config.scene_name).as_posix()
        )
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.web_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.web_camera)
        self.web_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.web_camera.trackbodyid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.web_camera.distance = 3.0
        self.web_camera.azimuth = 90.0
        self.web_camera.elevation = -20.0
        self.q_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.qref_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.n_plot_joint = 4

        # mujoco setup
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # parameters
        self.Nx = self.mj_model.nq + self.mj_model.nv
        self.Nu = self.mj_model.nu

        # get home keyframe
        self.default_q = self.mj_model.keyframe("home").qpos
        self.default_u = self.mj_model.keyframe("home").ctrl

        # communication setup
        # publisher
        self.time_shm = shared_memory.SharedMemory(
            name="time_shm", create=True, size=32
        )
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=self.Nx * 32
        )
        self.state_shared = np.ndarray(
            (self.Nx,), dtype=np.float32, buffer=self.state_shm.buf
        )
        # listener
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=True, size=self.n_acts * self.Nu * 32
        )
        self.acts_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.default_u
        self.refs_shm = shared_memory.SharedMemory(
            name="refs_shm", create=True, size=self.n_acts * self.Nu * 3 * 32
        )
        self.refs_shared = np.ndarray(
            (self.n_acts, self.Nu, 3), dtype=np.float32, buffer=self.refs_shm.buf
        )
        self.refs_shared[:] = 0.0
        self.plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=True, size=32
        )
        self.plan_time_shared = np.ndarray(
            1, dtype=np.float32, buffer=self.plan_time_shm.buf
        )
        self.plan_time_shared[0] = -self.ctrl_dt

        self.tau_shm = shared_memory.SharedMemory(
            name="tau_shm", create=True, size=self.n_acts * self.Nu * 32
        )
        self.tau_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.tau_shm.buf
        )

        # Runtime-adjustable Go2 objective weights shared with the planner and
        # the optional browser control panel.
        self.reward_weights_shm = shared_memory.SharedMemory(
            name="dial_reward_weights",
            create=True,
            size=3 * np.dtype(np.float32).itemsize,
        )
        self.reward_weights_shared = np.ndarray(
            (3,), dtype=np.float32, buffer=self.reward_weights_shm.buf
        )
        self.reward_weights_shared[:] = np.asarray(
            sim_config.reward_weights, dtype=np.float32
        )
        self.reset_shm = shared_memory.SharedMemory(
            name="dial_reset_counter",
            create=True,
            size=2 * np.dtype(np.uint64).itemsize,
        )
        self.reset_shared = np.ndarray(
            (2,), dtype=np.uint64, buffer=self.reset_shm.buf
        )
        self.reset_shared[:] = 0
        self._last_reset_counter = 0

    def _joint_target_torque(self, joint_target):
        """Evaluates the same joint-target PD law as ``BaseEnv.act2tau``."""

        joint_pos = self.mj_data.qpos[7 : 7 + self.Nu]
        joint_vel = self.mj_data.qvel[6 : 6 + self.Nu]
        torque = self.control_kp * (joint_target - joint_pos)
        torque -= self.control_kd * joint_vel
        return np.clip(
            torque,
            self.mj_model.actuator_ctrlrange[:, 0],
            self.mj_model.actuator_ctrlrange[:, 1],
        )

    def _reset_simulation(self):
        """Restore the initial pose and discard the currently buffered plan."""

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.t = 0.0
        self.time_shared[0] = 0.0
        self.state_shared[:] = np.concatenate(
            [self.mj_data.qpos, self.mj_data.qvel]
        )
        self.acts_shared[:] = self.default_u
        self.tau_shared[:] = 0.0
        self.refs_shared[:] = 0.0
        self.plan_time_shared[0] = -self.ctrl_dt
        self.q_history[:] = 0.0
        self.qref_history[:] = 0.0
        print("[INFO] Simulation reset to the home keyframe")

    def _start_web_viewer(self):
        app = Flask("dial_mpc_live_viewer")

        @app.get("/")
        def index():
            return (
                "<!doctype html><title>DIAL-MPC live viewer</title>"
                "<style>body{margin:0;background:#111}img{width:100vw;height:100vh;"
                "object-fit:contain}</style><img src='/stream.mjpg'>"
            )

        @app.get("/stream.mjpg")
        def stream():
            def frames():
                previous = None
                while True:
                    with self._web_frame_condition:
                        self._web_frame_condition.wait_for(
                            lambda: self._web_frame is not None
                            and self._web_frame is not previous
                        )
                        previous = self._web_frame
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + previous
                        + b"\r\n"
                    )

            return Response(
                frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        thread = threading.Thread(
            target=lambda: app.run(
                host=self.web_viewer_host,
                port=self.web_viewer_port,
                threaded=True,
                use_reloader=False,
            ),
            daemon=True,
        )
        thread.start()

    def _update_visualization(self, viewer, renderer):
        if viewer is not None:
            viewer.sync()
            return
        if self.t - self._last_web_frame_time < self.web_viewer_period:
            return
        renderer.update_scene(self.mj_data, camera=self.web_camera)
        pixels = renderer.render()
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="JPEG", quality=85)
        with self._web_frame_condition:
            self._web_frame = output.getvalue()
            self._web_frame_condition.notify_all()
        self._last_web_frame_time = self.t

    def main_loop(self):
        if self.plot:
            fig, axs = plt.subplots(self.n_plot_joint, 1, figsize=(12, 12))
            # plot history
            handles = []
            handles_ref = []
            # colors for each joint with rainbow
            colors = plt.cm.rainbow(np.linspace(0, 1, self.n_plot_joint))
            for i in range(self.n_plot_joint):
                handles.append(
                    axs[i].plot(
                        self.q_history[:, i],
                        color=colors[i],
                    )[0]
                )
                handles_ref.append(
                    axs[i].plot(
                        self.qref_history[:, i],
                        color=colors[i],
                        linestyle="--",
                    )[0]
                )
                # set ylim to [-0.5, 0.5]
                axs[i].set_ylim(
                    -1.0 + self.default_q[i + 7], 1.0 + self.default_q[i + 7]
                )
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel(f"Joint {i+1} Position")
            # show figure
            plt.show(block=False)

        viewer = None
        renderer = None
        if self.web_viewer_port:
            renderer = mujoco.Renderer(
                self.mj_model,
                height=self.web_viewer_height,
                width=self.web_viewer_width,
            )
            self._start_web_viewer()
            self._update_visualization(viewer, renderer)
        else:
            viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
            )

            cnt = 0
            viewer.user_scn.ngeom = 0
            for i in range(self.n_acts - 1):
                for j in range(self.mj_model.nu):
                    color = np.array(
                        [
                            1.0 * i / (self.n_acts - 1),
                            1.0 * j / self.mj_model.nu,
                            0.0,
                            1.0,
                        ]
                    )
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[cnt],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=np.zeros(3),
                        rgba=color,
                        pos=self.refs_shared[i, j, :],
                        mat=np.eye(3).flatten(),
                    )
                    cnt += 1
            viewer.user_scn.ngeom = cnt
            viewer.sync()
        while True:
            reset_counter = int(self.reset_shared[0])
            if reset_counter != self._last_reset_counter:
                self._reset_simulation()
                self._last_reset_counter = reset_counter
                # Publish completion only after state, time, and action buffers
                # all contain their reset values.
                self.reset_shared[1] = reset_counter
            if self.plot:
                # plot self.acts_shared
                for j in range(self.n_plot_joint):
                    # update plot
                    handles[j].set_ydata(self.acts_shared[:, j])
                    handles_ref[j].set_ydata(self.qref_history[:, j])
                plt.pause(0.001)
            if viewer is not None:
                for i in range(self.n_acts - 1):
                    for j in range(self.mj_model.nu):
                        r0 = self.refs_shared[i, j, :]
                        r1 = self.refs_shared[i + 1, j, :]
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[i * self.mj_model.nu + j],
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            0.02,
                            r0,
                            r1,
                        )
            if self.sync_mode:
                while self.t <= (self.plan_time_shared[0] + self.ctrl_dt):
                    if self.leg_control == "position":
                        self.mj_data.ctrl = self.acts_shared[0]
                    elif self.leg_control == "torque":
                        self.mj_data.ctrl = self._joint_target_torque(
                            self.acts_shared[0]
                        )
                    if self.record:
                        self.data.append(
                            np.concatenate(
                                [
                                    [self.t],
                                    self.mj_data.qpos,
                                    self.mj_data.qvel,
                                    self.mj_data.ctrl,
                                ]
                            )
                        )
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.t += self.sim_dt
                    # publish new state
                    q = self.mj_data.qpos
                    qd = self.mj_data.qvel
                    state = np.concatenate([q, qd])
                    self.time_shared[:] = self.t
                    self.state_shared[:] = state
                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                self._update_visualization(viewer, renderer)
            else:
                t0 = time.time()
                if self.plan_time_shared[0] < 0.0:
                    time.sleep(0.01)
                    continue
                delta_time = self.t - self.plan_time_shared[0]
                delta_step = int(delta_time / self.ctrl_dt)
                if delta_time > self.ctrl_dt / self.real_time_factor:
                    print(f"[WARN] Delayed by {delta_time*1000.0:.1f} ms")
                if delta_step >= self.n_acts or delta_step < 0:
                    delta_step = self.n_acts - 1

                if self.leg_control == "position":
                    self.mj_data.ctrl = self.acts_shared[delta_step]
                elif self.leg_control == "torque":
                    self.mj_data.ctrl = self._joint_target_torque(
                        self.acts_shared[delta_step]
                    )
                if self.record:
                    self.data.append(
                        np.concatenate(
                            [
                                [self.t],
                                self.mj_data.qpos,
                                self.mj_data.qvel,
                                self.mj_data.ctrl,
                            ]
                        )
                    )
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt
                q = self.mj_data.qpos
                qd = self.mj_data.qvel
                state = np.concatenate([q, qd])

                # publish new state
                self.time_shared[:] = self.t
                self.state_shared[:] = state

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                self._update_visualization(viewer, renderer)
                t1 = time.time()
                duration = t1 - t0
                if duration < self.sim_dt / self.real_time_factor:
                    time.sleep((self.sim_dt / self.real_time_factor - duration))
                else:
                    print("[WARN] Sim loop overruns")

    def close(self):
        for shm in (
            self.time_shm,
            self.state_shm,
            self.acts_shm,
            self.plan_time_shm,
            self.refs_shm,
            self.tau_shm,
            self.reward_weights_shm,
            self.reset_shm,
        ):
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                # An older observer may already have removed the name.  The
                # mapping is still released correctly when this owner exits.
                pass


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nSIMULATOR", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    group.add_argument(
        "--example",
        type=str,
        default=None,
        help="Example to run",
    )
    group.add_argument(
        "--list-examples",
        action="store_true",
        help="List available examples",
    )
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    parser.add_argument(
        "--web-viewer-port",
        type=int,
        default=None,
        help="Serve a headless MJPEG viewer on this port instead of opening a window",
    )
    args = parser.parse_args(args)

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.list_examples:
        print("Available examples:")
        for example in deploy_examples:
            print(f"  - {example}")
        return
    if args.example is not None:
        if args.example not in deploy_examples:
            print(f"Example {args.example} not found.")
            return
        config_dict = yaml.safe_load(
            open(get_example_path(args.example + ".yaml"), "r")
        )
    else:
        config_dict = yaml.safe_load(open(args.config, "r"))
    sim_config = load_dataclass_from_dict(DialSimConfig, config_dict)
    if args.web_viewer_port is not None:
        sim_config.web_viewer_port = args.web_viewer_port
    env_config = load_dataclass_from_dict(BaseEnvConfig, config_dict)
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    mujoco_env = DialSim(sim_config, env_config, dial_config)

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if mujoco_env.record:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = np.array(mujoco_env.data)
            output_dir = os.path.join(
                dial_config.output_dir,
                f"sim_{dial_config.env_name}_{env_config.task_name}_{timestamp}",
            )
            os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "states"), data)

        mujoco_env.close()


if __name__ == "__main__":
    main()
