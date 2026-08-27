from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path
from csm.omega import normalize_omega


@dataclass
class UnitreeGo2EnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = 30.0
    kd: Union[float, jax.Array] = 0.0
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "trot"
    include_foot_height_observation: bool = False
    enable_body_collisions: bool = False
    command_vx_min: float = -1.5
    command_vx_max: float = 1.5
    command_vy_min: float = -0.5
    command_vy_max: float = 0.5
    command_vyaw_min: float = -1.5
    command_vyaw_max: float = 1.5
    command_resample_steps: int = 500
    randomize_start_state: bool = False
    start_xy_noise: float = 0.0
    start_height_noise: float = 0.0
    start_rpy_noise: float = 0.0
    start_joint_position_noise: float = 0.0
    start_body_linear_velocity_noise: float = 0.0
    start_body_angular_velocity_noise: float = 0.0
    start_joint_velocity_noise: float = 0.0
    reward_weights: jax.Array = field(
        default_factory=lambda: jnp.array([1.0, 1.0, 1.0])
    )


class UnitreeGo2Env(BaseEnv):
    def __init__(self, config: UnitreeGo2EnvConfig):
        super().__init__(config)

        self._foot_radius = 0.0175

        self._gait = config.gait
        self._gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            "trot": jnp.array([0.45, 2, 0.08]),
            "canter": jnp.array([0.4, 4, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }

        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )

        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]

        self.joint_range = jnp.array(
            [
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ]
        )
        feet_site = [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        feet_site_id = [
            mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = jnp.array(feet_site_id)

    def make_system(self, config: UnitreeGo2EnvConfig) -> System:
        scene = (
            "mjx_scene_force_collision.xml"
            if config.enable_body_collisions
            else "mjx_scene_force.xml"
        )
        model_path = get_model_path("unitree_go2", scene)
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, state_rng, command_rng = jax.random.split(rng, 3)
        qpos, qvel = self._sample_initial_state(state_rng)
        pipeline_state = self.pipeline_init(qpos, qvel)

        default_vel_cmd = jnp.array(
            [self._config.default_vx, self._config.default_vy, 0.0]
        )
        default_ang_vel_cmd = jnp.array(
            [0.0, 0.0, self._config.default_vyaw]
        )
        vel_cmd, ang_vel_cmd = jax.lax.cond(
            jnp.asarray(self._config.randomize_tasks),
            lambda: self.sample_command(command_rng),
            lambda: (default_vel_cmd, default_ang_vel_cmd),
        )
        initial_scale = self._command_ramp_scale(0)

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.282, 0.0, 0.3]),
            "vel_cmd": vel_cmd,
            "ang_vel_cmd": ang_vel_cmd,
            "vel_tar": vel_cmd * initial_scale,
            "ang_vel_tar": ang_vel_cmd * initial_scale,
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
            "reward_weights": jnp.asarray(self._config.reward_weights),
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            pipeline_state = self.pipeline_step(state.pipeline_state, joint_targets)
        elif self._config.leg_control == "torque":
            pipeline_state = self.pd_pipeline_step(state.pipeline_state, action)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # Keep a sampled command until the next resampling boundary.  The old
        # implementation returned the default command on every non-boundary
        # step, so a randomized target survived for only one control tick.
        resample_steps = max(int(self._config.command_resample_steps), 1)
        should_resample = (
            state.info["randomize_target"]
            & (state.info["step"] > 0)
            & (state.info["step"] % resample_steps == 0)
        )
        vel_cmd, ang_vel_cmd = jax.lax.cond(
            should_resample,
            lambda: self.sample_command(cmd_rng),
            lambda: (state.info["vel_cmd"], state.info["ang_vel_cmd"]),
        )
        command_scale = self._command_ramp_scale(state.info["step"])
        state.info["vel_cmd"] = vel_cmd
        state.info["ang_vel_cmd"] = ang_vel_cmd
        state.info["vel_tar"] = vel_cmd * command_scale
        state.info["ang_vel_tar"] = ang_vel_cmd * command_scale

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt
        reward_air_time = jnp.sum((state.info["feet_air_time"] - 0.1) * first_contact)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1]
        R = math.quat_to_3x3(x.rot[self._torso_idx - 1])
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)
        reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = (
            state.info["yaw_tar"]
            + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        )
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # velocity reward
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        # stay alive reward
        reward_alive = 1.0 - state.done
        # reward
        # CSM-compatible objective basis.  The default weights [1, 1, 1]
        # exactly reproduce the original Go2 trot reward.
        reward_components = jnp.stack(
            [
                reward_vel + reward_ang_vel,
                reward_upright * 0.5 + reward_yaw * 0.3 + reward_height,
                reward_gaits * 0.1,
            ]
        )
        # See the push-recovery env: omega carries direction only, so that the
        # temperature is the sole scale.  DIAL's own update is invariant to the
        # weight scale, so this changes reported reward magnitudes and nothing
        # about the planner's behaviour.
        reward = jnp.dot(
            normalize_omega(state.info["reward_weights"]), reward_components
        )

        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        if self._config.terminate_on_joint_limit:
            done |= jnp.any(joint_angles < self.joint_range[:, 0])
            done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _command_ramp_scale(self, step: jax.Array | int) -> jax.Array:
        if self._config.ramp_up_time <= 0.0:
            return jnp.asarray(1.0)
        return jnp.clip(
            jnp.asarray(step) * self.dt / self._config.ramp_up_time, 0.0, 1.0
        )

    def _sample_initial_state(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Samples a bounded, physically plausible reset state when enabled."""

        qpos = self._init_q
        qvel = jnp.zeros(self._nv)
        if not self._config.randomize_start_state:
            return qpos, qvel

        keys = jax.random.split(rng, 7)
        xy_delta = jax.random.uniform(
            keys[0], (2,), minval=-1.0, maxval=1.0
        ) * self._config.start_xy_noise
        height_delta = jax.random.uniform(
            keys[1], (), minval=-1.0, maxval=1.0
        ) * self._config.start_height_noise
        rpy_delta = jax.random.uniform(
            keys[2], (3,), minval=-1.0, maxval=1.0
        ) * self._config.start_rpy_noise
        joint_delta = jax.random.uniform(
            keys[3], (self.action_size,), minval=-1.0, maxval=1.0
        ) * self._config.start_joint_position_noise

        qpos = qpos.at[:2].add(xy_delta)
        qpos = qpos.at[2].add(height_delta)
        rotation_delta = math.euler_to_quat(rpy_delta * 180.0 / jnp.pi)
        qpos = qpos.at[3:7].set(math.quat_mul(rotation_delta, qpos[3:7]))
        joint_margin = 1e-3
        joints = jnp.clip(
            qpos[7:] + joint_delta,
            self.joint_range[:, 0] + joint_margin,
            self.joint_range[:, 1] - joint_margin,
        )
        qpos = qpos.at[7:].set(joints)

        qvel = qvel.at[:3].set(
            jax.random.uniform(keys[4], (3,), minval=-1.0, maxval=1.0)
            * self._config.start_body_linear_velocity_noise
        )
        qvel = qvel.at[3:6].set(
            jax.random.uniform(keys[5], (3,), minval=-1.0, maxval=1.0)
            * self._config.start_body_angular_velocity_noise
        )
        qvel = qvel.at[6:].set(
            jax.random.uniform(
                keys[6], (self.action_size,), minval=-1.0, maxval=1.0
            )
            * self._config.start_joint_velocity_noise
        )
        return qpos, qvel

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        features = [
            state_info["vel_tar"],
            state_info["ang_vel_tar"],
            pipeline_state.ctrl,
            pipeline_state.qpos,
            vb,
            ab,
            pipeline_state.qvel[6:],
        ]
        if self._config.include_foot_height_observation:
            duty_ratio, cadence, amplitude = self._gait_params[self._gait]
            phases = self._gait_phase[self._gait]
            target_foot_height = get_foot_step(
                duty_ratio,
                cadence,
                amplitude,
                phases,
                state_info["step"] * self.dt,
            )
            actual_foot_height = pipeline_state.site_xpos[
                self._feet_site_id
            ][:, 2]
            features.extend([target_foot_height, actual_foot_height])
        obs = jnp.concatenate(features)
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1,
            (),
            minval=self._config.command_vx_min,
            maxval=self._config.command_vx_max,
        )
        lin_vel_y = jax.random.uniform(
            key2,
            (),
            minval=self._config.command_vy_min,
            maxval=self._config.command_vy_max,
        )
        ang_vel_yaw = jax.random.uniform(
            key3,
            (),
            minval=self._config.command_vyaw_min,
            maxval=self._config.command_vyaw_max,
        )
        new_lin_vel_cmd = jnp.stack([lin_vel_x, lin_vel_y, jnp.asarray(0.0)])
        new_ang_vel_cmd = jnp.stack(
            [jnp.asarray(0.0), jnp.asarray(0.0), ang_vel_yaw]
        )
        return new_lin_vel_cmd, new_ang_vel_cmd


@dataclass
class UnitreeGo2SeqJumpEnvConfig(UnitreeGo2EnvConfig):
    jump_dt: float = 1.0
    contact_targets: jax.Array = None
    contact_target_radius: jax.Array = None
    pose_target_sequence: jax.Array = None
    yaw_target_sequence: jax.Array = None


class UnitreeGo2SeqJumpEnv(UnitreeGo2Env):
    def __init__(
        self, config: UnitreeGo2SeqJumpEnvConfig = UnitreeGo2SeqJumpEnvConfig()
    ):
        super().__init__(config)
        if config.contact_targets is None or config.contact_target_radius is None:
            (
                self._contact_targets,
                self._contact_target_radius,
                self._pose_target_sequence,
                self._yaw_target_sequence,
            ) = UnitreeGo2SeqJumpEnv.generate_jumping_sequence(
                config.pose_target_sequence, config.yaw_target_sequence, 0.1
            )
        else:
            self._contact_targets = config.contact_targets
            self._contact_target_radius = config.contact_target_radius
            self._pose_target_sequence = config.pose_target_sequence
            self._yaw_target_sequence = config.yaw_target_sequence
        self.joint_range = jnp.array(
            [
                [-0.5, 0.5],
                [0.4, 2.0],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 2.0],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ]
        )

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, 0.27]),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
            "last_ctrl": jnp.zeros(12),
        }

        state_info["contact_stage"] = 0
        if not self._config.randomize_tasks:
            state_info["contact_targets"] = self._contact_targets
            state_info["contact_target_radius"] = self._contact_target_radius
            state_info["pose_target_sequence"] = self._pose_target_sequence
            state_info["yaw_target_sequence"] = self._yaw_target_sequence
        else:
            (
                state_info["contact_targets"],
                state_info["contact_target_radius"],
                state_info["pose_target_sequence"],
                state_info["yaw_target_sequence"],
            ) = self.sample_command(rng)

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)

        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
            pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        elif self._config.leg_control == "torque":
            # The reward terms below read `ctrl`; keep the start-of-step torque
            # for them while the physics closes the PD loop per substep.
            ctrl = self.act2tau(action, state.pipeline_state)
            pipeline_state = self.pd_pipeline_step(state.pipeline_state, action)
        else:
            raise ValueError("Invalid leg control type.")
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # done
        done = 0.0

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # position reward
        pose_target_sequence = state.info["pose_target_sequence"]
        pos_tar = pose_target_sequence[state.info["contact_stage"]]
        pos = x.pos[self._torso_idx - 1]
        reward_pos = -jnp.sum((pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_target_sequence = state.info["yaw_target_sequence"]
        yaw_tar = yaw_target_sequence[state.info["contact_stage"]]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        reward_yaw = -jnp.square(yaw - yaw_tar)
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))

        # contact reward
        reward_contact = 0.0
        penalty_contact = pipeline_state.contact.dist <= 0.001
        reward_1 = lambda x: 1.0 * x
        reward_0 = lambda x: 0.0
        contact_targets = state.info["contact_targets"]
        contact_target_radius = state.info["contact_target_radius"]
        for i in range(4):
            for j in range(len(contact_targets)):
                contact_dist = pipeline_state.contact.dist[i]
                contact_pt = pipeline_state.contact.pos[i]
                cond = (
                    jnp.sum((contact_pt[:2] - contact_targets[j, i, :2]) ** 2)
                    <= contact_target_radius[j, i] ** 2
                )  # & (z_feet[i] < 0.001)
                reward_contact += jax.lax.cond(
                    cond,
                    reward_1,
                    reward_0,
                    (j == state.info["contact_stage"])
                    * jnp.clip(contact_dist * -1.0 + 1.0, 0.0, 1.0),
                )
                penalty_contact = penalty_contact.at[i].set(
                    penalty_contact[i] & (~cond)
                )
        penalty_contact = jnp.sum(penalty_contact)
        # energy reward
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        # control rate reward
        reward_ctrl_rate = -jnp.sum((ctrl - state.info["last_ctrl"]) ** 2)
        # alive reward
        reward_alive = 1.0
        # reward
        reward = (
            reward_gaits * 0.0
            + reward_pos * 1.0
            + reward_upright * 1.0
            + reward_yaw * 0.3
            # + reward_pose * 0.0
            + reward_contact * 0.1
            - penalty_contact * 0.1
            + reward_energy * 0.0
            + reward_ctrl_rate * 0.0
            + reward_alive * 10.0
        )

        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        if self._config.terminate_on_joint_limit:
            done |= jnp.any(joint_angles < self.joint_range[:, 0])
            done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.1
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["contact_stage"] = jnp.minimum(
            jnp.floor(state.info["step"] * self.dt / self._config.jump_dt),
            len(contact_targets) - 1,
        ).astype(jnp.int32)
        state.info["last_ctrl"] = ctrl

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        quat = pipeline_state.qpos[3:7]
        rpy = math.quat_to_euler(quat)
        pose_target = state_info["pose_target_sequence"][state_info["contact_stage"]]
        yaw_target = state_info["yaw_target_sequence"][state_info["contact_stage"]]

        diff_position = x.pos[self._torso_idx - 1] - pose_target
        diff_yaw = rpy[2] - yaw_target
        diff_yaw = jnp.arctan2(jnp.sin(diff_yaw), jnp.cos(diff_yaw)).reshape(1)
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                state_info["last_ctrl"],
                diff_position,
                rpy[:2],
                diff_yaw,
                pipeline_state.qpos[7:],
                vb,
                ab,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def generate_jumping_sequence(
        com_pos: Sequence, com_heading: Sequence, foot_place_radius: float
    ):
        n_steps = com_pos.shape[0]
        contact_targets = []
        contact_target_radius = jnp.full((n_steps, 4), foot_place_radius)
        pose_target_sequence = jnp.array(com_pos)
        yaw_target_sequence = jnp.array(com_heading)
        assert n_steps == len(com_heading)

        for i in range(n_steps):
            contact_target = jnp.repeat(jnp.array([com_pos[i]]), 4, axis=0)
            offsets = jnp.array(
                [
                    [0.2, -0.135, 0.0],  # FR
                    [0.2, 0.135, 0.0],  # FL
                    [-0.2, -0.135, 0.0],  # RR
                    [-0.2, 0.135, 0.0],  # RL
                ]
            )
            R = math.quat_to_3x3(
                math.euler_to_quat(jnp.array([0.0, 0.0, com_heading[i] * 180 / jnp.pi]))
            )
            offsets = jnp.dot(offsets, R.T)
            contact_target = contact_target + offsets
            contact_targets.append(contact_target)
        contact_targets = jnp.array(contact_targets)

        return (
            contact_targets,
            contact_target_radius,
            pose_target_sequence,
            yaw_target_sequence,
        )

    def sample_command(
        self, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        com_pos_begin = jnp.array([0.0, 0.0, 0.27])
        com_yaw_begin = jnp.array([0.0])

        def randomize_com_pos(last_com_pos, rng):
            next_com_pos = last_com_pos.at[:2].add(
                jax.random.uniform(rng, (2,), minval=-0.65, maxval=0.65)
            )
            return next_com_pos, next_com_pos

        def randomize_com_yaw(last_com_yaw, rng):
            next_com_yaw = last_com_yaw + jax.random.uniform(
                rng, (1,), minval=-0.5, maxval=0.5
            )
            return next_com_yaw, next_com_yaw

        n_steps = 10
        keys = jax.random.split(rng, n_steps * 2)
        _, com_pos = jax.lax.scan(randomize_com_pos, com_pos_begin, keys[:n_steps])
        _, com_yaw = jax.lax.scan(randomize_com_yaw, com_yaw_begin, keys[n_steps:])
        com_pos = jnp.concatenate([com_pos_begin.reshape(1, 3), com_pos], axis=0)
        com_yaw = jnp.concatenate(
            [com_yaw_begin.reshape(1, 1), com_yaw], axis=0
        ).flatten()
        (
            contact_targets,
            contact_target_radius,
            pose_target_sequence,
            yaw_target_sequence,
        ) = UnitreeGo2SeqJumpEnv.generate_jumping_sequence(com_pos, com_yaw, 0.1)
        return (
            contact_targets,
            contact_target_radius,
            pose_target_sequence,
            yaw_target_sequence,
        )

    def update_viewer(self, viewer):
        cnt = viewer.user_scn.ngeom
        for i in range(self._contact_targets.shape[0]):
            for j in range(4):
                color = np.array([0.0, 1.0, 0.0, 0.5])
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[cnt],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=np.array([self._contact_target_radius[i, j], 0.01]),
                    rgba=color,
                    pos=self._contact_targets[i, j],
                    mat=np.eye(3).flatten(),
                )
                cnt += 1


class UnitreeGo2CrateEnvConfig(UnitreeGo2EnvConfig):
    pass


class UnitreeGo2CrateEnv(UnitreeGo2Env):
    def __init__(self, config: UnitreeGo2CrateEnvConfig = UnitreeGo2CrateEnvConfig()):
        super().__init__(config)
        self.joint_range = jnp.array(
            [
                [-0.25, 0.25],
                [-1.0, 1.4],
                [-2.7, -1.0],
                [-0.25, 0.25],
                [-1.0, 1.4],
                [-2.7, -1.0],
                [-0.25, 0.25],
                [0.0, 1.8],
                [-2.7, -1.0],
                [-0.25, 0.25],
                [0.0, 1.8],
                [-2.7, -1.0],
            ]
        )

    def make_system(self, config: UnitreeGo2EnvConfig) -> System:
        model_path = get_model_path("unitree_go2", "mjx_scene_force_crate.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
            pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
            pipeline_state = self.pd_pipeline_step(state.pipeline_state, action)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # done
        done = 0.0

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1]
        R = math.quat_to_3x3(x.rot[self._torso_idx - 1])
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)
        reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = state.info["yaw_tar"]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        reward_yaw = -jnp.square(yaw - yaw_tar)
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # velocity reward
        reward_vel = -jnp.sum(
            (xd.vel[self._torso_idx - 1] - state.info["vel_tar"]) ** 2
        )
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        # energy reward
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        # pitch reward
        rpy = math.quat_to_euler(x.rot[self._torso_idx - 1])
        pitch_tar = -0.7854
        pitch = rpy[1]
        reward_pitch = -jnp.square(pitch - pitch_tar)
        reward_roll = -jnp.square(rpy[0])

        # contact reward
        reward_contact = 0.0
        penalty_contact = pipeline_state.contact.dist <= 0.001
        reward_1 = lambda x: 1.0 * x
        reward_0 = lambda x: 0.0
        contact_indices = [16, 17, 18, 19]
        for i in range(4):
            # contact_idx = 26 + 4 + 2 + 2 * 2 * (i+1) + i
            # contact_idx = (4 + 2 + 2 * 2 * (i+1) + i) * 2 + 1
            contact_idx = contact_indices[i]
            contact_dist = pipeline_state.contact.dist[contact_idx]
            contact_pt = pipeline_state.contact.pos[contact_idx]
            cond = (
                (contact_pt[0] > 1.0)
                & (contact_pt[0] < 1.6)
                & (contact_pt[1] > -0.45)
                & (contact_pt[1] < 0.45)
                & (contact_pt[2] > 0.59)
                & (contact_pt[2] < 0.61)
            )
            reward_contact += jax.lax.cond(cond, reward_1, reward_0, 1.0)
            penalty_contact = penalty_contact.at[i].set(penalty_contact[i] & (~cond))
        penalty_contact = jnp.sum(penalty_contact)

        # reward
        reward = (
            reward_gaits * 0.0
            + reward_pos * 1.0
            + reward_upright * 0.01
            + reward_yaw * 0.3
            # + reward_pose * 0.0
            + reward_vel * 0.0
            + reward_height * 0.0
            + reward_energy * 0.0000
            + reward_pitch * 0.0
            + reward_roll * 0.0
            + reward_contact * 0.02
            - penalty_contact * 0.0
        )
        # jax.debug.print("{geom}", geom=pipeline_state.contact.geom)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        state.info["pos_tar"] = jnp.array([1.45, 0.0, 0.87])
        state.info["vel_tar"] = jnp.array([0.0, 0.0, 0.0])
        state.info["ang_vel_tar"] = jnp.array([0.0, 0.0, 0.0])
        state.info["yaw_tar"] = 0.0
        return state


@dataclass
class UnitreeGo2PushRecoverEnvConfig(UnitreeGo2EnvConfig):
    """Stand-and-resist task used to screen a disturbance-rejection basis.

    The robot holds a nominal stance and is hit with a velocity impulse on the
    floating base.  Nothing is hidden from the planner: the push lands at reset,
    so DIAL simply sees a disturbed state and has to react to it.
    """

    enable_body_collisions: bool = True
    terminate_on_joint_limit: bool = False
    gait: str = "stand"
    default_vx: float = 0.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 0.0
    nominal_height: float = 0.30
    # Reset impulse magnitude (m/s on the base).  Sampled in
    # [push_scale_min, 1] * push_linear_velocity so that one screening run
    # covers a band of disturbance sizes rather than a single operating point.
    push_linear_velocity: float = 1.0
    push_angular_velocity: float = 0.0
    push_scale_min: float = 0.25
    # Row normalizers.  Each row reads about -1 at a "moderate" deviation:
    # 0.2 rad of tilt, 0.10 m of base offset, 0.10 m of slip per foot, and
    # 0.3 rad on every joint.  Comparable row scales are what make a weight
    # grid meaningful -- otherwise one row dominates at every omega.
    tilt_scale: float = 0.04
    base_scale: float = 0.01
    foot_scale: float = 0.04
    shape_scale: float = 1.0
    foot_lift_penalty: float = 0.5
    reward_weights: jax.Array = field(
        default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0])
    )


class UnitreeGo2PushRecoverEnv(UnitreeGo2Env):
    """Four-row disturbance-rejection basis.

    The rows deliberately live at three different levels of the kinematic
    chain, because rows drawn from a single level are collinear and cannot
    move the argmax:

      0. torso tilt          -- orientation only, position excluded
      1. base pose           -- hold the CoM over its start point
      2. contact             -- keep each foot where it started, in world frame
      3. shape               -- keep the joints at the nominal stance

    Rows 2 and 3 are separated only by the floating base: world-frame foot
    position and base-relative joint angle are the same quantity once the base
    is pinned.  Keeping row 2 in world coordinates is what makes "plant the
    feet and bend" and "hold the shape and step" distinct solutions.
    """

    def __init__(self, config: UnitreeGo2PushRecoverEnvConfig):
        super().__init__(config)
        # Nominal contact geometry, evaluated once from the home keyframe.  The
        # reset pose is fixed, so these world-frame targets are constants.
        init_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))
        self._nominal_foot_pos = jnp.asarray(
            init_state.site_xpos[self._feet_site_id]
        )
        self._nominal_base_xy = jnp.asarray(
            init_state.x.pos[self._torso_idx - 1][:2]
        )

    def _sample_push(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        heading_key, scale_key, ang_key = jax.random.split(rng, 3)
        heading = jax.random.uniform(heading_key, (), minval=-jnp.pi, maxval=jnp.pi)
        scale = jax.random.uniform(
            scale_key, (), minval=self._config.push_scale_min, maxval=1.0
        )
        speed = self._config.push_linear_velocity * scale
        linear = jnp.stack(
            [speed * jnp.cos(heading), speed * jnp.sin(heading), jnp.asarray(0.0)]
        )
        angular = jax.random.uniform(
            ang_key, (3,), minval=-1.0, maxval=1.0
        ) * self._config.push_angular_velocity
        return linear, angular

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        state = super().reset(rng)
        rng, push_rng = jax.random.split(state.info["rng"])
        linear, angular = self._sample_push(push_rng)

        qpos = state.pipeline_state.qpos
        qvel = state.pipeline_state.qvel.at[:3].add(linear).at[3:6].add(angular)
        pipeline_state = self.pipeline_init(qpos, qvel)

        state.info["rng"] = rng
        state.info["pos_tar"] = jnp.array(
            [self._nominal_base_xy[0], self._nominal_base_xy[1],
             self._config.nominal_height]
        )
        state.info["vel_tar"] = jnp.zeros(3)
        state.info["ang_vel_tar"] = jnp.zeros(3)
        state.info["push_linear"] = linear
        state.info["push_speed"] = jnp.linalg.norm(linear)
        state.info["reward_weights"] = normalize_omega(
            state.info["reward_weights"]
        )
        state.info["reward_terms"] = jnp.zeros(4)
        state.info["feet_lifted"] = jnp.zeros(())

        obs = self._get_obs(pipeline_state, state.info)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

    def step(self, state: State, action: jax.Array) -> State:
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
            pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        else:
            pipeline_state = self.pd_pipeline_step(state.pipeline_state, action)
        x = pipeline_state.x
        obs = self._get_obs(pipeline_state, state.info)

        pos = x.pos[self._torso_idx - 1]
        up = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(up, x.rot[self._torso_idx - 1])

        # Row 0 -- torso tilt.  Orientation only.  Splitting this from the base
        # position row is the whole point: holding the CoM against a push needs
        # a horizontal ground reaction force, and that force tilts the torso.
        reward_tilt = -jnp.sum(jnp.square(vec - up)) / self._config.tilt_scale

        # Row 1 -- base pose: keep the CoM over where it started.
        base_err = jnp.sum(
            jnp.square(pos[:2] - self._nominal_base_xy)
        ) + jnp.square(pos[2] - self._config.nominal_height)
        reward_base = -base_err / self._config.base_scale

        # Row 2 -- contact: keep each foot on its original world-frame spot,
        # plus an explicit price on breaking contact at all.
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_err = jnp.sum(jnp.square(foot_pos - self._nominal_foot_pos))
        lifted = (foot_pos[:, 2] - self._foot_radius) > 1e-3
        n_lifted = jnp.sum(lifted.astype(jnp.float32))
        reward_feet = -(
            foot_err / self._config.foot_scale
            + self._config.foot_lift_penalty * n_lifted
        )

        # Row 3 -- shape: base-relative joint configuration.
        joint_angles = pipeline_state.q[7:]
        reward_shape = -jnp.sum(
            jnp.square(joint_angles - self._default_pose)
        ) / self._config.shape_scale

        reward_components = jnp.stack(
            [reward_tilt, reward_base, reward_feet, reward_shape]
        )
        # Unit weights by construction: the objective depends only on
        # omega / T, so letting omega carry a free scale would make the
        # temperature meaningless.  Normalising on read and storing the result
        # keeps `state.info` honest about what was used.
        weights = normalize_omega(state.info["reward_weights"])
        reward = jnp.dot(weights, reward_components)

        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= pos[2] < 0.15
        done = done.astype(jnp.float32)

        state.info["step"] += 1
        state.info["reward_weights"] = weights
        state.info["reward_terms"] = reward_components
        state.info["feet_lifted"] = n_lifted

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )


@dataclass
class UnitreeGo2GaitChoiceEnvConfig(UnitreeGo2EnvConfig):
    """Velocity-command locomotion where the weight picks *how* to walk.

    The walking basis failed because every row was a deviation penalty against
    the same nominal trot.  Rows that share an argmin leave omega nothing to
    do but change the curvature of one basin, and the cross-evaluation showed
    exactly that.  So this task deletes the gait reference: there is no
    ``z_feet_tar``, and the target foot height is kept out of the observation.
    Nothing tells the robot which gait to use.  The rows say only what the
    operator cares about, and the gait is what omega is left to choose.

    The conflict between the rows is mechanical rather than tuned.  At a
    commanded speed v a gait is fixed by its period T and its duty factor D
    (the fraction of the cycle a foot spends on the ground), and steady
    locomotion forces

        mean normal force per stance foot   F = m g / (4 D)
        ballistic torso bounce              dz = g (1 - 2D)^2 T^2 / 8
        swing power per leg                 P ~ m_l v^2 / ((1 - D)^2 T)

    Reading the signs off those three: transport efficiency wants D down and
    T up, payload stability wants D up and T down, contact gentleness wants D
    up and T up.  Rows 1 and 2 are antipodal in both coordinates and row 3 is
    parallel to neither, so three non-parallel directions compete in a
    two-dimensional space and no gait is optimal for all of them.  That is the
    property the walking rows did not have.
    """

    # No gait reference anywhere.  "stand" only selects the (unused) phase
    # table; the reward never reads it and the observation never carries it.
    gait: str = "stand"
    include_foot_height_observation: bool = False
    enable_body_collisions: bool = True
    terminate_on_joint_limit: bool = False

    # The trade-off only exists near the actuation envelope.  At a stroll all
    # four rows agree on the same comfortable trot, which is the walking task
    # all over again, so the default command is deliberately brisk.
    default_vx: float = 1.2
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 0.4
    nominal_height: float = 0.30

    # Command envelope.  Wider than the trot env's because the point of the
    # task is that one score field has to cover the whole joystick, turns
    # included, not a single straight line.
    command_vx_min: float = -0.6
    command_vx_max: float = 1.6
    command_vy_min: float = -0.6
    command_vy_max: float = 0.6
    command_vyaw_min: float = -1.6
    command_vyaw_max: float = 1.6

    # Row normalizers.  Seeds only: the screen measures each row's standard
    # deviation across the DIAL sample cloud and rescales so all four are
    # comparable, which is what makes the *direction* of omega meaningful and
    # lets a temperature be quoted.
    track_scale: float = 0.10
    energy_scale: float = 100.0
    payload_scale: float = 1.00
    contact_scale: float = 0.50

    # Rows 1-3 are priced per unit of *delivered* command rather than per
    # second when this is positive.  Priced per second, the realised speed
    # explained 92% of the energy row and 93% of the contact row: the cheapest
    # way to lower any rate per unit time is to move less, one shared direction
    # that collapsed a nominally four-row basis to an effective rank of 1.85.
    #
    # Dividing by the raw speed removes that but opens a worse hole.  A penalty
    # over |v| always shrinks as |v| grows, and inside a 0.64 s horizon the
    # cheapest way to make |v| large is to launch the body ballistically --
    # measured, the planner did exactly that: 100% flight phase, no foot
    # contact at all, an apparent cost of transport of 0.33 against a trot's
    # 1.0, and a fall within two control steps.
    #
    # So the divisor is clipped at the commanded speed and expressed as a
    # fraction of it.  At the commanded speed the rows read their per-second
    # value; below it they inflate (a standstill pays 20x); above it nothing
    # further is gained, which closes the ballistic exploit.  With no
    # translation commanded the cap collapses to the floor and the rows are
    # per-second again, which is the right reading when "per metre" is
    # meaningless.
    speed_floor: float = 0.0
    # Row 1: copper loss coefficient, W per (N m)^2.  A Go2 joint is a 0.1 ohm
    # motor behind a 6.33:1 gear with k_t about 0.3 N m / A, so holding torque
    # costs R (tau / (6.33 k_t))^2 ~ 0.03 tau^2.  Without it "energy" would be
    # free at standstill and the row would price only motion.
    copper_coeff: float = 0.03
    # Search assistance, kept strictly out of the objective.  Sampling MPC will
    # not discover a gait from a standing plan: standing is a local optimum, and
    # a sampled step pays support, impulse and swing work immediately while the
    # progress it buys arrives later.  The original DIAL-MPC trot config solves
    # this with a foot-height reference -- which is exactly the prescription
    # that made the walking basis non-discriminative.  So the reference lives
    # here, switched off by default, and is used only to drive a bootstrap
    # instance that puts the planner into a trotting basin.  Every measured
    # number comes from an instance with this at zero.
    bootstrap_gait_weight: float = 0.0
    # Row 3: price of a missing stance foot.  The mean over a cycle is
    # 4 (1 - D), so this term is a linear price on the duty factor itself.
    support_weight: float = 0.25

    reward_weights: jax.Array = field(
        default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0])
    )


class UnitreeGo2GaitChoiceEnv(UnitreeGo2Env):
    """Four rows describing what the operator cares about, not how to walk.

      0. command tracking    -- body-frame linear velocity and yaw rate
      1. transport efficiency -- positive joint work plus copper loss
      2. payload stability   -- what a carried camera or cargo feels
      3. contact gentleness  -- touchdown impulse and stance support

    Rows 1-3 are all trivially satisfied by standing still, exactly as rows
    1-3 of the push-recovery basis were satisfied by not moving.  What makes
    them separable there was the impulse; what makes them separable here is
    row 0, which is present at weight 1 in every basis weight the screen and
    the training grid use.  A pure e_i corner is allowed to be degenerate.
    """

    def __init__(self, config: UnitreeGo2GaitChoiceEnvConfig):
        super().__init__(config)
        self._total_mass = float(jnp.sum(self.sys.mj_model.body_mass))

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def _get_obs(self, pipeline_state, state_info) -> jax.Array:
        """Command, proprioception and contact -- no world position, no phase.

        The rows are functions of velocity, acceleration and contact alone, so
        the optimal policy is stationary in time and in place.  Feeding the
        world x/y of a robot that walks away forever would be the one input
        guaranteed to leave the training distribution, and the gait phase is
        deliberately absent because prescribing it is what broke the walking
        basis.
        """
        x, xd = pipeline_state.x, pipeline_state.xd
        torso = self._torso_idx - 1
        vb = global_to_body_velocity(xd.vel[torso], x.rot[torso])
        ab = global_to_body_velocity(xd.ang[torso], x.rot[torso])
        foot_z = (
            pipeline_state.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        )
        return jnp.concatenate(
            [
                state_info["vel_tar"][:2],
                state_info["ang_vel_tar"][2:3],
                pipeline_state.ctrl,
                pipeline_state.qpos[2:],          # height, quaternion, joints
                vb,
                ab,
                pipeline_state.qvel[6:],
                foot_z,
                (foot_z < 1e-3).astype(jnp.float32),
            ]
        )

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        state = super().reset(rng)
        pipeline_state = state.pipeline_state
        torso = self._torso_idx - 1
        foot_z = (
            pipeline_state.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        )
        # Rows 2 and 3 read accelerations and touchdown speeds, which are
        # differences across a control step.  Seeding the "last" fields from
        # the reset state itself makes the first step's differences zero
        # instead of a spurious spike.
        state.info["last_vz"] = pipeline_state.xd.vel[torso][2]
        state.info["last_foot_z"] = foot_z
        state.info["last_contact"] = foot_z < 1e-3
        state.info["reward_weights"] = normalize_omega(
            state.info["reward_weights"]
        )
        state.info["reward_terms"] = jnp.zeros(4)
        state.info["n_contact"] = jnp.sum((foot_z < 1e-3).astype(jnp.float32))
        state.info["power"] = jnp.zeros(())
        return state.replace(obs=self._get_obs(pipeline_state, state.info))

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
            pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        else:
            pipeline_state = self.pd_pipeline_step(state.pipeline_state, action)
        x, xd = pipeline_state.x, pipeline_state.xd
        torso = self._torso_idx - 1

        resample_steps = max(int(self._config.command_resample_steps), 1)
        should_resample = (
            state.info["randomize_target"]
            & (state.info["step"] > 0)
            & (state.info["step"] % resample_steps == 0)
        )
        vel_cmd, ang_vel_cmd = jax.lax.cond(
            should_resample,
            lambda: self.sample_command(cmd_rng),
            lambda: (state.info["vel_cmd"], state.info["ang_vel_cmd"]),
        )
        command_scale = self._command_ramp_scale(state.info["step"])
        vel_tar = vel_cmd * command_scale
        ang_vel_tar = ang_vel_cmd * command_scale

        vb = global_to_body_velocity(xd.vel[torso], x.rot[torso])
        # brax reports xd.ang in rad/s already.  The trot env multiplies it by
        # pi/180 before comparing against the yaw-rate command, which is
        # harmless there only because every shipped config commands zero yaw
        # rate.  This task commands turns, so the conversion is dropped.
        ab = global_to_body_velocity(xd.ang[torso], x.rot[torso])

        # Row 0 -- command tracking.  The mission, and the only row that is
        # not satisfied by standing still.
        track_err = jnp.sum(jnp.square(vb[:2] - vel_tar[:2])) + jnp.square(
            ab[2] - ang_vel_tar[2]
        )
        reward_track = -track_err / self._config.track_scale

        # Row 1 -- transport efficiency.  Positive mechanical work (no
        # regeneration) plus resistive loss.  Wants long strides at low
        # cadence and will happily let the torso bounce, because ballistic
        # flight is free.
        tau = pipeline_state.ctrl
        joint_vel = pipeline_state.qvel[6:]
        power = jnp.sum(jnp.maximum(tau * joint_vel, 0.0)) + (
            self._config.copper_coeff * jnp.sum(jnp.square(tau))
        )
        if self._config.speed_floor > 0.0:
            floor = self._config.speed_floor
            cap = jnp.maximum(jnp.linalg.norm(vel_tar[:2]), floor)
            progress = jnp.clip(
                jnp.linalg.norm(xd.vel[torso][:2]), floor, cap
            ) / cap
        else:
            progress = 1.0
        reward_energy = -power / progress / self._config.energy_scale

        # Row 2 -- payload stability.  Vertical acceleration and roll/pitch
        # rate are what a carried camera or cargo actually feels.  No height
        # setpoint and no attitude setpoint: prescribing a posture here would
        # reintroduce the walking basis's shared argmin.
        accel_z = (xd.vel[torso][2] - state.info["last_vz"]) / self.dt
        payload_err = jnp.square(accel_z / 9.81) + jnp.sum(jnp.square(ab[:2]))
        reward_payload = -payload_err / progress / self._config.payload_scale

        # Row 3 -- contact gentleness.  Touchdown vertical speed prices how
        # hard the foot lands; the stance deficit averages 4 (1 - D) over a
        # cycle, which is a direct linear price on the duty factor and hence,
        # through F = m g / (4 D), on the peak normal force.
        foot_z = (
            pipeline_state.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        )
        foot_zdot = (foot_z - state.info["last_foot_z"]) / self.dt
        contact = foot_z < 1e-3
        touchdown = contact & jnp.logical_not(state.info["last_contact"])
        impact = jnp.sum(
            jnp.square(jnp.maximum(-foot_zdot, 0.0))
            * touchdown.astype(jnp.float32)
        )
        n_contact = jnp.sum(contact.astype(jnp.float32))
        reward_contact = -(
            impact + self._config.support_weight * (4.0 - n_contact)
        ) / progress / self._config.contact_scale

        reward_components = jnp.stack(
            [reward_track, reward_energy, reward_payload, reward_contact]
        )
        weights = normalize_omega(state.info["reward_weights"])
        reward = jnp.dot(weights, reward_components)

        # Bootstrap only.  A Python-level branch on a config float, so it is
        # static under jit and simply absent from the measured instance.
        if self._config.bootstrap_gait_weight > 0.0:
            duty_ratio, cadence, amplitude = self._gait_params[self._config.gait]
            phases = self._gait_phase[self._config.gait]
            z_tar = get_foot_step(
                duty_ratio, cadence, amplitude, phases,
                state.info["step"] * self.dt,
            )
            z_now = pipeline_state.site_xpos[self._feet_site_id][:, 2]
            reward = reward - self._config.bootstrap_gait_weight * 0.1 * jnp.sum(
                ((z_tar - z_now) / 0.05) ** 2
            )

        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[torso]), up) < 0.0
        done |= x.pos[torso][2] < 0.18
        done = done.astype(jnp.float32)

        state.info["rng"] = rng
        state.info["step"] += 1
        state.info["vel_cmd"] = vel_cmd
        state.info["ang_vel_cmd"] = ang_vel_cmd
        state.info["vel_tar"] = vel_tar
        state.info["ang_vel_tar"] = ang_vel_tar
        state.info["last_vz"] = xd.vel[torso][2]
        state.info["last_foot_z"] = foot_z
        state.info["last_contact"] = contact
        state.info["reward_weights"] = weights
        state.info["reward_terms"] = reward_components
        state.info["n_contact"] = n_contact
        state.info["power"] = power

        obs = self._get_obs(pipeline_state, state.info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )


brax_envs.register_environment("unitree_go2_walk", UnitreeGo2Env)
brax_envs.register_environment("unitree_go2_seq_jump", UnitreeGo2SeqJumpEnv)
brax_envs.register_environment("unitree_go2_crate_climb", UnitreeGo2CrateEnv)
brax_envs.register_environment(
    "unitree_go2_push_recover", UnitreeGo2PushRecoverEnv
)
brax_envs.register_environment(
    "unitree_go2_gait_choice", UnitreeGo2GaitChoiceEnv
)
