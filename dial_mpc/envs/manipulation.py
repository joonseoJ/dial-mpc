from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple, Union, List

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
class AllegroReorientEnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = 1.0
    kd: Union[float, jax.Array] = 0.1


class AllegroReorientEnv(BaseEnv):
    def __init__(self, config: AllegroReorientEnvConfig):
        super().__init__(config)

        self._object_body_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "object"
        )
        self._init_q = jnp.array(self.sys.mj_model.keyframe("in_hand_reorient").qpos)

    def make_system(self, config: AllegroReorientEnvConfig) -> System:
        model_path = get_model_path("wonik_allegro", "scene_left.xml")
        mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.5]),
            "pos_tar": jnp.array([0.0, 0.0, 0.13]),
            "step": 0,
        }

        obs = jnp.zeros(1)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            raise NotImplementedError
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # reward
        ball_ang_vel = xd.ang[self._object_body_idx - 1] * jnp.pi / 180.0
        ball_pos = x.pos[self._object_body_idx - 1]
        reward_ang_vel = -jnp.sum(jnp.square(ball_ang_vel - state.info["ang_vel_tar"]))
        reward_pos = -jnp.sum(jnp.square(ball_pos - state.info["pos_tar"]))
        reward_joint_angle_deviation = -jnp.sum(jnp.square(pipeline_state.q[7:] - self._init_q[7:]))

        reward = (reward_ang_vel * 1.0
                  + reward_pos * 5.0
                  + reward_joint_angle_deviation * 0.1)
        # done
        done = jnp.zeros(1)
        done = jnp.where(state.info["step"] >= 100, 1, done)

        # update state
        state_info = {
            "rng": rng,
            "ang_vel_tar": state.info["ang_vel_tar"],
            "pos_tar": state.info["pos_tar"],
            "step": state.info["step"] + 1,
        }

        obs = jnp.zeros(1)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + self._init_q[7:] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

brax_envs.register_environment("allegro_reorient", AllegroReorientEnv)


@dataclass
class AllegroInHandEnvConfig(BaseEnvConfig):
    """Reorient an object in the hand, with the weight deciding how.

    This is the task the locomotion bases were reaching for and kept missing.
    Two things it has that they did not.

    Sampling MPC is not a stylistic choice here.  A multi-finger hand makes and
    breaks contact every few control steps, the dynamics are non-smooth at each
    of those events, and gradients taken through contact are known to point the
    wrong way.  Flat-ground trotting, by contrast, is the least contact-rich
    thing DIAL-MPC does, and nothing about it argues against a learned policy or
    a reduced-order controller.

    And the weight is a question someone actually asks.  "How much do I care
    about foot placement versus torso attitude while being shoved" is not a
    preference, it is one objective cut into pieces.  "This one is fragile" is a
    preference, it comes from the object rather than from the algorithm, and it
    changes from grasp to grasp -- which is exactly the setting where composing
    trained score fields at run time beats retraining per object.

    The conflicts are physical, and unusually dense for a four-row basis:

      * progress vs force -- a moment on the object needs tangential force, and
        the friction cone bounds it by the normal force, ||f_t|| <= mu f_n.  To
        turn the object faster you must squeeze it harder.  Exact.
      * progress vs security -- turning further than one grasp allows requires
        finger gaiting, and a lifted finger contributes nothing to the grasp
        wrench.  Security drops exactly when progress happens.  Exact.
      * security vs force -- holding on wants more normal force for friction
        margin; not crushing the object wants less.  The classic grasping
        trade-off, and the two rows sit on opposite sides of it.
      * progress vs posture -- reaching around the object needs finger
        configurations far from the comfortable nominal.

    Rows 1-3 are all satisfied by holding still at the nominal pose, and row 0
    is what makes that a failure.  That is the push-recovery shape, with the
    goal orientation playing the disturbance's part -- except that here
    "hold still" is legibly the wrong answer rather than a subtle one.
    """

    kp: Union[float, jax.Array] = 1.0
    kd: Union[float, jax.Array] = 0.1
    leg_control: str = "position"
    dt: float = 0.02
    timestep: float = 0.005

    # The goal orientation turns continuously about this axis at this rate.  A
    # fixed goal was tried first and DIAL reached a half turn in three seconds,
    # after which the episode had nothing left to trade off; a turning goal
    # keeps the four rows in competition for the whole episode and makes the
    # task's difficulty a single dial.  The reference is a pose rather than an
    # angular velocity so that row 0 has a proper argmin and falling behind is
    # priced.
    goal_rate: float = 1.0
    goal_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    randomize_tasks: bool = False

    # Contact geometry, read off the model: the object is a 25 mm sphere and the
    # fingertips are 12 mm capsules whose centres sit 19 mm (35 mm for the
    # thumb) along the tip body's z axis.
    object_radius: float = 0.025
    fingertip_radius: float = 0.012
    fingertip_offset: float = 0.019
    thumbtip_offset: float = 0.035
    fingertip_half_length: float = 0.010
    thumbtip_half_length: float = 0.008
    # Physics steps run with the nominal grasp before an episode starts.  The
    # keyframe leaves the object floating a centimetre above the fingertips;
    # letting it settle first means every weight is asked to reorient an object
    # the hand is already holding, rather than one it is still catching.
    settle_steps: int = 20

    # Half-width of the action box, in radians about the nominal grasp.  At the
    # original 0.4 rad a sample perturbs every one of sixteen joints by up to
    # 23 degrees at once, and 92-98% of the proposal cloud threw the ball out of
    # the hand inside a 0.4 s horizon -- the softmax was ranking ways to drop it.
    # A fingertip sits about 90 mm from its joint, so 0.15 rad still moves it
    # 13 mm against a 25 mm ball.
    action_range: float = 0.15

    nominal_object_height: float = 0.14
    drop_height: float = 0.06
    drop_distance: float = 0.09

    # Row 1 charges this per finger that is off the object.  A grasp held by two
    # fingers is not the same grasp as one held by four, and the count is the
    # part of security that finger gaiting actually spends.
    contact_weight: float = 0.004
    # Row 2 deadband, in squared metres of penetration.  A grasp that holds at
    # all presses to some degree; pricing that would make the row an argument
    # against holding the object rather than against crushing it.  Measured on
    # a nominal grasp before it is set.
    force_baseline: float = 0.0

    # Row normalizers.  Seeds; the screen measures each row's spread across the
    # sample cloud and equalises at constant total magnitude.
    progress_scale: float = 0.05
    security_scale: float = 0.004
    force_scale: float = 2.0e-6
    posture_scale: float = 0.5

    reward_weights: jax.Array = field(
        default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0])
    )


class AllegroInHandEnv(BaseEnv):
    """Four ways to spend a reorientation.

      0. progress -- close the angle to the goal orientation
      1. security -- keep the object where a grasp can hold it, on all four tips
      2. force    -- do not squeeze harder than the job needs
      3. posture  -- keep the fingers near a configuration they can work from
    """

    FINGERTIPS = ("ff_tip", "mf_tip", "rf_tip", "th_tip")

    def __init__(self, config: AllegroInHandEnvConfig):
        super().__init__(config)
        model = self.sys.mj_model
        self._object_body_idx = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY.value, "object"
        )
        tips = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in self.FINGERTIPS
        ]
        assert not any(i == -1 for i in tips), "fingertip body not found"
        self._tip_idx = jnp.array(tips)
        # The contact capsule sits along the tip body's own z axis, further out
        # on the thumb than on the fingers.
        self._tip_offset = jnp.array(
            [config.fingertip_offset] * 3 + [config.thumbtip_offset]
        )
        self._tip_half = jnp.array(
            [config.fingertip_half_length] * 3 + [config.thumbtip_half_length]
        )
        self._touch_distance = config.object_radius + config.fingertip_radius

        self._init_q = jnp.array(model.keyframe("in_hand_reorient").qpos)
        self._nominal_hand = self._init_q[7:]
        # Where the object comes to rest in the nominal grasp.  Row 1 measures
        # drift from here, so it has to be the held position and not the
        # keyframe's floating one.
        self._nominal_object = jnp.asarray(
            self._settle(self.pipeline_init(self._init_q, jnp.zeros(self._nv)))
            .x.pos[self._object_body_idx - 1]
        )

    def make_system(self, config: AllegroInHandEnvConfig) -> System:
        model_path = get_model_path("wonik_allegro", "scene_left.xml")
        mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    # ------------------------------------------------------------------ #
    def _settle(self, pipeline_state):
        """Close the nominal grasp and let the object come to rest in it."""

        def body(carry, _):
            return self.pipeline_step(carry, self._nominal_hand), None

        settled, _ = jax.lax.scan(
            body, pipeline_state, None, length=int(self._config.settle_steps)
        )
        return settled

    def _goal_axis(self, rng: jax.Array) -> jax.Array:
        axis = jnp.asarray(self._config.goal_axis, dtype=jnp.float32)
        if self._config.randomize_tasks:
            axis = jax.random.normal(rng, (3,))
        return axis / jnp.maximum(jnp.linalg.norm(axis), 1e-9)

    def _goal_quat(self, state_info, base: jax.Array) -> jax.Array:
        # The rate lives in `info` rather than in the config so a viewer can
        # turn it while the hand is running: how fast the object is wanted is a
        # task knob, exactly like the weights are.
        angle = state_info["goal_rate"] * state_info["step"] * self.dt
        return math.quat_mul(
            math.quat_rot_axis(state_info["goal_axis"], angle), base
        )

    def _tip_segment(self, x) -> Tuple[jax.Array, jax.Array]:
        """Endpoints of each fingertip's contact capsule, in world frame."""

        pos = x.pos[self._tip_idx - 1]
        rot = x.rot[self._tip_idx - 1]
        zero = jnp.zeros(4)
        near = jnp.stack([zero, zero, self._tip_offset - self._tip_half], -1)
        far = jnp.stack([zero, zero, self._tip_offset + self._tip_half], -1)
        return (pos + jax.vmap(math.rotate)(near, rot),
                pos + jax.vmap(math.rotate)(far, rot))

    def _contact(self, x) -> Tuple[jax.Array, jax.Array]:
        """Penetration depth per fingertip, as a stand-in for normal force.

        MuJoCo's contact is soft, so depth is monotone in the force it carries.
        Reading it off the geometry instead of the solver keeps the row a
        differentiable function of the state and free of any contact-array
        bookkeeping under vmap.
        """

        object_pos = x.pos[self._object_body_idx - 1]
        near, far = self._tip_segment(x)
        axis = far - near
        # Closest point on the capsule's axis segment.  Measuring to the capsule
        # centre instead misses by up to a half-length, which on this hand is
        # 10 mm against a 37 mm touch distance -- enough to report a settled
        # grasp as no contact at all.
        length_sq = jnp.maximum(jnp.sum(jnp.square(axis), axis=-1), 1e-12)
        t = jnp.clip(
            jnp.sum((object_pos - near) * axis, axis=-1) / length_sq, 0.0, 1.0
        )
        closest = near + axis * t[:, None]
        distance = jnp.linalg.norm(closest - object_pos, axis=-1)
        penetration = jnp.maximum(self._touch_distance - distance, 0.0)
        return penetration, penetration > 1e-4

    def _get_obs(self, pipeline_state, state_info) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        index = self._object_body_idx - 1
        object_pos = x.pos[index]
        object_rot = x.rot[index]
        goal = self._goal_quat(state_info, state_info["goal_base"])
        penetration, contact = self._contact(x)
        return jnp.concatenate(
            [
                object_pos - self._nominal_object,
                object_rot,
                xd.vel[index],
                xd.ang[index],
                goal,
                math.quat_mul(math.quat_inv(object_rot), goal),
                pipeline_state.q[7:],
                pipeline_state.qd[6:],
                pipeline_state.ctrl,
                penetration * 100.0,
                contact.astype(jnp.float32),
            ]
        )

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, goal_rng = jax.random.split(rng)
        pipeline_state = self._settle(
            self.pipeline_init(self._init_q, jnp.zeros(self._nv))
        )
        base = pipeline_state.x.rot[self._object_body_idx - 1]
        state_info = {
            "rng": rng,
            "goal_axis": self._goal_axis(goal_rng),
            "goal_rate": jnp.asarray(self._config.goal_rate, dtype=jnp.float32),
            "goal_base": base,
            "step": 0,
            "reward_weights": normalize_omega(
                jnp.asarray(self._config.reward_weights)
            ),
            "reward_terms": jnp.zeros(4),
            "n_contact": jnp.zeros(()),
            "angle_error": jnp.zeros(()),
            "squeeze": jnp.zeros(()),
            "goal_quat": base,
            "spin_rate": jnp.zeros(()),
        }
        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done, {}, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        ctrl = self.act2joint(action)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x = pipeline_state.x
        index = self._object_body_idx - 1
        object_pos = x.pos[index]
        object_rot = x.rot[index]
        state.info["step"] += 1
        goal = self._goal_quat(state.info, state.info["goal_base"])

        penetration, contact = self._contact(x)
        n_contact = jnp.sum(contact.astype(jnp.float32))

        # Row 0 -- progress.  1 - |q . q_goal| is zero at the goal, one at a
        # half turn away, and blind to the quaternion's sign ambiguity.  The
        # squared form reads the same at both ends but its gradient vanishes at
        # exactly half a turn, which is where this task starts.
        alignment = jnp.abs(jnp.dot(object_rot, goal))
        angle_error = 1.0 - alignment
        reward_progress = -angle_error / self._config.progress_scale

        # Row 1 -- security.  Where the object sits relative to where a grasp
        # can hold it, plus what finger gaiting spends: a lifted finger carries
        # none of the grasp wrench.
        # Bounded at the drop threshold.  An object on its way to the floor
        # makes this term unbounded, and unbounded is what it was: across the
        # proposal cloud the security row had 835 times the spread of the
        # progress row, so no weight on any other row could move the argmax.
        # A drop is what `done` is for; the row should price holding on.
        drift = jnp.minimum(
            jnp.sum(jnp.square(object_pos - self._nominal_object)),
            self._config.drop_distance ** 2,
        )
        reward_security = -(
            drift + self._config.contact_weight * (4.0 - n_contact)
        ) / self._config.security_scale

        # Row 2 -- force.  Squared penetration stands in for squeeze; the
        # deadband keeps the row about crushing rather than about holding.
        squeeze = jnp.sum(jnp.square(penetration))
        reward_force = -jnp.maximum(
            squeeze - self._config.force_baseline, 0.0
        ) / self._config.force_scale

        # Row 3 -- posture.  Reaching around the object costs finger
        # configurations far from one they can work from again.
        reward_posture = -jnp.sum(
            jnp.square(pipeline_state.q[7:] - self._nominal_hand)
        ) / self._config.posture_scale

        reward_components = jnp.stack(
            [reward_progress, reward_security, reward_force, reward_posture]
        )
        weights = normalize_omega(state.info["reward_weights"])
        reward = jnp.dot(weights, reward_components)

        dropped = object_pos[2] < self._config.drop_height
        dropped |= (
            jnp.linalg.norm(object_pos - self._nominal_object)
            > self._config.drop_distance
        )
        done = dropped.astype(jnp.float32)

        state.info["goal_quat"] = goal
        # What the object actually turned, about the axis it was asked to turn
        # about.  Angle error alone cannot tell a robot that is tracking from
        # one that has fallen a full turn behind.
        state.info["spin_rate"] = jnp.dot(
            pipeline_state.xd.ang[index], state.info["goal_axis"]
        )
        state.info["reward_weights"] = weights
        state.info["reward_terms"] = reward_components
        state.info["n_contact"] = n_contact
        state.info["angle_error"] = angle_error
        state.info["squeeze"] = squeeze

        obs = self._get_obs(pipeline_state, state.info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        """Actions are offsets around the nominal grasp, not absolute angles.

        Centring the action box on the pose the hand starts in is what makes a
        zero plan a held grasp rather than an open hand, so the annealing starts
        from something that is at least holding the object.
        """

        targets = (self._nominal_hand
                   + act * self._config.action_scale * self._config.action_range)
        return jnp.clip(
            targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )


brax_envs.register_environment("allegro_in_hand", AllegroInHandEnv)
