"""Go2 whose objective rows are gaits, not deviation penalties.

Every basis so far -- walking, push-recovery, the crate -- had one ultimate
goal and rows that only tuned the detail of reaching it, so the rows shared an
argmin and omega decided almost nothing.  Gait style is the opposite: trot puts
the diagonal feet down together, pace the same-side feet, bound the front and
hind pairs.  These are mutually exclusive, and no weight vector can satisfy two
at once, so omega is forced to make a genuine choice between equal
alternatives.

Six rows, one per gait, each priced as how well the realised foot-height
profile matches that gait's phase pattern.  A small posture floor (upright and
height) is added to every row identically, so the robot stands and steps under
any weight and omega only picks the stepping *pattern*.  The gaits share one
cadence, duty ratio and amplitude on purpose: then the rows differ only in
phase, which is what makes "trot vs pace" a clean question rather than a
confound with speed.

Composition is the point.  Fit a field for pure trot and one for pure pace and
their linear mixture is a gait halfway between -- a continuous interpolation an
RL policy would have to be retrained for at every ratio, and that a sampling
planner produces for free because it plans the phase ahead.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from brax import math
from brax.base import System
from brax.envs.base import State
import brax.envs as brax_envs

from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
from dial_mpc.utils.function_utils import (
    global_to_body_velocity,
    get_foot_step,
)
from csm.omega import normalize_omega


# Foot order is [FL, FR, RL, RR].  A phase is the fraction of the cycle each
# foot's stance is offset by; feet that share a phase strike together.
# Four gaits, the contact classes DIAL separated cleanly in verification:
# trot (diagonal +0.94), pace (lateral +0.38), bound (front/hind +0.76), and
# walk (four-beat sequential).  gallop was dropped because DIAL rendered it
# identically to bound, and canter because its pattern stayed weak and blurred
# into trot -- keeping either would have handed omega two rows that mean the
# same motion, exactly the degeneracy this task exists to avoid.
GAIT_NAMES = ("walk", "trot", "pace", "bound")
GAIT_PHASES = jnp.array([
    [0.0, 0.5, 0.75, 0.25],   # walk  -- four-beat, one foot at a time
    [0.0, 0.5, 0.5, 0.0],     # trot  -- diagonal pairs (FL+RR, FR+RL)
    [0.0, 0.5, 0.0, 0.5],     # pace  -- lateral pairs (FL+RL, FR+RR)
    [0.0, 0.0, 0.5, 0.5],     # bound -- front pair then hind pair
])


@dataclass
class UnitreeGo2GaitEnvConfig(UnitreeGo2EnvConfig):
    # Four gait rows; a uniform weight asks for the average of all four, which
    # no single stepping pattern satisfies, so DIAL settles on a blended gait.
    reward_weights: jax.Array = field(default_factory=lambda: jnp.ones(4))
    # Shared across gaits so the rows differ only in phase.
    gait_cadence: float = 2.0
    gait_duty: float = 0.5
    gait_amplitude: float = 0.08
    # How strongly each row insists on the robot standing (common to all rows).
    posture_floor: float = 0.3
    # How strongly each row insists on tracking the forward command.
    track_floor: float = 0.3
    gait_scale: float = 1.0


class UnitreeGo2GaitEnv(UnitreeGo2Env):
    """Go2 with a six-gait objective; omega selects the stepping pattern."""

    def _get_obs(self, pipeline_state, state_info):
        # The parent observation carries the pose, velocities and command but no
        # sense of *when* in the gait cycle the robot is -- and a gait is a
        # periodic function of that phase.  bound survived without it because
        # its front/hind pairing is recoverable from the state alone, but trot
        # and pace need to know which diagonal or lateral pair is due, which
        # only the phase tells them.  All four gaits share one cadence, so a
        # single phase clock suffices.  The current foot heights are added too,
        # so the field reads the contact state directly instead of inferring it
        # from joint angles.
        obs = super()._get_obs(pipeline_state, state_info)
        t = state_info["step"] * self.dt
        phi = 2.0 * jnp.pi * t * self._config.gait_cadence
        phase = jnp.array([jnp.sin(phi), jnp.cos(phi)])
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        return jnp.concatenate([obs, phase, z_feet])

    def reset(self, rng: jax.Array) -> State:
        # The parent seeds reward_terms with three zeros for its three-row
        # objective; ours has six, and the rollout scan needs the carried
        # shape to match what step writes back.
        state = super().reset(rng)
        info = {**state.info,
                "reward_terms": jnp.zeros(4),
                "reward_weights": normalize_omega(
                    jnp.asarray(self._config.reward_weights))}
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        if self._config.leg_control == "position":
            joint_targets = self.act2joint(action)
            pipeline_state = self.pipeline_step(state.pipeline_state, joint_targets)
        else:
            pipeline_state = self.pd_pipeline_step(state.pipeline_state, action)
        x, xd = pipeline_state.x, pipeline_state.xd
        obs = self._get_obs(pipeline_state, state.info)

        # command ramp only -- gait tasks run at one fixed forward command
        command_scale = self._command_ramp_scale(state.info["step"])
        state.info["vel_tar"] = state.info["vel_cmd"] * command_scale
        state.info["ang_vel_tar"] = state.info["ang_vel_cmd"] * command_scale

        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        t = state.info["step"] * self.dt
        # one target height profile per gait, all at the shared cadence/amp
        def one_gait(phases):
            return get_foot_step(
                self._config.gait_duty, self._config.gait_cadence,
                self._config.gait_amplitude, phases, t,
            )
        z_tar = jax.vmap(one_gait)(GAIT_PHASES)            # (6, 4)
        gait_rows = -jnp.sum(((z_tar - z_feet[None, :]) / 0.05) ** 2, axis=1)

        # posture + command floor, identical in every row so omega only moves
        # the gait pattern
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        floor = (self._config.posture_floor * (reward_upright * 0.5 + reward_height)
                 + self._config.track_floor * reward_vel)

        reward_components = (
            floor + gait_rows * 0.1 / self._config.gait_scale
        )
        weights = normalize_omega(state.info["reward_weights"])
        reward = jnp.dot(weights, reward_components)

        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)

        # contact bookkeeping the observation helper reads
        foot_contact_z = z_feet - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt = contact | state.info["last_contact"]

        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_tar[GAIT_NAMES.index("trot")]
        state.info["feet_air_time"] += self.dt
        state.info["feet_air_time"] *= ~contact_filt
        state.info["last_contact"] = contact
        state.info["reward_terms"] = reward_components
        state.info["reward_weights"] = weights

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )


brax_envs.register_environment("unitree_go2_gait", UnitreeGo2GaitEnv)
