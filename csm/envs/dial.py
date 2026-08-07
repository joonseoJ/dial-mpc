"""Built-in CSM objective decompositions for DIAL-MPC environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brax import math

from csm.data_collection import scalar_reward_cost
from csm.envs.base import DIALObjectiveSpec
from dial_mpc.utils.function_utils import global_to_body_velocity


GO2_TROT_OBJECTIVES = ("tracking", "stability", "gait")


def make_go2_trot_cost(env):
    """Build the three-component cost used by the Go2 trot CSM example."""

    torso_idx = int(env._torso_idx) - 1

    def cost(state, action):
        del action
        pipeline = state.pipeline_state
        x, xd = pipeline.x, pipeline.xd
        vel_target = state.info["vel_tar"]
        ang_vel_target = state.info["ang_vel_tar"]

        body_vel = global_to_body_velocity(xd.vel[torso_idx], x.rot[torso_idx])
        body_ang_vel = global_to_body_velocity(
            xd.ang[torso_idx] * jnp.pi / 180.0, x.rot[torso_idx]
        )
        tracking = (
            jnp.sum(jnp.square(body_vel[:2] - vel_target[:2]))
            + jnp.square(body_ang_vel[2] - ang_vel_target[2])
        )

        up = jnp.array([0.0, 0.0, 1.0])
        upright = jnp.sum(jnp.square(math.rotate(up, x.rot[0]) - up))
        yaw_target = (
            state.info["yaw_tar"]
            + ang_vel_target[2] * env.dt * (state.info["step"] - 1)
        )
        yaw = math.quat_to_euler(x.rot[torso_idx])[2]
        delta_yaw = jnp.atan2(jnp.sin(yaw - yaw_target), jnp.cos(yaw - yaw_target))
        height = jnp.square(x.pos[torso_idx, 2] - state.info["pos_tar"][2])
        stability = 0.5 * upright + 0.3 * jnp.square(delta_yaw) + height

        gait = 0.1 * jnp.sum(
            jnp.square((state.info["z_feet_tar"] - state.info["z_feet"]) / 0.05)
        )
        return jnp.stack([tracking, stability, gait])

    return cost


def get_objective_spec(env) -> DIALObjectiveSpec:
    """Return a built-in objective decomposition for a DIAL environment.

    Unknown environments remain fully usable through their scalar reward.
    Custom environments can instead pass their own ``objective_fn`` directly
    to :class:`csm.data_collection.DIALScoreCollector`.
    """

    if type(env).__name__ == "UnitreeGo2Env":
        return DIALObjectiveSpec(GO2_TROT_OBJECTIVES, make_go2_trot_cost(env))
    return DIALObjectiveSpec(("negative_reward",), scalar_reward_cost)


__all__ = ["GO2_TROT_OBJECTIVES", "get_objective_spec", "make_go2_trot_cost"]
