"""Maximum-entropy locomotion prior used by Residual-MPPI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cloudpickle
import jax
import jax.numpy as jnp
from brax import math
from brax.envs.base import State, Wrapper
from brax.training.acme import running_statistics
from brax.training.agents.sac import networks as sac_networks

from dial_mpc.utils.function_utils import global_to_body_velocity


@dataclass(frozen=True)
class RobustPriorRewardConfig:
    """Reward scales for a broad, stable command-conditioned locomotion prior."""

    alive: float = 4.0
    tracking: float = 1.0
    upright: float = 1.0
    height: float = 0.5
    gait: float = 0.25
    smoothness: float = 0.05
    action: float = 0.01
    fall: float = 10.0
    tracking_temperature: float = 0.25
    upright_temperature: float = 0.25
    height_temperature: float = 0.01
    gait_temperature: float = 1.0


@dataclass(frozen=True)
class RecoveryPriorConfig:
    """State distribution and reward for a locomotion-compatible safety prior."""

    nominal_reset_probability: float = 0.25
    tilted_reset_probability: float = 0.25
    fallen_reset_probability: float = 0.50
    tilted_angle_degrees: float = 60.0
    fallen_angle_jitter_degrees: float = 15.0
    fallen_angle_scale: float = 1.0
    fallen_pose_start: int = 0
    fallen_pose_count: int = 10
    # Spawn disturbed poses above the ground and let them make contact under
    # physics.  Initializing a rotated nominal pose at standing torso height
    # or lower can place a leg/body geom below the floor and create an invalid
    # recovery state.
    tilted_height: float = 0.38
    fallen_height: float = 0.38
    reset_joint_noise: float = 0.12
    reset_linear_velocity: float = 0.25
    reset_angular_velocity: float = 0.5
    reset_joint_velocity: float = 0.5
    push_interval_steps: int = 125
    push_probability: float = 0.75
    push_linear_velocity: float = 0.75
    push_angular_velocity: float = 1.5
    health: float = 2.0
    health_progress: float = 8.0
    recovered: float = 4.0
    recovery_event: float = 12.0
    tracking: float = 2.0
    gait: float = 0.5
    smoothness: float = 0.03
    action: float = 0.005
    joint_limit: float = 1.0
    tracking_temperature: float = 0.25
    gait_temperature: float = 1.0
    recovered_up_threshold: float = 0.75
    recovered_height_threshold: float = 0.23


class RecoveryGo2PriorEnv(Wrapper):
    """Go2 task that continues through falls and explicitly learns recovery.

    Reset modes are sampled independently for every vectorized environment:
    nominal locomotion, a large attitude disturbance, or a prone/supine/side
    pose.  A fall never terminates an episode, because doing so would remove
    precisely the transitions required to learn how to stand up again.
    """

    metric_names = (
        "prior_health",
        "prior_health_progress",
        "prior_recovered",
        "prior_recovery_event",
        "prior_tracking",
        "prior_gait",
        "prior_smoothness",
        "prior_action",
        "prior_joint_limit",
        "prior_reset_mode",
    )

    def __init__(self, env, config: RecoveryPriorConfig | None = None):
        super().__init__(env)
        self.recovery_config = config or RecoveryPriorConfig()
        self._prior_torso_idx = int(env._torso_idx) - 1

    def _health(self, pipeline_state) -> tuple[jax.Array, jax.Array, jax.Array]:
        torso = self._prior_torso_idx
        up = jnp.asarray([0.0, 0.0, 1.0])
        up_dot = jnp.dot(math.rotate(up, pipeline_state.x.rot[torso]), up)
        upright = jnp.clip((up_dot + 1.0) * 0.5, 0.0, 1.0)
        height = pipeline_state.x.pos[torso, 2]
        height_score = jax.nn.sigmoid((height - 0.18) / 0.035)
        health = 0.65 * upright + 0.35 * height_score
        recovered = (up_dot > self.recovery_config.recovered_up_threshold) & (
            height > self.recovery_config.recovered_height_threshold
        )
        return health, recovered.astype(jnp.float32), up_dot

    def _sample_mode(self, key: jax.Array) -> jax.Array:
        config = self.recovery_config
        probabilities = jnp.asarray(
            [
                config.nominal_reset_probability,
                config.tilted_reset_probability,
                config.fallen_reset_probability,
            ]
        )
        probabilities = probabilities / jnp.sum(probabilities)
        return jax.random.choice(key, 3, p=probabilities)

    def reset(self, rng: jax.Array) -> State:
        rng, mode_key = jax.random.split(rng)
        return self.reset_with_mode(rng, self._sample_mode(mode_key))

    def reset_with_mode(
        self,
        rng: jax.Array,
        mode: jax.Array | int,
        fallen_index_override: jax.Array | int | None = None,
    ) -> State:
        """Resets to mode 0=nominal, 1=tilted, or 2=fallen."""
        keys = jax.random.split(rng, 9)
        state = self.env.reset(keys[0])
        mode = jnp.asarray(mode, dtype=jnp.int32)
        config = self.recovery_config

        nominal_qpos = state.pipeline_state.qpos
        qpos = jnp.asarray(self.env._init_q)
        qvel = jnp.zeros_like(state.pipeline_state.qvel)

        tilted_rpy = jax.random.uniform(keys[1], (3,), minval=-1.0, maxval=1.0)
        tilted_rpy = tilted_rpy.at[2].set(
            jax.random.uniform(keys[2], (), minval=-180.0, maxval=180.0)
        )
        tilted_rpy = tilted_rpy.at[:2].multiply(config.tilted_angle_degrees)

        fallen_count = max(min(int(config.fallen_pose_count), 10), 1)
        fallen_start = max(min(int(config.fallen_pose_start), 10 - fallen_count), 0)
        fallen_index = jax.random.randint(
            keys[3], (), fallen_start, fallen_start + fallen_count
        )
        if fallen_index_override is not None:
            fallen_index = jnp.asarray(fallen_index_override, dtype=jnp.int32)
        fallen_attitudes = jnp.asarray(
            [
                [90.0, 0.0, 0.0],
                [-90.0, 0.0, 0.0],
                [180.0, 0.0, 0.0],
                [0.0, 90.0, 0.0],
                [0.0, -90.0, 0.0],
                [180.0, 30.0, 0.0],
                # Repeat the two upside-down families so hard recovery states
                # occupy 60% of the default fallen-reset distribution.
                [180.0, 0.0, 0.0],
                [180.0, 30.0, 0.0],
                [180.0, 0.0, 0.0],
                [180.0, 30.0, 0.0],
            ]
        )
        fallen_rpy = fallen_attitudes[fallen_index] * config.fallen_angle_scale
        fallen_rpy = fallen_rpy + jax.random.uniform(
            keys[4], (3,), minval=-1.0, maxval=1.0
        ) * config.fallen_angle_jitter_degrees
        fallen_rpy = fallen_rpy.at[2].set(
            jax.random.uniform(keys[5], (), minval=-180.0, maxval=180.0)
        )

        rpy = jnp.where(mode == 1, tilted_rpy, fallen_rpy)
        rotated = math.quat_mul(math.euler_to_quat(rpy), qpos[3:7])
        qpos = qpos.at[3:7].set(rotated)
        reset_height = jnp.where(
            mode == 1, config.tilted_height, config.fallen_height
        )
        qpos = qpos.at[2].set(
            reset_height
            + jax.random.uniform(keys[6], (), minval=-0.015, maxval=0.015)
        )
        joint_noise = jax.random.uniform(
            keys[7], (self.action_size,), minval=-1.0, maxval=1.0
        ) * config.reset_joint_noise
        joints = jnp.clip(
            qpos[7:] + joint_noise,
            self.env.joint_range[:, 0] + 1e-3,
            self.env.joint_range[:, 1] - 1e-3,
        )
        qpos = qpos.at[7:].set(joints)
        velocity_noise = jax.random.uniform(
            keys[8], qvel.shape, minval=-1.0, maxval=1.0
        )
        velocity_scale = jnp.concatenate(
            [
                jnp.full(3, config.reset_linear_velocity),
                jnp.full(3, config.reset_angular_velocity),
                jnp.full(self.action_size, config.reset_joint_velocity),
            ]
        )
        qvel = velocity_noise * velocity_scale

        qpos = jnp.where(mode == 0, nominal_qpos, qpos)
        qvel = jnp.where(mode == 0, state.pipeline_state.qvel, qvel)
        pipeline_state = self.env.pipeline_init(qpos, qvel)
        health, recovered, _ = self._health(pipeline_state)
        info = {
            **state.info,
            "prior_rng": keys[8],
            "prior_last_action": jnp.zeros(self.action_size),
            "prior_previous_health": health,
            "prior_was_recovered": recovered,
            "prior_reset_mode": mode.astype(jnp.float32),
            "prior_fallen_index": fallen_index.astype(jnp.float32),
        }
        obs = self.env._get_obs(pipeline_state, info)
        metrics = {
            **state.metrics,
            **{name: jnp.asarray(0.0) for name in self.metric_names},
            "prior_reset_mode": mode.astype(jnp.float32),
        }
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.asarray(0.0),
            done=jnp.asarray(0.0),
            info=info,
            metrics=metrics,
        )

    def _apply_push(self, state: State) -> State:
        config = self.recovery_config
        rng, event_key, velocity_key = jax.random.split(state.info["prior_rng"], 3)
        interval = max(int(config.push_interval_steps), 1)
        at_boundary = (state.info["step"] > 0) & (
            state.info["step"] % interval == 0
        )
        apply_push = at_boundary & (
            jax.random.uniform(event_key) < config.push_probability
        )
        delta = jax.random.uniform(
            velocity_key, state.pipeline_state.qvel.shape, minval=-1.0, maxval=1.0
        )
        scale = jnp.concatenate(
            [
                jnp.full(3, config.push_linear_velocity),
                jnp.full(3, config.push_angular_velocity),
                jnp.zeros(self.action_size),
            ]
        )
        qvel = state.pipeline_state.qvel + delta * scale
        qvel = jnp.where(apply_push, qvel, state.pipeline_state.qvel)
        pipeline_state = state.pipeline_state.replace(qvel=qvel)
        return state.replace(
            pipeline_state=pipeline_state,
            info={**state.info, "prior_rng": rng},
        )

    def step(self, state: State, action: jax.Array) -> State:
        config = self.recovery_config
        previous_action = state.info["prior_last_action"]
        previous_health = state.info["prior_previous_health"]
        was_recovered = state.info["prior_was_recovered"]
        state = self._apply_push(state)
        next_state = self.env.step(state, action)
        pipeline = next_state.pipeline_state
        torso = self._prior_torso_idx
        health, recovered, _ = self._health(pipeline)

        body_velocity = global_to_body_velocity(
            pipeline.xd.vel[torso], pipeline.x.rot[torso]
        )
        body_angular_velocity = global_to_body_velocity(
            pipeline.xd.ang[torso] * jnp.pi / 180.0, pipeline.x.rot[torso]
        )
        tracking_error = (
            jnp.sum(jnp.square(body_velocity[:2] - next_state.info["vel_tar"][:2]))
            + jnp.square(
                body_angular_velocity[2] - next_state.info["ang_vel_tar"][2]
            )
        )
        gait_error = jnp.mean(
            jnp.square(
                (next_state.info["z_feet_tar"] - next_state.info["z_feet"]) / 0.05
            )
        )
        recovery_event = recovered * (1.0 - was_recovered)
        joint_angles = pipeline.qpos[7:]
        lower_violation = jnp.maximum(self.env.joint_range[:, 0] - joint_angles, 0.0)
        upper_violation = jnp.maximum(joint_angles - self.env.joint_range[:, 1], 0.0)
        joint_limit_cost = jnp.mean(jnp.square(lower_violation + upper_violation))

        terms = {
            "prior_health": config.health * health,
            "prior_health_progress": config.health_progress
            * (health - previous_health),
            "prior_recovered": config.recovered * recovered,
            "prior_recovery_event": config.recovery_event * recovery_event,
            "prior_tracking": config.tracking
            * recovered
            * jnp.exp(-tracking_error / config.tracking_temperature),
            "prior_gait": config.gait
            * recovered
            * jnp.exp(-gait_error / config.gait_temperature),
            "prior_smoothness": -config.smoothness
            * jnp.mean(jnp.square(action - previous_action)),
            "prior_action": -config.action * jnp.mean(jnp.square(action)),
            "prior_joint_limit": -config.joint_limit * joint_limit_cost,
            "prior_reset_mode": next_state.info["prior_reset_mode"],
        }
        reward = sum(
            value for name, value in terms.items() if name != "prior_reset_mode"
        )
        info = {
            **next_state.info,
            "prior_last_action": action,
            "prior_previous_health": health,
            "prior_was_recovered": recovered,
        }
        return next_state.replace(
            reward=reward,
            # Do not terminate on a fall: the subsequent transitions are the
            # recovery supervision. EpisodeWrapper still supplies truncation.
            done=jnp.asarray(0.0),
            info=info,
            metrics={**next_state.metrics, **terms},
        )


class RobustGo2PriorEnv(Wrapper):
    """Reweights Go2 for survival-first but non-degenerate locomotion.

    Survival, uprightness and recovery dominate the objective, while a modest
    command and gait reward keeps the stochastic policy's support over useful
    walking actions instead of allowing the trivial stationary solution.
    """

    metric_names = (
        "prior_alive",
        "prior_tracking",
        "prior_upright",
        "prior_height",
        "prior_gait",
        "prior_smoothness",
        "prior_action",
        "prior_fall",
    )

    def __init__(self, env, reward_config: RobustPriorRewardConfig | None = None):
        super().__init__(env)
        self.reward_config = reward_config or RobustPriorRewardConfig()
        self._prior_torso_idx = int(env._torso_idx) - 1

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        info = {**state.info, "prior_last_action": jnp.zeros(self.action_size)}
        metrics = {
            **state.metrics,
            **{name: jnp.asarray(0.0) for name in self.metric_names},
        }
        return state.replace(info=info, metrics=metrics)

    def step(self, state: State, action: jax.Array) -> State:
        previous_action = state.info["prior_last_action"]
        next_state = self.env.step(state, action)
        config = self.reward_config
        pipeline = next_state.pipeline_state
        x, xd = pipeline.x, pipeline.xd
        torso = self._prior_torso_idx

        body_velocity = global_to_body_velocity(xd.vel[torso], x.rot[torso])
        body_angular_velocity = global_to_body_velocity(
            xd.ang[torso] * jnp.pi / 180.0, x.rot[torso]
        )
        tracking_error = (
            jnp.sum(
                jnp.square(body_velocity[:2] - next_state.info["vel_tar"][:2])
            )
            + jnp.square(
                body_angular_velocity[2] - next_state.info["ang_vel_tar"][2]
            )
        )
        up = jnp.asarray([0.0, 0.0, 1.0])
        upright_error = jnp.sum(jnp.square(math.rotate(up, x.rot[torso]) - up))
        height_error = jnp.square(
            x.pos[torso, 2] - next_state.info["pos_tar"][2]
        )
        gait_error = jnp.mean(
            jnp.square(
                (next_state.info["z_feet_tar"] - next_state.info["z_feet"])
                / 0.05
            )
        )

        alive = 1.0 - next_state.done
        tracking = jnp.exp(-tracking_error / config.tracking_temperature)
        upright = jnp.exp(-upright_error / config.upright_temperature)
        height = jnp.exp(-height_error / config.height_temperature)
        gait = jnp.exp(-gait_error / config.gait_temperature)
        smoothness_cost = jnp.mean(jnp.square(action - previous_action))
        action_cost = jnp.mean(jnp.square(action))
        fall = next_state.done

        terms = {
            "prior_alive": config.alive * alive,
            "prior_tracking": config.tracking * tracking,
            "prior_upright": config.upright * upright,
            "prior_height": config.height * height,
            "prior_gait": config.gait * gait,
            "prior_smoothness": -config.smoothness * smoothness_cost,
            "prior_action": -config.action * action_cost,
            "prior_fall": -config.fall * fall,
        }
        reward = sum(terms.values(), jnp.asarray(0.0))
        next_last_action = jnp.where(next_state.done, jnp.zeros_like(action), action)
        info = {**next_state.info, "prior_last_action": next_last_action}
        metrics = {**next_state.metrics, **terms}
        return next_state.replace(reward=reward, info=info, metrics=metrics)


@dataclass
class RLPriorPolicy:
    """Portable SAC actor with exact tanh-Gaussian likelihood evaluation."""

    params: object
    observation_size: int
    action_size: int
    hidden_layer_sizes: tuple[int, ...] = (512, 512, 256)
    init_noise_std: float = 0.3
    state_dependent_std: bool = True
    layer_norm: bool = True
    action_epsilon: float = 1e-6

    def _networks(self):
        return sac_networks.make_sac_networks(
            self.observation_size,
            self.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=jax.nn.swish,
            distribution_type="tanh_normal",
            init_noise_std=self.init_noise_std,
            state_dependent_std=self.state_dependent_std,
            policy_network_layer_norm=self.layer_norm,
            q_network_layer_norm=self.layer_norm,
        )

    def distribution_parameters(self, observation: jax.Array) -> jax.Array:
        networks = self._networks()
        return networks.policy_network.apply(*self.params, observation)

    def mode(self, observation: jax.Array) -> jax.Array:
        networks = self._networks()
        logits = networks.policy_network.apply(*self.params, observation)
        return networks.parametric_action_distribution.mode(logits)

    def sample(self, observation: jax.Array, rng: jax.Array) -> jax.Array:
        networks = self._networks()
        logits = networks.policy_network.apply(*self.params, observation)
        return networks.parametric_action_distribution.sample(logits, rng)

    def log_prob(self, observation: jax.Array, action: jax.Array) -> jax.Array:
        """Returns log pi(action|observation), including the tanh Jacobian."""
        networks = self._networks()
        logits = networks.policy_network.apply(*self.params, observation)
        bounded = jnp.clip(
            action, -1.0 + self.action_epsilon, 1.0 - self.action_epsilon
        )
        raw_action = networks.parametric_action_distribution.inverse_postprocess(
            bounded
        )
        return networks.parametric_action_distribution.log_prob(logits, raw_action)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            cloudpickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "RLPriorPolicy":
        with Path(path).open("rb") as stream:
            policy = cloudpickle.load(stream)
        if not isinstance(policy, cls):
            raise TypeError("checkpoint is not an RL prior policy")
        return policy


@dataclass
class SafetyCritic:
    """SAC Q-function used as a survival safety critic for Residual-MPPI.

    Evaluates Q_survival(s, a) — the expected discounted future survival reward.
    Higher Q means the action keeps the robot alive longer.  Uses the
    conservative estimate (min of twin critics).
    """

    q_params: object  # (normalizer_params, target_q_network_params)
    observation_size: int
    action_size: int
    hidden_layer_sizes: tuple[int, ...] = (512, 512, 256)
    layer_norm: bool = True

    def _q_network(self):
        networks = sac_networks.make_sac_networks(
            self.observation_size,
            self.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=jax.nn.swish,
            q_network_layer_norm=self.layer_norm,
        )
        return networks.q_network

    def q_value(self, observation: jax.Array, action: jax.Array) -> jax.Array:
        """Returns conservative Q(s, a) — min across twin critics."""
        q_net = self._q_network()
        q_values = q_net.apply(*self.q_params, observation, action)
        return jnp.min(q_values, axis=-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            cloudpickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "SafetyCritic":
        with Path(path).open("rb") as stream:
            critic = cloudpickle.load(stream)
        if not isinstance(critic, cls):
            raise TypeError("checkpoint is not a SafetyCritic")
        return critic


def make_sac_network_factory(
    hidden_layer_sizes: Sequence[int] = (512, 512, 256),
    init_noise_std: float = 0.3,
    state_dependent_std: bool = True,
    layer_norm: bool = True,
):
    """Creates the network factory shared by SAC training and saved actors."""

    hidden_layer_sizes = tuple(int(size) for size in hidden_layer_sizes)

    def factory(observation_size, action_size, preprocess_observations_fn):
        return sac_networks.make_sac_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=jax.nn.swish,
            distribution_type="tanh_normal",
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            policy_network_layer_norm=layer_norm,
            q_network_layer_norm=layer_norm,
        )

    return factory
