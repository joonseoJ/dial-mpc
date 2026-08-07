import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig


def _bare_env(config):
    env = object.__new__(UnitreeGo2Env)
    env._config = config
    env._n_frames = 1
    env.sys = SimpleNamespace(
        opt=SimpleNamespace(timestep=0.02), act_size=lambda: 12
    )
    env._nv = 18
    env._init_q = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]), jnp.zeros(12)]
    )
    env.joint_range = jnp.tile(jnp.array([[-1.0, 1.0]]), (12, 1))
    return env


class Go2RandomizationTest(unittest.TestCase):
    def test_command_samples_respect_configured_bounds(self):
        config = UnitreeGo2EnvConfig(
            command_vx_min=0.4,
            command_vx_max=1.1,
            command_vy_min=-0.2,
            command_vy_max=0.2,
            command_vyaw_min=-0.5,
            command_vyaw_max=0.5,
        )
        env = _bare_env(config)
        keys = jax.random.split(jax.random.PRNGKey(7), 128)
        linear, angular = jax.vmap(env.sample_command)(keys)
        self.assertTrue(bool(jnp.all((linear[:, 0] >= 0.4) & (linear[:, 0] <= 1.1))))
        self.assertTrue(bool(jnp.all((linear[:, 1] >= -0.2) & (linear[:, 1] <= 0.2))))
        self.assertTrue(bool(jnp.all((angular[:, 2] >= -0.5) & (angular[:, 2] <= 0.5))))
        np.testing.assert_allclose(np.asarray(linear[:, 2]), 0.0)
        np.testing.assert_allclose(np.asarray(angular[:, :2]), 0.0)

    def test_randomized_reset_state_is_bounded_and_reproducible(self):
        config = UnitreeGo2EnvConfig(
            randomize_start_state=True,
            start_xy_noise=0.05,
            start_height_noise=0.015,
            start_rpy_noise=0.05,
            start_joint_position_noise=0.08,
            start_body_linear_velocity_noise=0.15,
            start_body_angular_velocity_noise=0.15,
            start_joint_velocity_noise=0.30,
        )
        env = _bare_env(config)
        key = jax.random.PRNGKey(11)
        qpos_a, qvel_a = env._sample_initial_state(key)
        qpos_b, qvel_b = env._sample_initial_state(key)
        np.testing.assert_allclose(qpos_a, qpos_b)
        np.testing.assert_allclose(qvel_a, qvel_b)
        self.assertGreater(float(jnp.linalg.norm(qpos_a - env._init_q)), 0.0)
        self.assertGreater(float(jnp.linalg.norm(qvel_a)), 0.0)
        self.assertLessEqual(float(jnp.max(jnp.abs(qpos_a[:2]))), 0.05)
        self.assertLessEqual(float(jnp.abs(qpos_a[2] - 0.3)), 0.015)
        self.assertTrue(bool(jnp.all(qpos_a[7:] > -1.0)))
        self.assertTrue(bool(jnp.all(qpos_a[7:] < 1.0)))

    def test_disabled_reset_randomization_preserves_nominal_state(self):
        env = _bare_env(UnitreeGo2EnvConfig(randomize_start_state=False))
        qpos, qvel = env._sample_initial_state(jax.random.PRNGKey(3))
        np.testing.assert_allclose(qpos, env._init_q)
        np.testing.assert_allclose(qvel, 0.0)

    def test_command_ramp_handles_positive_and_negative_commands_symmetrically(self):
        env = _bare_env(UnitreeGo2EnvConfig(ramp_up_time=1.0))
        scale = env._command_ramp_scale(25)
        self.assertAlmostEqual(float(scale), 0.5, places=6)
        command = jnp.array([0.8, -0.2, 0.0])
        np.testing.assert_allclose(command * scale, jnp.array([0.4, -0.1, 0.0]))


if __name__ == "__main__":
    unittest.main()
