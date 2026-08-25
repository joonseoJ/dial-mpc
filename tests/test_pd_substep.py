"""The per-substep PD loop must only differ where it is supposed to."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax import envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


def _env(timestep, pd_substep):
    config = yaml.safe_load(open(get_example_path("unitree_go2_trot.yaml")))
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(config["env_name"]), config, convert_list_to_array=True
    )
    env_config = dataclasses.replace(
        env_config, timestep=timestep, pd_substep=pd_substep
    )
    return brax_envs.get_environment(config["env_name"], config=env_config)


def _rollout(env, steps=12, seed=0):
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1)
    rewards, positions = [], []
    for i in range(steps):
        action = jnp.sin(jnp.arange(env.action_size) * 0.7 + i * 0.3) * 0.5
        state = step(state, action)
        rewards.append(float(state.reward))
        positions.append(np.asarray(state.pipeline_state.q))
    return np.asarray(rewards), np.asarray(positions)


class PDSubstepTest(unittest.TestCase):
    def test_single_substep_paths_are_identical(self):
        """With n_frames == 1 there is nothing to recompute, so both agree."""
        env_off = _env(0.02, False)
        env_on = _env(0.02, True)
        self.assertEqual(env_off._n_frames, 1)
        self.assertEqual(env_on._n_frames, 1)
        r_off, q_off = _rollout(env_off)
        r_on, q_on = _rollout(env_on)
        np.testing.assert_allclose(r_on, r_off, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(q_on, q_off, rtol=1e-6, atol=1e-6)

    def test_flags_default_off_and_the_example_opts_in_explicitly(self):
        """Other configs keep the original behaviour unless they ask for it."""
        from dial_mpc.config.base_env_config import BaseEnvConfig

        defaults = BaseEnvConfig()
        self.assertFalse(defaults.pd_substep)
        self.assertTrue(defaults.terminate_on_joint_limit)
        self.assertEqual(defaults.timestep, 0.02)

        # The trot example opts into the corrected plant on purpose.
        config = yaml.safe_load(open(get_example_path("unitree_go2_trot.yaml")))
        self.assertTrue(config["pd_substep"])
        self.assertFalse(config["terminate_on_joint_limit"])
        self.assertEqual(config["timestep"], 0.01)

    def test_multi_substep_paths_differ(self):
        """With n_frames > 1 the recomputed torque must actually change things."""
        r_off, q_off = _rollout(_env(0.01, False))
        r_on, q_on = _rollout(_env(0.01, True))
        self.assertEqual(r_off.shape, r_on.shape)
        self.assertGreater(float(np.max(np.abs(q_on - q_off))), 1e-6)
        # ...but stay a small perturbation, not a different simulation.
        self.assertLess(float(np.max(np.abs(q_on - q_off))), 0.5)

    def test_joint_limit_no_longer_ends_the_trot_episode(self):
        """A joint touching its limit is solver slack, not a fall."""
        config = yaml.safe_load(open(get_example_path("unitree_go2_trot.yaml")))
        env_config = load_dataclass_from_dict(
            dial_envs.get_config(config["env_name"]),
            config,
            convert_list_to_array=True,
        )
        self.assertFalse(env_config.terminate_on_joint_limit)
        env = brax_envs.get_environment(config["env_name"], config=env_config)
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        # Drive a joint hard against its upper limit; the episode must survive.
        step = jax.jit(env.step)
        for _ in range(6):
            state = step(state, jnp.ones(env.action_size))
        self.assertEqual(float(state.done), 0.0)

    def test_rollouts_stay_finite(self):
        for timestep, pd in ((0.02, False), (0.01, False), (0.01, True)):
            rewards, positions = _rollout(_env(timestep, pd))
            self.assertTrue(np.all(np.isfinite(rewards)), (timestep, pd))
            self.assertTrue(np.all(np.isfinite(positions)), (timestep, pd))


if __name__ == "__main__":
    unittest.main()
