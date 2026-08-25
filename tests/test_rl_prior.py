import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics

from csm.rl_prior import (
    RecoveryGo2PriorEnv,
    RLPriorPolicy,
    make_sac_network_factory,
)
from csm.exact_cli import _load_config
from brax import envs as brax_envs
import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import load_dataclass_from_dict


class RLPriorPolicyTest(unittest.TestCase):
    def _policy(self):
        factory = make_sac_network_factory(
            (8, 8), init_noise_std=0.3, state_dependent_std=True
        )
        networks = factory(5, 2, running_statistics.normalize)
        normalizer = running_statistics.init_state(jnp.zeros(5))
        actor = networks.policy_network.init(jax.random.PRNGKey(0))
        return RLPriorPolicy(
            params=(normalizer, actor),
            observation_size=5,
            action_size=2,
            hidden_layer_sizes=(8, 8),
        )

    def test_tanh_policy_actions_and_log_probability(self):
        policy = self._policy()
        observation = jnp.zeros(5)
        mode = policy.mode(observation)
        sample = policy.sample(observation, jax.random.PRNGKey(1))
        self.assertTrue(bool(jnp.all(jnp.abs(mode) <= 1.0)))
        self.assertTrue(bool(jnp.all(jnp.abs(sample) <= 1.0)))
        self.assertTrue(bool(jnp.isfinite(policy.log_prob(observation, mode))))
        self.assertTrue(
            bool(
                jnp.isfinite(
                    policy.log_prob(observation, jnp.asarray([1.0, -1.0]))
                )
            )
        )

    def test_checkpoint_round_trip_preserves_actor(self):
        policy = self._policy()
        observation = jnp.arange(5, dtype=jnp.float32) / 10.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.pkl"
            policy.save(path)
            restored = RLPriorPolicy.load(path)
        np.testing.assert_allclose(
            restored.mode(observation), policy.mode(observation), rtol=1e-6
        )
        np.testing.assert_allclose(
            restored.log_prob(observation, restored.mode(observation)),
            policy.log_prob(observation, policy.mode(observation)),
            rtol=1e-6,
        )


class RecoveryPriorEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = type(
            "Args", (), {"example": "unitree_go2_trot_randomized", "config": None}
        )()
        config = _load_config(args)
        env_config = load_dataclass_from_dict(
            dial_envs.get_config(config["env_name"]),
            config,
            convert_list_to_array=True,
        )
        cls.env = RecoveryGo2PriorEnv(
            brax_envs.get_environment(config["env_name"], config=env_config)
        )

    def test_reset_modes_cover_healthy_and_fallen_states(self):
        nominal = self.env.reset_with_mode(jax.random.PRNGKey(10), 0)
        fallen = self.env.reset_with_mode(jax.random.PRNGKey(11), 2)
        nominal_health, nominal_recovered, _ = self.env._health(
            nominal.pipeline_state
        )
        fallen_health, fallen_recovered, _ = self.env._health(
            fallen.pipeline_state
        )
        self.assertGreater(float(nominal_health), float(fallen_health))
        self.assertEqual(float(nominal_recovered), 1.0)
        self.assertEqual(float(fallen_recovered), 0.0)

    def test_fall_is_not_terminal_and_reward_is_finite(self):
        state = self.env.reset_with_mode(jax.random.PRNGKey(12), 2)
        next_state = self.env.step(state, jnp.zeros(self.env.action_size))
        self.assertEqual(float(next_state.done), 0.0)
        self.assertTrue(bool(jnp.isfinite(next_state.reward)))


if __name__ == "__main__":
    unittest.main()
