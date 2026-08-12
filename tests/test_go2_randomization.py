import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
from csm.energy_cli import (
    _dagger_beta_schedule,
    _parse_episode_lengths,
    _resolve_closed_loop_horizons,
    _sample_episode_limit,
    _split_integer,
)
from csm.architectures import CompositionalEnergyMLP
from csm.energy_data import make_closed_loop_windows
from csm.energy_policy import CompositionalEnergyPolicy
from csm.exact_cli import _summarize_rollout_records
from csm.energy_training import (
    _deployment_strata,
    _project_rms,
    _sample_influence,
    _sample_stratified_deployment_indices,
)


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


class _FixedBudgetEnergy:
    action_size = 1
    horizon = 2
    magnitude_head = object()

    def __call__(self, plan, observation):
        del observation
        return jnp.asarray([jnp.sum(jnp.square(plan - 1.0))])

    def predict_update_magnitude(self, plan, observation, omega, maximum):
        del plan, observation, omega, maximum
        return jnp.asarray(0.02)


class _IdentityNormalizer:
    def __call__(self, value, use_running_average=True):
        del use_running_average
        return value


class Go2RandomizationTest(unittest.TestCase):
    def test_magnitude_is_complete_inference_budget(self):
        policy = CompositionalEnergyPolicy(
            model=_FixedBudgetEnergy(),
            normalizer=_IdentityNormalizer(),
            cost_mean=jnp.zeros(1),
            cost_std=jnp.ones(1),
            mode_weights=jnp.ones((1, 1)),
            u_min=-jnp.ones(1),
            u_max=jnp.ones(1),
            num_steps=8,
            step_size=1.0,
            momentum=0.5,
            lock_first_action=False,
            trust_radius=0.05,
            magnitude_is_total_budget=True,
        )
        initial = jnp.zeros((2, 1))
        result = policy.apply(
            initial,
            jnp.zeros(1),
            jax.random.PRNGKey(0),
            omega=jnp.ones(1),
        )
        final_rms = jnp.sqrt(jnp.mean(jnp.square(result - initial)))
        self.assertLessEqual(float(final_rms), 0.02001)
        self.assertGreater(float(final_rms), 0.019)

    def test_bounded_magnitude_head_respects_deployment_radius(self):
        model = CompositionalEnergyMLP(
            action_size=2,
            observation_size=3,
            horizon=2,
            num_objectives=3,
            encoder_hidden=(8, 8),
            head_hidden=(4,),
            magnitude_hidden=(4,),
            rngs=nnx.Rngs(4),
        )
        magnitude = model.predict_update_magnitude(
            jnp.zeros((5, 2, 2)),
            jnp.zeros((5, 3)),
            jnp.tile(jnp.array([[2.0, 2.0, 1.0]]), (5, 1)),
            maximum=0.05,
        )
        self.assertEqual(magnitude.shape, (5,))
        self.assertTrue(bool(jnp.all(magnitude >= 0.0)))
        self.assertTrue(bool(jnp.all(magnitude <= 0.05)))
        self.assertFalse(hasattr(model, "log_update_scale"))

    def test_closed_loop_horizon_curriculum_and_relabel_split(self):
        self.assertEqual(
            _resolve_closed_loop_horizons(5, (4, 4, 6, 8, 12)),
            [4, 4, 6, 8, 12],
        )
        self.assertEqual(
            _resolve_closed_loop_horizons(3, (4, 6), override=None),
            [4, 6, 6],
        )
        self.assertEqual(
            _resolve_closed_loop_horizons(3, (4, 6), override=5),
            [5, 5, 5],
        )
        self.assertEqual(_split_integer(400, 4), [100, 100, 100, 100])
        self.assertEqual(_split_integer(10, 3), [4, 3, 3])

    def test_hard_gradient_influence_is_capped_without_changing_target(self):
        directions = jnp.stack(
            [jnp.ones((2, 3)), 10.0 * jnp.ones((2, 3))]
        )
        influence = _sample_influence(directions, cap=2.0)
        np.testing.assert_allclose(influence, jnp.array([1.0, 0.2]), rtol=1e-5)
        np.testing.assert_allclose(directions[1], 10.0)

    def test_deployment_sampling_balances_magnitude_strata(self):
        updates = jnp.zeros((6, 2, 1))
        updates = updates.at[:, 1, 0].set(
            jnp.array([0.02, 0.03, 0.06, 0.065, 0.08, 0.09])
        )
        strata = _deployment_strata(updates, radius=0.05)
        np.testing.assert_array_equal(strata[0], [0, 1])
        np.testing.assert_array_equal(strata[1], [2, 3])
        np.testing.assert_array_equal(strata[2], [4, 5])
        sampled = np.asarray(
            _sample_stratified_deployment_indices(
                jax.random.PRNGKey(3), strata, batch_size=8
            )
        )
        self.assertEqual(len(sampled), 8)
        self.assertTrue(np.all(np.isin(sampled[:2], strata[0])))
        self.assertTrue(np.all(np.isin(sampled[2:4], strata[1])))
        self.assertTrue(np.all(np.isin(sampled[4:], strata[2])))

    def test_checkpoint_selection_uses_worst_mode_score(self):
        records = []
        for mode, survived_steps in ((0, 100), (0, 100), (1, 20), (1, 40)):
            records.append(
                {
                    "mode": mode,
                    "survived_steps": survived_steps,
                    "fell": survived_steps < 100,
                    "mean_vx": 0.8,
                    "tracking_rmse": 0.0,
                    "mean_tilt": 0.0,
                    "action_jerk": 0.0,
                }
            )
        metrics = _summarize_rollout_records(
            records,
            steps=100,
            minimum_mean_vx=0.3,
            common_seeds=True,
            worst_mode_selection=True,
        )
        self.assertEqual(metrics["selection_aggregation"], "worst_mode")
        self.assertTrue(metrics["selection_common_seeds"])
        self.assertAlmostEqual(metrics["selection_score"], 30.0)
        self.assertAlmostEqual(metrics["mean_survived_steps"], 65.0)

    def test_deployment_update_uses_rms_trust_projection(self):
        update = 4.0 * jnp.ones((2, 3, 4))
        projected = _project_rms(update, radius=0.05)
        rms = jnp.sqrt(jnp.mean(jnp.square(projected), axis=(-2, -1)))
        np.testing.assert_allclose(rms, jnp.full((2,), 0.05), rtol=1e-5)

    def test_closed_loop_windows_pad_only_after_rollout_end(self):
        warm = jnp.arange(3 * 2 * 2).reshape(3, 2, 2)
        observations = jnp.arange(3 * 5).reshape(3, 5)
        omegas = jnp.tile(jnp.array([[2.0, 2.0, 1.0]]), (3, 1))
        teachers = warm + 1
        windows = make_closed_loop_windows(
            warm, observations, omegas, teachers, sequence_length=2
        )
        self.assertEqual(windows.observations.shape, (3, 2, 5))
        np.testing.assert_allclose(windows.valid, [[1, 1], [1, 1], [1, 0]])
        np.testing.assert_allclose(windows.observations[2, 1], observations[2])

    def test_short_long_episode_pool_and_student_only_schedule(self):
        lengths = _parse_episode_lengths("80,80,400")
        self.assertEqual(lengths, (80, 80, 400))
        rng = jax.random.PRNGKey(19)
        sampled = []
        for _ in range(64):
            rng, limit = _sample_episode_limit(rng, lengths)
            sampled.append(limit)
        self.assertEqual(set(sampled), {80, 400})
        self.assertEqual(
            _dagger_beta_schedule(5, 0.5, 2),
            [0.5, 0.5, 0.5, 0.0, 0.0],
        )

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
        config = UnitreeGo2EnvConfig(randomize_start_state=False)
        self.assertFalse(config.include_foot_height_observation)
        env = _bare_env(config)
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
