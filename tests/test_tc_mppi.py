import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.core.tc_mppi import TimeCorrelatedGaussian, difference_operator


class TimeCorrelatedGaussianTest(unittest.TestCase):
    def setUp(self):
        self.sampler = TimeCorrelatedGaussian(
            horizon=5,
            history_length=2,
            dt=0.08,
            derivative_weights=(1.0, 0.0, 1e-4),
        )

    def test_second_difference_operator(self):
        operator = difference_operator(length=4, order=2, dt=1.0)
        np.testing.assert_allclose(
            operator,
            np.array([[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]]),
        )

    def test_covariance_sqrt_matches_inverse_precision(self):
        covariance = self.sampler.covariance_sqrt @ self.sampler.covariance_sqrt.T
        expected = jnp.linalg.inv(self.sampler.tail_precision)
        np.testing.assert_allclose(covariance, expected, rtol=1e-5, atol=1e-6)
        # The derivative cost must actually correlate distinct time nodes.
        off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
        self.assertGreater(float(jnp.max(jnp.abs(off_diagonal))), 0.0)

    def test_conditional_means_and_importance_ratio_are_jittable(self):
        history = jnp.array([[-0.2, 0.1], [-0.1, 0.05]])
        estimate = jnp.linspace(-0.1, 0.3, 10).reshape(5, 2)
        prior, sampling = jax.jit(self.sampler.conditional_means)(history, estimate)
        self.assertEqual(prior.shape, estimate.shape)
        self.assertEqual(sampling.shape, estimate.shape)
        zero_ratio = self.sampler.log_importance_ratio(
            sampling[None], prior, prior, jnp.ones(5)
        )
        np.testing.assert_allclose(zero_ratio, 0.0, atol=1e-7)

    def test_noise_scale_preserves_shape(self):
        epsilon = jnp.ones((7, 5, 3))
        scales = jnp.linspace(0.2, 1.0, 5)
        correlated = self.sampler.transform_standard_normal(epsilon, scales)
        self.assertEqual(correlated.shape, epsilon.shape)


if __name__ == "__main__":
    unittest.main()
