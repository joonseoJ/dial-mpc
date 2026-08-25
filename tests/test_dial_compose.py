"""Composition of independently trained basis score fields."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from csm.architectures import StandardNormalizer
from csm.dial_score import (
    ComposedDialScorePolicy,
    DialScoreMLP,
    DialScorePolicy,
    dial_factors,
    dial_sigma_control,
    factor_to_t,
)

HORIZON, ACTION_SIZE, OBS_SIZE = 5, 3, 7
BASIS = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0]])


def _policy(seed):
    sigma_control = dial_sigma_control(0.9, HORIZON)
    factors = dial_factors(0.5, 2)
    model = DialScoreMLP(
        action_size=ACTION_SIZE,
        observation_size=OBS_SIZE,
        horizon=HORIZON,
        sigma_control=sigma_control,
        hidden=(32, 32),
        rngs=nnx.Rngs(seed),
        factor_min=float(jnp.min(factors)),
        factor_max=float(jnp.max(factors)),
    )
    return DialScorePolicy(
        model=model,
        normalizer=StandardNormalizer(OBS_SIZE),
        factors=factors,
        shift_matrix=jnp.eye(HORIZON),
    )


def _composed():
    return ComposedDialScorePolicy(
        policies=tuple(_policy(seed) for seed in range(len(BASIS))),
        mode_weights=jnp.asarray(BASIS),
        pinv_mode_weights=jnp.asarray(np.linalg.pinv(BASIS)),
    )


class CompositionCoefficientTest(unittest.TestCase):
    def test_basis_rows_select_themselves(self):
        policy = _composed()
        for row, omega in enumerate(BASIS):
            c = np.asarray(policy.coefficients(jnp.asarray(omega)))
            expected = np.zeros(len(BASIS))
            expected[row] = 1.0
            np.testing.assert_allclose(c, expected, atol=1e-5)

    def test_coefficients_sum_to_one(self):
        policy = _composed()
        for omega in ([1.0, 1.0, 1.0], [2.0, 1.0, 1.5], [5.0, 2.0, 1.0]):
            c = np.asarray(policy.coefficients(jnp.asarray(omega)))
            np.testing.assert_allclose(float(c.sum()), 1.0, rtol=1e-5)

    def test_composition_is_scale_invariant_like_the_teacher(self):
        """DIAL's update ignores the scale of omega, so composition must too."""
        policy = _composed()
        base = np.asarray(policy.coefficients(jnp.asarray([1.0, 2.0, 3.0])))
        for alpha in (0.1, 5.0, 100.0):
            scaled = np.asarray(
                policy.coefficients(jnp.asarray([alpha, 2 * alpha, 3 * alpha]))
            )
            np.testing.assert_allclose(scaled, base, rtol=1e-4)

    def test_normalisation_changes_magnitude_not_direction(self):
        """The raw least-squares solution points the right way but is mis-scaled.

        ``(2,2,2)`` asks for exactly the behaviour of ``(1,1,1)`` -- the teacher
        ignores the scale of omega -- yet least squares returns coefficients
        twice as large, which would double the applied update.
        """
        pinv = np.linalg.pinv(BASIS)
        omega = np.array([2.0, 2.0, 2.0])
        raw = omega @ pinv
        self.assertAlmostEqual(float(raw.sum()), 2.0, places=4)
        policy = _composed()
        normalised = np.asarray(policy.coefficients(jnp.asarray(omega)))
        # A pure rescale by 1/sum: direction untouched, magnitude corrected.
        # Comparing this way stays valid when a coefficient is exactly zero.
        np.testing.assert_allclose(normalised * raw.sum(), raw, atol=1e-5)
        self.assertNotAlmostEqual(float(raw.sum()), 1.0, places=3)


class ComposedApplyTest(unittest.TestCase):
    def test_apply_matches_the_manual_composition(self):
        policy = _composed()
        omega = jnp.asarray([2.0, 1.0, 1.5])
        obs = jax.random.normal(jax.random.PRNGKey(3), (OBS_SIZE,))
        plan = jnp.zeros((HORIZON, ACTION_SIZE))
        c = policy.coefficients(omega)
        model = policy.policies[0].model

        manual = plan
        for factor in np.asarray(policy.policies[0].factors):
            t = factor_to_t(factor, model.factor_min, model.factor_max).reshape(1)
            sigma_sq = jnp.square(model.sigma(t))
            score = sum(
                c[i] * p.model.score(manual, p.normalizer(obs), t)
                for i, p in enumerate(policy.policies)
            )
            manual = jnp.clip(manual + sigma_sq * score, -1.0, 1.0)
        np.testing.assert_allclose(
            np.asarray(policy.apply(plan, obs, omega)),
            np.asarray(manual),
            atol=1e-6,
        )

    def test_at_a_basis_row_it_reduces_to_that_single_field(self):
        """Composition must not perturb a field that is used on its own."""
        policy = _composed()
        obs = jax.random.normal(jax.random.PRNGKey(4), (OBS_SIZE,))
        plan = jax.random.uniform(
            jax.random.PRNGKey(5), (HORIZON, ACTION_SIZE), minval=-0.5, maxval=0.5
        )
        for row, omega in enumerate(BASIS):
            np.testing.assert_allclose(
                np.asarray(policy.apply(plan, obs, jnp.asarray(omega))),
                np.asarray(policy.policies[row].apply(plan, obs)),
                atol=1e-5,
            )

    def test_output_stays_bounded_and_locks_node_zero(self):
        policy = _composed()
        obs = jax.random.normal(jax.random.PRNGKey(6), (OBS_SIZE,))
        plan = jax.random.uniform(
            jax.random.PRNGKey(7), (HORIZON, ACTION_SIZE), minval=-1.0, maxval=1.0
        )
        out = policy.apply(plan, obs, jnp.asarray([1.0, 1.0, 1.0]), passes=3)
        self.assertLessEqual(float(jnp.max(jnp.abs(out))), 1.0 + 1e-6)
        np.testing.assert_allclose(
            np.asarray(out[0]), np.asarray(plan[0]), atol=1e-6
        )

    def test_jit_and_pickle_round_trip(self):
        policy = _composed()
        obs = jnp.zeros((OBS_SIZE,))
        plan = jnp.zeros((HORIZON, ACTION_SIZE))
        omega = jnp.asarray([1.0, 1.0, 1.0])
        expected = policy.apply(plan, obs, omega)
        jitted = jax.jit(lambda p, o, w: policy.apply(p, o, w))
        np.testing.assert_allclose(
            np.asarray(jitted(plan, obs, omega)), np.asarray(expected), atol=1e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "composed.pkl"
            policy.save(path)
            loaded = ComposedDialScorePolicy.load(path)
        np.testing.assert_allclose(
            np.asarray(loaded.apply(plan, obs, omega)),
            np.asarray(expected),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()


class SelectionAndOmegaSamplingTest(unittest.TestCase):
    """On-policy checkpoint selection and per-episode omega draws."""

    def _data(self, n=512, seed=0):
        from csm.dial_score import DialScoreData, dial_sigma_control
        sigma = dial_sigma_control(0.9, HORIZON)
        factors = dial_factors(0.5, 2)
        rng = jax.random.PRNGKey(seed)
        rng, u_rng, o_rng, l_rng = jax.random.split(rng, 4)
        u = jax.random.uniform(u_rng, (n, HORIZON, ACTION_SIZE), minval=-1, maxval=1)
        obs = jax.random.normal(o_rng, (n, OBS_SIZE))
        level = jax.random.randint(l_rng, (n,), 0, 2)
        delta = (factors[level][:, None, None]
                 * sigma.reshape(1, HORIZON, 1) * (0.2 - u))
        delta = delta.at[:, 0].set(0.0)
        return DialScoreData(u=u, factor=factors[level], level=level,
                             delta=delta, obs=obs)

    def test_selection_fn_drives_checkpoint_choice(self):
        """The kept weights follow the selection score, not validation loss."""
        import optax
        from csm.architectures import StandardNormalizer
        from csm.dial_score import DialScoreMLP, dial_sigma_control, fit_dial_score

        data = self._data()
        normalizer = StandardNormalizer(OBS_SIZE)
        normalizer.fit(data.obs)
        model = DialScoreMLP(
            action_size=ACTION_SIZE, observation_size=OBS_SIZE, horizon=HORIZON,
            sigma_control=dial_sigma_control(0.9, HORIZON), hidden=(32, 32),
            rngs=nnx.Rngs(0), factor_min=0.5, factor_max=1.0)

        seen = []
        # Best score lands on the first call, so the final weights must be the
        # ones snapshotted there rather than the last (better-fitting) ones.
        scores = [5.0, 1.0, 1.0, 1.0]

        def selection(step):
            seen.append(step)
            return scores[min(len(seen) - 1, len(scores) - 1)]

        result = fit_dial_score(
            model, nnx.Optimizer(model, optax.adam(1e-3)), data,
            normalizer=normalizer, batch_size=64, num_iters=400,
            rng=jax.random.PRNGKey(1), validation_fraction=0.2,
            eval_every=100, selection_fn=selection, selection_every=100,
            desc="selection test")
        self.assertEqual(seen, [100, 200, 300, 400])
        self.assertAlmostEqual(result.best["selection"], 5.0)

    def test_selection_absent_keeps_validation_behaviour(self):
        import optax
        from csm.architectures import StandardNormalizer
        from csm.dial_score import DialScoreMLP, dial_sigma_control, fit_dial_score

        data = self._data(seed=2)
        normalizer = StandardNormalizer(OBS_SIZE)
        normalizer.fit(data.obs)
        model = DialScoreMLP(
            action_size=ACTION_SIZE, observation_size=OBS_SIZE, horizon=HORIZON,
            sigma_control=dial_sigma_control(0.9, HORIZON), hidden=(32, 32),
            rngs=nnx.Rngs(0), factor_min=0.5, factor_max=1.0)
        result = fit_dial_score(
            model, nnx.Optimizer(model, optax.adam(1e-3)), data,
            normalizer=normalizer, batch_size=64, num_iters=200,
            rng=jax.random.PRNGKey(1), validation_fraction=0.2, eval_every=100,
            desc="no selection")
        self.assertNotIn("selection", result.best)
        self.assertIn("val_loss", result.best)

    def test_sampled_omegas_are_convex_combinations_of_the_basis(self):
        """Drawn weights must stay inside the basis hull, where composition
        interpolates rather than extrapolates."""
        basis = jnp.asarray(BASIS)
        keys = jax.random.split(jax.random.PRNGKey(0), 200)
        coefficients = jax.vmap(
            lambda k: jax.random.dirichlet(k, jnp.ones(len(BASIS))))(keys)
        omegas = coefficients @ basis
        np.testing.assert_allclose(
            np.asarray(coefficients.sum(axis=1)), 1.0, rtol=1e-5)
        self.assertGreaterEqual(float(coefficients.min()), 0.0)
        # every draw is reachable with coefficients that sum to one
        policy = _composed()
        for omega in np.asarray(omegas)[:20]:
            c = np.asarray(policy.coefficients(jnp.asarray(omega)))
            np.testing.assert_allclose(float(c.sum()), 1.0, rtol=1e-4)
            np.testing.assert_allclose(c @ np.asarray(BASIS), omega, rtol=1e-3)
