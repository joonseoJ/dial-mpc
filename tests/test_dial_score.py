import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from csm.architectures import StandardNormalizer
from csm.dial_score import (
    DialScoreData,
    concat_dial_score_data,
    DialScoreMLP,
    DialScorePolicy,
    dial_factors,
    dial_sigma_control,
    factor_to_t,
    fit_dial_score,
    level_loss_weights,
    load_dial_score_data,
    save_dial_score_data,
    score_regression_loss,
    t_to_factor,
)


HORIZON = 5
ACTION_SIZE = 3
OBS_SIZE = 7
TRAJ_FACTOR = 0.5
NUM_LEVELS = 2


def _schedule():
    sigma_control = dial_sigma_control(0.9, HORIZON)
    factors = dial_factors(TRAJ_FACTOR, NUM_LEVELS)
    return sigma_control, factors, float(jnp.min(factors)), float(jnp.max(factors))


def _model(hidden=(64, 64), seed=0):
    sigma_control, _, factor_min, factor_max = _schedule()
    return DialScoreMLP(
        action_size=ACTION_SIZE,
        observation_size=OBS_SIZE,
        horizon=HORIZON,
        sigma_control=sigma_control,
        hidden=hidden,
        rngs=nnx.Rngs(seed),
        factor_min=factor_min,
        factor_max=factor_max,
    )


class ScheduleTest(unittest.TestCase):
    def test_sigma_control_matches_dial(self):
        sigma_control = dial_sigma_control(0.9, HORIZON, sigma_scale=2.0)
        expected = 2.0 * 0.9 ** np.arange(HORIZON)[::-1]
        np.testing.assert_allclose(
            np.asarray(sigma_control), expected, rtol=1e-6
        )
        # Node 0 is the least noisy, the plan tail the most.
        self.assertLess(float(sigma_control[0]), float(sigma_control[-1]))

    def test_factor_conditioning_round_trip(self):
        _, factors, factor_min, factor_max = _schedule()
        t = factor_to_t(factors, factor_min, factor_max)
        np.testing.assert_allclose(np.asarray(t), [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(t_to_factor(t, factor_min, factor_max)),
            np.asarray(factors),
            rtol=1e-6,
        )

    def test_single_level_conditioning_is_well_defined(self):
        factors = dial_factors(TRAJ_FACTOR, 1)
        t = factor_to_t(factors, 1.0, 1.0)
        self.assertTrue(np.all(np.isfinite(np.asarray(t))))
        np.testing.assert_allclose(np.asarray(t), [0.0], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(t_to_factor(t, 1.0, 1.0)), [1.0], rtol=1e-6
        )


class DialScoreMLPTest(unittest.TestCase):
    def test_score_is_update_over_sigma_squared(self):
        model = _model()
        sigma_control, factors, factor_min, factor_max = _schedule()
        u = jnp.zeros((HORIZON, ACTION_SIZE))
        y = jnp.zeros((OBS_SIZE,))
        for factor in factors:
            t = factor_to_t(factor, factor_min, factor_max).reshape(1)
            sigma = model.sigma(t)
            np.testing.assert_allclose(
                np.asarray(sigma)[:, 0],
                np.asarray(sigma_control) * float(factor),
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(model.score(u, y, t) * jnp.square(sigma)),
                np.asarray(model.update(u, y, t)),
                atol=1e-6,
            )

    def test_first_node_update_is_locked_to_zero(self):
        model = _model()
        _, _, factor_min, factor_max = _schedule()
        u = jax.random.normal(jax.random.PRNGKey(1), (4, HORIZON, ACTION_SIZE))
        y = jax.random.normal(jax.random.PRNGKey(2), (4, OBS_SIZE))
        t = jnp.full((4, 1), 0.5)
        delta = model.update(u, y, t)
        self.assertEqual(delta.shape, (4, HORIZON, ACTION_SIZE))
        np.testing.assert_allclose(np.asarray(delta[:, 0]), 0.0, atol=0.0)
        # Later nodes must still be free.
        self.assertGreater(float(jnp.max(jnp.abs(delta[:, 1:]))), 0.0)

    def test_batched_and_unbatched_agree(self):
        model = _model()
        u = jax.random.normal(jax.random.PRNGKey(3), (HORIZON, ACTION_SIZE))
        y = jax.random.normal(jax.random.PRNGKey(4), (OBS_SIZE,))
        t = jnp.array([0.25])
        single = model.update(u, y, t)
        batched = model.update(u[None], y[None], t[None])
        np.testing.assert_allclose(
            np.asarray(single), np.asarray(batched[0]), atol=1e-6
        )


def _synthetic_dataset(num_samples=2048, seed=0):
    """A learnable score field: delta depends smoothly on (u, obs, level)."""

    sigma_control, factors, factor_min, factor_max = _schedule()
    rng = jax.random.PRNGKey(seed)
    rng, u_rng, obs_rng, level_rng, gain_rng = jax.random.split(rng, 5)
    u = jax.random.uniform(
        u_rng, (num_samples, HORIZON, ACTION_SIZE), minval=-1.0, maxval=1.0
    )
    obs = jax.random.normal(obs_rng, (num_samples, OBS_SIZE))
    level = jax.random.randint(level_rng, (num_samples,), 0, NUM_LEVELS)
    factor = factors[level]
    gain = jax.random.normal(gain_rng, (OBS_SIZE, ACTION_SIZE)) * 0.2

    target = obs @ gain  # (N, nu)
    # A contraction toward the observation-dependent target, scaled by the
    # level's per-node sigma, so both nodes and levels carry structure.
    scale = factor[:, None, None] * sigma_control.reshape(1, HORIZON, 1)
    delta = scale * (target[:, None, :] - u)
    delta = delta.at[:, 0].set(0.0)
    return DialScoreData(u=u, factor=factor, level=level, delta=delta, obs=obs)


class FitDialScoreTest(unittest.TestCase):
    def test_score_regression_fits_a_known_field(self):
        data = _synthetic_dataset()
        normalizer = StandardNormalizer(OBS_SIZE)
        normalizer.fit(data.obs)
        model = _model(hidden=(128, 128))
        optimizer = nnx.Optimizer(model, optax.adam(3e-3))
        result = fit_dial_score(
            model,
            optimizer,
            data,
            normalizer=normalizer,
            batch_size=256,
            num_iters=1500,
            rng=jax.random.PRNGKey(5),
            validation_fraction=0.2,
            eval_every=250,
            desc="test fit",
        )
        self.assertLess(result.best["val_relative_rms"], 0.15)
        self.assertGreater(result.best["val_cosine"], 0.95)
        # Every annealing level must be fit, not just the dominant one.
        self.assertEqual(set(result.per_level), {0, 1})
        for metrics in result.per_level.values():
            self.assertGreater(metrics["cosine"], 0.9)
        # The best checkpoint is restored, so the final metrics match it.
        self.assertLessEqual(
            result.final["loss"], result.best["val_loss"] * 1.01 + 1e-9
        )

    def test_loss_is_sigma_fourth_weighted_update_error(self):
        model = _model()
        data = _synthetic_dataset(num_samples=16, seed=1)
        t = factor_to_t(data.factor, model.factor_min, model.factor_max)[:, None]
        loss = score_regression_loss(model, data.u, data.obs, t, data.delta)
        expected = jnp.mean(
            jnp.square(model.update(data.u, data.obs, t) - data.delta)
        )
        np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)

    def test_level_balance_zero_is_exactly_the_plain_loss(self):
        data = _synthetic_dataset(num_samples=64, seed=4)
        weights = level_loss_weights(data, 0.0)
        np.testing.assert_allclose(np.asarray(weights), 1.0, atol=0.0)

        model = _model()
        t = factor_to_t(data.factor, model.factor_min, model.factor_max)[:, None]
        plain = score_regression_loss(model, data.u, data.obs, t, data.delta)
        weighted = score_regression_loss(
            model, data.u, data.obs, t, data.delta, weights[data.level]
        )
        np.testing.assert_allclose(float(plain), float(weighted), rtol=1e-6)

    def test_level_balance_one_equalizes_level_contributions(self):
        data = _synthetic_dataset(num_samples=4096, seed=5)
        weights = level_loss_weights(data, 1.0)
        levels = np.asarray(data.level)
        delta = np.asarray(data.delta)

        # The fine level has intrinsically smaller updates, so it must be the
        # one that gets scaled up.
        power = [
            float(np.mean(np.square(delta[levels == level])))
            for level in (0, 1)
        ]
        self.assertLess(power[1], power[0])
        self.assertGreater(float(weights[1]), float(weights[0]))

        # After weighting, each level contributes equal power to the loss.
        contributions = [
            power[level] * float(weights[level]) for level in (0, 1)
        ]
        np.testing.assert_allclose(
            contributions[0], contributions[1], rtol=1e-5
        )
        # And the mean weight over the dataset stays 1, so the loss scale — and
        # therefore the usable learning rate — is unchanged.
        np.testing.assert_allclose(
            float(np.mean(np.asarray(weights)[levels])), 1.0, rtol=1e-5
        )

    def test_level_balance_rejects_out_of_range(self):
        data = _synthetic_dataset(num_samples=16, seed=6)
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                level_loss_weights(data, bad)

    def test_level_weighted_fit_improves_the_weak_level(self):
        data = _synthetic_dataset(num_samples=4096, seed=7)
        normalizer = StandardNormalizer(OBS_SIZE)
        normalizer.fit(data.obs)

        def run(balance):
            model = _model(hidden=(64, 64), seed=3)
            weights = level_loss_weights(data, balance)
            return fit_dial_score(
                model,
                nnx.Optimizer(model, optax.adam(3e-3)),
                data,
                normalizer=normalizer,
                batch_size=256,
                num_iters=600,
                rng=jax.random.PRNGKey(8),
                validation_fraction=0.2,
                eval_every=300,
                level_weights=None if balance == 0.0 else weights,
                desc=f"balance {balance}",
            )

        plain = run(0.0)
        balanced = run(1.0)
        # Same data, same init, same schedule: only the loss weighting differs,
        # and the fine level is what it is meant to help.
        self.assertLess(
            balanced.per_level[1]["relative_rms"],
            plain.per_level[1]["relative_rms"],
        )

    def test_single_sample_dataset_does_not_crash(self):
        data = _synthetic_dataset(num_samples=1, seed=2)
        normalizer = StandardNormalizer(OBS_SIZE)
        model = _model(hidden=(8,))
        result = fit_dial_score(
            model,
            nnx.Optimizer(model, optax.adam(1e-3)),
            data,
            normalizer=normalizer,
            batch_size=1,
            num_iters=2,
            rng=jax.random.PRNGKey(6),
            validation_fraction=0.1,
            eval_every=1,
            desc="tiny fit",
        )
        self.assertTrue(np.isfinite(result.final["loss"]))


class DialScorePolicyTest(unittest.TestCase):
    def _policy(self, dt=1.0):
        sigma_control, factors, _, _ = _schedule()
        model = _model()
        normalizer = StandardNormalizer(OBS_SIZE)
        shift_matrix = jnp.eye(HORIZON)
        return (
            DialScorePolicy(
                model=model,
                normalizer=normalizer,
                factors=factors,
                shift_matrix=shift_matrix,
                dt=dt,
            ),
            factors,
        )

    def test_apply_reproduces_the_manual_denoising_loop(self):
        policy, factors = self._policy()
        model = policy.model
        obs = jax.random.normal(jax.random.PRNGKey(7), (OBS_SIZE,))
        plan = jnp.zeros((HORIZON, ACTION_SIZE))

        manual = plan
        for factor in factors:
            t = factor_to_t(factor, model.factor_min, model.factor_max).reshape(1)
            sigma_sq = jnp.square(model.sigma(t))
            manual = jnp.clip(
                manual + sigma_sq * model.score(manual, obs, t), -1.0, 1.0
            )
        np.testing.assert_allclose(
            np.asarray(policy.apply(plan, obs)), np.asarray(manual), atol=1e-6
        )

    def test_apply_stays_in_normalized_bounds_and_locks_node_zero(self):
        policy, _ = self._policy(dt=50.0)
        obs = jax.random.normal(jax.random.PRNGKey(8), (OBS_SIZE,))
        plan = jax.random.uniform(
            jax.random.PRNGKey(9), (HORIZON, ACTION_SIZE), minval=-1.0, maxval=1.0
        )
        refined = policy.apply(plan, obs, passes=3)
        self.assertLessEqual(float(jnp.max(jnp.abs(refined))), 1.0 + 1e-6)
        np.testing.assert_allclose(
            np.asarray(refined[0]), np.asarray(plan[0]), atol=1e-6
        )

    def test_jit_and_pickle_round_trip(self):
        policy, _ = self._policy()
        obs = jnp.zeros((OBS_SIZE,))
        plan = jnp.zeros((HORIZON, ACTION_SIZE))
        expected = policy.apply(plan, obs)
        jitted = jax.jit(policy.apply)
        np.testing.assert_allclose(
            np.asarray(jitted(plan, obs)), np.asarray(expected), atol=1e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pkl"
            policy.save(path)
            loaded = DialScorePolicy.load(path)
        np.testing.assert_allclose(
            np.asarray(loaded.apply(plan, obs)), np.asarray(expected), atol=1e-6
        )

    def test_shift_applies_the_node_map(self):
        sigma_control, factors, _, _ = _schedule()
        shift_matrix = jnp.roll(jnp.eye(HORIZON), -1, axis=0).at[-1].set(0.0)
        policy = DialScorePolicy(
            model=_model(),
            normalizer=StandardNormalizer(OBS_SIZE),
            factors=factors,
            shift_matrix=shift_matrix,
        )
        plan = jnp.arange(HORIZON * ACTION_SIZE, dtype=jnp.float32).reshape(
            HORIZON, ACTION_SIZE
        )
        shifted = policy.shift(plan)
        np.testing.assert_allclose(
            np.asarray(shifted[:-1]), np.asarray(plan[1:]), atol=1e-6
        )
        np.testing.assert_allclose(np.asarray(shifted[-1]), 0.0, atol=1e-6)


class DatasetIOTest(unittest.TestCase):
    def test_save_load_round_trip(self):
        data = _synthetic_dataset(num_samples=8, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            save_dial_score_data(path, data)
            loaded = load_dial_score_data(path)
        for field in DialScoreData._fields:
            original = getattr(data, field)
            if original is None:
                continue      # unrecorded physics is written as a placeholder
            np.testing.assert_allclose(
                np.asarray(getattr(loaded, field)),
                np.asarray(original),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()


class PhysicsStateAndBankTest(unittest.TestCase):
    """Recorded physics state and episode reseeding."""

    def _data(self, n, with_state=True):
        from csm.dial_score import DialScoreData
        base = _synthetic_dataset(num_samples=n)
        if with_state:
            return base._replace(
                qpos=jnp.zeros((n, 19)), qvel=jnp.zeros((n, 18)),
                step=jnp.arange(n, dtype=jnp.int32))
        return base

    def test_old_datasets_load_and_report_missing_state(self):
        """A dataset written before the fields existed must stay usable."""
        import numpy as _np
        data = _synthetic_dataset(num_samples=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            # write only the original five arrays
            with open(path, "wb") as stream:
                _np.savez_compressed(stream, **{
                    f: _np.asarray(getattr(data, f))
                    for f in ("u", "factor", "level", "delta", "obs")})
            loaded = load_dial_score_data(path)
        self.assertEqual(loaded.size, 8)
        self.assertFalse(loaded.has_physics_state)
        np.testing.assert_array_equal(np.asarray(loaded.step), -1)

    def test_round_trip_preserves_physics_state(self):
        data = self._data(6)
        self.assertTrue(data.has_physics_state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d.npz"
            save_dial_score_data(path, data)
            loaded = load_dial_score_data(path)
        self.assertTrue(loaded.has_physics_state)
        np.testing.assert_array_equal(
            np.asarray(loaded.step), np.asarray(data.step))

    def test_mixing_recorded_and_unrecorded_drops_the_claim(self):
        """Concatenating must not produce data that looks relabellable but isn't."""
        merged = concat_dial_score_data([self._data(4), self._data(4, False)])
        self.assertEqual(merged.size, 8)
        self.assertFalse(merged.has_physics_state)

    def test_state_bank_samples_and_refreshes_rng(self):
        from csm.dial_score import StateBank

        class FakeState:
            def __init__(self, tag, info):
                self.tag, self.info = tag, info
            def replace(self, info):
                return FakeState(self.tag, info)

        bank = StateBank(capacity=3)
        for i in range(5):
            bank.add(FakeState(i, {"rng": jnp.zeros(2, dtype=jnp.uint32)}))
        # capacity drops the oldest: banked states go stale as the policy moves
        self.assertEqual(len(bank), 3)
        self.assertEqual([s.tag for s in bank.states], [2, 3, 4])
        drawn = bank.sample(jax.random.PRNGKey(0))
        # a fresh rng, otherwise every restart replays the same trajectory
        self.assertFalse(
            bool(jnp.array_equal(drawn.info["rng"], jnp.zeros(2, dtype=jnp.uint32)))
        )

    def test_mixed_reset_falls_back_when_bank_is_empty(self):
        from csm.dial_score import StateBank, mixed_reset_fn
        sentinel = object()
        fresh = lambda rng: sentinel
        self.assertIs(mixed_reset_fn(fresh, None, 0.5), fresh)
        self.assertIs(mixed_reset_fn(fresh, StateBank(), 0.5), fresh)
        self.assertIs(mixed_reset_fn(fresh, StateBank([1, 2]), 0.0), fresh)

    def test_mixed_reset_uses_both_sources(self):
        from csm.dial_score import StateBank, mixed_reset_fn

        class FakeState:
            def __init__(self, tag): self.tag, self.info = tag, {"rng": jnp.zeros(2, jnp.uint32)}
            def replace(self, info):
                s = FakeState(self.tag); s.info = info; return s

        bank = StateBank([FakeState("banked")])
        reset = mixed_reset_fn(lambda rng: FakeState("fresh"), bank, 0.5)
        tags = {reset(jax.random.PRNGKey(i)).tag for i in range(40)}
        self.assertEqual(tags, {"banked", "fresh"})
