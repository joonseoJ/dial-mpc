import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from csm.gibbs_data import (
    GibbsDataset,
    anchor_decoder,
    compose_anchor_logits,
    load_gibbs_dataset,
    save_gibbs_dataset,
    standardized_gibbs_logits,
)


class AnchorGibbsAlgebraTest(unittest.TestCase):
    def setUp(self):
        self.anchors = jnp.asarray(
            [[2.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 2.0]]
        )
        self.decoder = anchor_decoder(self.anchors)

    def test_unseen_raw_cost_matches_direct_objective_composition(self):
        costs = jnp.asarray(
            [[0.2, 1.1, 0.7], [0.8, 0.4, 1.3], [1.2, 0.3, 0.1]]
        )
        anchor_costs = costs @ self.anchors.T
        omega = jnp.asarray([3.0, 1.5, 0.5])
        composed = compose_anchor_logits(anchor_costs, omega, self.decoder)
        np.testing.assert_allclose(
            composed, costs @ omega, rtol=1e-6, atol=2e-6
        )

    def test_unseen_standardization_matches_true_dial(self):
        costs = jnp.asarray(
            [[0.2, 1.1, 0.7], [0.8, 0.4, 1.3], [1.2, 0.3, 0.1]]
        )
        omega = jnp.asarray([1.3, 2.1, 0.6])
        anchor_costs = costs @ self.anchors.T
        composed = compose_anchor_logits(anchor_costs, omega, self.decoder)
        actual, actual_scale = standardized_gibbs_logits(
            composed, 0.05, candidate_axis=0
        )
        expected, expected_scale = standardized_gibbs_logits(
            costs @ omega, 0.05, candidate_axis=0
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_scale, expected_scale, rtol=1e-5)

    def test_positive_weight_scale_is_removed_before_composition(self):
        omega = jnp.asarray([1.0, 2.0, 3.0])
        target_sum = 5.0
        normalized = omega * target_sum / omega.sum()
        doubled = 2.0 * omega * target_sum / (2.0 * omega).sum()
        np.testing.assert_allclose(normalized, doubled, atol=1e-7)

    def test_dataset_round_trip(self):
        dataset = GibbsDataset(
            observations=jnp.zeros((1, 2)),
            queries=jnp.zeros((1, 2, 1)),
            candidates=jnp.zeros((1, 3, 2, 1)),
            anchor_costs=jnp.zeros((1, 3, 3)),
            anchor_updates=jnp.zeros((1, 3, 2, 1)),
            anchor_scales=jnp.ones((1, 3)),
            factors=jnp.ones((1,)),
            priorities=jnp.ones((1,)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            save_gibbs_dataset(path, dataset)
            loaded = load_gibbs_dataset(path)
        np.testing.assert_array_equal(loaded.candidates, dataset.candidates)


if __name__ == "__main__":
    unittest.main()
