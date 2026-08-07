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
)


class AnchorGibbsAlgebraTest(unittest.TestCase):
    def setUp(self):
        self.anchors = jnp.asarray(
            [[2.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 2.0]]
        )
        self.decoder = anchor_decoder(self.anchors)

    def test_unseen_logit_matches_direct_objective_composition(self):
        costs = jnp.asarray(
            [[0.2, 1.1, 0.7], [0.8, 0.4, 1.3], [1.2, 0.3, 0.1]]
        )
        anchor_logits = -(costs @ self.anchors.T)
        omega = jnp.asarray([3.0, 1.5, 0.5])
        composed = compose_anchor_logits(anchor_logits, omega, self.decoder)
        np.testing.assert_allclose(
            composed, -(costs @ omega), rtol=1e-6, atol=2e-6
        )

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
            anchor_logits=jnp.zeros((1, 3, 3)),
            anchor_updates=jnp.zeros((1, 3, 2, 1)),
            factors=jnp.ones((1,)),
            logit_scales=jnp.ones((1,)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            save_gibbs_dataset(path, dataset)
            loaded = load_gibbs_dataset(path)
        np.testing.assert_array_equal(loaded.candidates, dataset.candidates)


if __name__ == "__main__":
    unittest.main()
