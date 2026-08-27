"""The two properties that make a stored collection worth storing.

A cloud dataset is only reusable if relabelling it reproduces exactly what the
collector would have produced, and if a round trip through disk changes
nothing.  Both are checked here on a deliberately tiny collection: the point is
the arithmetic, not the physics.
"""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import make_controller
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

from csm.basis_screen import build_omegas
from csm.cloud_data import load_clouds, make_relabeler, save_clouds
from csm.parallel_collect import collect, sample_drive_omegas


EXAMPLE = "unitree_go2_push_recover"
TEMPERATURE = 0.25


def _tiny():
    config = yaml.safe_load(open(get_example_path(EXAMPLE + ".yaml")))
    dial_config = load_dataclass_from_dict(DialConfig, config)
    dial_config = dataclasses.replace(dial_config, Nsample=63, Hsample=8, Hnode=3)
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(dial_config.env_name), config, convert_list_to_array=True
    )
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    return dial_config, env, make_controller(dial_config, env)


class CloudCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dial_config, cls.env, cls.mbdpi = _tiny()
        cls.basis = jnp.stack(
            [build_omegas(4)[f"boost{i}"] for i in range(4)]
        )
        cls.data, cls.stats = collect(
            cls.env, cls.mbdpi, cls.dial_config, cls.basis,
            jax.random.PRNGKey(0), num_steps=2, num_envs=2, repeats=2,
            perturb_scales=(0.25, 1.0), temperature=TEMPERATURE, std_normalize=False,
            episode_steps=4, init_passes=1, steps_per_call=2, progress=False,
        )
        relabel, ess = make_relabeler(cls.mbdpi, cls.dial_config)
        # Held in a dict: a plain callable assigned onto the class would be
        # rebound as a method and swallow the test case as its first argument.
        cls.fns = {"relabel": relabel, "ess": ess}

    def test_shapes_line_up(self):
        data, stats = self.data, self.stats
        expected = (
            stats["queries_per_level"] * stats["levels"]
            * stats["num_envs"] * 2
        )
        self.assertEqual(data.size, expected)
        self.assertEqual(data.terms.shape[1], stats["repeats"])
        self.assertEqual(data.terms.shape[-1], 4)
        self.assertEqual(data.sample_rng.shape[:2], data.terms.shape[:2])
        self.assertEqual(int(data.level.max()), stats["levels"] - 1)
        # Every query at a level shares that level's annealing factor.
        for level in range(stats["levels"]):
            factors = np.unique(np.asarray(data.factor)[np.asarray(data.level) == level])
            self.assertEqual(factors.size, 1)

    def test_relabelling_is_deterministic_and_weight_dependent(self):
        first, _ = self.fns["relabel"](self.data, self.basis[0], TEMPERATURE, False)
        again, _ = self.fns["relabel"](self.data, self.basis[0], TEMPERATURE, False)
        self.assertTrue(bool(jnp.array_equal(first, again)))

        other, _ = self.fns["relabel"](self.data, self.basis[2], TEMPERATURE, False)
        self.assertGreater(
            float(jnp.linalg.norm(first - other)) / float(jnp.linalg.norm(first)),
            1e-3,
            "different basis rows produced the same labels",
        )

    def test_temperature_is_free_to_change_after_collection(self):
        """The whole reason clouds are stored rather than labels."""

        sharp = float(jnp.mean(self.fns["ess"](self.data, self.basis[0], 0.1, False)))
        soft = float(jnp.mean(self.fns["ess"](self.data, self.basis[0], 1.0, False)))
        self.assertLess(sharp, soft)
        cold, _ = self.fns["relabel"](self.data, self.basis[0], 0.1, False)
        warm, _ = self.fns["relabel"](self.data, self.basis[0], 1.0, False)
        self.assertGreater(float(jnp.linalg.norm(cold - warm)), 0.0)

    def test_repeat_count_is_free_to_change_after_collection(self):
        one, spread_one = self.fns["relabel"](self.data, self.basis[1], TEMPERATURE, False, 1)
        two, spread_two = self.fns["relabel"](self.data, self.basis[1], TEMPERATURE, False, 2)
        # A single cloud has no Monte-Carlo spread to report, and averaging two
        # of them has to move the estimate.
        self.assertAlmostEqual(float(jnp.max(spread_one)), 0.0, places=6)
        self.assertGreater(float(jnp.mean(spread_two)), 0.0)
        self.assertGreater(float(jnp.linalg.norm(one - two)), 0.0)

    def test_disk_round_trip_changes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clouds.npz"
            save_clouds(path, self.data)
            back = load_clouds(path)
        before, _ = self.fns["relabel"](self.data, self.basis[3], TEMPERATURE, False)
        after, _ = self.fns["relabel"](back, self.basis[3], TEMPERATURE, False)
        self.assertTrue(bool(jnp.array_equal(before, after)))


class DriveWeightTest(unittest.TestCase):
    def test_basis_rows_are_not_starved(self):
        """Dirichlet alone under-samples the rows the basis is made of."""

        basis = jnp.stack([build_omegas(4)[f"boost{i}"] for i in range(4)])
        _, drawn = sample_drive_omegas(
            jax.random.PRNGKey(0), basis, 512, mix_basis_rows=0.5
        )
        matches = np.isclose(
            np.asarray(drawn)[:, None, :], np.asarray(basis)[None], atol=1e-5
        ).all(-1).any(-1)
        self.assertGreater(matches.mean(), 0.35)
        self.assertLess(matches.mean(), 0.65)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(drawn), axis=-1), 1.0, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
