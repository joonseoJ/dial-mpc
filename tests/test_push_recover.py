"""Guards on the disturbance-rejection basis.

The failure this suite exists to prevent is collinearity.  A basis whose rows
are different projections of one quantity cannot move the argmax, so no weight
vector changes behaviour and every downstream composition result is noise.  The
Go2 trot basis failed exactly that way -- three rows, all squared errors against
the same nominal trot -- and it cost a lot of GPU time to find out empirically.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import brax.envs as brax_envs
from brax import math

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import make_controller
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

from csm.dial_lean import make_sampler


EXAMPLE = "unitree_go2_push_recover"


def _build():
    config_dict = yaml.safe_load(open(get_example_path(EXAMPLE + ".yaml")))
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(dial_config.env_name),
        config_dict,
        convert_list_to_array=True,
    )
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    return dial_config, env


class PushRecoverBasisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dial_config, cls.env = _build()
        # Held in a dict: a jitted callable assigned straight onto the class
        # would be rebound as a method and receive the test case as its first
        # argument.
        cls.fns = {"step": jax.jit(cls.env.step), "reset": jax.jit(cls.env.reset)}

    def _terms_from(self, qpos):
        """Reward rows after one zero-action step from a perturbed pose."""

        env = self.env
        state = self.fns["reset"](jax.random.PRNGKey(0))
        pipeline_state = env.pipeline_init(qpos, jnp.zeros(env._nv))
        state = state.replace(pipeline_state=pipeline_state)
        return np.asarray(self.fns["step"](state, jnp.zeros(env.action_size)).info["reward_terms"])

    def test_rows_have_distinct_sensitivity_signatures(self):
        """No two rows may react to the same set of deviations.

        Rows cannot be tested for exclusive ownership of one quantity -- a rigid
        robot that pitches also drags its feet through the world, so the contact
        row must move.  What separates the rows is which deviations each one is
        *blind* to, and every pair here differs on at least one:

                       tilt   translate   bend
            row0 tilt   yes      no         no
            row1 base   no       yes        no
            row2 feet   yes      yes        yes
            row3 shape  no       no         yes
        """

        env = self.env
        base = self._terms_from(env._init_q)
        tilt = base - self._terms_from(
            env._init_q.at[3:7].set(
                math.quat_mul(
                    math.euler_to_quat(jnp.array([0.0, 12.0, 0.0])),
                    env._init_q[3:7],
                )
            )
        )
        translate = base - self._terms_from(env._init_q.at[0].add(0.12))
        bend = base - self._terms_from(env._init_q.at[7:].add(0.25))

        blind, sees = 0.05, 0.3
        for name, row, expected in (
            ("tilt", 0, (True, False, False)),
            ("base", 1, (False, True, False)),
            ("feet", 2, (True, True, True)),
            ("shape", 3, (False, False, True)),
        ):
            for label, drop, want in zip(
                ("tilt", "translate", "bend"), (tilt, translate, bend), expected
            ):
                value = abs(float(drop[row]))
                if want:
                    self.assertGreater(
                        value, sees,
                        f"row{row} ({name}) should react to {label}, saw {value:.4f}",
                    )
                else:
                    self.assertLess(
                        value, blind,
                        f"row{row} ({name}) should ignore {label}, saw {value:.4f}",
                    )

    def test_contact_row_is_world_framed_and_shape_row_is_not(self):
        """The separation that keeps rows 2 and 3 from collapsing into one.

        Translating the whole robot leaves every joint angle untouched but moves
        every foot in the world.  If the contact row were written in base
        coordinates it would be blind to this, become a copy of the shape row,
        and "plant the feet and bend" would stop being distinguishable from
        "hold the shape and step".
        """

        env = self.env
        base = self._terms_from(env._init_q)
        translated = self._terms_from(env._init_q.at[0].add(0.12))

        shape_change = abs(base[3] - translated[3])
        contact_change = abs(base[2] - translated[2])
        self.assertLess(shape_change, 1e-3, "shape row moved under pure translation")
        self.assertGreater(
            contact_change, 1.0,
            f"contact row is insensitive to translation ({contact_change})",
        )

    def test_reward_is_the_weighted_row_sum(self):
        env = self.env
        weights = jnp.array([0.3, 1.7, 0.5, 2.1])
        state = self.fns["reset"](jax.random.PRNGKey(1))
        state.info["reward_weights"] = weights
        stepped = self.fns["step"](state, jnp.zeros(env.action_size))
        self.assertAlmostEqual(
            float(stepped.reward),
            float(jnp.dot(weights, stepped.info["reward_terms"])),
            places=5,
        )

    def test_push_is_applied_at_reset(self):
        speeds = [
            float(self.fns["reset"](jax.random.PRNGKey(s)).info["push_speed"])
            for s in range(8)
        ]
        lo = self.env._config.push_linear_velocity * self.env._config.push_scale_min
        hi = self.env._config.push_linear_velocity
        self.assertTrue(all(lo - 1e-5 <= s <= hi + 1e-5 for s in speeds), speeds)
        self.assertGreater(np.std(speeds), 0.0, "push magnitude is not randomized")


class LeanSamplerTest(unittest.TestCase):
    """The lean rollout must draw DIAL's proposal, not an approximation of it.

    Only the sampling is checked for exact equality.  The rollout itself cannot
    be: this plant is chaotic enough that recompiling the same function changes
    individual sample returns by O(0.1) over 24 steps purely through float32
    rounding, so a bitwise rollout comparison would test XLA's fusion choices
    rather than the algorithm.
    """

    def test_proposal_matches_reverse_once(self):
        dial_config, env = _build()
        mbdpi = make_controller(dial_config, env)
        plan = jnp.zeros((dial_config.Hnode + 1, mbdpi.nu))
        noise = mbdpi.sigma_control * 0.5
        rng = jax.random.PRNGKey(11)

        _, sample_rng = jax.random.split(rng)
        eps = jax.random.normal(
            sample_rng, (dial_config.Nsample, dial_config.Hnode + 1, mbdpi.nu)
        )
        expected = eps * noise[None, :, None] + plan
        expected = expected.at[:, 0].set(plan[0, :])
        expected = jnp.concatenate([expected, plan[None]], axis=0)
        expected = jnp.clip(expected, -1.0, 1.0)

        got = make_sampler(mbdpi, dial_config)(sample_rng, plan, noise)
        self.assertTrue(bool(jnp.array_equal(expected, got)))


if __name__ == "__main__":
    unittest.main()
