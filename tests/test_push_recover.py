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

    def test_reward_uses_the_unit_normalised_weights(self):
        """Whatever scale comes in, the reward is computed at unit length.

        The objective depends only on `omega / T`, so a free scale on omega
        would leave the temperature undefined.  Normalising here means no
        caller -- planner, viewer, sweep -- can reintroduce that ambiguity, and
        the stored weights always say what was actually used.
        """

        env = self.env
        weights = jnp.array([0.3, 1.7, 0.5, 2.1])
        unit = weights / jnp.linalg.norm(weights)
        state = self.fns["reset"](jax.random.PRNGKey(1))
        state.info["reward_weights"] = weights
        stepped = self.fns["step"](state, jnp.zeros(env.action_size))
        self.assertAlmostEqual(
            float(stepped.reward),
            float(jnp.dot(unit, stepped.info["reward_terms"])),
            places=5,
        )
        np.testing.assert_allclose(
            np.asarray(stepped.info["reward_weights"]), np.asarray(unit), atol=1e-6
        )

    def test_reward_is_invariant_to_the_incoming_weight_scale(self):
        env = self.env
        rewards = []
        for scale in (0.5, 1.0, 7.0):
            state = self.fns["reset"](jax.random.PRNGKey(1))
            state.info["reward_weights"] = jnp.array([1.0, 2.0, 0.5, 1.5]) * scale
            rewards.append(
                float(self.fns["step"](state, jnp.zeros(env.action_size)).reward)
            )
        self.assertAlmostEqual(rewards[0], rewards[1], places=5)
        self.assertAlmostEqual(rewards[1], rewards[2], places=5)

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


class OmegaConventionTest(unittest.TestCase):
    """Unit weights and the nu solve that per-field temperatures require."""

    def test_normalization_keeps_direction_and_survives_zero(self):
        from csm.omega import normalize_omega

        raw = jnp.array([3.0, 1.0, 1.0, 1.0])
        unit = normalize_omega(raw)
        self.assertAlmostEqual(float(jnp.linalg.norm(unit)), 1.0, places=6)
        self.assertAlmostEqual(
            float(jnp.dot(unit, raw) / jnp.linalg.norm(raw)), 1.0, places=6
        )
        np.testing.assert_allclose(
            np.asarray(normalize_omega(jnp.zeros(4))), np.zeros(4)
        )

    def test_nu_solve_reduces_and_retempers(self):
        from csm.omega import normalize_omega_np, nu_coefficients

        basis = np.stack([
            normalize_omega_np(np.eye(4)[i] * 2.0 + 1.0) for i in range(4)
        ])
        temps = np.array([0.25, 0.30, 0.37, 0.39])

        for j in range(4):
            a = nu_coefficients(basis, temps, basis[j], temps[j])
            np.testing.assert_allclose(a, np.eye(4)[j], atol=1e-9)

        # Same field, sharper: one coefficient, scaled by the temperature ratio.
        a = nu_coefficients(basis, temps, basis[2], 0.25)
        expected = np.eye(4)[2] * (temps[2] / 0.25)
        np.testing.assert_allclose(a, expected, atol=1e-9)

    def test_omega_solve_is_wrong_when_temperatures_differ(self):
        """The failure mode nu space exists to fix.

        With mixed temperatures the legacy solve reproduces neither the
        requested sharpness nor, because each field is scaled differently, the
        requested direction.
        """

        from csm.omega import (
            normalize_omega_np, nu_coefficients, nu_matrix, omega_coefficients,
        )

        basis = np.stack([
            normalize_omega_np(np.eye(4)[i] * 2.0 + 1.0) for i in range(4)
        ])
        temps = np.array([0.25, 0.30, 0.37, 0.39])
        target, target_temp = normalize_omega_np(np.ones(4)), 0.30

        matrix = nu_matrix(basis, temps)
        wanted = target / target_temp
        np.testing.assert_allclose(
            nu_coefficients(basis, temps, target, target_temp) @ matrix,
            wanted, atol=1e-9,
        )
        got = omega_coefficients(basis, target) @ matrix
        self.assertGreater(np.abs(got - wanted).max() / np.abs(wanted).max(), 0.1)
        # Not merely mis-scaled: the realised direction is uneven too.
        self.assertGreater(got.max() / got.min() - 1.0, 0.05)


class ComposedPolicyCoefficientTest(unittest.TestCase):
    """The composition solve as the deployed policy actually performs it."""

    @staticmethod
    def _policy(temperatures=None):
        from csm.basis_screen import build_omegas
        from csm.dial_score import ComposedDialScorePolicy
        from csm.omega import nu_matrix

        basis = np.stack([np.asarray(build_omegas(4)[f"boost{i}"]) for i in range(4)])
        common = dict(
            policies=(), mode_weights=jnp.asarray(basis),
            pinv_mode_weights=jnp.asarray(np.linalg.pinv(basis)),
        )
        if temperatures is None:
            return basis, ComposedDialScorePolicy(**common)
        return basis, ComposedDialScorePolicy(
            basis_temperatures=jnp.asarray(temperatures),
            temperature=float(temperatures[0]),
            pinv_nu_weights=jnp.asarray(
                np.linalg.pinv(nu_matrix(basis, temperatures))
            ),
            **common,
        )

    def test_reduces_to_a_single_field(self):
        basis, policy = self._policy([0.25] * 4)
        for row in range(4):
            np.testing.assert_allclose(
                np.asarray(policy.coefficients(jnp.asarray(basis[row]))),
                np.eye(4)[row], atol=1e-6,
            )

    def test_incoming_weight_scale_is_irrelevant(self):
        """Normalisation at the boundary, seen from the composition side."""

        _, policy = self._policy([0.25] * 4)
        one = np.asarray(policy.coefficients(jnp.asarray([1.0, 2.0, 0.5, 1.5])))
        seven = np.asarray(policy.coefficients(jnp.asarray([7.0, 14.0, 3.5, 10.5])))
        np.testing.assert_allclose(one, seven, atol=1e-6)

    def test_retempering_scales_the_coefficients(self):
        _, policy = self._policy([0.25] * 4)
        target = jnp.asarray([1.0, 1.0, 1.0, 1.0])
        base = np.asarray(policy.coefficients(target))
        sharper = np.asarray(policy.coefficients(target, 0.125))
        np.testing.assert_allclose(sharper, 2.0 * base, rtol=1e-5)

    def test_magnitude_is_recovered_not_guessed(self):
        """What the sum-to-one rescale could only approximate.

        The nu solve reproduces the requested natural parameter exactly; the
        legacy solve normalises the coefficients to sum to one, which is a
        different vector whenever that sum is not already one.
        """

        from csm.omega import nu_matrix

        basis, policy = self._policy([0.25] * 4)
        _, legacy = self._policy(None)
        target = np.ones(4) / 2.0                      # unit uniform
        nu = nu_matrix(basis, [0.25] * 4)

        exact = np.asarray(policy.coefficients(jnp.asarray(target))) @ nu
        np.testing.assert_allclose(exact, target / 0.25, rtol=1e-5)

        guessed = np.asarray(legacy.coefficients(jnp.asarray(target))) @ nu
        self.assertAlmostEqual(
            float(np.asarray(legacy.coefficients(jnp.asarray(target))).sum()),
            1.0, places=5,
        )
        self.assertGreater(
            np.abs(guessed - target / 0.25).max() / (target / 0.25).max(), 0.1
        )
