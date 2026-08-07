import unittest
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import DIALTCMPPI


class _X(NamedTuple):
    pos: jax.Array


class _PipelineState(NamedTuple):
    q: jax.Array
    qd: jax.Array
    x: _X


class _State(NamedTuple):
    pipeline_state: _PipelineState
    reward: jax.Array


class _FakeEnv:
    action_size = 2

    def step(self, state, action):
        q = state.pipeline_state.q + jnp.pad(action, (0, 1))
        pipeline_state = _PipelineState(q=q, qd=q, x=_X(pos=q[None]))
        return _State(pipeline_state, -jnp.sum(action**2))


class DialTCIntegrationTest(unittest.TestCase):
    def test_dial_mean_mode_keeps_first_node_and_has_no_ratio_gap(self):
        config = DialConfig(
            Nsample=8,
            Hsample=4,
            Hnode=2,
            Ndiffuse=1,
            time_correlated=True,
            tc_history_length=2,
            tc_mean_mode="dial",
            tc_derivative_weights=(1.0, 0.0, 3e-5),
            tc_importance_scale=0.0,
        )
        controller = DIALTCMPPI(config, _FakeEnv())
        pipeline_state = _PipelineState(
            q=jnp.zeros(3), qd=jnp.zeros(3), x=_X(pos=jnp.zeros((1, 3)))
        )
        state = _State(pipeline_state, jnp.asarray(0.0))
        mean = jnp.array([[0.2, -0.1], [0.1, 0.0], [0.0, 0.1]])

        _, updated, info = controller.reverse_once(
            state,
            jax.random.PRNGKey(0),
            mean,
            jnp.ones(3) * 0.4,
            jnp.zeros((2, 2)),
        )
        updated.block_until_ready()

        expected_first = np.broadcast_to(np.asarray(mean[0]), (9, 2))
        np.testing.assert_allclose(info["Y0s"][:, 0], expected_first, atol=1e-6)
        np.testing.assert_allclose(info["tc_log_importance_ratio"], 0.0, atol=1e-7)
        self.assertTrue(bool(jnp.isfinite(updated).all()))


if __name__ == "__main__":
    unittest.main()
