"""Backward-compatible import location for the DIAL-native collector.

The old implementation expected raw ``mujoco.mjx.Data`` plus a Hydrax task.
DIAL-MPC owns state transitions through its Brax environments, so callers
should construct :class:`csm.data_collection.DIALScoreCollector` instead.
"""

from csm.data_collection import DIALScoreCollector


def build_mjx_score_collector(env, horizon, num_samples=1024, temperature=0.05):
    """Return a DIAL-native collector (legacy function name)."""

    return DIALScoreCollector(
        env,
        horizon=horizon,
        num_samples=num_samples,
        temperature=temperature,
    )


__all__ = ["DIALScoreCollector", "build_mjx_score_collector"]
