"""DIAL-MPC objective adapters used by Compositional Score Matching."""

from csm.envs.base import DIALObjectiveSpec
from csm.envs.dial import (
    GO2_TROT_OBJECTIVES,
    get_objective_spec,
    make_go2_trot_cost,
)

__all__ = [
    "DIALObjectiveSpec",
    "GO2_TROT_OBJECTIVES",
    "get_objective_spec",
    "make_go2_trot_cost",
]
