"""Compositional Score Matching for DIAL-MPC.

The package is intentionally JAX/Brax-native.  Importing it has no global XLA
side effects and does not require the Hydrax, GPC, Warp, or Torch projects from
which the original prototype was extracted.
"""

from csm.data_collection import (
    DIALScoreCollector,
    ScoreDataPoint,
    build_sigma_schedule,
    stack_score_data,
)
from csm.policy import CompositionalPolicy

__all__ = [
    "CompositionalPolicy",
    "DIALScoreCollector",
    "ScoreDataPoint",
    "build_sigma_schedule",
    "stack_score_data",
]
