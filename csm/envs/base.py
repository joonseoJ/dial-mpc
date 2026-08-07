"""Objective specifications connecting DIAL environments to CSM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax


@dataclass(frozen=True)
class DIALObjectiveSpec:
    """Names and unweighted stage-cost function for a DIAL environment."""

    names: tuple[str, ...]
    cost: Callable[[object, jax.Array], jax.Array]

    @property
    def num_objectives(self) -> int:
        return len(self.names)
