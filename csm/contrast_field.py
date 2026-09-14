"""A gait field split into what every gait shares and what makes it that gait.

The four gait labels are nearly the same vector: on stored clouds their pairwise
cosine is 0.88-0.94 and the part that distinguishes them is 12-21% of the label
norm, while the fit's own error is ~30% of it.  Regressing onto `label_i`
directly therefore spends the whole error budget on the shared part and loses
the gait inside it -- every fitted field comes out as the same floor field, and
which gait appears in closed loop is decided by the collection driver rather
than by omega (see memory gait-dagger-rotation).

The fix is to regress the two parts separately.  Write

    label_i = m + r_i,     m = mean_j label_j,     r_i = label_i - m

which is exact arithmetic with `sum_i r_i = 0`.  Fit one network for `m` -- the
large, easy, shared update -- and one per gait for `r_i`.  Each residual
network's relative error is then measured against the residual itself, so a 30%
fit resolves the gait instead of burying it.

Composition is untouched.  The composed controller mixes fields with
coefficients rescaled to sum to one, so

    sum_i a_i (m + r_i) = m + sum_i a_i r_i

which is the label for the mixed weight by the same arithmetic.  The score's
linearity in nu is a property of the labels, and this decomposition is applied
to the labels, so nothing about the composition changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import jax

from csm.dial_score import DialScorePolicy


@dataclass
class ContrastField:
    """`base + residual`, presented as one field.

    Carries exactly the interface the composed controller uses from a
    :class:`DialScorePolicy` -- `factors`, `shift_matrix` and `delta` -- so it
    drops into `ComposedDialScorePolicy.policies` unchanged.
    """

    base: DialScorePolicy
    residual: DialScorePolicy

    @property
    def factors(self) -> jax.Array:
        return self.base.factors

    @property
    def shift_matrix(self) -> jax.Array:
        return self.base.shift_matrix

    @property
    def temperature(self):
        return self.base.temperature

    @property
    def level_scales(self):
        return self.base.level_scales

    def delta(self, plan: jax.Array, obs: jax.Array, t: jax.Array) -> jax.Array:
        return self.base.delta(plan, obs, t) + self.residual.delta(plan, obs, t)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as stream:
            cloudpickle.dump(self, stream)

    @staticmethod
    def load(path: str | Path) -> "ContrastField":
        with open(path, "rb") as stream:
            return cloudpickle.load(stream)
