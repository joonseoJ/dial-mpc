"""Assemble a composed policy from four explicitly chosen single-gait fields.

Neither run holds all four gaits: DIAL-driven collection reproduces bound,
student(trotting)-driven DAgger reproduces trot (see memory gait-dagger-rotation,
refined by the per-gait result).  Because each e_i field is a standalone
DialScorePolicy and composition merely selects field_i when driving gait i,
the best field for each gait can be taken from whichever run produced it and
stacked into one identity-basis composed policy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from csm.dial_score import ComposedDialScorePolicy, DialScorePolicy
from csm.omega import nu_matrix

field_paths = [Path(p) for p in sys.argv[1:5]]  # e0 e1 e2 e3
out = Path(sys.argv[5])
# The composed policy carries the temperature its fields were labelled at; it
# must match them, so take it from the fields rather than assuming one.

fields = []
for i, p in enumerate(field_paths):
    fields.append(DialScorePolicy.load(p))
    print(f"e{i}: {p}")

temps = {float(getattr(f, "temperature", None) or 0) for f in fields}
assert len(temps) == 1 and 0 not in temps, f"fields disagree on temperature: {temps}"
T = temps.pop()
print(f"temperature from fields: {T}")
basis = np.eye(4, dtype=np.float32)
nu = nu_matrix(basis, [T] * 4)
policy = ComposedDialScorePolicy(
    policies=tuple(fields),
    mode_weights=jnp.asarray(basis),
    pinv_mode_weights=jnp.asarray(np.linalg.pinv(basis)),
    basis_temperatures=jnp.asarray([T] * 4),
    temperature=T,
    pinv_nu_weights=jnp.asarray(np.linalg.pinv(nu)),
)
out.parent.mkdir(parents=True, exist_ok=True)
policy.save(out)
print(f"saved {out}")
