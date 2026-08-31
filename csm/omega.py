"""Weight-vector conventions: unit omega for direction, temperature for sharpness.

A Gibbs objective is `exp(-(omega . C) / T)`, so only the ratio `omega / T`
enters -- doubling the weights and doubling the temperature is the same
distribution.  Left alone, that redundancy makes "the weights" and "the
temperature" jointly meaningless, and it is what forced DIAL to divide sample
returns by their own standard deviation in the first place: with an arbitrary
weight scale, no fixed temperature could work.

The convention here removes the redundancy by fiat.  Every weight vector
entering the planner or a score field is normalised to unit length, so:

    omega   carries the direction  -- *what* is being optimised
    1 / T   carries the length     -- *how sharply*

and the natural parameter that actually determines the distribution is

    nu = omega / T

Composition then has to be solved in nu, not in omega.  A field trained at
`(omega_i, T_i)` learned the score for `nu_i`, temperature included; asking for
`omega*` at `T*` means finding coefficients with `sum a_i nu_i = nu*`.  Solving
in omega instead is only correct when every field shares one temperature, and
silently returns both the wrong sharpness and the wrong direction when they do
not.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def normalize_omega(omega):
    """Project a weight vector onto the unit sphere, direction preserved.

    A zero vector has no direction to preserve, so it is passed through rather
    than amplified into noise by a division by ~0.
    """

    omega = jnp.asarray(omega, dtype=jnp.float32)
    norm = jnp.linalg.norm(omega)
    return jnp.where(norm > 1e-8, omega / jnp.maximum(norm, 1e-8), omega)


def normalize_omega_np(omega) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    norm = np.linalg.norm(omega)
    return omega / norm if norm > 1e-8 else omega


def nu(omega, temperature):
    """The natural parameter the distribution actually depends on."""

    return normalize_omega(omega) / jnp.asarray(temperature, dtype=jnp.float32)


def nu_matrix(basis_omegas, basis_temperatures) -> np.ndarray:
    """Rows `omega_i / T_i` for a basis of fields."""

    omegas = np.stack([normalize_omega_np(w) for w in basis_omegas])
    temps = np.asarray(basis_temperatures, dtype=float).reshape(-1, 1)
    return omegas / temps


def nu_coefficients(
    basis_omegas, basis_temperatures, target_omega, target_temperature
) -> np.ndarray:
    """Least-squares coefficients that reproduce a target in nu space.

    Exact whenever the nu rows are linearly independent, which needs at least
    as many reward rows as basis fields.  Reduction follows for free: asking
    for field `j` at its own temperature returns `e_j`, and asking for field
    `j` at a different temperature returns `(T_j / T*) e_j` -- the same field,
    retempered.
    """

    matrix = nu_matrix(basis_omegas, basis_temperatures)
    target = normalize_omega_np(target_omega) / float(target_temperature)
    return target @ np.linalg.pinv(matrix)


def mixture_from_pinv(omega, temperature, pinv_nu, pinv_mode=None):
    """Composition coefficients from a pseudoinverse computed at fit time.

    `nu_coefficients` solves this at fit time in numpy; this is the same solve
    for a policy that already carries the pseudoinverse, traceable under jit.
    Three call sites used to inline it -- the policy, the viewer and the
    collector's student driver -- and only two of them carried the legacy
    branch, so a policy fitted before temperatures were recorded worked in one
    and crashed in another.

    With `pinv_nu` the solve is in nu = omega / T and the magnitude is
    recovered.  Without it, the policy predates that and the only thing left to
    pin the scale with is the sum-to-one rescale the old fits used.
    """

    omega = jnp.asarray(omega, dtype=jnp.float32)
    if pinv_nu is None:
        raw = omega @ jnp.asarray(pinv_mode)
        total = jnp.sum(raw)
        # A zero sum means omega is orthogonal to the basis span; leave the raw
        # coefficients rather than dividing by ~0 and returning garbage.
        return jnp.where(jnp.abs(total) > 1e-6, raw / total, raw)
    direction = normalize_omega(omega)
    return (direction / jnp.asarray(temperature, dtype=jnp.float32)) @ (
        jnp.asarray(pinv_nu)
    )


def omega_coefficients(basis_omegas, target_omega) -> np.ndarray:
    """The legacy solve: least squares in omega, rescaled to sum to one.

    Kept for comparison only.  The sum-to-one step exists because DIAL's
    standard-deviation normalisation destroys the magnitude that least squares
    would otherwise recover, and it is what makes the result wrong as soon as
    the fields do not share a temperature.
    """

    basis = np.stack([np.asarray(w, dtype=float) for w in basis_omegas])
    coefficients = np.asarray(target_omega, dtype=float) @ np.linalg.pinv(basis)
    total = coefficients.sum()
    return coefficients / total if abs(total) > 1e-6 else coefficients
