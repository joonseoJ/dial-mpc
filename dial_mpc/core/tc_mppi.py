"""Time-correlated Gaussian utilities for DIAL-TC-MPPI.

The implementation follows equations (13)--(23) of Lee and Lee,
"Time-Correlated Model Predictive Path Integral" (ICRA 2025).  It operates
on DIAL-MPC's action nodes; the existing spline then maps the correlated
nodes to the simulator-rate action sequence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jax import numpy as jnp


def difference_operator(length: int, order: int, dt: float) -> np.ndarray:
    """Returns the finite-difference operator D^(order) from paper eq. (13)."""

    if length <= 0:
        raise ValueError("length must be positive")
    if order < 0 or order >= length:
        raise ValueError("order must satisfy 0 <= order < length")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    operator = np.eye(length, dtype=np.float64)
    for _ in range(order):
        rows = operator.shape[0] - 1
        first_difference = np.zeros((rows, rows + 1), dtype=np.float64)
        indices = np.arange(rows)
        first_difference[indices, indices] = -1.0 / dt
        first_difference[indices, indices + 1] = 1.0 / dt
        operator = first_difference @ operator
    return operator


class TimeCorrelatedGaussian:
    """Conditional time-correlated Gaussian over one DIAL action-node plan.

    Matrices are scalar in the action dimension and shared by every actuator.
    This is equivalent to the Kronecker product with an identity action-cost
    matrix, while avoiding a large ``(horizon * action_size)^2`` allocation.
    """

    def __init__(
        self,
        horizon: int,
        history_length: int,
        dt: float,
        derivative_weights: Sequence[float],
        regularization: float = 1e-6,
    ):
        if history_length < 1:
            raise ValueError("history_length must be at least one")
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        if len(derivative_weights) != history_length + 1:
            raise ValueError(
                "derivative_weights must contain history_length + 1 values"
            )
        if derivative_weights[0] <= 0.0:
            raise ValueError("the zero-order derivative weight must be positive")
        if any(weight < 0.0 for weight in derivative_weights):
            raise ValueError("derivative weights must be non-negative")

        self.horizon = horizon
        self.history_length = history_length
        total = history_length + horizon

        differences = [
            difference_operator(total, order, dt)
            for order in range(history_length + 1)
        ]
        precision = sum(
            float(weight) * (operator.T @ operator)
            for weight, operator in zip(derivative_weights, differences)
        )
        precision += regularization * np.eye(total)

        # The paper uses derivatives 0..d-1 for the prior gradient and only
        # derivative 0 for the estimated/sampling gradient (eq. 17).
        prior_gradient = -sum(
            float(derivative_weights[order])
            * (differences[order].T @ differences[order])
            for order in range(history_length)
        )
        estimate_gradient = -float(derivative_weights[0]) * np.eye(total)

        d = history_length
        self.precision = jnp.asarray(precision)
        self.tail_precision = jnp.asarray(precision[d:, d:])
        self.tail_head_precision = jnp.asarray(precision[d:, :d])
        self.prior_gradient_tail = jnp.asarray(prior_gradient[d:, :])
        self.estimate_gradient_tail = jnp.asarray(estimate_gradient[d:, :])
        self.tail_cholesky = jnp.linalg.cholesky(self.tail_precision)
        self.covariance_sqrt = jnp.linalg.solve(
            self.tail_cholesky.T, jnp.eye(horizon)
        )

    def conditional_means(self, history, estimate, reference=None):
        """Computes the prior and sampling means from paper eqs. (18)-(19)."""

        history = jnp.asarray(history)
        estimate = jnp.asarray(estimate)
        if reference is None:
            reference = jnp.zeros_like(estimate)
        else:
            reference = jnp.asarray(reference)

        zero_reference_history = jnp.zeros_like(history)
        full_reference = jnp.concatenate(
            [zero_reference_history, reference], axis=0
        )
        full_estimate = jnp.concatenate([history, estimate], axis=0)
        boundary = self.tail_head_precision @ history
        prior_rhs = boundary + self.prior_gradient_tail @ full_reference
        estimate_rhs = boundary + self.estimate_gradient_tail @ full_estimate
        prior_mean = -jnp.linalg.solve(self.tail_precision, prior_rhs)
        sampling_mean = -jnp.linalg.solve(self.tail_precision, estimate_rhs)
        return prior_mean, sampling_mean

    def transform_standard_normal(self, epsilon, noise_scale):
        """Maps iid noise to N(0, S H_tt^-1 S), with S=diag(noise_scale)."""

        # If H = L L^T, L^-T epsilon has covariance H^-1.
        correlated = jnp.einsum("ts,ksa->kta", self.covariance_sqrt, epsilon)
        return correlated * noise_scale[None, :, None]

    def log_importance_ratio(
        self, actions, prior_mean, sampling_mean, noise_scale
    ):
        """Returns (barU-hatU)^T S^-1 H_tt S^-1 V from paper eq. (23)."""

        safe_scale = jnp.maximum(noise_scale, 1e-6)
        scaled_difference = (prior_mean - sampling_mean) / safe_scale[:, None]
        scaled_actions = actions / safe_scale[None, :, None]
        precision_difference = self.tail_precision @ scaled_difference
        return jnp.einsum("ta,kta->k", precision_difference, scaled_actions)
