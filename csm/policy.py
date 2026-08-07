"""CompositionalPolicy: score-based inference with linear score composition.

Training (DSM)
--------------
N score networks ``s_θ,j`` are learned, one per predefined training mode ``ωⱼ``.
Each is trained via denoising score matching (DSM) to approximate::

    s_θ,j(U_σ, σ, x) ≈ ∇_U log p_σ,j(U|x) = −ε/σ

where ``U_σ = U* + σε`` corrupts the best MPPI trajectory at noise level σ.

Mode matrix and pseudoinverse
------------------------------
The N training mode weight vectors form the mode matrix::

    W  (shape N × k)   — rows are the training mode weight vectors

At inference with target cost weight ``ω`` (shape k), the composition
coefficients ``c`` (shape N) are computed via the pseudoinverse::

    c = ω @ pinv(W)

The composed score function is::

    s_total(U_σ, σ, x) = Σⱼ cⱼ · s_θ,j(U_σ, σ, x)
                       ≈ ∇_U log p_σ(U|x, ω)

This inherits the linearity of the exponential-family score::

    ∇_U log p_σ(U|x, ω) = Σᵢ ωᵢ · ∇_U log p_σ,i(U|x)

When ω lies in the row span of W the reconstruction is exact; otherwise ``c``
minimises ``‖c W − ω‖²`` (least-squares approximation).

Inference (annealed score ascent)
----------------------------------
Consistent with the MPPI update formula ``δU ≈ σ² ∇_U log p_σ(U|x)``, the
inference performs annealed score ascent from σ_max → σ_min::

    for σ in [σ_max, …, σ_min]:
        U ← clip(U + dt · σ² · s_total(U, σ, x), −1, 1)

Starting with ``U = prev_norm + σ_max · (1 − warm_start) · ε`` and annealing
σ downward gives coarse-to-fine refinement that mirrors MPPI's behaviour.

Alternative: DDIM-style deterministic sampling
-----------------------------------------------
A faster deterministic alternative replaces the Langevin update with the
DDIM denoising step (Tweedie estimate + interpolation)::

    U_clean_hat = U + σ² · s_total(U, σ)
    U_next = (σ_next/σ) · U + (1 − σ_next/σ) · U_clean_hat

This requires fewer annealing steps but is harder to warm-start.

Special case — one-hot training
---------------------------------
When W = I (pendulum), ``pinv(I) = I`` and ``c = ω``.  The inference is then
standard annealed score ascent with direct weight composition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import cloudpickle
from flax import nnx
from flax.struct import dataclass

from csm.architectures import CompositionalDenoisingMLP


@dataclass
class CompositionalPolicy:
    """A pickle-able Compositional Score Modeling policy.

    Generates action sequences via annealed score ascent consistent with
    the MPPI update formula ``δU ≈ σ² ∇_U log p_σ(U|x, ω)``.

    Unlike the GPC :class:`~gpc.policy.Policy`, the observation ``y`` does
    **not** include cost weights.  Instead, ``ω`` is mapped to composition
    coefficients via ``pinv(mode_weights)`` at the score level, preserving
    the mathematical linearity of the exponential-family score function.

    Attributes:
        model: The :class:`CompositionalDenoisingMLP` with shared encoder and
            N independent heads (one per training mode).  Each head predicts
            the score ``∇_U log p_σ,j(U|x) ≈ −ε/σ``.
        normalizer: Observation normalization module (e.g. ``nnx.BatchNorm``).
        u_min: Minimum action values, shape ``(nu,)``.
        u_max: Maximum action values, shape ``(nu,)``.
        mode_weights: Training mode weight matrix of shape ``(N, k)``.  Row
            ``j`` is the raw cost weight vector ``ωⱼ`` used to collect data
            for head ``j``.  At inference, ``pinv(mode_weights)`` maps the
            target cost weight ``ω`` (shape ``(k,)``) to composition
            coefficients ``c`` (shape ``(N,)``).
        pinv_mode_weights: Precomputed pseudoinverse of ``mode_weights``,
            shape ``(k, N)``.  Stored explicitly so that inference never calls
            ``jnp.linalg.pinv`` (which uses cuSolver / SVD) inside a JIT-
            compiled function.  This avoids a runtime cuSolver handle conflict
            when JAX and Warp share the same GPU.  Compute once with
            ``jnp.array(np.linalg.pinv(np.asarray(mode_weights)))`` before
            constructing the policy.
        sigma_min: Minimum noise level (end of annealing schedule).
        sigma_max: Maximum noise level (start of annealing schedule).
        num_steps: Number of annealing steps from σ_max → σ_min.
        dt: Step-size multiplier for the annealed score update.
            Actual step size at each level: ``dt · σ²``.
    """

    model: CompositionalDenoisingMLP
    normalizer: nnx.Module
    u_min: jax.Array
    u_max: jax.Array
    mode_weights: jax.Array
    pinv_mode_weights: jax.Array
    sigma_min: float = 1e-2
    sigma_max: float = 1.0
    num_steps: int = 50
    dt: float = 1.0
    predicts_update: bool = False
    shift_matrix: jax.Array | None = None
    lock_first_action: bool = False
    inference_sigmas: jax.Array | None = None

    def save(self, path: Union[Path, str]) -> None:
        """Save the policy, including Flax initializer closures.

        Args:
            path: Destination file path.
        """
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "CompositionalPolicy":
        """Load a policy from a cloudpickle file.

        Handles policies saved before ``pinv_mode_weights`` was added as an
        explicit field: if the attribute is absent, it is recomputed from
        ``mode_weights`` using numpy (no GPU/cuSolver required) and a new
        policy object is returned with all fields populated.

        Args:
            path: Source file path.

        Returns:
            The loaded :class:`CompositionalPolicy` instance.
        """
        with open(path, "rb") as f:
            obj = cloudpickle.load(f)

        # Training checkpoints are saved as dicts with a "policy" key.
        if isinstance(obj, dict) and "policy" in obj:
            policy = obj["policy"]
        else:
            policy = obj

        if not hasattr(policy, "pinv_mode_weights"):
            pinv_mw = jnp.array(np.linalg.pinv(np.asarray(policy.mode_weights)))
            policy = CompositionalPolicy(
                policy.model,
                policy.normalizer,
                policy.u_min,
                policy.u_max,
                policy.mode_weights,
                pinv_mw,
                sigma_min=policy.sigma_min,
                sigma_max=policy.sigma_max,
                num_steps=policy.num_steps,
                dt=policy.dt,
            )

        return policy

    def apply(
        self,
        prev: jax.Array,
        y: jax.Array,
        rng: jax.Array,
        warm_start_level: float = 0.0,
        *,
        omega: jax.Array,
    ) -> jax.Array:
        """Generate an action sequence for target cost weights ``omega``.

        Composition coefficients are computed from ``omega`` via the
        pseudoinverse of the training mode weight matrix::

            c = omega @ pinv(mode_weights)   # shape (N,)

        The initial sample is::

            U₀ = prev_norm + σ_max · (1 − warm_start) · noise

        where ``prev_norm`` is the previous action sequence normalised to
        ``[−1, 1]``.  Annealed score ascent then refines ``U₀``::

            for σ in [σ_max, …, σ_min]:
                t = log(σ/σ_min) / log(σ_max/σ_min)    # ∈ [0, 1]
                s = Σⱼ cⱼ · s_θ,j(U, t, y)             # composed score
                U ← clip(U + dt · σ² · s,  −1, 1)

        Each step corresponds to one MPPI-style update at the current noise
        level σ (step size ``dt · σ²``).  Decreasing σ gives coarse-to-fine
        refinement.

        Typical usage — JIT-compile with fixed omega::

            jit_apply = jax.jit(
                functools.partial(policy.apply, omega=omega, warm_start_level=1.0)
            )
            U = jit_apply(prev, y, rng)

        Args:
            prev: Previous action sequence of shape ``(H, nu)``, in
                ``[u_min, u_max]``.  Normalised internally before use.
            y: Current observation of shape ``(obs_size,)``.
            rng: JAX PRNG key for the initial noise sample.
            warm_start_level: Warm-start degree in ``[0, 1]``.
                ``1.0`` starts from ``prev_norm`` (no added noise — pure
                refinement). ``0.0`` starts from ``prev_norm + σ_max · noise``
                (maximum exploration from the warm-start position).
            omega: Target cost weight vector of shape ``(k,)`` (keyword-only),
                in the same scale as the rows of ``mode_weights``.

        Returns:
            Optimised action sequence of shape ``(H, nu)`` in ``[u_min, u_max]``.
        """
        # Map target cost weights → composition coefficients via pseudoinverse.
        # pinv_mode_weights is precomputed at policy construction time using
        # numpy (LAPACK), so no cuSolver call happens here inside JIT.
        # shape: (k,) @ (k, N) → (N,)
        c = omega @ self.pinv_mode_weights

        # Normalise observation using running statistics (inference mode).
        y = self.normalizer(y, use_running_average=True)

        # Normalise prev from [u_min, u_max] to [−1, 1].
        act_mean = (self.u_max + self.u_min) / 2.0
        act_scale = (self.u_max - self.u_min) / 2.0
        prev_norm = (prev - act_mean) / act_scale

        # Initialise: add σ_max-level noise scaled by (1 − warm_start_level).
        warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        noise = jax.random.normal(rng, prev.shape)
        U = prev_norm + self.sigma_max * (1.0 - warm_start_level) * noise
        U = jnp.clip(U, -1.0, 1.0)

        # Pre-compute the annealing schedule.
        if getattr(self, "inference_sigmas", None) is not None:
            sigmas = jnp.asarray(self.inference_sigmas)
        else:
            sigmas = jnp.exp(
                jnp.linspace(
                    jnp.log(self.sigma_max),
                    jnp.log(self.sigma_min),
                    self.num_steps,
                )
            )  # (num_steps,), decreasing

        log_sigma_range = jnp.log(self.sigma_max) - jnp.log(self.sigma_min)

        def _score_step(
            U: jax.Array, sigma: jax.Array
        ) -> Tuple[jax.Array, None]:
            # Map σ → t ∈ [0,1] matching the training schedule.
            # t=1 → σ_max (high noise),  t=0 → σ_min (near clean).
            t = jnp.array(
                [(jnp.log(sigma) - jnp.log(self.sigma_min)) / log_sigma_range]
            )  # shape (1,)

            # Composed score: Σⱼ cⱼ · s_θ,j(U, t, y)
            s = self.model(U, y, t, c)  # (H, nu)

            # New policies directly predict the bounded MPPI update delta_U.
            # Older checkpoints remain loadable and retain score semantics.
            delta = s if getattr(self, "predicts_update", False) else sigma**2 * s
            if getattr(self, "lock_first_action", False):
                delta = delta.at[0].set(jnp.zeros_like(delta[0]))
            U = jnp.clip(U + self.dt * delta, -1.0, 1.0)
            return U, None

        # Anneal from σ_max → σ_min.
        U, _ = jax.lax.scan(_score_step, U, sigmas)

        # Rescale from [−1, 1] to [u_min, u_max].
        return U * act_scale + act_mean

    def shift(self, sequence: jax.Array) -> jax.Array:
        """Shift a plan by one DIAL environment step for warm-starting."""

        sequence = jnp.asarray(sequence)
        if getattr(self, "shift_matrix", None) is not None:
            return jnp.einsum("ij,ja->ia", self.shift_matrix, sequence)
        shifted = jnp.roll(sequence, -1, axis=0)
        return shifted.at[-1].set(jnp.zeros_like(shifted[-1]))
