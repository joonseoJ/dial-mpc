"""Neural network architectures for Compositional Score Modeling (CSM).

The key architectural element is :class:`CompositionalDenoisingMLP`: a shared
encoder with k independent output heads, one per cost objective.

Mathematical foundation
-----------------------
For multi-objective MPPI the optimal trajectory distribution is::

    P(τ | ω) ∝ exp(-1/λ · ωᵀ C(τ))

where ``C(τ) ∈ ℝᵏ`` is the per-objective cost vector and ``ω ∈ ℝᵏ`` is the
preference weight vector.  Taking the gradient of the log-probability yields a
**score function that is linear in ω**::

    ∇_τ log P(τ | ω) = -1/λ · Σᵢ ωᵢ ∇_τ Cᵢ(τ)

This linearity is the core insight: instead of feeding ``ω`` into a black-box
network, we learn k independent basis vector fields ``vᵢ(s, τ) ≈ ∇_τ Cᵢ(τ)``
and combine them at inference time::

    v_total(s, τ, ω) = Σᵢ ωᵢ · vᵢ(s, τ)

The shared encoder captures common state representations while independent
heads prevent objective interference.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """A simple multi-layer perceptron with swish activations.

    All hidden layers apply the swish activation function.  The final
    (output) layer is linear.
    """

    def __init__(self, layer_sizes: Sequence[int], rngs: nnx.Rngs) -> None:
        """Initialize the MLP.

        Args:
            layer_sizes: Sizes of all layers, including input and output.
            rngs: Random number generators for weight initialization.
        """
        self.num_hidden = len(layer_sizes) - 2
        for i, (in_sz, out_sz) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            setattr(self, f"l{i}", nnx.Linear(in_sz, out_sz, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass."""
        for i in range(self.num_hidden):
            x = nnx.swish(getattr(self, f"l{i}")(x))
        return getattr(self, f"l{self.num_hidden}")(x)


class IdentityNormalizer(nnx.Module):
    """Normalizer with the same call interface as Flax BatchNorm."""

    def __call__(
        self, x: jax.Array, use_running_average: bool = True
    ) -> jax.Array:
        del use_running_average
        return x


class StandardNormalizer(nnx.Module):
    """Fixed observation standardization fitted from collected DIAL states."""

    def __init__(self, observation_size: int, epsilon: float = 1e-6) -> None:
        self.mean = nnx.Variable(jnp.zeros((observation_size,)))
        self.var = nnx.Variable(jnp.ones((observation_size,)))
        self.epsilon = float(epsilon)

    def fit(self, observations: jax.Array) -> None:
        """Update statistics from raw observations."""

        observations = jnp.asarray(observations)
        if observations.ndim != 2 or observations.shape[1] != self.mean.value.shape[0]:
            raise ValueError(
                "observations must have shape (samples, observation_size)"
            )
        self.mean.value = jnp.mean(observations, axis=0)
        self.var.value = jnp.var(observations, axis=0)

    def __call__(
        self, x: jax.Array, use_running_average: bool = True
    ) -> jax.Array:
        del use_running_average
        return (x - self.mean.value) / jnp.sqrt(self.var.value + self.epsilon)


class CompositionalDenoisingMLP(nnx.Module):
    """Shared encoder + N independent heads for compositional score modeling.

    Each head ``j`` predicts the score of the noised trajectory distribution
    for training mode ``j``::

        s_θ,j(U_σ, t, y) ≈ ∇_U log p_σ,j(U|y) = −ε/σ

    Architecture::

        input = concat([U_flat, y, t])           # (H·nu + obs + 1,)
        latent = encoder(input)                   # (latent_size,)
        s_j    = head_j(latent)  for j=0..N-1    # each (H·nu,)

    where ``t = log(σ/σ_min) / log(σ_max/σ_min) ∈ [0,1]`` is the noise-level
    conditioning signal (t=0 near-clean, t=1 heavily noised).

    Inference::

        c = omega @ pinv(W)          # composition coefficients (N,)
        s_total = Σⱼ cⱼ · s_θ,j     # (H, nu), composed score

    The shared encoder learns common state/action representations while the
    independent heads each specialise in one training mode's score direction.
    During training, only one head contributes to the loss per gradient step
    (the head matching the current mode).  Because ``forward_all`` stacks head
    outputs and ``jax.lax.dynamic_index_in_dim`` selects head ``j``, the
    backward pass propagates gradients only to head ``j`` and the shared
    encoder — cross-mode gradient interference is naturally prevented without
    explicit stop-gradient calls.

    Attributes:
        action_size: Dimension of a single action vector ``u``.
        observation_size: Dimension of the observation vector ``y``.
        horizon: Number of action steps in the sequence ``U = [u₀, …, u_{H-1}]``.
        num_objectives: Number of cost objectives ``k``.
        encoder: Shared MLP backbone that maps ``(U_flat, y, t)`` to a latent.
        head{i}: k independent MLP heads, one per objective.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        num_objectives: int,
        encoder_hidden: Sequence[int],
        head_hidden: Sequence[int],
        rngs: nnx.Rngs,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        edm: bool = False,
        sigma_data: float = 0.5,
        sigma_min: float = 1e-2,
        sigma_max: float = 1.0,
    ) -> None:
        """Initialize the compositional denoising network.

        Args:
            action_size: Dimension of the action space ``u``.
            observation_size: Dimension of the observation vector ``y``.
            horizon: Number of steps in the action sequence.
            num_objectives: Number of independent cost objectives ``k``.
            encoder_hidden: Hidden layer sizes for the shared encoder.  The
                last value is the latent dimension passed to each head.
            head_hidden: Hidden layer sizes for each independent head
                (excluding the input latent and the output layer).
            rngs: Random number generators for weight initialization.
            fourier_features: If ``> 0``, map the network input through a fixed
                random Fourier feature embedding ``[sin(2π x B), cos(2π x B)]``
                (Tancik et al. 2020) of this many frequencies before the
                encoder, to counter MLP spectral bias and fit the
                high-frequency small-σ score.  ``0`` disables it (default).
            fourier_scale: Std of the random Fourier projection ``B``; larger
                values emphasise higher frequencies.
            edm: If ``True``, apply EDM-style σ-preconditioning (Karras et al.
                2022): scale the network input by ``c_in(σ)`` and its output by
                the score scale ``c_out(σ) = σ_data / (σ·√(σ²+σ_data²))``, so
                the backbone predicts a unit-variance quantity at every σ and
                the large small-σ score comes from the ``c_out`` scaling rather
                than the (bounded) network output.  A pure output *scale*
                (no denoiser skip term) is used so per-head scores stay linear
                in the composition weights.  Default off.
            sigma_data: EDM data-scale hyper-parameter σ_data.
            sigma_min: Minimum σ, used to recover σ from the conditioning
                ``t = log(σ/σ_min)/log(σ_max/σ_min)`` (EDM only).
            sigma_max: Maximum σ (EDM only).
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.horizon = horizon
        self.num_objectives = num_objectives

        # Preconditioning / feature-embedding configuration.
        self.fourier_features = int(fourier_features)
        self.fourier_scale = float(fourier_scale)
        self.edm = bool(edm)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        input_size = horizon * action_size + observation_size + 1
        latent_size = encoder_hidden[-1]
        output_size = horizon * action_size

        # Optional fixed random Fourier feature projection of the raw input.
        # Stored as a non-trainable buffer so it is preserved on save/load but
        # never updated by the optimizer.
        if self.fourier_features > 0:
            # Normalise by sqrt(input_size) so a projection row x·B_k has std
            # ~ fourier_scale·rms(x) regardless of input dimensionality — i.e.
            # fourier_scale is an input-dim-independent frequency bandwidth
            # (a feature completes ~fourier_scale cycles over a unit change in
            # x).  Without this normalisation the dot-product variance grows
            # with input_size, driving the phases to tens of radians so the
            # sin/cos features become noise and the encoder learns nothing.
            b = (
                jax.random.normal(rngs.params(), (input_size, self.fourier_features))
                * (self.fourier_scale / jnp.sqrt(input_size))
            )
            self.fourier_B = nnx.Variable(b)
            encoder_in = 2 * self.fourier_features
        else:
            encoder_in = input_size

        # Shared encoder: (encoder_in,) → (latent_size,)
        self.encoder = MLP(
            [encoder_in] + list(encoder_hidden), rngs=rngs
        )

        # k independent heads: (latent_size,) → (output_size,)
        for i in range(num_objectives):
            setattr(
                self,
                f"head{i}",
                MLP([latent_size] + list(head_hidden) + [output_size], rngs=rngs),
            )

    # ------------------------------------------------------------------ #
    # Preconditioning helpers (EDM)
    # ------------------------------------------------------------------ #

    def _sigma_from_t(self, t: jax.Array) -> jax.Array:
        """Recover σ from the log-uniform conditioning ``t`` ∈ [0, 1]."""
        log_min = jnp.log(self.sigma_min)
        log_max = jnp.log(self.sigma_max)
        return jnp.exp(log_min + t * (log_max - log_min))  # same shape as t

    def _c_in(self, sigma: jax.Array) -> jax.Array:
        """EDM input scaling ``1/√(σ²+σ_data²)``."""
        return 1.0 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def _c_out(self, sigma: jax.Array) -> jax.Array:
        """EDM score-output scaling ``σ_data/(σ·√(σ²+σ_data²))``.

        This is the EDM denoiser ``c_out`` divided by σ² — i.e. the natural
        magnitude of the *score* ``(D − u)/σ²`` — so the backbone output stays
        ~unit-variance across σ.  It is a per-sample scalar, so multiplying
        each head by it preserves linearity of the composed score in ω.
        """
        return self.sigma_data / (sigma * jnp.sqrt(sigma**2 + self.sigma_data**2))

    # ------------------------------------------------------------------ #
    # Core forward methods
    # ------------------------------------------------------------------ #

    def _encode(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Compute the shared latent representation.

        Args:
            u: Action sequence of shape ``(*batch, H, nu)``.
            y: Observation of shape ``(*batch, obs_size)``.
            t: Flow time of shape ``(*batch, 1)``.

        Returns:
            Latent of shape ``(*batch, latent_size)``.
        """
        batches = u.shape[:-2]
        # EDM input preconditioning: scale u by c_in(σ) (per-sample scalar).
        if getattr(self, "edm", False):
            sigma = self._sigma_from_t(t)  # (*batch, 1)
            u = u * self._c_in(sigma).reshape(batches + (1, 1))
        u_flat = u.reshape(batches + (self.horizon * self.action_size,))
        x = jnp.concatenate([u_flat, y, t], axis=-1)
        # Optional random Fourier feature embedding of the raw input.
        if getattr(self, "fourier_features", 0) > 0:
            proj = 2.0 * jnp.pi * (x @ self.fourier_B.value)
            x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        return self.encoder(x)

    def forward_all(
        self, u: jax.Array, y: jax.Array, t: jax.Array
    ) -> jax.Array:
        """Compute all k basis vector fields from the shared latent.

        Args:
            u: Action sequence of shape ``(*batch, H, nu)``.
            y: Observation of shape ``(*batch, obs_size)``.
            t: Flow time of shape ``(*batch, 1)``.

        Returns:
            Stacked vector fields of shape ``(k, *batch, H, nu)``, one per
            objective.  The first axis indexes the objective.
        """
        batches = u.shape[:-2]
        latent = self._encode(u, y, t)

        # Each head produces (*batch, H*nu); stack along a new leading axis.
        heads = jnp.stack(
            [getattr(self, f"head{i}")(latent) for i in range(self.num_objectives)],
            axis=0,
        )  # (k, *batch, H*nu)

        scores = heads.reshape(
            (self.num_objectives,) + batches + (self.horizon, self.action_size)
        )
        # EDM output preconditioning: the per-head score is c_out(σ)·F.  c_out
        # is a per-sample scalar broadcast over (k, H, nu), so the composed
        # score Σⱼ cⱼ·scoreⱼ stays linear in the composition weights.
        if getattr(self, "edm", False):
            sigma = self._sigma_from_t(t)  # (*batch, 1)
            c_out = self._c_out(sigma).reshape((1,) + batches + (1, 1))
            scores = scores * c_out
        return scores

    def __call__(
        self,
        u: jax.Array,
        y: jax.Array,
        t: jax.Array,
        omega: jax.Array,
    ) -> jax.Array:
        """Compute the combined vector field for preference weights ``omega``.

        At inference time, the combined vector field is the linear combination::

            v_total = Σᵢ ωᵢ · vᵢ(u, y, t)

        which inherits the mathematical linearity guarantee of the exponential
        family optimal trajectory distribution.

        Args:
            u: Action sequence of shape ``(*batch, H, nu)``.
            y: Observation of shape ``(*batch, obs_size)``.
            t: Flow time of shape ``(*batch, 1)``.
            omega: Preference weight vector of shape ``(k,)``.  Need not sum
                to 1; the caller is responsible for normalization if desired.

        Returns:
            Combined vector field of shape ``(*batch, H, nu)``.
        """
        all_v = self.forward_all(u, y, t)  # (k, *batch, H, nu)
        k = self.num_objectives
        # Broadcast omega: (k,) → (k, *1..., 1, 1) to match all_v dimensions
        omega_bc = omega.reshape([k] + [1] * (all_v.ndim - 1))
        return (all_v * omega_bc).sum(axis=0)  # (*batch, H, nu)


class VectorValueMLP(nnx.Module):
    """Shared encoder + k independent scalar value heads.

    Estimates the infinite-horizon discounted cost vector under a fixed
    TCMPPI policy with cost weights ``ω``::

        V_θ(x) ∈ ℝᵏ,    V_θ(x)_i = E_π[Σ_{t=0}^∞ γ^t c_i(x_t, u_t) | x₀ = x]

    where ``c_i(x, u)`` is the raw (unweighted) stage cost for objective ``i``
    and the expectation is over trajectories induced by TCMPPI with weights ``ω``.

    This satisfies the Bellman equation component-wise::

        V_θ(x_t) = c(x_t, u_t) + γ V_{θ⁻}(x_{t+1})

    and is trained with the TD loss::

        L(θ) = E_{(x_t,u_t,x_{t+1})~D}[‖V_θ(x_t) − (c_t + γ V_{θ⁻}(x_{t+1}))‖²]

    where ``θ⁻`` denotes the periodically-updated target network parameters.

    Architecture::

        input  = y               # (*batch, obs_size)
        latent = encoder(y)      # (*batch, latent_size)
        V_i    = vh_i(latent)    # (*batch, 1)  per objective
        output = concat(V_i)     # (*batch, k)

    Attributes:
        obs_size: Dimension of the observation vector ``y``.
        num_objectives: Number of cost objectives ``k``.
        encoder: Shared MLP backbone mapping obs to a latent.
        vh{i}: k independent scalar value heads.
    """

    def __init__(
        self,
        obs_size: int,
        num_objectives: int,
        encoder_hidden: Sequence[int],
        head_hidden: Sequence[int],
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the vector value network.

        Args:
            obs_size: Dimension of the observation vector ``y``.
            num_objectives: Number of independent cost objectives ``k``.
            encoder_hidden: Hidden layer sizes for the shared encoder.
                The last value is the latent dimension passed to each head.
            head_hidden: Hidden layer sizes for each value head
                (excluding the latent input and the scalar output).
            rngs: Random number generators for weight initialization.
        """
        self.obs_size = obs_size
        self.num_objectives = num_objectives

        latent_size = encoder_hidden[-1]
        self.encoder = MLP([obs_size] + list(encoder_hidden), rngs=rngs)
        for i in range(num_objectives):
            setattr(
                self,
                f"vh{i}",
                MLP([latent_size] + list(head_hidden) + [1], rngs=rngs),
            )

    def __call__(self, y: jax.Array) -> jax.Array:
        """Predict the k-dimensional discounted cost value vector.

        Args:
            y: Observation of shape ``(*batch, obs_size)``.

        Returns:
            Value vector of shape ``(*batch, k)``.
        """
        latent = self.encoder(y)
        return jnp.concatenate(
            [getattr(self, f"vh{i}")(latent) for i in range(self.num_objectives)],
            axis=-1,
        )


class CompositionalEnergyMLP(nnx.Module):
    """Shared trajectory encoder with one scalar energy per objective.

    Unlike a linearly-composed MPPI update, the raw multi-objective trajectory
    cost really is compositional.  This module therefore predicts

    ``E(o, U) = [E_tracking, E_stability, E_gait]``

    and leaves preference composition to the scalar energy
    ``E_omega = omega @ E``.  Control is obtained by differentiating that
    scalar with respect to the complete action-node sequence ``U``.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        num_objectives: int,
        encoder_hidden: Sequence[int],
        head_hidden: Sequence[int],
        rngs: nnx.Rngs,
    ) -> None:
        self.action_size = int(action_size)
        self.observation_size = int(observation_size)
        self.horizon = int(horizon)
        self.num_objectives = int(num_objectives)
        # Converts a physical cost gradient into a controller displacement.
        # Sobolev supervision determines the energy gradients themselves;
        # DIAL magnitude calibration learns only this separate positive scale.
        self.log_update_scale = nnx.Param(jnp.log(jnp.asarray(0.1)))

        input_size = self.horizon * self.action_size + self.observation_size
        latent_size = int(encoder_hidden[-1])
        self.encoder = MLP(
            [input_size] + list(encoder_hidden), rngs=rngs
        )
        for i in range(self.num_objectives):
            setattr(
                self,
                f"energy_head{i}",
                MLP([latent_size] + list(head_hidden) + [1], rngs=rngs),
            )

    def forward_all(self, u: jax.Array, y: jax.Array) -> jax.Array:
        """Return normalized scalar energies with shape ``(*batch, k)``."""
        batches = u.shape[:-2]
        u_flat = u.reshape(batches + (self.horizon * self.action_size,))
        latent = self.encoder(jnp.concatenate([u_flat, y], axis=-1))
        return jnp.concatenate(
            [
                getattr(self, f"energy_head{i}")(latent)
                for i in range(self.num_objectives)
            ],
            axis=-1,
        )

    def __call__(self, u: jax.Array, y: jax.Array) -> jax.Array:
        return self.forward_all(u, y)
