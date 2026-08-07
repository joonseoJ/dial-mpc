"""Training utilities for Compositional Score Modeling (CSM).

Two training objectives are available:

1. **DSM** (Denoising Score Matching) — the original approach.
2. **Score Regression** — direct regression onto multi-point MPPI score
   estimates (Direction 5).

DSM (legacy)
------------
For each head ``i``, given a target trajectory ``U*`` (MPPI best):

1. Sample σ ~ LogUniform(σ_min, σ_max)
2. Corrupt: ``U_σ = U* + σ · ε``,  ``ε ~ N(0, I)``
3. Train ``s_θ,i(U_σ, σ, x) ≈ −ε/σ``

Loss: ``L_i = ‖σ · s_θ,i(U_σ, σ, x) + ε‖²``

Score Regression (Direction 5)
------------------------------
Instead of learning from synthetically corrupted MPPI outputs, we directly
regress the MPPI-estimated score at multiple query points::

    L_i = ‖s_θ,i(U_point, t_σ, x) − score_target‖²

where ``score_target = δU / σ²`` is the MPPI update divided by σ², computed
at ``(U_point, σ)`` via :func:`~csm.data_collection.collect_multipoint_mppi_scores`.

This better preserves the linearity of the Gibbs score in ``ω``, because the
training targets come from the Gibbs distribution directly (via MPPI) rather
than from the MPPI *output* distribution (via DSM).

Gradient routing
----------------
At each step one objective index ``i`` is sampled.  ``CompositionalDenoisingMLP
.forward_all`` stacks all head outputs; ``jax.lax.dynamic_index_in_dim``
selects head ``i``.  Gradients flow only to head ``i`` and the shared encoder
— all other heads get zero gradient, matching the CSM requirement.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm.auto import tqdm

from csm.architectures import CompositionalDenoisingMLP


def fit_compositional_policy(
    obs_per_obj: Sequence[jax.Array],
    acts_per_obj: Sequence[jax.Array],
    model: CompositionalDenoisingMLP,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_iters: int,
    rng: jax.Array,
    sigma_min: float = 1e-2,
    sigma_max: float = 1.0,
) -> jax.Array:
    """Train each head on its mode's demonstration data using DSM.

    At each gradient step one mode index ``i`` is sampled.  A mini-batch from
    dataset ``i`` is corrupted with noise level σ ~ LogUniform(σ_min, σ_max)
    and the head ``i`` score prediction is trained via the noise-prediction
    form of the DSM loss:

        L_i = ‖σ · s_θ,i(U + σε, t_σ, y) + ε‖²

    where ``t_σ = log(σ/σ_min) / log(σ_max/σ_min) ∈ [0,1]`` is passed to the
    network as a scalar conditioning signal.

    Args:
        obs_per_obj: Sequence of N observation arrays, one per mode.
            Each has shape ``(T_i, obs_size)``.
        acts_per_obj: Sequence of N best-trajectory arrays from MPPI,
            each of shape ``(T_i, H, nu)``.
        model: :class:`~csm.architectures.CompositionalDenoisingMLP` to train
            in place.
        optimizer: Flax NNX optimizer, updated in place.
        batch_size: Mini-batch size per gradient step.
        num_iters: Gradient steps **per mode** (total = ``num_iters * N``).
        rng: JAX PRNG key.
        sigma_min: Minimum noise level (near-clean trajectories).
        sigma_max: Maximum noise level (heavily corrupted trajectories).

    Returns:
        Scalar loss from the final gradient step.
    """
    k = model.num_objectives
    log_sigma_min = jnp.log(sigma_min)
    log_sigma_max = jnp.log(sigma_max)

    def _loss_fn(
        model: CompositionalDenoisingMLP,
        obs: jax.Array,
        act: jax.Array,
        noise: jax.Array,
        t: jax.Array,
        obj_idx: jax.Array,
    ) -> jax.Array:
        """DSM noise-prediction loss for mode head ``obj_idx``.

        Args:
            obs: Observations, shape ``(batch, obs_size)``.
            act: Target (best MPPI) trajectories, shape ``(batch, H, nu)``.
            noise: Standard Gaussian noise, shape ``(batch, H, nu)``.
            t: Log-sigma index in [0, 1], shape ``(batch, 1)``.
                t=0 → σ_min (nearly clean),  t=1 → σ_max (heavily noised).
            obj_idx: Scalar mode index selecting which head to update.
        """
        # Map t → σ via log-linear schedule
        sigma = jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))
        # sigma: (batch, 1)

        # Corrupt target trajectory: U_σ = U* + σ · ε
        # sigma[..., None] broadcasts to (batch, 1, 1) against (batch, H, nu)
        noised_act = act + sigma[..., None] * noise

        # Forward all heads — score prediction s_i ≈ ∇_U log p_σ,i = −ε/σ
        all_s = model.forward_all(noised_act, obs, t)  # (k, batch, H, nu)
        k_sz, *_ = all_s.shape
        all_s_flat = all_s.reshape(k_sz, -1)  # (k, batch·H·nu)

        # Select head obj_idx — gradient only to this head and shared encoder.
        s_i = jax.lax.dynamic_index_in_dim(
            all_s_flat, obj_idx, axis=0, keepdims=False
        )  # (batch·H·nu,)

        # Reshape for per-element sigma weighting
        batch_size_ = act.shape[0]
        H, nu = act.shape[1], act.shape[2]
        s_i_3d = s_i.reshape(batch_size_, H, nu)  # (batch, H, nu)

        # Noise-prediction error: at optimum σ·s_i = −ε
        # sigma[..., None]: (batch, 1, 1) broadcasts with (batch, H, nu)
        noise_pred_err = sigma[..., None] * s_i_3d + noise  # (batch, H, nu)

        return jnp.mean(jnp.square(noise_pred_err))

    total_steps = num_iters * k
    last_loss = jnp.zeros(())

    pbar = tqdm(
        range(total_steps),
        desc="DSM training",
        unit="step",
        leave=True,
        dynamic_ncols=True,
    )
    for step in pbar:
        rng, obj_rng, batch_rng, noise_rng, t_rng = jax.random.split(rng, 5)

        # Sample a random mode for this step.
        obj_idx = jax.random.randint(obj_rng, (), 0, k)

        # Draw a mini-batch from the selected mode's dataset.
        N_i = obs_per_obj[int(obj_idx)].shape[0]
        batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, N_i)
        batch_obs = obs_per_obj[int(obj_idx)][batch_idx]
        batch_act = acts_per_obj[int(obj_idx)][batch_idx]

        # Sample Gaussian noise and noise level t ∈ [0, 1].
        noise = jax.random.normal(noise_rng, batch_act.shape)
        t = jax.random.uniform(t_rng, (batch_size, 1))

        # Gradient step — only head obj_idx and the encoder are updated.
        last_loss, grad = nnx.value_and_grad(_loss_fn)(
            model,
            batch_obs,
            batch_act,
            noise,
            t,
            obj_idx,
        )
        optimizer.update(grad)

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{float(last_loss):.6f}"})

    return last_loss


def fit_score_regression(
    u_points_per_obj: Sequence[jax.Array],
    sigmas_per_obj: Sequence[jax.Array],
    scores_per_obj: Sequence[jax.Array],
    obs_per_obj: Sequence[jax.Array],
    model: CompositionalDenoisingMLP,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_iters: int,
    rng: jax.Array,
    sigma_min: float = 1e-2,
    sigma_max: float = 1.0,
    use_sigma_weighting: bool = True,
    sigma_weight_power: float = 4.0,
    predict_update: bool = True,
    targets_are_updates: bool = False,
    normalizer: nnx.Module | None = None,
    onpolicy_u_points_per_obj: Sequence[jax.Array] | None = None,
    onpolicy_sigmas_per_obj: Sequence[jax.Array] | None = None,
    onpolicy_scores_per_obj: Sequence[jax.Array] | None = None,
    onpolicy_obs_per_obj: Sequence[jax.Array] | None = None,
    onpolicy_frac: float = 0.0,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, jax.Array], None] | None = None,
) -> jax.Array:
    """Train each head via direct score regression on MPPI score estimates.

    At each gradient step one mode index ``i`` is sampled.  A mini-batch from
    dataset ``i`` is used to regress the network score prediction onto the
    MPPI-estimated score target::

        L_i = σ⁴ · ‖s_θ,i(U_point, t_σ, y) − score_target‖²

    Since the controller applies ``delta_U = σ² * score``, σ⁴ weighting
    makes this exactly mean-squared regression of the MPPI control update.
    This prevents tiny, high-variance small-σ scores from dominating the
    updates that actually determine the robot action.

    Observation normalisation
    -------------------------
    If ``normalizer`` is provided, each mini-batch of observations is
    normalised with ``normalizer(obs, use_running_average=True)`` **before**
    the gradient step (outside the differentiated function, since the
    normaliser carries no trainable parameters).  This lets callers store
    **raw** observations in the dataset and normalise consistently at train
    time with whatever running statistics the normaliser currently holds —
    the same statistics the policy uses at inference — instead of baking a
    (possibly stale) normalisation into the stored data.

    Args:
        u_points_per_obj: Per-mode query points, each ``(T_i, H, nu)`` in
            normalised ``[-1, 1]`` space.
        sigmas_per_obj: Per-mode sigma values, each ``(T_i,)``.
        scores_per_obj: Per-mode MPPI score targets, each ``(T_i, H, nu)``
            in normalised space.
        obs_per_obj: Per-mode observations, each ``(T_i, obs_size)``.  Raw
            (un-normalised) when ``normalizer`` is given, otherwise assumed
            already normalised.
        model: :class:`CompositionalDenoisingMLP` to train in place.
        optimizer: Flax NNX optimizer, updated in place.
        batch_size: Mini-batch size per gradient step.
        num_iters: Gradient steps **per mode** (total = ``num_iters * N``).
        rng: JAX PRNG key.
        sigma_min: Minimum noise level (for t_σ computation).
        sigma_max: Maximum noise level (for t_σ computation).
        use_sigma_weighting: If ``True``, apply sigma-dependent weighting
            when regressing raw scores.
        sigma_weight_power: Sigma exponent used for weighting.  The default
            ``4`` directly minimizes squared MPPI update error because
            ``delta_U = sigma² * score``.
        predict_update: Regress the bounded MPPI update ``delta_U`` directly
            instead of the numerically ill-conditioned ``delta_U / sigma²``.
            Composition remains linear because sigma is shared by all heads.
        normalizer: Optional observation normaliser applied to each mini-batch
            (inference mode).  If ``None``, observations are used as given.
        onpolicy_*_per_obj: Optional second per-mode dataset (e.g. DAgger
            on-policy corrections) sampled with a fixed per-batch fraction so
            it is not drowned out when the primary (expert) dataset is far
            larger.  All four must be given together.
        onpolicy_frac: Fraction of each mini-batch drawn from the on-policy
            dataset (the rest from the primary dataset).  Applied per mode
            only where that mode's on-policy pool is non-empty; ``0`` (default)
            disables on-policy sampling entirely.
        checkpoint_every: Invoke ``checkpoint_callback`` after every this many
            total gradient steps.  ``0`` disables intermediate checkpoints.
        checkpoint_callback: Callback receiving ``(completed_steps, loss)``.

    Returns:
        Scalar loss from the final gradient step.
    """
    k = model.num_objectives
    _use_onpolicy = (
        onpolicy_u_points_per_obj is not None and onpolicy_frac > 0.0
    )
    log_sigma_min = jnp.log(sigma_min)
    log_sigma_max = jnp.log(sigma_max)
    log_sigma_range = log_sigma_max - log_sigma_min

    def _loss_fn(
        model: CompositionalDenoisingMLP,
        u_point: jax.Array,
        sigma: jax.Array,
        score_target: jax.Array,
        obs: jax.Array,
        obj_idx: jax.Array,
    ) -> jax.Array:
        batch_sz = u_point.shape[0]
        H, nu = u_point.shape[1], u_point.shape[2]

        t = (jnp.log(sigma) - log_sigma_min) / log_sigma_range  # (batch,)
        t = t.reshape(batch_sz, 1)  # (batch, 1)

        all_s = model.forward_all(u_point, obs, t)  # (k, batch, H, nu)
        k_sz = all_s.shape[0]
        all_s_flat = all_s.reshape(k_sz, -1)

        s_i = jax.lax.dynamic_index_in_dim(
            all_s_flat, obj_idx, axis=0, keepdims=False
        )
        s_i_3d = s_i.reshape(batch_sz, H, nu)

        if predict_update:
            target = (
                score_target
                if targets_are_updates
                else sigma[:, None, None] ** 2 * score_target
            )
            residual = s_i_3d - target
        else:
            residual = s_i_3d - score_target  # (batch, H, nu)
        per_sample = jnp.mean(jnp.square(residual), axis=(-2, -1))  # (batch,)

        if use_sigma_weighting and not predict_update:
            per_sample = sigma**sigma_weight_power * per_sample

        return jnp.mean(per_sample)

    @nnx.jit
    def _train_step(
        model: CompositionalDenoisingMLP,
        optimizer: nnx.Optimizer,
        batch_u: jax.Array,
        batch_sigma: jax.Array,
        batch_score: jax.Array,
        batch_obs: jax.Array,
        obj_idx: jax.Array,
    ) -> jax.Array:
        """Compile forward, backward, and Adam update as one GPU program."""

        loss, grad = nnx.value_and_grad(_loss_fn)(
            model,
            batch_u,
            batch_sigma,
            batch_score,
            batch_obs,
            obj_idx,
        )
        optimizer.update(grad)
        return loss

    total_steps = num_iters * k
    last_loss = jnp.zeros(())

    pbar = tqdm(
        range(total_steps),
        desc="Score regression training",
        unit="step",
        leave=True,
        dynamic_ncols=True,
    )
    for step in pbar:
        rng, obj_rng, batch_rng = jax.random.split(rng, 3)

        obj_idx = jax.random.randint(obj_rng, (), 0, k)
        i = int(obj_idx)

        # Split the batch between the primary (expert) and on-policy pools so
        # a small on-policy set is not diluted by a much larger expert set.
        n_op = 0
        if _use_onpolicy and onpolicy_obs_per_obj[i].shape[0] > 0:
            n_op = int(round(batch_size * onpolicy_frac))
        n_ex = batch_size - n_op

        N_i = obs_per_obj[i].shape[0]
        ex_idx = jax.random.randint(batch_rng, (n_ex,), 0, N_i)
        batch_u = u_points_per_obj[i][ex_idx]
        batch_sigma = sigmas_per_obj[i][ex_idx]
        batch_score = scores_per_obj[i][ex_idx]
        batch_obs = obs_per_obj[i][ex_idx]

        if n_op > 0:
            rng, op_rng = jax.random.split(rng)
            N_op = onpolicy_obs_per_obj[i].shape[0]
            op_idx = jax.random.randint(op_rng, (n_op,), 0, N_op)
            batch_u = jnp.concatenate(
                [batch_u, onpolicy_u_points_per_obj[i][op_idx]]
            )
            batch_sigma = jnp.concatenate(
                [batch_sigma, onpolicy_sigmas_per_obj[i][op_idx]]
            )
            batch_score = jnp.concatenate(
                [batch_score, onpolicy_scores_per_obj[i][op_idx]]
            )
            batch_obs = jnp.concatenate(
                [batch_obs, onpolicy_obs_per_obj[i][op_idx]]
            )

        # Normalise observations outside the differentiated function — the
        # normaliser holds no trainable parameters and runs in inference mode.
        if normalizer is not None:
            batch_obs = normalizer(batch_obs, use_running_average=True)

        last_loss = _train_step(
            model,
            optimizer,
            batch_u,
            batch_sigma,
            batch_score,
            batch_obs,
            obj_idx,
        )

        completed_steps = step + 1
        if (
            checkpoint_every > 0
            and checkpoint_callback is not None
            and completed_steps % checkpoint_every == 0
        ):
            last_loss.block_until_ready()
            checkpoint_callback(completed_steps, last_loss)

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{float(last_loss):.6f}"})

    return last_loss
