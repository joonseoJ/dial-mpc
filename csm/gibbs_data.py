"""Anchor-factorized Gibbs data collected from DIAL-TC-MPPI proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def anchor_decoder(anchor_weights: jax.Array) -> jax.Array:
    """Returns the minimum-norm map ``omega -> alpha`` with ``W.T @ alpha=omega``."""

    weights = jnp.asarray(anchor_weights, dtype=jnp.float32)
    if weights.ndim != 2:
        raise ValueError("anchor weights must have shape (anchors, objectives)")
    gram = weights.T @ weights
    if int(jnp.linalg.matrix_rank(gram)) != weights.shape[1]:
        raise ValueError("anchor weights must span the objective space")
    return weights @ jnp.linalg.inv(gram)


def compose_anchor_logits(
    anchor_logits: jax.Array,
    omega: jax.Array,
    decoder: jax.Array,
) -> jax.Array:
    """Composes candidate logits before Gibbs normalization.

    ``anchor_logits`` has anchors on its last axis.  This is deliberately
    different from linearly combining already-normalized weights or noised
    scores, neither of which preserves exponential-family composition.
    """

    alpha = jnp.asarray(decoder) @ jnp.asarray(omega)
    return jnp.einsum("...a,a->...", anchor_logits, alpha)


@dataclass
class GibbsDataset:
    """Grouped candidate sets for anchor-Gibbs distribution supervision."""

    observations: jax.Array  # (Q, obs)
    queries: jax.Array  # (Q, H, action)
    candidates: jax.Array  # (Q, M, H, action)
    anchor_logits: jax.Array  # (Q, M, A)
    anchor_updates: jax.Array  # (Q, A, H, action), selected-bank teacher
    factors: jax.Array  # (Q,)
    logit_scales: jax.Array  # (Q,)


def concatenate_gibbs_datasets(
    datasets: Sequence[GibbsDataset],
) -> GibbsDataset:
    if not datasets:
        raise ValueError("at least one Gibbs dataset is required")
    return GibbsDataset(
        **{
            name: jnp.concatenate([getattr(dataset, name) for dataset in datasets])
            for name in GibbsDataset.__dataclass_fields__
        }
    )


def save_gibbs_dataset(path: Path | str, dataset: GibbsDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            name: np.asarray(getattr(dataset, name))
            for name in GibbsDataset.__dataclass_fields__
        },
    )


def load_gibbs_dataset(path: Path | str) -> GibbsDataset:
    with np.load(path) as archive:
        return GibbsDataset(
            **{
                name: jnp.asarray(archive[name])
                for name in GibbsDataset.__dataclass_fields__
            }
        )


class DIALTCGibbsTeacher:
    """Exact Gibbs labels over the proposal of a DIAL-TC-MPPI controller.

    Dynamics are rolled out once per candidate.  The resulting vector costs
    generate every anchor distribution with a *shared*, weight-independent
    logit scale.  Consequently an arbitrary preference can be composed at the
    clean candidate-logit level before the nonlinear softmax.
    """

    def __init__(
        self,
        planner: object,
        objective_fn: Callable,
        anchor_weights: jax.Array,
        num_model_candidates: int = 64,
        scale_floor: float = 1e-3,
    ) -> None:
        if planner.tc_sampler is None:
            raise ValueError("AFGS requires a DIAL-TC-MPPI planner")
        if num_model_candidates < 2:
            raise ValueError("num_model_candidates must be at least two")
        if num_model_candidates > planner.args.Nsample + 1:
            raise ValueError("model candidates cannot exceed expert candidates")
        self.planner = planner
        self.anchor_weights = jnp.asarray(anchor_weights, dtype=jnp.float32)
        self.decoder = anchor_decoder(self.anchor_weights)
        self.num_model_candidates = int(num_model_candidates)
        self.scale_floor = float(scale_floor)

        env_step = planner.env.step
        node2u = planner.node2u_vmap

        def rollout_cost(state, nodes):
            actions = node2u(nodes)

            def scan_step(carry, action):
                next_state = env_step(carry, action)
                return next_state, jnp.asarray(objective_fn(next_state, action))

            _, costs = jax.lax.scan(scan_step, state, actions)
            # DIAL uses the horizon mean reward.  Mean objective costs keep the
            # same temperature convention and avoid a hidden horizon factor.
            return jnp.mean(costs, axis=0)

        self._rollout_costs = jax.jit(jax.vmap(rollout_cost, in_axes=(None, 0)))

    def _all_logits(self, costs, tc_log_ratio):
        weighted_costs = costs @ self.anchor_weights.T
        # One scalar scale is shared by every anchor for this query.  Per-mode
        # standardization would destroy linear composition in omega.
        scale = jnp.maximum(
            jnp.sqrt(jnp.mean(jnp.var(weighted_costs, axis=0))), self.scale_floor
        )
        logits = -weighted_costs / (scale * self.planner.args.temp_sample)
        logits = logits + (
            self.planner.args.tc_importance_scale * tc_log_ratio[:, None]
        )
        return logits, scale

    def _select_candidates(self, candidates, logits, rng):
        count = self.num_model_candidates
        # Always retain the proposal center (the final candidate).  Half of
        # the remainder covers candidates important to at least one anchor;
        # the other half preserves broad proposal coverage.
        remaining = count - 1
        elite_count = remaining // 2
        random_count = remaining - elite_count
        anchor_probabilities = jax.nn.softmax(logits, axis=0)
        importance = jnp.max(anchor_probabilities, axis=1)
        importance = importance.at[-1].set(-jnp.inf)
        elite = jax.lax.top_k(importance, elite_count)[1]
        rng, sample_rng = jax.random.split(rng)
        random = jax.random.randint(
            sample_rng, (random_count,), 0, candidates.shape[0] - 1
        )
        indices = jnp.concatenate(
            [elite, random, jnp.asarray([candidates.shape[0] - 1])]
        )
        return candidates[indices], logits[indices], rng

    def collect_query(
        self,
        state,
        query,
        factor,
        rng,
        behavior_omega,
        action_history,
    ) -> tuple[GibbsDataset, jax.Array, jax.Array]:
        """Collects one structured query and returns the full expert update."""

        noise_scale = self.planner.sigma_control * jnp.asarray(factor)
        rng, candidates, _, _, tc_ratio = self.planner.sample_nodes(
            rng, query, noise_scale, action_history
        )
        costs = self._rollout_costs(state, candidates)
        all_logits, logit_scale = self._all_logits(costs, tc_ratio)

        behavior_logits = compose_anchor_logits(
            all_logits, behavior_omega, self.decoder
        )
        behavior_weights = jax.nn.softmax(behavior_logits)
        expert_query = jnp.einsum("m,mha->ha", behavior_weights, candidates)

        selected, selected_logits, rng = self._select_candidates(
            candidates, all_logits, rng
        )
        selected_weights = jax.nn.softmax(selected_logits, axis=0)
        selected_means = jnp.einsum("ma,mhd->ahd", selected_weights, selected)
        anchor_updates = selected_means - query[None]
        centered_logits = selected_logits - jnp.mean(
            selected_logits, axis=0, keepdims=True
        )
        dataset = GibbsDataset(
            observations=state.obs[None],
            queries=query[None],
            candidates=selected[None],
            anchor_logits=centered_logits[None],
            anchor_updates=anchor_updates[None],
            factors=jnp.asarray([factor], dtype=jnp.float32),
            logit_scales=jnp.asarray([logit_scale], dtype=jnp.float32),
        )
        return dataset, expert_query, rng
