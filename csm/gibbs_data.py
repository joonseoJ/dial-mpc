"""Raw anchor-cost data collected from exact DIAL-TC-MPPI proposals."""

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
    anchor_values: jax.Array,
    omega: jax.Array,
    decoder: jax.Array,
) -> jax.Array:
    """Composes raw anchor values for an arbitrary objective weight.

    The name is retained for API compatibility.  Version-2 AFGS passes raw
    anchor costs here, then performs DIAL's weight-dependent standardization.
    """

    alpha = jnp.asarray(decoder) @ jnp.asarray(omega)
    return jnp.einsum("...a,a->...", anchor_values, alpha)


def standardized_gibbs_logits(
    weighted_costs: jax.Array,
    temperature: float | jax.Array,
    scale_floor: float = 1e-6,
    *,
    candidate_axis: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Returns the exact reward-standardized DIAL logits and their scale."""

    costs = jnp.asarray(weighted_costs)
    centered = costs - jnp.mean(costs, axis=candidate_axis, keepdims=True)
    scale = jnp.maximum(jnp.std(costs, axis=candidate_axis), scale_floor)
    expanded_scale = jnp.expand_dims(scale, axis=candidate_axis)
    return -centered / (expanded_scale * temperature), scale


@dataclass
class GibbsDataset:
    """Deployment-distributed candidate banks with raw anchor-cost labels."""

    observations: jax.Array  # (Q, obs)
    queries: jax.Array  # (Q, H, action)
    candidates: jax.Array  # (Q, M, H, action)
    anchor_costs: jax.Array  # (Q, M, anchors), centered raw horizon costs
    anchor_updates: jax.Array  # (Q, anchors, H, action), exact bank update
    anchor_scales: jax.Array  # (Q, anchors), per-bank DIAL reward std
    factors: jax.Array  # (Q,)
    priorities: jax.Array  # (Q,), pre-fall replay priority


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
        missing = set(GibbsDataset.__dataclass_fields__) - set(archive.files)
        if missing:
            raise ValueError(
                f"{path} uses the legacy shared-scale AFGS format; "
                "recollect it with the current dial-afgs command "
                f"(missing {sorted(missing)})"
            )
        return GibbsDataset(
            **{
                name: jnp.asarray(archive[name])
                for name in GibbsDataset.__dataclass_fields__
            }
        )


class DIALTCGibbsTeacher:
    """DIAL-TC expert plus raw anchor costs on deployment-like banks.

    The behavior update uses all DIAL candidates and exactly reproduces its
    per-weight reward standardization.  Training banks are independent random
    subsets of the same proposal (plus its center), without elite selection.
    """

    def __init__(
        self,
        planner: object,
        objective_fn: Callable,
        anchor_weights: jax.Array,
        num_model_candidates: int = 128,
        banks_per_query: int = 4,
        scale_floor: float = 1e-6,
    ) -> None:
        if planner.tc_sampler is None:
            raise ValueError("AFGS requires a DIAL-TC-MPPI planner")
        if planner.args.tc_importance_scale != 0.0:
            raise ValueError(
                "AFGS currently requires tc_importance_scale=0 so the "
                "standalone student exactly reproduces the proposal correction"
            )
        if num_model_candidates < 2:
            raise ValueError("model candidates must be at least two")
        if num_model_candidates > planner.args.Nsample + 1:
            raise ValueError("model candidates cannot exceed expert candidates")
        if banks_per_query < 1:
            raise ValueError("banks_per_query must be positive")
        if banks_per_query * (num_model_candidates - 1) > planner.args.Nsample:
            raise ValueError(
                "independent random banks require banks_per_query * "
                "(model_candidates - 1) <= expert samples"
            )
        self.planner = planner
        self.anchor_weights = jnp.asarray(anchor_weights, dtype=jnp.float32)
        self.decoder = anchor_decoder(self.anchor_weights)
        self.num_model_candidates = int(num_model_candidates)
        self.banks_per_query = int(banks_per_query)
        self.scale_floor = float(scale_floor)

        env_step = planner.env.step
        node2u = planner.node2u_vmap

        def rollout_cost(state, nodes):
            actions = node2u(nodes)

            def scan_step(carry, action):
                next_state = env_step(carry, action)
                return next_state, jnp.asarray(objective_fn(next_state, action))

            _, costs = jax.lax.scan(scan_step, state, actions)
            return jnp.mean(costs, axis=0)

        self._rollout_costs = jax.jit(jax.vmap(rollout_cost, in_axes=(None, 0)))

    def _random_banks(self, candidates, anchor_costs, rng):
        """Samples banks distributed exactly like the standalone proposal."""

        rng, bank_rng = jax.random.split(rng)
        random_count = self.num_model_candidates - 1
        sampled_count = candidates.shape[0] - 1
        random_indices = jax.random.permutation(bank_rng, sampled_count)[
            : self.banks_per_query * random_count
        ].reshape(self.banks_per_query, random_count)
        center_indices = jnp.full(
            (self.banks_per_query, 1), sampled_count, dtype=random_indices.dtype
        )
        indices = jnp.concatenate([random_indices, center_indices], axis=1)
        return candidates[indices], anchor_costs[indices], rng

    def collect_query(
        self,
        state,
        query,
        factor,
        rng,
        behavior_omega,
        action_history,
    ) -> tuple[GibbsDataset, jax.Array, jax.Array]:
        """Collects random banks and returns the full true-DIAL expert update."""

        noise_scale = self.planner.sigma_control * jnp.asarray(factor)
        rng, candidates, _, _, _ = self.planner.sample_nodes(
            rng, query, noise_scale, action_history
        )
        objective_costs = self._rollout_costs(state, candidates)
        anchor_costs = objective_costs @ self.anchor_weights.T

        behavior_costs = objective_costs @ behavior_omega
        behavior_logits, _ = standardized_gibbs_logits(
            behavior_costs,
            self.planner.args.temp_sample,
            self.scale_floor,
            candidate_axis=0,
        )
        behavior_weights = jax.nn.softmax(behavior_logits)
        expert_query = jnp.einsum("m,mha->ha", behavior_weights, candidates)

        banks, bank_anchor_costs, rng = self._random_banks(
            candidates, anchor_costs, rng
        )
        centered_costs = bank_anchor_costs - jnp.mean(
            bank_anchor_costs, axis=1, keepdims=True
        )
        scales = jnp.maximum(
            jnp.std(bank_anchor_costs, axis=1), self.scale_floor
        )
        logits = -centered_costs / (
            scales[:, None, :] * self.planner.args.temp_sample
        )
        bank_weights = jax.nn.softmax(logits, axis=1)
        bank_means = jnp.einsum("bma,bmhd->bahd", bank_weights, banks)
        anchor_updates = bank_means - query[None, None]
        bank_count = self.banks_per_query
        dataset = GibbsDataset(
            observations=jnp.broadcast_to(
                state.obs, (bank_count, state.obs.shape[-1])
            ),
            queries=jnp.broadcast_to(query, (bank_count,) + query.shape),
            candidates=banks,
            anchor_costs=centered_costs,
            anchor_updates=anchor_updates,
            anchor_scales=scales,
            factors=jnp.full((bank_count,), factor, dtype=jnp.float32),
            priorities=jnp.ones((bank_count,), dtype=jnp.float32),
        )
        return dataset, expert_query, rng
