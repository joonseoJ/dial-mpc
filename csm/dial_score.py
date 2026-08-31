"""Single-weight DIAL-MPC score matching with exactly one MLP.

This module is the deliberately minimal version of the CSM idea: one network,
one preference weight, trained by **score regression** and deployed by
**denoising**.  It exists to establish that a single MLP can reproduce
DIAL-MPC's annealed planner before any compositional machinery is layered on
top.  Nothing here is compositional and nothing here predicts an energy.

What DIAL-MPC actually computes
-------------------------------
:meth:`dial_mpc.core.dial_core.MBDPI.reverse_once` is one Gaussian MPPI update
on the *spline node* plan ``Y`` (shape ``(Hnode+1, nu)``) with a **per-node**
noise vector::

    noise_scale = sigma_control * factor,
    sigma_control = sigma_scale * horizon_diffuse_factor ** [Hnode, ..., 1, 0]

    Y_s = clip(Y + noise_scale[:, None] * eps_s, -1, 1),  Y_s[0] = Y[0]
    w   = softmax((rew_s - rew_Y) / std(rew) / temp_sample)
    Y'  = sum_s w_s Y_s

Within one control step DIAL anneals ``factor`` down the geometric schedule
``traj_diffuse_factor ** [0, 1, ..., Ndiffuse-1]``.  So DIAL is exactly an
annealed denoiser over a *fixed* per-node noise profile whose only scalar
degree of freedom is ``factor``.

Score parameterisation
----------------------
For a Gaussian corruption with diagonal per-node standard deviation ``sigma``,
Tweedie's identity makes the MPPI update the score up to ``sigma**2``::

    delta(Y)      = E[Y' - Y]                      (the DIAL update)
    score(Y)      = delta(Y) / sigma**2            (~ grad_Y log p_sigma(Y))
    denoise step  = Y + sigma**2 * score(Y) = Y + delta(Y)

:class:`DialScoreMLP` therefore holds a single MLP that emits ``delta`` and
divides by ``sigma(t)**2`` to expose :meth:`DialScoreMLP.score`.  The score is
the object being learned; the ``1/sigma**2`` factor is only a *conditioning*
choice.  Predicting the score directly with an unscaled head is what makes
naive score MLPs fail here: over DIAL's schedule ``sigma**2`` spans more than
an order of magnitude, so the raw score target's scale explodes at small
``factor`` while the update it produces stays bounded.

Training objective (score regression)
-------------------------------------
Regression onto the teacher score with ``sigma**4`` weighting::

    L = E[ || sigma**2 * (s_theta(Y, y, t) - delta_teacher / sigma**2) ||^2 ]

The weighting makes this exactly mean-squared error of the *control update*
that the robot will apply, which is the quantity deployment cares about, while
``s_theta`` remains an honest score field.

Teacher targets come from :meth:`MBDPI.reverse_once` itself, averaged over
``repeats`` independent sample batches.  No MPPI equation is reimplemented, so
reward normalisation, query inclusion, spline interpolation, first-node
locking, clipping, and Gibbs weighting are exactly DIAL's.

Inference (denoising)
---------------------
:meth:`DialScorePolicy.apply` runs DIAL's annealing loop with the network in
place of the sampler::

    for factor in traj_diffuse_factor ** [0, 1, ...]:
        sigma = sigma_control * factor
        Y <- clip(Y + dt * sigma**2 * s_theta(Y, y, t(factor)), -1, 1)

and :meth:`DialScorePolicy.shift` warm-starts the next control step through
DIAL's own ``node2u -> roll -> u2node`` spline map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.struct import dataclass
from tqdm.auto import tqdm

from csm.architectures import MLP

from csm.omega import mixture_from_pinv, normalize_omega


# --------------------------------------------------------------------------- #
# DIAL annealing schedule
# --------------------------------------------------------------------------- #


def dial_sigma_control(
    horizon_diffuse_factor: float, horizon: int, sigma_scale: float = 1.0
) -> jax.Array:
    """Return DIAL's per-node noise profile ``sigma_control`` of shape ``(H,)``.

    ``horizon`` is the number of spline nodes, i.e. ``Hnode + 1``.  Node 0 gets
    the smallest noise and the last node the largest, matching
    ``horizon_diffuse_factor ** arange(H)[::-1]``.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not 0.0 < horizon_diffuse_factor <= 1.0:
        raise ValueError("horizon_diffuse_factor must lie in (0, 1]")
    powers = jnp.arange(horizon, dtype=jnp.float32)[::-1]
    return jnp.asarray(sigma_scale, dtype=jnp.float32) * (
        jnp.asarray(horizon_diffuse_factor, dtype=jnp.float32) ** powers
    )


def dial_factors(traj_diffuse_factor: float, num_levels: int) -> jax.Array:
    """Return DIAL's within-step annealing factors ``gamma ** [0, ..., L-1]``."""

    if num_levels < 1:
        raise ValueError("num_levels must be at least 1")
    if not 0.0 < traj_diffuse_factor <= 1.0:
        raise ValueError("traj_diffuse_factor must lie in (0, 1]")
    return jnp.asarray(traj_diffuse_factor, dtype=jnp.float32) ** jnp.arange(
        num_levels, dtype=jnp.float32
    )


def _log_span(factor_min: float, factor_max: float) -> jax.Array:
    span = jnp.log(jnp.asarray(factor_max)) - jnp.log(jnp.asarray(factor_min))
    # A single annealing level degenerates to factor_min == factor_max; keep the
    # map well defined instead of dividing by zero.
    return jnp.where(span > 1e-12, span, 1.0)


def factor_to_t(
    factor: jax.Array | float, factor_min: float, factor_max: float
) -> jax.Array:
    """Map an annealing factor to the network conditioning ``t`` in ``[0, 1]``.

    ``t = 0`` is the finest level (``factor_min``) and ``t = 1`` the coarsest
    (``factor_max``), log-uniformly spaced so DIAL's geometric schedule becomes
    evenly spaced in ``t``.
    """

    factor = jnp.asarray(factor, dtype=jnp.float32)
    return (jnp.log(factor) - jnp.log(jnp.asarray(factor_min))) / _log_span(
        factor_min, factor_max
    )


def t_to_factor(
    t: jax.Array | float, factor_min: float, factor_max: float
) -> jax.Array:
    """Inverse of :func:`factor_to_t`."""

    t = jnp.asarray(t, dtype=jnp.float32)
    return jnp.exp(jnp.log(jnp.asarray(factor_min)) + t * _log_span(
        factor_min, factor_max
    ))


def build_shift_matrix(planner: object) -> jax.Array:
    """Return DIAL's warm-start map as a matrix over spline nodes.

    ``MBDPI.shift`` interpolates nodes to dense controls, rolls them one control
    step, zeros the tail, and projects back onto nodes.  Every actuator is
    processed independently and each stage is linear, so the whole operation is
    one ``(H, H)`` matrix applied as ``shift_matrix @ Y``.
    """

    horizon = int(planner.args.Hnode) + 1

    def shift_column(column: jax.Array) -> jax.Array:
        controls = planner.node2u(column)
        controls = jnp.roll(controls, -1).at[-1].set(0.0)
        return planner.u2node(controls)

    return jax.vmap(shift_column)(jnp.eye(horizon, dtype=jnp.float32)).T


# --------------------------------------------------------------------------- #
# The single network
# --------------------------------------------------------------------------- #


class DialScoreMLP(nnx.Module):
    """One MLP modelling DIAL's annealed score for a single preference weight.

    ``input  = concat([Y_flat, y, t])``  with ``t`` the annealing conditioning
    ``output = delta``, and ``score = delta / sigma(t)**2``.

    Node 0 is DIAL's locked control: every sample in ``reverse_once`` shares
    ``Y[0]``, so the exact update there is identically zero.  That structure is
    hard-coded rather than learned.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        sigma_control: jax.Array,
        hidden: Sequence[int],
        rngs: nnx.Rngs,
        factor_min: float = 1.0,
        factor_max: float = 1.0,
        lock_first_node: bool = True,
    ) -> None:
        self.action_size = int(action_size)
        self.observation_size = int(observation_size)
        self.horizon = int(horizon)
        self.factor_min = float(factor_min)
        self.factor_max = float(factor_max)
        self.lock_first_node = bool(lock_first_node)

        sigma_control = jnp.asarray(sigma_control, dtype=jnp.float32)
        if sigma_control.shape != (self.horizon,):
            raise ValueError(
                f"sigma_control must have shape ({self.horizon},), "
                f"got {sigma_control.shape}"
            )
        if jnp.any(sigma_control <= 0.0):
            raise ValueError("sigma_control entries must be positive")
        # Non-trainable buffer: part of the score parameterisation, not a
        # parameter, so the optimizer never touches it and it survives pickling.
        self.sigma_control = nnx.Variable(sigma_control)

        plan_size = self.horizon * self.action_size
        self.net = MLP(
            [plan_size + self.observation_size + 1] + list(hidden) + [plan_size],
            rngs=rngs,
        )

    def sigma(self, t: jax.Array) -> jax.Array:
        """Per-node noise standard deviation, shape ``(*batch, H, 1)``."""

        factor = t_to_factor(t, self.factor_min, self.factor_max)  # (*batch, 1)
        return factor[..., None] * self.sigma_control.value.reshape(
            self.horizon, 1
        )

    def update(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Predict DIAL's bounded plan update, shape ``(*batch, H, nu)``."""

        batches = u.shape[:-2]
        u_flat = u.reshape(batches + (self.horizon * self.action_size,))
        features = jnp.concatenate([u_flat, y, t], axis=-1)
        delta = self.net(features).reshape(
            batches + (self.horizon, self.action_size)
        )
        if self.lock_first_node:
            delta = delta.at[..., 0, :].set(0.0)
        return delta

    def score(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Score ``grad_U log p_sigma(U|y) = delta / sigma**2``."""

        return self.update(u, y, t) / jnp.square(self.sigma(t))

    def __call__(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        return self.score(u, y, t)


# --------------------------------------------------------------------------- #
# Exact DIAL teacher
# --------------------------------------------------------------------------- #


class DialScoreTeacher:
    """Averaged ``MBDPI.reverse_once`` updates for one fixed reward weight."""

    def __init__(
        self,
        planner: object,
        reward_weights: jax.Array,
        repeats: int = 4,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        self.planner = planner
        self.repeats = int(repeats)
        self.reward_weights = jnp.asarray(reward_weights, dtype=jnp.float32)
        self.sigma_control = jnp.asarray(planner.sigma_control, dtype=jnp.float32)
        self._mean_update = jax.jit(self._mean_update_impl)

    def with_reward_weights(self, state: object) -> object:
        """Return ``state`` with this teacher's single preference weight."""

        info = dict(state.info)
        info["reward_weights"] = self.reward_weights
        return state.replace(info=info)

    def _mean_update_impl(
        self,
        state: object,
        query: jax.Array,
        noise_scale: jax.Array,
        rng: jax.Array,
        weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        info_dict = dict(state.info)
        info_dict["reward_weights"] = weights
        state = state.replace(info=info_dict)

        def body(carry, key):
            _, refined, info = self.planner.reverse_once(
                state, key, query, noise_scale
            )
            ess = 1.0 / jnp.sum(jnp.square(info["weights"]))
            return carry, (refined - query, ess)

        keys = jax.random.split(rng, self.repeats)
        _, (deltas, esses) = jax.lax.scan(body, None, keys)
        mean = jnp.mean(deltas, axis=0)
        # Monte-Carlo spread of the target itself: the noise floor that score
        # regression cannot go below.
        spread = jnp.sqrt(jnp.mean(jnp.var(deltas, axis=0)))
        return mean, spread, jnp.mean(esses)

    def targets(
        self,
        state: object,
        query: jax.Array,
        factor: float | jax.Array,
        rng: jax.Array,
        weights: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(delta, target_spread, effective_sample_size)``.

        ``weights`` overrides this teacher's preference weight for one call.
        """

        noise_scale = self.sigma_control * jnp.asarray(factor, dtype=jnp.float32)
        weights = (
            self.reward_weights
            if weights is None
            else jnp.asarray(weights, dtype=jnp.float32)
        )
        return self._mean_update(state, query, noise_scale, rng, weights)

    def targets_basis(
        self,
        state: object,
        query: jax.Array,
        factor: float | jax.Array,
        rng: jax.Array,
        basis: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Label one query under every basis weight with common random numbers.

        Passing the same ``rng`` to each weight makes ``reverse_once`` draw the
        same samples and roll the same physics -- ``reward_weights`` enters only
        the scalar reward -- so the returned updates differ solely through the
        Gibbs weighting.  That removes sampling noise from every comparison
        between basis rows and makes one rollout pay for all ``k`` labels.

        Returns ``(deltas, spreads, esses)`` with a leading basis axis.
        """

        basis = jnp.asarray(basis, dtype=jnp.float32)
        results = [
            self.targets(state, query, factor, rng, weights=row) for row in basis
        ]
        return (
            jnp.stack([item[0] for item in results]),
            jnp.stack([jnp.asarray(item[1]) for item in results]),
            jnp.stack([jnp.asarray(item[2]) for item in results]),
        )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class DialScoreData(NamedTuple):
    """Score-regression dataset for one preference weight.

    ``qpos``/``qvel``/``step`` record the physics state each label was taken at.
    They are not used for training; they exist so a dataset can be *relabelled*
    under a different preference weight, teacher temperature, or annealing
    schedule instead of being recollected.  Measured on this environment, those
    three fields are exactly what the teacher needs: rebuilding a state from
    ``qpos + qvel`` alone reproduces the teacher with cosine 0.730, adding
    ``step`` takes it to 1.0000, and every other ``info`` field is derived from
    ``step`` and the config, so storing them adds nothing.

    ``step = -1`` marks a dataset collected before these fields existed; those
    cannot be relabelled (see :meth:`has_physics_state`).
    """

    u: jax.Array  # (N, H, nu) query plans in normalized node coordinates
    factor: jax.Array  # (N,) annealing factor of each query
    level: jax.Array  # (N,) annealing level index of each query
    delta: jax.Array  # (N, H, nu) exact DIAL update targets
    obs: jax.Array  # (N, obs_size) raw observations
    # Default to None so callers that do not record physics stay ergonomic;
    # anything written to disk is always concrete.
    qpos: jax.Array | None = None  # (N, nq) generalized positions
    qvel: jax.Array | None = None  # (N, nv) generalized velocities
    step: jax.Array | None = None  # (N,) env step index, -1 where unrecorded

    @property
    def size(self) -> int:
        return int(self.u.shape[0])

    @property
    def has_physics_state(self) -> bool:
        """Whether every row carries the state needed to relabel it."""

        if self.qpos is None or self.step is None:
            return False
        return bool(self.qpos.shape[-1] > 0 and jnp.all(self.step >= 0))

    @staticmethod
    def without_physics_state(size: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Placeholder physics fields for a dataset that did not record them."""

        return (
            jnp.zeros((size, 0), dtype=jnp.float32),
            jnp.zeros((size, 0), dtype=jnp.float32),
            jnp.full((size,), -1, dtype=jnp.int32),
        )


def concat_dial_score_data(parts: Sequence[DialScoreData]) -> DialScoreData:
    """Concatenate datasets from several collection rounds."""

    parts = [part for part in parts if part.size > 0]
    if not parts:
        raise ValueError("no non-empty datasets to concatenate")
    if not all(part.has_physics_state for part in parts):
        # Mixing recorded and unrecorded rows would silently produce a dataset
        # that looks relabellable but is not, so drop the fields explicitly.
        total = sum(part.size for part in parts)
        placeholder = dict(
            zip(("qpos", "qvel", "step"), DialScoreData.without_physics_state(total))
        )
        merged = {
            field: jnp.concatenate([getattr(part, field) for part in parts], axis=0)
            for field in DialScoreData._fields
            if field not in placeholder
        }
        return DialScoreData(**merged, **placeholder)
    return DialScoreData(
        *(
            jnp.concatenate([getattr(part, field) for part in parts], axis=0)
            for field in DialScoreData._fields
        )
    )


def save_dial_score_data(path: str | Path, data: DialScoreData) -> None:
    """Save a dataset atomically as a plain NPZ of numpy arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    placeholder = dict(
        zip(("qpos", "qvel", "step"), DialScoreData.without_physics_state(data.size))
    )
    payload = {
        field: np.asarray(
            placeholder[field]
            if getattr(data, field) is None
            else getattr(data, field)
        )
        for field in DialScoreData._fields
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(destination)


def load_dial_score_data(path: str | Path) -> DialScoreData:
    """Load a dataset written by :func:`save_dial_score_data`."""

    with np.load(path) as payload:
        fields = {}
        for field in DialScoreData._fields:
            if field in payload:
                fields[field] = jnp.asarray(payload[field])
        size = int(fields["u"].shape[0])
        if "step" not in fields:
            # Written before the physics state was recorded.
            missing = DialScoreData.without_physics_state(size)
            fields.update(zip(("qpos", "qvel", "step"), missing))
        return DialScoreData(**fields)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


class StateBank:
    """Environment states to restart episodes from.

    Two gaps motivate this.  The horizon present in the data is capped by how
    long the *student* survives, not by ``episode_steps`` -- a fall resets the
    episode regardless -- so simply asking for longer episodes cannot reach late
    states.  And failures repeat in a narrow band (first fall clusters near step
    500), so uniform fresh resets spend most of their budget away from where the
    policy actually breaks.

    Seeding episodes from banked states decouples *reaching* a state from
    *surviving* to it: DIAL walks 3000 steps without falling, so its deep states
    can be handed to the student directly, and the student's own pre-fall states
    can be replayed to concentrate labels where they matter.
    """

    def __init__(self, states: list | None = None, capacity: int = 2000) -> None:
        self.states = list(states or [])
        self.capacity = int(capacity)

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state: object) -> None:
        self.states.append(state)
        if len(self.states) > self.capacity:
            # Drop the oldest: banked states go stale as the policy changes.
            self.states = self.states[-self.capacity :]

    def sample(self, rng: jax.Array) -> object:
        if not self.states:
            raise ValueError("state bank is empty")
        index = int(jax.random.randint(rng, (), 0, len(self.states)))
        state = self.states[index]
        # Refresh the env's own rng, otherwise every restart from this state
        # replays an identical trajectory.
        info = dict(state.info)
        info["rng"] = jax.random.fold_in(rng, index)
        return state.replace(info=info)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as stream:
            cloudpickle.dump(self.states, stream)

    @staticmethod
    def load(path: str | Path, capacity: int = 2000) -> "StateBank":
        with open(path, "rb") as stream:
            return StateBank(cloudpickle.load(stream), capacity=capacity)


def mixed_reset_fn(
    env_reset: Callable[[jax.Array], object],
    bank: "StateBank | None",
    fraction: float,
) -> Callable[[jax.Array], object]:
    """Restart from the bank with probability ``fraction``, else fresh.

    Keeping a share of fresh resets matters: banked states are correlated and
    concentrated near failures, and training only on those trades nominal
    walking for recovery behaviour.
    """

    if bank is None or len(bank) == 0 or fraction <= 0.0:
        return env_reset

    def reset_fn(rng: jax.Array) -> object:
        choose, pick = jax.random.split(rng)
        if float(jax.random.uniform(choose)) < fraction:
            return bank.sample(pick)
        return env_reset(pick)

    return reset_fn


StudentUpdate = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


class CollectionStats(NamedTuple):
    """Diagnostics from one collection rollout.

    ``relative_label_noise`` is the single most useful number here.  DIAL's
    Gibbs weights are nearly winner-take-all (``mean_effective_samples`` is
    typically a few dozen out of thousands), so one ``reverse_once`` call is a
    high-variance estimate of the update.  Averaging ``repeats`` calls leaves a
    residual label noise of ``spread / sqrt(repeats - 1)``; dividing by the
    target's own RMS gives the floor that validation ``relative_rms`` cannot go
    below no matter how well the network fits.  Halving that floor costs four
    times as many teacher calls, and raising ``Nsample`` barely helps because
    the effective sample size is what limits the estimate.
    """

    mean_reward: float
    total_reward: float
    num_steps: int
    num_resets: int
    num_episodes: int
    mean_target_spread: float
    mean_effective_samples: float
    target_rms: float
    # ``None`` when a single teacher repeat leaves no way to estimate the
    # spread, rather than a NaN that would poison the JSON report.
    label_noise: float | None
    relative_label_noise: float | None


class DialScoreCollector:
    """Roll DIAL's own control loop and record exact score targets.

    With ``student=None`` the annealing chain advances on the teacher's update,
    so round 0 is literally a DIAL-MPC rollout and its mean reward is the
    baseline the student has to match.  Passing a ``student`` callable turns the
    same loop into DAgger: the visited queries come from the network while every
    target still comes from DIAL.
    """

    def __init__(
        self,
        env: object,
        planner: object,
        teacher: DialScoreTeacher,
        factors: jax.Array,
        perturbations: int = 4,
        perturb_scale: float = 1.0,
        reset_on_done: bool = True,
    ) -> None:
        if perturbations < 0:
            raise ValueError("perturbations cannot be negative")
        if perturb_scale < 0.0:
            raise ValueError("perturb_scale cannot be negative")
        self.env = env
        self.planner = planner
        self.teacher = teacher
        self.factors = jnp.asarray(factors, dtype=jnp.float32)
        self.factor_min = float(jnp.min(self.factors))
        self.factor_max = float(jnp.max(self.factors))
        self.perturbations = int(perturbations)
        self.perturb_scale = float(perturb_scale)
        self.reset_on_done = bool(reset_on_done)

        self.horizon = int(planner.args.Hnode) + 1
        self.action_size = int(env.action_size)
        self.sigma_control = jnp.asarray(planner.sigma_control, dtype=jnp.float32)

        self._reset = jax.jit(env.reset)
        self._step = jax.jit(env.step)
        shift_matrix = build_shift_matrix(planner)
        self.shift_matrix = shift_matrix
        self._shift = jax.jit(
            lambda plan: jnp.einsum("ij,ja->ia", shift_matrix, plan)
        )

    def rollout(
        self,
        rng: jax.Array,
        num_steps: int,
        student: StudentUpdate | None = None,
        init_passes: int = 1,
        episode_steps: int | None = None,
        reset_fn: Callable[[jax.Array], object] | None = None,
        failure_bank: "StateBank | None" = None,
        failure_lookback: int = 30,
        desc: str = "DIAL score collection",
    ) -> tuple[DialScoreData, CollectionStats]:
        """Collect ``num_steps`` control steps of score targets.

        ``episode_steps`` forces a reset every that many steps.  One long
        episode from a deterministic reset covers a single state trajectory no
        matter how many steps it is, which is the usual reason a fitted score
        field collapses off that corridor; short episodes turn the same budget
        into many independent initial conditions.
        """

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if init_passes < 1:
            raise ValueError("init_passes must be at least 1")
        if episode_steps is not None and episode_steps < 1:
            raise ValueError("episode_steps must be at least 1")

        num_levels = int(self.factors.shape[0])
        queries_per_level = 1 + self.perturbations
        reset_fn = reset_fn or self._reset
        recent: list[object] = []

        rng, reset_rng = jax.random.split(rng)
        state = self.teacher.with_reward_weights(reset_fn(reset_rng))
        plan = jnp.zeros((self.horizon, self.action_size), dtype=jnp.float32)

        rows_qpos: list[np.ndarray] = []
        rows_qvel: list[np.ndarray] = []
        rows_step: list[int] = []
        rows_u: list[np.ndarray] = []
        rows_factor: list[float] = []
        rows_level: list[int] = []
        rows_delta: list[np.ndarray] = []
        rows_obs: list[np.ndarray] = []
        rewards: list[float] = []
        spreads: list[float] = []
        esses: list[float] = []
        num_resets = 0
        num_episodes = 1
        episode_step = 0

        planned_episodes = (
            1
            if episode_steps is None
            else -(-num_steps // episode_steps)  # ceiling division
        )
        # An estimate: unplanned resets after a fall add further init passes.
        total_queries = (
            (num_steps - planned_episodes + planned_episodes * init_passes)
            * num_levels
            * queries_per_level
        )
        progress = tqdm(
            total=total_queries, desc=desc, unit="query", dynamic_ncols=True
        )

        def record(query, factor, level, delta, obs, physics):
            rows_u.append(np.asarray(query))
            rows_factor.append(float(factor))
            rows_level.append(int(level))
            rows_delta.append(np.asarray(delta))
            rows_obs.append(np.asarray(obs))
            rows_qpos.append(physics[0])
            rows_qvel.append(physics[1])
            rows_step.append(physics[2])

        for step_idx in range(num_steps):
            # DIAL runs extra annealing passes on the first control step of an
            # episode, including after a reset.  Tiling the same factor set keeps
            # the noise range unchanged instead of continuing the geometric decay
            # into levels the student would otherwise never see again.
            passes = init_passes if episode_step == 0 else 1
            obs = state.obs
            physics = (
                np.asarray(state.pipeline_state.q),
                np.asarray(state.pipeline_state.qd),
                int(state.info["step"]),
            )
            recent.append(state)
            if len(recent) > failure_lookback:
                recent = recent[-failure_lookback:]
            for _ in range(passes):
                for level in range(num_levels):
                    factor = float(self.factors[level])
                    noise = self.sigma_control * factor

                    rng, teacher_rng = jax.random.split(rng)
                    delta, spread, ess = self.teacher.targets(
                        state, plan, factor, teacher_rng
                    )
                    delta.block_until_ready()
                    record(plan, factor, level, delta, obs, physics)
                    spreads.append(float(spread))
                    esses.append(float(ess))

                    for _ in range(self.perturbations):
                        rng, noise_rng, teacher_rng = jax.random.split(rng, 3)
                        eps = jax.random.normal(noise_rng, plan.shape)
                        query = jnp.clip(
                            plan + self.perturb_scale * noise[:, None] * eps,
                            -1.0,
                            1.0,
                        )
                        # DIAL never perturbs the locked node, so the score
                        # there stays exactly zero for the perturbed query too.
                        query = query.at[0].set(plan[0])
                        query_delta, query_spread, query_ess = (
                            self.teacher.targets(
                                state, query, factor, teacher_rng
                            )
                        )
                        query_delta.block_until_ready()
                        record(query, factor, level, query_delta, obs, physics)
                        spreads.append(float(query_spread))
                        esses.append(float(query_ess))

                    if student is None:
                        step_delta = delta
                    else:
                        t = factor_to_t(
                            factor, self.factor_min, self.factor_max
                        ).reshape(1)
                        step_delta = student(plan, obs, t)
                    plan = jnp.clip(plan + step_delta, -1.0, 1.0)
                    progress.update(queries_per_level)

            state = self._step(state, plan[0])
            rewards.append(float(state.reward))
            progress.set_postfix(
                step=f"{step_idx + 1}/{num_steps}",
                reward=f"{rewards[-1]:.3f}",
                refresh=False,
            )
            fell = self.reset_on_done and float(state.done) > 0.5
            episode_over = (
                episode_steps is not None and episode_step + 1 >= episode_steps
            )
            if fell and failure_bank is not None:
                # Bank the run-up to the fall, not the fallen state itself: the
                # useful labels are where the policy still had a choice.
                for banked in recent:
                    failure_bank.add(banked)
            if fell or episode_over:
                rng, reset_rng = jax.random.split(rng)
                state = self.teacher.with_reward_weights(reset_fn(reset_rng))
                plan = jnp.zeros_like(plan)
                episode_step = 0
                recent = []
                num_resets += int(fell)
                if step_idx + 1 < num_steps:
                    num_episodes += 1
            else:
                plan = self._shift(plan)
                episode_step += 1

        progress.close()

        data = DialScoreData(
            u=jnp.asarray(np.stack(rows_u)),
            factor=jnp.asarray(np.asarray(rows_factor, dtype=np.float32)),
            level=jnp.asarray(np.asarray(rows_level, dtype=np.int32)),
            delta=jnp.asarray(np.stack(rows_delta)),
            obs=jnp.asarray(np.stack(rows_obs)),
            qpos=jnp.asarray(np.stack(rows_qpos)),
            qvel=jnp.asarray(np.stack(rows_qvel)),
            step=jnp.asarray(np.asarray(rows_step, dtype=np.int32)),
        )
        mean_spread = float(np.mean(spreads))
        target_rms = float(np.sqrt(np.mean(np.square(np.asarray(rows_delta)))))
        repeats = int(self.teacher.repeats)
        # Residual noise of the *averaged* target.  The biased per-repeat
        # variance estimates (repeats-1)/repeats of the per-call variance, and
        # the mean of `repeats` calls has variance sigma**2 / repeats, so the two
        # factors collapse to spread**2 / (repeats - 1).
        label_noise = (
            float(mean_spread / np.sqrt(repeats - 1)) if repeats > 1 else None
        )
        stats = CollectionStats(
            mean_reward=float(np.mean(rewards)),
            total_reward=float(np.sum(rewards)),
            num_steps=int(num_steps),
            num_resets=int(num_resets),
            num_episodes=int(num_episodes),
            mean_target_spread=mean_spread,
            mean_effective_samples=float(np.mean(esses)),
            target_rms=target_rms,
            label_noise=label_noise,
            relative_label_noise=(
                None
                if label_noise is None
                else label_noise / (target_rms + 1e-12)
            ),
        )
        return data, stats


    def rollout_basis(
        self,
        rng: jax.Array,
        num_steps: int,
        basis: jax.Array,
        student: Callable | None = None,
        init_passes: int = 1,
        episode_steps: int | None = None,
        drive_row: int = 0,
        randomize_drive_omega: bool = False,
        reset_fn: Callable[[jax.Array], object] | None = None,
        failure_bank: "StateBank | None" = None,
        failure_lookback: int = 30,
        desc: str = "DIAL basis collection",
    ) -> tuple[list[DialScoreData], CollectionStats]:
        """Collect one dataset per basis weight from a single set of rollouts.

        Every query is labelled under all ``k`` basis weights with the same rng,
        so the rollouts and physics are shared and only the Gibbs weighting
        differs.  One environment step therefore pays for ``k`` labels, and the
        per-row targets are directly comparable because they see identical
        samples.

        ``drive_row`` selects which basis weight advances the annealing chain and
        the environment when no ``student`` is given; it should be the weight the
        deployed controller will actually run at.

        ``randomize_drive_omega`` draws a fresh preference weight for every
        episode as a uniform convex combination of the basis rows, and passes it
        to ``student`` so the *composed* controller drives.  Composition is
        evaluated at states no single basis weight visits, so collecting only
        along one row leaves exactly the distribution composition needs
        uncovered.  ``student`` then takes ``(plan, obs, t, omega)``.

        Returns one :class:`DialScoreData` per basis row, all sharing the same
        queries, observations, factors, and levels.
        """

        basis = jnp.asarray(basis, dtype=jnp.float32)
        num_rows = int(basis.shape[0])
        if not 0 <= drive_row < num_rows:
            raise ValueError(f"drive_row must lie in [0, {num_rows})")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if episode_steps is not None and episode_steps < 1:
            raise ValueError("episode_steps must be at least 1")

        num_levels = int(self.factors.shape[0])
        queries_per_level = 1 + self.perturbations
        if randomize_drive_omega and student is None:
            raise ValueError(
                "randomize_drive_omega needs a student that accepts omega"
            )

        def sample_omega(key):
            if not randomize_drive_omega:
                return basis[drive_row]
            # Uniform on the simplex, so the drawn weight is a convex
            # combination of the basis rows -- the region composition
            # interpolates over.
            coefficients = jax.random.dirichlet(key, jnp.ones(num_rows))
            return coefficients @ basis

        def with_drive(state, weights):
            info = dict(state.info)
            info["reward_weights"] = weights
            return state.replace(info=info)

        reset_fn = reset_fn or self._reset
        recent: list[object] = []
        rng, reset_rng, omega_rng = jax.random.split(rng, 3)
        drive_weights = sample_omega(omega_rng)
        drive_omegas = [np.asarray(drive_weights)]
        state = with_drive(reset_fn(reset_rng), drive_weights)
        plan = jnp.zeros((self.horizon, self.action_size), dtype=jnp.float32)

        rows_u: list[np.ndarray] = []
        rows_factor: list[float] = []
        rows_level: list[int] = []
        rows_delta: list[np.ndarray] = []   # each (k, H, nu)
        rows_obs: list[np.ndarray] = []
        rows_qpos: list[np.ndarray] = []
        rows_qvel: list[np.ndarray] = []
        rows_step: list[int] = []
        rewards: list[float] = []
        spreads: list[float] = []
        esses: list[float] = []
        num_resets = 0
        num_episodes = 1
        episode_step = 0

        planned = 1 if episode_steps is None else -(-num_steps // episode_steps)
        progress = tqdm(
            total=(num_steps - planned + planned * init_passes)
            * num_levels
            * queries_per_level,
            desc=desc,
            unit="query",
            dynamic_ncols=True,
        )

        def record(query, factor, level, deltas, obs, physics):
            rows_u.append(np.asarray(query))
            rows_factor.append(float(factor))
            rows_level.append(int(level))
            rows_delta.append(np.asarray(deltas))
            rows_obs.append(np.asarray(obs))
            rows_qpos.append(physics[0])
            rows_qvel.append(physics[1])
            rows_step.append(physics[2])

        for step_idx in range(num_steps):
            passes = init_passes if episode_step == 0 else 1
            obs = state.obs
            physics = (
                np.asarray(state.pipeline_state.q),
                np.asarray(state.pipeline_state.qd),
                int(state.info["step"]),
            )
            recent.append(state)
            if len(recent) > failure_lookback:
                recent = recent[-failure_lookback:]
            for _ in range(passes):
                for level in range(num_levels):
                    factor = float(self.factors[level])
                    noise = self.sigma_control * factor

                    rng, teacher_rng = jax.random.split(rng)
                    deltas, spread, ess = self.teacher.targets_basis(
                        state, plan, factor, teacher_rng, basis
                    )
                    deltas.block_until_ready()
                    record(plan, factor, level, deltas, obs, physics)
                    spreads.append(float(jnp.mean(spread)))
                    esses.append(float(jnp.mean(ess)))

                    for _ in range(self.perturbations):
                        rng, noise_rng, teacher_rng = jax.random.split(rng, 3)
                        eps = jax.random.normal(noise_rng, plan.shape)
                        query = jnp.clip(
                            plan + self.perturb_scale * noise[:, None] * eps,
                            -1.0,
                            1.0,
                        )
                        query = query.at[0].set(plan[0])
                        q_deltas, q_spread, q_ess = self.teacher.targets_basis(
                            state, query, factor, teacher_rng, basis
                        )
                        q_deltas.block_until_ready()
                        record(query, factor, level, q_deltas, obs, physics)
                        spreads.append(float(jnp.mean(q_spread)))
                        esses.append(float(jnp.mean(q_ess)))

                    if student is None:
                        step_delta = deltas[drive_row]
                    else:
                        t = factor_to_t(
                            factor, self.factor_min, self.factor_max
                        ).reshape(1)
                        step_delta = (
                            student(plan, obs, t, drive_weights)
                            if randomize_drive_omega
                            else student(plan, obs, t)
                        )
                    plan = jnp.clip(plan + step_delta, -1.0, 1.0)
                    progress.update(queries_per_level)

            state = self._step(state, plan[0])
            rewards.append(float(state.reward))
            progress.set_postfix(
                step=f"{step_idx + 1}/{num_steps}",
                reward=f"{rewards[-1]:.3f}",
                refresh=False,
            )
            fell = self.reset_on_done and float(state.done) > 0.5
            episode_over = (
                episode_steps is not None and episode_step + 1 >= episode_steps
            )
            if fell and failure_bank is not None:
                for banked in recent:
                    failure_bank.add(banked)
            if fell or episode_over:
                rng, reset_rng, omega_rng = jax.random.split(rng, 3)
                drive_weights = sample_omega(omega_rng)
                state = with_drive(reset_fn(reset_rng), drive_weights)
                plan = jnp.zeros_like(plan)
                episode_step = 0
                recent = []
                num_resets += int(fell)
                if step_idx + 1 < num_steps:
                    num_episodes += 1
                    drive_omegas.append(np.asarray(drive_weights))
            else:
                plan = self._shift(plan)
                episode_step += 1

        progress.close()

        if randomize_drive_omega:
            spread = np.stack(drive_omegas)
            spread = spread / spread.sum(axis=1, keepdims=True)
            print(
                f"  drive omegas: {len(drive_omegas)} episodes, "
                f"simplex mean {np.round(spread.mean(axis=0), 3).tolist()}, "
                f"std {np.round(spread.std(axis=0), 3).tolist()}"
            )
        stacked = np.stack(rows_delta)                     # (N, k, H, nu)
        shared = dict(
            u=jnp.asarray(np.stack(rows_u)),
            factor=jnp.asarray(np.asarray(rows_factor, dtype=np.float32)),
            level=jnp.asarray(np.asarray(rows_level, dtype=np.int32)),
            obs=jnp.asarray(np.stack(rows_obs)),
        )
        shared.update(
            qpos=jnp.asarray(np.stack(rows_qpos)),
            qvel=jnp.asarray(np.stack(rows_qvel)),
            step=jnp.asarray(np.asarray(rows_step, dtype=np.int32)),
        )
        datasets = [
            DialScoreData(delta=jnp.asarray(stacked[:, row]), **shared)
            for row in range(num_rows)
        ]

        mean_spread = float(np.mean(spreads))
        target_rms = float(np.sqrt(np.mean(np.square(stacked))))
        repeats = int(self.teacher.repeats)
        label_noise = (
            float(mean_spread / np.sqrt(repeats - 1)) if repeats > 1 else None
        )
        stats = CollectionStats(
            mean_reward=float(np.mean(rewards)),
            total_reward=float(np.sum(rewards)),
            num_steps=int(num_steps),
            num_resets=int(num_resets),
            num_episodes=int(num_episodes),
            mean_target_spread=mean_spread,
            mean_effective_samples=float(np.mean(esses)),
            target_rms=target_rms,
            label_noise=label_noise,
            relative_label_noise=(
                None if label_noise is None else label_noise / (target_rms + 1e-12)
            ),
        )
        return datasets, stats


# --------------------------------------------------------------------------- #
# Score regression training
# --------------------------------------------------------------------------- #


def score_regression_loss(
    model: DialScoreMLP,
    u: jax.Array,
    y: jax.Array,
    t: jax.Array,
    delta_target: jax.Array,
    weight: jax.Array | None = None,
) -> jax.Array:
    """``sigma**4``-weighted regression onto the teacher score.

    The teacher score is ``delta_target / sigma**2``.  Weighting its squared
    error by ``sigma**4`` makes the loss identical to mean-squared error of the
    plan update that will actually be applied, so no annealing level dominates
    purely because its score is numerically large.

    ``weight`` is an optional per-sample multiplier, shape ``(batch,)``, used by
    :func:`level_loss_weights` to rebalance the annealing levels.  Leaving it
    ``None`` keeps the plain update-MSE objective.
    """

    sigma_sq = jnp.square(model.sigma(t))  # (batch, H, 1)
    score_error = model.score(u, y, t) - delta_target / sigma_sq
    per_sample = jnp.mean(jnp.square(sigma_sq * score_error), axis=(-2, -1))
    if weight is not None:
        per_sample = per_sample * weight
    return jnp.mean(per_sample)


def level_loss_weights(data: DialScoreData, balance: float) -> jax.Array:
    """Per-level loss multipliers that rebalance the annealing levels.

    The plain objective is mean-squared error of the applied update, so a level
    whose updates are intrinsically smaller contributes proportionally less
    squared error and gets optimized less hard.  DIAL's fine level (small
    ``factor``) is exactly that case, and it is consistently the weakest head.

    ``balance`` interpolates between the two honest objectives rather than
    hiding the trade-off:

    * ``0`` — untouched update-MSE.  Every returned weight is exactly ``1``.
    * ``1`` — full equalization: each level's error is divided by that level's
      target power, so the loss becomes mean *relative* error and every level
      contributes equally regardless of its update size.

    Weights are renormalized to average ``1`` over the dataset, so the loss
    magnitude and hence the usable learning rate do not shift with ``balance``.
    """

    if not 0.0 <= balance <= 1.0:
        raise ValueError("balance must lie in [0, 1]")
    levels = np.asarray(data.level)
    delta = np.asarray(data.delta)
    num_levels = int(levels.max()) + 1
    weights = np.ones(num_levels, dtype=np.float32)
    if balance == 0.0:
        return jnp.asarray(weights)

    overall = float(np.mean(np.square(delta)))
    counts = np.zeros(num_levels, dtype=np.float64)
    for level in range(num_levels):
        mask = levels == level
        counts[level] = mask.sum()
        if counts[level] == 0:
            continue
        power = float(np.mean(np.square(delta[mask])))
        weights[level] = (overall / max(power, 1e-12)) ** balance
    # Renormalize by the dataset-weighted mean so E[weight] == 1.
    mean_weight = float(np.sum(counts * weights) / max(counts.sum(), 1.0))
    return jnp.asarray(weights / max(mean_weight, 1e-12))


def score_regression_metrics(
    model: DialScoreMLP,
    u: jax.Array,
    y: jax.Array,
    t: jax.Array,
    delta_target: jax.Array,
) -> dict[str, jax.Array]:
    """Loss plus scale-free accuracy of the predicted update."""

    predicted = model.update(u, y, t)
    error = predicted - delta_target
    target_power = jnp.mean(jnp.square(delta_target))
    flat_pred = predicted.reshape(predicted.shape[0], -1)
    flat_target = delta_target.reshape(delta_target.shape[0], -1)
    cosine = jnp.sum(flat_pred * flat_target, axis=-1) / (
        jnp.linalg.norm(flat_pred, axis=-1) * jnp.linalg.norm(flat_target, axis=-1)
        + 1e-12
    )
    return {
        "loss": jnp.mean(jnp.square(error)),
        "relative_rms": jnp.sqrt(
            jnp.mean(jnp.square(error)) / (target_power + 1e-12)
        ),
        "cosine": jnp.mean(cosine),
    }


def _split_indices(
    size: int, validation_fraction: float, rng: jax.Array
) -> tuple[jax.Array, jax.Array]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    permutation = jax.random.permutation(rng, size)
    num_val = int(round(size * validation_fraction))
    num_val = min(max(num_val, 1 if size > 1 else 0), size - 1 if size > 1 else 0)
    if num_val == 0:
        # Too small to hold anything out; report training accuracy instead of
        # silently claiming a validation number that does not exist.
        return permutation, permutation
    return permutation[num_val:], permutation[:num_val]


class FitResult(NamedTuple):
    """Outcome of :func:`fit_dial_score`."""

    history: list[dict[str, float]]
    best: dict[str, float]
    final: dict[str, float]
    per_level: dict[int, dict[str, float]]


def fit_dial_score(
    model: DialScoreMLP,
    optimizer: nnx.Optimizer,
    data: DialScoreData,
    *,
    normalizer: nnx.Module,
    batch_size: int,
    num_iters: int,
    rng: jax.Array,
    validation_fraction: float = 0.1,
    eval_every: int = 500,
    eval_chunk: int = 4096,
    keep_best: bool = True,
    level_weights: jax.Array | None = None,
    selection_fn: Callable[[int], float] | None = None,
    selection_every: int = 0,
    desc: str = "score regression",
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> FitResult:
    """Fit the single MLP by score regression on exact DIAL updates.

    Observations are normalised once up front with the fitted ``normalizer``
    (which holds no trainable parameters), and the conditioning ``t`` is derived
    from each sample's annealing factor with the model's own schedule, so the
    training and inference conditioning cannot drift apart.

    ``level_weights`` optionally rebalances the annealing levels in the training
    loss (see :func:`level_loss_weights`).  Reported metrics stay unweighted, so
    two runs with different weightings remain comparable on the same yardstick.

    ``selection_fn`` replaces validation loss as the checkpoint criterion.  It is
    called with the completed step count every ``selection_every`` steps and must
    return a score where higher is better, normally from a short closed-loop
    rollout.  Validation loss repeatedly failed to predict control here -- one
    run improved fine-level validation 30% while its closed-loop cost rose 13%
    -- because it is measured on states the teacher visited, not states the
    policy reaches.  Validation metrics are still recorded either way.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_iters < 1:
        raise ValueError("num_iters must be at least 1")

    obs = normalizer(data.obs, use_running_average=True)
    t = factor_to_t(data.factor, model.factor_min, model.factor_max)[:, None]
    sample_weight = (
        None
        if level_weights is None
        else jnp.asarray(level_weights)[data.level]
    )

    rng, split_rng = jax.random.split(rng)
    train_idx, val_idx = _split_indices(data.size, validation_fraction, split_rng)

    @nnx.jit
    def train_step(
        model: DialScoreMLP,
        optimizer: nnx.Optimizer,
        u: jax.Array,
        y: jax.Array,
        t: jax.Array,
        delta: jax.Array,
        weight: jax.Array | None,
    ) -> jax.Array:
        loss, grad = nnx.value_and_grad(score_regression_loss)(
            model, u, y, t, delta, weight
        )
        optimizer.update(grad)
        return loss

    @nnx.jit
    def eval_chunk_metrics(
        model: DialScoreMLP,
        u: jax.Array,
        y: jax.Array,
        t: jax.Array,
        delta: jax.Array,
    ) -> dict[str, jax.Array]:
        return score_regression_metrics(model, u, y, t, delta)

    def evaluate(indices: jax.Array) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for start in range(0, int(indices.shape[0]), eval_chunk):
            chunk = indices[start : start + eval_chunk]
            metrics = eval_chunk_metrics(
                model, data.u[chunk], obs[chunk], t[chunk], data.delta[chunk]
            )
            weight = int(chunk.shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight
            count += weight
        return {key: value / count for key, value in totals.items()}

    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_params = None
    best_selection = -float("inf")
    last_loss = 0.0

    def snapshot() -> None:
        nonlocal best_params
        if keep_best:
            best_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    progress = tqdm(
        range(num_iters), desc=desc, unit="step", dynamic_ncols=True
    )
    for step in progress:
        rng, batch_rng = jax.random.split(rng)
        batch = train_idx[
            jax.random.randint(
                batch_rng, (batch_size,), 0, int(train_idx.shape[0])
            )
        ]
        last_loss = float(
            train_step(
                model,
                optimizer,
                data.u[batch],
                obs[batch],
                t[batch],
                data.delta[batch],
                None if sample_weight is None else sample_weight[batch],
            )
        )

        completed = step + 1
        if eval_every > 0 and (
            completed % eval_every == 0 or completed == num_iters
        ):
            metrics = evaluate(val_idx)
            record = {
                "step": completed,
                "train_loss": last_loss,
                "val_loss": metrics["loss"],
                "val_relative_rms": metrics["relative_rms"],
                "val_cosine": metrics["cosine"],
            }
            history.append(record)
            if selection_fn is None and (
                best_metrics is None
                or record["val_loss"] < best_metrics["val_loss"]
            ):
                best_metrics = record
                snapshot()
            progress.set_postfix(
                train=f"{last_loss:.3e}",
                val=f"{record['val_loss']:.3e}",
                rms=f"{record['val_relative_rms']:.3f}",
                cos=f"{record['val_cosine']:.3f}",
                refresh=False,
            )
            if callback is not None:
                callback(completed, record)
        if (
            selection_fn is not None
            and selection_every > 0
            and (completed % selection_every == 0 or completed == num_iters)
        ):
            score = float(selection_fn(completed))
            if history and history[-1]["step"] == completed:
                history[-1]["selection"] = score
            if score > best_selection:
                best_selection = score
                snapshot()
                if history and history[-1]["step"] == completed:
                    best_metrics = history[-1]
            progress.set_postfix(
                train=f"{last_loss:.3e}", select=f"{score:.4f}",
                best=f"{best_selection:.4f}", refresh=False,
            )
        elif eval_every <= 0 or completed % eval_every != 0:
            progress.set_postfix(train=f"{last_loss:.3e}", refresh=False)
    progress.close()

    if keep_best and best_params is not None:
        nnx.update(model, best_params)

    final = evaluate(val_idx)
    per_level: dict[int, dict[str, float]] = {}
    for level in sorted({int(value) for value in np.asarray(data.level)}):
        level_idx = jnp.asarray(
            np.flatnonzero(np.asarray(data.level)[np.asarray(val_idx)] == level)
        )
        if level_idx.shape[0] == 0:
            continue
        per_level[level] = evaluate(val_idx[level_idx])

    if best_metrics is None:
        best_metrics = {
            "step": num_iters,
            "train_loss": last_loss,
            "val_loss": final["loss"],
            "val_relative_rms": final["relative_rms"],
            "val_cosine": final["cosine"],
        }
    if selection_fn is not None:
        best_metrics = dict(best_metrics or {})
        best_metrics["selection"] = best_selection
    return FitResult(
        history=history, best=best_metrics, final=final, per_level=per_level
    )


# --------------------------------------------------------------------------- #
# Denoising policy
# --------------------------------------------------------------------------- #


@dataclass
class DialScorePolicy:
    """Pickle-able denoising controller built on one score MLP.

    Plans live in DIAL's normalized spline-node coordinates, so no actuator
    rescaling happens anywhere: ``apply`` consumes and returns a plan that
    ``env.step`` can be driven with directly through ``plan[0]``.

    Attributes:
        model: The single :class:`DialScoreMLP`.
        normalizer: Fixed observation standardization.
        factors: Annealing schedule applied per control step, coarse to fine.
        shift_matrix: DIAL's ``node2u -> roll -> u2node`` warm-start map.
        dt: Step-size multiplier on the denoising update; ``1.0`` reproduces
            DIAL's update magnitude exactly.
    """

    model: DialScoreMLP
    normalizer: nnx.Module
    factors: jax.Array
    shift_matrix: jax.Array
    dt: float = 1.0
    # Sampling temperature the labels were generated at, and the per-level
    # multiplier applied to it.  A field is the score of a specific Gibbs
    # distribution, so the temperature is part of what it *is*: composing two
    # fields fitted at different temperatures without carrying them gets both
    # the sharpness and the direction wrong.
    temperature: float | None = None
    level_scales: jax.Array | None = None

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as stream:
            cloudpickle.dump(self, stream)

    @staticmethod
    def load(path: str | Path) -> "DialScorePolicy":
        with open(path, "rb") as stream:
            return cloudpickle.load(stream)

    def delta(
        self, plan: jax.Array, obs: jax.Array, t: jax.Array
    ) -> jax.Array:
        """Predicted DIAL update at one annealing level, from a raw observation."""

        y = self.normalizer(obs, use_running_average=True)
        return self.model.update(plan, y, t)

    def apply(
        self, plan: jax.Array, obs: jax.Array, passes: int = 1
    ) -> jax.Array:
        """Refine ``plan`` by annealed denoising at the current observation.

        ``passes`` tiles the annealing schedule, mirroring DIAL's larger
        ``Ndiffuse_init`` on the first control step.
        """

        if passes < 1:
            raise ValueError("passes must be at least 1")
        y = self.normalizer(obs, use_running_average=True)
        factors = jnp.tile(jnp.asarray(self.factors), passes)
        factor_min = self.model.factor_min
        factor_max = self.model.factor_max

        def denoise_step(plan: jax.Array, factor: jax.Array):
            t = factor_to_t(factor, factor_min, factor_max).reshape(1)
            sigma_sq = jnp.square(self.model.sigma(t))
            score = self.model.score(plan, y, t)
            return jnp.clip(plan + self.dt * sigma_sq * score, -1.0, 1.0), None

        plan, _ = jax.lax.scan(
            denoise_step, jnp.clip(plan, -1.0, 1.0), factors
        )
        return plan

    def shift(self, plan: jax.Array) -> jax.Array:
        """Warm-start the next control step with DIAL's own spline shift."""

        return jnp.einsum("ij,ja->ia", jnp.asarray(self.shift_matrix), plan)


@dataclass
class ComposedDialScorePolicy:
    """Linear composition of independently trained single-weight score fields.

    Each basis field is a complete :class:`DialScorePolicy`, not a head on a
    shared trunk.  That is deliberate: if the composed controller misbehaves,
    every basis field can be evaluated and watched on its own with the existing
    tools, so a shared encoder is never a candidate explanation.

    Composition
    -----------
    The exact Gibbs score is linear in ``omega`` because ``log Z(omega)`` does
    not depend on ``U``.  What DIAL supplies is that score smoothed at noise
    ``sigma`` and estimated with a finite softmax, which is linear in direction
    but **invariant to the scale of omega**: ``reverse_once`` divides the sample
    rewards by their own standard deviation, so ``omega`` and ``alpha * omega``
    produce the same Gibbs weights and hence the same update.

    Least squares alone therefore gets the direction right and the magnitude
    wrong whenever the coefficients do not sum to one — measured on this plant,
    ``(1,1,1)`` composed with cosine 0.9997 but 40% relative error.  Rescaling
    the coefficients to sum to one removes it (40% -> 2.4%) and cannot change
    the direction.  Keep that normalisation.

    Attributes:
        policies: One trained field per basis row, in the row order of
            ``mode_weights``.
        mode_weights: Basis preference weights, shape ``(k, num_objectives)``.
        pinv_mode_weights: Precomputed pseudoinverse, shape
            ``(num_objectives, k)``.  Stored so inference never calls
            ``jnp.linalg.pinv`` inside a jit.
    """

    policies: tuple[DialScorePolicy, ...]
    mode_weights: jax.Array
    pinv_mode_weights: jax.Array
    # One temperature per basis field, and the temperature composition targets
    # default to.  Absent (None) means the fields were fitted under DIAL's
    # spread normalisation, where the weight scale is inert and the legacy
    # sum-to-one solve is the only thing that can recover a magnitude.
    basis_temperatures: jax.Array | None = None
    temperature: float | None = None
    pinv_nu_weights: jax.Array | None = None

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as stream:
            cloudpickle.dump(self, stream)

    @staticmethod
    def load(path: str | Path) -> "ComposedDialScorePolicy":
        with open(path, "rb") as stream:
            return cloudpickle.load(stream)

    def coefficients(
        self, omega: jax.Array, target_temperature: jax.Array | None = None
    ) -> jax.Array:
        """Composition coefficients for a target weight and sharpness.

        Field ``i`` was fitted at ``(omega_i, T_i)`` and therefore learned the
        score of the natural parameter ``nu_i = omega_i / T_i`` -- temperature
        included.  Asking for ``omega*`` at ``T*`` means solving

            sum_i a_i nu_i = omega* / T*

        which is the same least squares as before, in nu rather than omega.
        With one shared temperature the two differ only by a scale, but that
        scale is exactly the magnitude the old sum-to-one rescale was guessing
        at; solving in nu recovers it instead.  Any per-level temperature
        profile cancels here, because it multiplies every field and the target
        alike, so these coefficients are the same at every annealing level.

        Falls back to the legacy sum-normalised omega solve for policies saved
        before temperatures were recorded, where the spread normalisation had
        already destroyed the magnitude.
        """

        pinv_nu = getattr(self, "pinv_nu_weights", None)
        temperature = (
            self.temperature if target_temperature is None else target_temperature
        )
        return mixture_from_pinv(
            omega, temperature, pinv_nu, self.pinv_mode_weights
        )

    def shift(self, plan: jax.Array) -> jax.Array:
        return self.policies[0].shift(plan)

    def apply(
        self, plan: jax.Array, obs: jax.Array, omega: jax.Array, passes: int = 1
    ) -> jax.Array:
        """Denoise ``plan`` under the composed score for ``omega``."""

        if passes < 1:
            raise ValueError("passes must be at least 1")
        coefficients = self.coefficients(omega)
        reference = self.policies[0]
        model = reference.model
        factor_min, factor_max = model.factor_min, model.factor_max
        observations = [
            policy.normalizer(obs, use_running_average=True)
            for policy in self.policies
        ]
        factors = jnp.tile(jnp.asarray(reference.factors), passes)

        def denoise_step(plan: jax.Array, factor: jax.Array):
            t = factor_to_t(factor, factor_min, factor_max).reshape(1)
            sigma_sq = jnp.square(model.sigma(t))
            score = sum(
                coefficients[i] * policy.model.score(plan, observations[i], t)
                for i, policy in enumerate(self.policies)
            )
            step = reference.dt * sigma_sq * score
            return jnp.clip(plan + step, -1.0, 1.0), None

        plan, _ = jax.lax.scan(
            denoise_step, jnp.clip(plan, -1.0, 1.0), factors
        )
        return plan


__all__ = [
    "ComposedDialScorePolicy",
    "StateBank",
    "mixed_reset_fn",
    "CollectionStats",
    "DialScoreCollector",
    "DialScoreData",
    "DialScoreMLP",
    "DialScorePolicy",
    "DialScoreTeacher",
    "FitResult",
    "build_shift_matrix",
    "concat_dial_score_data",
    "dial_factors",
    "dial_sigma_control",
    "factor_to_t",
    "fit_dial_score",
    "level_loss_weights",
    "load_dial_score_data",
    "save_dial_score_data",
    "score_regression_loss",
    "score_regression_metrics",
    "t_to_factor",
]
