"""Batched DIAL collection that stores clouds instead of labels.

Three things separate this from `DialScoreCollector`, in descending order of
how much they matter.

**One rollout serves every basis row.**  A rollout does not depend on the
weight vector -- `reward_weights` enters only the scalar reward -- so labelling
`k` rows by calling the teacher `k` times rolls the same physics `k` times.
Reading the per-row cost off each sample instead makes the basis free, which on
a four-row basis is a straight 4x.

**One jitted call per control pass.**  The serial collector blocks on the device
once per query per row: at two annealing levels, five queries and four rows that
is eighty round trips per environment step.  Everything inside a pass is one
program here, and several environments run inside it, so the device stops
waiting on Python.

**Clouds are what gets written down.**  Per-sample per-row costs are stored, not
finished updates, so the temperature, the weight vector, the repeat count and
the choice of standard-deviation normalisation are all decided *after*
collection by `csm.cloud_data.make_relabeler` -- a softmax, no physics.  The
sampled trajectories are recoverable from one rng key, so they are not stored.

Measured on this plant, throughput saturates near 1.7M environment-steps per
second anywhere above ~16k concurrent rollouts, so batching past that buys
nothing on its own.  What it buys is the removal of redundant work and of
dispatch stalls, and those are why this is faster.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp

from pathlib import Path

from csm.cloud_data import DialCloudData, save_clouds
from csm.dial_lean import make_rollout, make_sampler, mppi_weights
from csm.dial_score import factor_to_t
from csm.omega import mixture_from_pinv, normalize_omega


def make_pass(
    env,
    mbdpi,
    dial_config,
    perturb_scales,
    repeats: int,
    temperature: float,
    std_normalize: bool = False,
    student: Callable | None = None,
    level_scales=None,
):
    """One annealing pass over every level, batched across environments.

    Returns a function mapping `(states, plans, rngs, omegas)` to updated plans
    and the clouds visited on the way.  `student`, when given, takes
    `(plan, obs, t, omega)` and replaces the teacher as the thing that advances
    the plan -- the labels still come from DIAL either way.
    """

    sample = make_sampler(mbdpi, dial_config)
    rollout = make_rollout(env, lambda s: s.info["reward_terms"])
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))
    sigma = mbdpi.sigma_control
    factors = dial_config.traj_diffuse_factor ** jnp.arange(dial_config.Ndiffuse)
    factor_min = float(jnp.min(factors))
    factor_max = float(jnp.max(factors))
    scales = jnp.asarray(perturb_scales, dtype=jnp.float32)
    n_query = 1 + int(scales.shape[0])
    temps = (jnp.ones_like(factors) if level_scales is None
             else jnp.asarray(level_scales, dtype=factors.dtype)) * temperature

    def queries_for(plan, noise, rng):
        """The plan itself plus copies displaced by a ladder of distances.

        One displacement per perturbed slot rather than one shared scale.  The
        student is fitted on `(plan, obs, t)`, so a query's distance is part of
        its *input*, not a hidden change of target -- unlike the temperature,
        which has to stay a property of the field.  A ladder therefore buys
        clean labels near the plan and coverage far from it in one collection.
        """

        eps = jax.random.normal(rng, (scales.shape[0],) + plan.shape)
        displaced = plan + scales[:, None, None] * noise[None, :, None] * eps
        displaced = jnp.clip(displaced, -1.0, 1.0).at[:, 0].set(plan[0])
        return jnp.concatenate([plan[None], displaced], axis=0)

    def cloud_terms(state, query, noise, key):
        nodes = sample(key, query, noise)
        return rollout_vmap(state, mbdpi.node2u_vvmap(nodes)).mean(axis=1)

    def level_pass(carry, level):
        factor, temp = level
        state, plan, rng, omega = carry
        noise = sigma * factor
        rng, query_rng, cloud_rng = jax.random.split(rng, 3)
        queries = queries_for(plan, noise, query_rng)
        keys = jax.random.split(cloud_rng, n_query * repeats).reshape(
            n_query, repeats, 2
        )
        terms = jax.vmap(
            jax.vmap(cloud_terms, in_axes=(None, None, None, 0)),
            in_axes=(None, 0, None, 0),
        )(state, queries, noise, keys)          # (Q, R, N, k)

        # The drive update is the base query's label under this environment's
        # own weight vector, averaged over repeats exactly as the teacher does.
        returns = terms[0] @ omega              # (R, N)
        weights = mppi_weights(returns, temp, std_normalize, axis=-1)
        base_nodes = jax.vmap(lambda k: sample(k, plan, noise))(keys[0])
        drive = jnp.einsum("rn,rnij->ij", weights, base_nodes) / repeats - plan

        if student is not None:
            drive = student(
                plan, state.obs,
                factor_to_t(factor, factor_min, factor_max).reshape(1), omega,
            )

        new_plan = jnp.clip(plan + drive, -1.0, 1.0)
        record = {"u": queries, "terms": terms, "sample_rng": keys,
                  "noise": jnp.broadcast_to(noise, (n_query,) + noise.shape),
                  "factor": jnp.full((n_query,), factor)}
        return (state, new_plan, rng, omega), record

    def one_env(state, plan, rng, omega):
        (_, plan, rng, _), record = jax.lax.scan(
            level_pass, (state, plan, rng, omega), (factors, temps)
        )
        return plan, rng, record

    @jax.jit
    def run(states, plans, rngs, omegas):
        return jax.vmap(one_env)(states, plans, rngs, omegas)

    return run, int(factors.shape[0]), n_query


def make_driver(env):
    """Batched reset and step, plus in-place reset of whichever envs went down."""

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    @jax.jit
    def replace_fallen(states, fresh, mask):
        return jax.tree.map(
            lambda old, new: jnp.where(
                mask.reshape((-1,) + (1,) * (old.ndim - 1)), new, old
            ),
            states,
            fresh,
        )

    @jax.jit
    def set_weights(states, omegas):
        info = dict(states.info)
        info["reward_weights"] = jax.vmap(normalize_omega)(omegas)
        return states.replace(info=info)

    return reset, step, replace_fallen, set_weights


def sample_drive_omegas(rng, basis, num_envs: int, mix_basis_rows: float = 0.5):
    """Weights the environments are driven at during collection.

    A plain Dirichlet draw over the basis rows starves the rows themselves --
    measured previously at [0.37, 0.42, 0.22] on a three-row simplex -- and the
    basis rows are deployment targets, so a fraction of the environments is
    pinned to them outright.
    """

    basis = jnp.asarray(basis)
    k = basis.shape[0]
    rng, pick_rng, dirichlet_rng, which_rng = jax.random.split(rng, 4)
    mixed = jax.random.dirichlet(dirichlet_rng, jnp.ones(k), (num_envs,)) @ basis
    rows = basis[jax.random.randint(which_rng, (num_envs,), 0, k)]
    take_row = jax.random.uniform(pick_rng, (num_envs,)) < mix_basis_rows
    chosen = jnp.where(take_row[:, None], rows, mixed)
    return rng, jax.vmap(normalize_omega)(chosen)


def make_disturber(push_interval: int, push_probability: float,
                   speed_min: float, speed_max: float):
    """Shove the base between control steps, where the planner cannot see it.

    A push written into `env.step` would be visible to DIAL's rollouts -- they
    call the same function -- so every sampled trajectory would brace for a
    disturbance it has no business knowing about.  Applying it outside the
    planner keeps the shove a surprise, which is the whole task.

    Pushes inside an episode also fix what the collected states look like.  With
    a single shove at reset, a 150-step episode spends its first second
    recovering and the rest standing still, so most of the dataset would be
    quiet standing and the transient this task is about would be rare.
    """

    def disturb(states, rng, index):
        if push_interval < 1:
            return states
        num_envs = states.done.shape[0]
        fire_key, hit_key, heading_key, speed_key = jax.random.split(rng, 4)
        due = (index % push_interval == 0) & (index > 0)
        hit = jax.random.uniform(hit_key, (num_envs,)) < push_probability
        heading = jax.random.uniform(
            heading_key, (num_envs,), minval=-jnp.pi, maxval=jnp.pi
        )
        speed = jax.random.uniform(
            speed_key, (num_envs,), minval=speed_min, maxval=speed_max
        )
        impulse = jnp.stack(
            [speed * jnp.cos(heading), speed * jnp.sin(heading),
             jnp.zeros_like(speed)], axis=-1
        )
        apply = (due & hit)[:, None]
        pipeline_state = states.pipeline_state
        qvel = pipeline_state.qvel.at[:, :3].add(
            jnp.where(apply, impulse, 0.0)
        )
        return states.replace(pipeline_state=pipeline_state.replace(qvel=qvel))

    return disturb


def make_block(env, mbdpi, run_pass, episode_steps: int, disturb=None):
    """Scan several control steps inside one jitted call.

    Host round trips, not device throughput, are what the batched pass is left
    fighting: at 8 environments a single control step is ~5 s of rollout but the
    loop around it costs about as much again.  Everything below -- stepping,
    detecting falls, resetting, resampling the drive weight, warm-starting the
    plan -- runs on device, so a block of K steps costs one dispatch instead
    of K.

    A fall inside a block resets that environment in place without the extra
    annealing passes an episode boundary gets.  The plan is zeroed either way,
    so the difference is a few control steps of a colder start, and keeping the
    boundary out of the scan is what lets `init_passes` stay a Python integer.
    """

    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)

    def block(carry, inputs):
        states, plans, pass_rngs, omegas, basis, mix = carry
        key, index = inputs
        observed = (states.obs, states.pipeline_state.q,
                    states.pipeline_state.qd, states.info["step"])

        plans, pass_rngs, record = run_pass(states, plans, pass_rngs, omegas)
        states = step(states, plans[:, 0])
        plans = jax.vmap(mbdpi.shift)(plans)

        key, reset_key, omega_key, push_key = jax.random.split(key, 4)
        if disturb is not None:
            states = disturb(states, push_key, index)
        num_envs = plans.shape[0]
        fresh = reset(jax.random.split(reset_key, num_envs))
        _, new_omegas = sample_drive_omegas(omega_key, basis, num_envs, mix)
        info = dict(fresh.info)
        info["reward_weights"] = new_omegas
        fresh = fresh.replace(info=info)

        down = states.done > 0.5
        states = jax.tree.map(
            lambda old, new: jnp.where(
                down.reshape((-1,) + (1,) * (old.ndim - 1)), new, old
            ),
            states, fresh,
        )
        omegas = jnp.where(down[:, None], new_omegas, omegas)
        plans = jnp.where(down[:, None, None], jnp.zeros_like(plans), plans)

        carry = (states, plans, pass_rngs, omegas, basis, mix)
        return carry, (record, observed, states.reward, down)

    @jax.jit
    def run(states, plans, pass_rngs, omegas, basis, mix, keys, indices):
        carry = (states, plans, pass_rngs, omegas, basis, mix)
        carry, out = jax.lax.scan(block, carry, (keys, indices))
        return carry[0], carry[1], carry[2], carry[3], out

    return run


def _pack(chunks) -> DialCloudData:
    merged = {
        name: jnp.asarray(np.concatenate([chunk[name] for chunk in chunks]))
        for name in chunks[0]
    }
    return DialCloudData(
        u=merged["u"], factor=merged["factor"], level=merged["level"],
        obs=merged["obs"], qpos=merged["qpos"], qvel=merged["qvel"],
        step=merged["step"], sample_rng=merged["sample_rng"],
        terms=merged["terms"], noise=merged["noise"],
    )


def _flush(chunks, shard_dir: Path, index: int) -> Path:
    path = shard_dir / f"clouds_{index:04d}.npz"
    save_clouds(path, _pack(chunks))
    return path


def _shard_size(path: str) -> int:
    with np.load(path) as payload:
        return int(payload["u"].shape[0])


def collect(
    env,
    mbdpi,
    dial_config,
    basis,
    rng,
    num_steps: int,
    num_envs: int = 8,
    perturb_scales=(0.25, 0.5, 1.0, 1.5),
    repeats: int = 2,
    temperature: float = 0.25,
    std_normalize: bool = False,
    level_scales=None,
    episode_steps: int = 150,
    init_passes: int = 5,
    steps_per_call: int = 25,
    push_interval: int = 40,
    push_probability: float = 0.8,
    push_speed_min: float = 0.25,
    push_speed_max: float = 1.2,
    shard_dir=None,
    shard_every: int = 0,
    student: Callable | None = None,
    mix_basis_rows: float = 0.5,
    progress: bool = True,
):
    """Run `num_steps` synchronized control steps across `num_envs` copies.

    Episode windows are shared, so `init_passes` applies to every environment at
    once at a boundary and never becomes a per-environment branch that vmap
    would have to evaluate on both sides.  Between boundaries the loop runs on
    device in blocks of `steps_per_call`.
    """

    run_pass, num_levels, n_query = make_pass(
        env, mbdpi, dial_config, perturb_scales, repeats,
        temperature, std_normalize, student, level_scales,
    )
    disturb = make_disturber(
        push_interval, push_probability, push_speed_min, push_speed_max
    )
    run_block = make_block(env, mbdpi, run_pass, episode_steps, disturb)
    reset = jax.jit(jax.vmap(env.reset))

    @jax.jit
    def set_weights(states, omegas):
        info = dict(states.info)
        info["reward_weights"] = jax.vmap(normalize_omega)(omegas)
        return states.replace(info=info)

    basis = jnp.asarray(basis)
    mix = jnp.asarray(mix_basis_rows, dtype=jnp.float32)
    rng, reset_rng, omega_rng, pass_rng = jax.random.split(rng, 4)
    states = set_weights(
        reset(jax.random.split(reset_rng, num_envs)),
        sample_drive_omegas(omega_rng, basis, num_envs, mix_basis_rows)[1],
    )
    omegas = states.info["reward_weights"]
    plans = jnp.zeros(
        (num_envs, dial_config.Hnode + 1, int(env.action_size)), dtype=jnp.float32
    )
    pass_rngs = jax.random.split(pass_rng, num_envs)

    per_query = num_levels * n_query
    chunks: list[dict] = []
    shards: list[str] = []
    shard_dir = None if shard_dir is None else Path(shard_dir)
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    falls = 0
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=num_steps, desc="parallel DIAL collection", unit="step",
                   dynamic_ncols=True)

    def stash(record, observed, blocks: int):
        """(B, E, L, Q, ...) -> one row per (block, env, level, query)."""

        flat = {name: np.asarray(value).reshape((-1,) + value.shape[4:])
                for name, value in record.items()}
        obs, qpos, qvel, env_step = (np.asarray(v) for v in observed)
        flat["obs"] = np.repeat(obs.reshape(-1, obs.shape[-1]), per_query, axis=0)
        flat["qpos"] = np.repeat(qpos.reshape(-1, qpos.shape[-1]), per_query, axis=0)
        flat["qvel"] = np.repeat(qvel.reshape(-1, qvel.shape[-1]), per_query, axis=0)
        flat["step"] = np.repeat(env_step.reshape(-1), per_query, axis=0)
        flat["level"] = np.tile(
            np.repeat(np.arange(num_levels), n_query), blocks * num_envs
        )
        chunks.append(flat)

    done_steps = 0
    while done_steps < num_steps:
        at_boundary = done_steps % episode_steps == 0
        # Extra annealing passes at an episode start: the plan is zero there and
        # one pass per control step would leave the first seconds unplanned.
        if at_boundary and init_passes > 1:
            for _ in range(init_passes - 1):
                plans, pass_rngs, record = run_pass(states, plans, pass_rngs, omegas)
                observed = (states.obs, states.pipeline_state.q,
                            states.pipeline_state.qd, states.info["step"])
                stash(jax.tree.map(lambda x: x[None], record),
                      tuple(v[None] for v in observed), 1)

        block = min(steps_per_call, num_steps - done_steps,
                    episode_steps - done_steps % episode_steps)
        rng, block_rng = jax.random.split(rng)
        states, plans, pass_rngs, omegas, (record, observed, reward, down) = run_block(
            states, plans, pass_rngs, omegas, basis, mix,
            jax.random.split(block_rng, block),
            jnp.arange(done_steps, done_steps + block) % episode_steps,
        )
        stash(record, observed, block)
        rewards.extend(np.asarray(reward).mean(axis=1).tolist())
        falls += int(np.asarray(down).sum())
        done_steps += block

        if done_steps % episode_steps == 0 and done_steps < num_steps:
            rng, reset_rng, omega_rng = jax.random.split(rng, 3)
            rng_new, new_omegas = sample_drive_omegas(
                omega_rng, basis, num_envs, mix_basis_rows
            )
            states = set_weights(
                reset(jax.random.split(reset_rng, num_envs)), new_omegas
            )
            omegas = states.info["reward_weights"]
            plans = jnp.zeros_like(plans)

        # Flushing as we go keeps a multi-hour collection from living or dying
        # with the process, and keeps peak host memory bounded.
        if (shard_dir is not None and shard_every > 0
                and done_steps % shard_every == 0):
            shards.append(str(_flush(chunks, shard_dir, len(shards))))
            chunks = []

        if bar is not None:
            bar.set_postfix(reward=f"{rewards[-1]:.3f}", falls=falls,
                            shards=len(shards), refresh=False)
            bar.update(block)

    if bar is not None:
        bar.close()

    if shard_dir is not None:
        if chunks:
            shards.append(str(_flush(chunks, shard_dir, len(shards))))
            chunks = []
        data = None
    else:
        data = _pack(chunks)
    stats = {
        "queries": (sum(_shard_size(path) for path in shards)
                    if data is None else data.size),
        "shards": shards,
        "env_steps": num_steps * num_envs,
        "mean_reward": float(np.mean(rewards)),
        "falls": falls,
        "num_envs": num_envs,
        "repeats": repeats,
        "levels": num_levels,
        "queries_per_level": n_query,
        "perturb_scales": list(map(float, perturb_scales)),
        "level_scales": (None if level_scales is None
                         else list(map(float, level_scales))),
        "steps_per_call": steps_per_call,
    }
    return data, stats


def make_student_driver(policy, target_temperature=None):
    """Let a trained composed policy advance the plan during collection.

    DAgger's whole point is to label the states the student actually reaches,
    so the student drives while DIAL keeps supplying the labels -- the clouds
    are still rolled out at every query either way, only the plan the queries
    are centred on changes.

    Each environment carries its own weight vector, so the composition
    coefficients are recomputed inside the loop rather than fixed up front.
    That is a 4x4 solve against a stored pseudoinverse, which costs nothing
    beside a 2049-sample rollout.
    """

    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    pinv_nu = jnp.asarray(policy.pinv_nu_weights)
    temperature = jnp.asarray(
        policy.temperature if target_temperature is None else target_temperature,
        dtype=jnp.float32,
    )

    def student(plan, obs, t, omega):
        coefficients = mixture_from_pinv(
            omega, temperature, pinv_nu, policy.pinv_mode_weights
        )
        update = None
        for index, field in enumerate(fields):
            term = coefficients[index] * field.delta(plan, obs, t)
            update = term if update is None else update + term
        return update

    return student
