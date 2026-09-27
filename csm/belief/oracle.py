"""Oracle robust MPPI (baseline B2) and the gating experiment E0.

B2 plans by rolling every sampled plan under every particle of the belief and
weighting by it:

    C_w(V_k) = sum_m w_m c(V_k, theta_m)
    U <- U + sum_k softmax(-C_w/lam)_k eps_k

This is the quality ceiling the learned method approximates, and it costs `M`
times a nominal plan.  **E0 asks whether that cost buys anything**: run it at
`M in {1, 2, ..., 64}` and look at success against wall clock.  If success
saturates by `M = 8` the premise of the whole approach is weak and the task
needs redesigning before anything is learned; if it keeps climbing to 32 or 64
the premise holds.

The belief here is the prior, not a filtered posterior.  E0 is about whether
hedging over a spread of hypotheses helps at all, so it deliberately runs the
hardest case -- the controller knows only the box `theta` was drawn from.  A
particle filter can only narrow that, so E0 is an upper bound on how much `M`
can matter, which is the right thing for a gate to be.

Raw Gibbs throughout: no division of the returns by their own spread.  That
division makes the update invariant to the scale of the weight, which is
exactly what a belief's sum-to-one normalisation is there to fix, and it would
make the temperature mean something different for every belief.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

from csm.belief import plant as P


def make_planner(model, bare, ids, n_sample, w: P.Costs = P.Costs(),
                 substeps: int = P.PLAN_SUBSTEPS):
    """One MPPI update at a belief: `(data, U, theta, w, sigma, lam, goal, key) -> U`.

    `substeps` is the **planner's** integration fidelity and defaults to the
    coarser `P.PLAN_SUBSTEPS`; `make_world` below always steps the real plant at
    `P.SUBSTEPS`, so nothing reported here rides on the planner's shortcut.

    `lam` is a traced argument, not a closure constant.  Closing over it forces
    a separate compilation per annealing level, and the thing being compiled is
    a double-vmapped MJX rollout over `n_sample x M` plans -- minutes each, with
    XLA holding the interpreter lock for long stretches, which starves any
    render thread sharing the process.  Nothing about the temperature needs to
    be static.
    """
    cost_mat, _ = P.make_cost_matrix(model, bare, ids, w, substeps)

    @jax.jit
    def update(data, U, theta, belief, sigma, lam, goal, key):
        eps = sigma * jax.random.normal(key, (n_sample, P.DIM_U))
        eps = eps.at[-1].set(0.0)               # the proposal centre is scored too
        V = jnp.clip(U[None, :] + eps, -1.0, 1.0)
        c = cost_mat(data, V, theta, goal)      # (K, M)
        Cw = c @ belief
        om = jax.nn.softmax(-(Cw - Cw.min()) / lam)
        ess = 1.0 / jnp.sum(om ** 2) / om.shape[0]
        return jnp.clip(U + om @ eps, -1.0, 1.0), ess

    return update


def make_world(model, bare, ids, w: P.Costs = P.Costs()):
    """The real plant: steps under the *true* theta and reports its cost rows."""
    @jax.jit
    def step(data, u, theta_true, goal):
        mdl = P.apply_theta(model, ids, theta_true)
        prev = data.cvel[ids["payload"], 3:]
        nd, bias = P.step(mdl, bare, ids, data, u)
        rows = P.stage_cost(mdl, ids, nd, prev, u, theta_true, goal, bias, w)
        return nd, rows

    return step


def make_e0(model, bare, ids, n_ep, n_sample, w: P.Costs = P.Costs(),
            substeps: int = P.PLAN_SUBSTEPS):
    """Every episode advanced in lockstep, one dispatch per annealing level.

    E0 wants success *and* wall clock against `M`, and run episode by episode
    the small-`M` end lands below the dispatch floor: measured on this plant,
    128, 256 and 512 plans all cost the same 0.133 s, because what sets the
    floor there is the rollout's sequential depth (`H * substeps` kernel
    launches) rather than the physics.  At `M = 1` a sequential episode is 512
    rollouts, squarely in that regime, so a sequential E0 would report `M = 1`
    and `M = 8` as costing the same and the cost axis of the gate would be
    meaningless.  Episodes are independent, so batching them is a vmap.

    The belief is shared across episodes -- it is one prior -- while the true
    parameters differ per episode, which is exactly the asymmetry E0 is about.
    """
    _, cost_fn = P.make_rollout(model, bare, ids, w, substeps)

    @jax.jit
    def update(datas, Us, th_b, belief, sigma, lam, goal, key):
        eps = sigma * jax.random.normal(key, (n_ep, n_sample, P.DIM_U))
        eps = eps.at[:, -1, :].set(0.0)         # the proposal centre is scored
        V = jnp.clip(Us[:, None, :] + eps, -1.0, 1.0)

        def per_ep(data, v):
            c = jax.vmap(jax.vmap(cost_fn, in_axes=(None, 0, None, None)),
                         in_axes=(None, None, 0, None), out_axes=1)(
                             data, v, th_b, goal)          # (K, M)
            return c @ belief

        Cw = jax.vmap(per_ep)(datas, V)                    # (n_ep, K)
        om = jax.nn.softmax(-(Cw - Cw.min(axis=1, keepdims=True)) / lam, axis=1)
        ess = 1.0 / jnp.sum(om ** 2, axis=1) / n_sample
        return jnp.clip(Us + jnp.einsum("ek,ekd->ed", om, eps), -1.0, 1.0), ess

    @jax.jit
    def world(datas, us, th_true, goal):
        """The real plant, at `P.SUBSTEPS` -- never the planner's shortcut."""
        def one(d, u, th):
            mdl = P.apply_theta(model, ids, th)
            prev = d.cvel[ids["payload"], 3:]
            nd, bias = P.step(mdl, bare, ids, d, u, P.SUBSTEPS)
            return nd, P.stage_cost(mdl, ids, nd, prev, u, th, goal, bias, w)
        return jax.vmap(one)(datas, us, th_true)

    @jax.jit
    def init(q0, th_true):
        d0 = mjx.make_data(model).replace(qpos=q0)
        datas = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_ep,) + jnp.shape(x)), d0)
        return jax.vmap(lambda d, th: mjx.forward(
            P.apply_theta(model, ids, th), d))(datas, th_true)

    return init, update, world


def run_e0(ids, init, update, world, th_true, th_b, bel, goal, q0, steps,
           sigma_sched, lam_sched, key):
    """One batched run: `(gaps, profiles, mean ESS, seconds per control step)`."""
    datas = init(q0, th_true)
    n_ep = th_true.shape[0]
    Us = jnp.zeros((n_ep, P.DIM_U))
    prof = jnp.zeros((n_ep, 7))
    ess, wall = [], 0.0
    for t in range(steps):
        t0 = time.time()
        for sg, lam in zip(sigma_sched, lam_sched):
            key, k = jax.random.split(key)
            Us, e = update(datas, Us, th_b, bel, sg, lam, goal, k)
            ess.append(e)
        Us = jax.block_until_ready(Us)
        if t > 0:                         # step 0 carries the compile
            wall += time.time() - t0
        datas, rows = world(datas, Us[:, :P.NU], th_true, goal)
        prof = prof + rows
        Us = jax.vmap(P.shift_nodes)(Us)
    gaps = jnp.linalg.norm(datas.xpos[:, ids["payload"]] - goal, axis=-1)
    return (np.asarray(gaps), np.asarray(prof),
            float(np.mean([float(x.mean()) for x in ess])),
            wall / max(steps - 1, 1))


def run_episode(model, ids, update, world, theta_true, theta_belief, belief,
                goal, q0, steps, passes, sigma_sched, lam_sched, key,
                w: P.Costs = P.Costs()):
    """One closed-loop episode.  Returns the per-row violation profile."""
    data = mjx.make_data(model).replace(qpos=q0)
    data = mjx.forward(P.apply_theta(model, ids, theta_true), data)
    U = jnp.zeros(P.DIM_U)
    prof = jnp.zeros(7)
    ess_log = []
    for t in range(steps):
        for sg, lam in zip(sigma_sched, lam_sched):
            for _ in range(passes):
                key, k = jax.random.split(key)
                U, ess = update(data, U, theta_belief, belief, sg, lam, goal, k)
                ess_log.append(float(ess))
        data, rows = world(data, U[:P.NU], theta_true, goal)
        prof = prof + rows
        U = P.shift_nodes(U)
    p_final = data.xpos[ids["payload"]]
    return np.asarray(prof), float(jnp.linalg.norm(p_final - goal)), \
        float(np.mean(ess_log))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--episodes", type=int, default=16,
                    help="true-theta draws, all advanced in one dispatch")
    ap.add_argument("--belief-seeds", type=int, default=3,
                    help="independent draws of the M particles.  At M = 1 the "
                         "belief *is* a single sample of the prior, so one "
                         "seed measures that draw's luck rather than M.")
    ap.add_argument("--steps", type=int, default=30,
                    help="control steps; 30 is 1.5 s, and the transport needs "
                         "about 0.5 s of it")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--lam", type=float, nargs="+", default=list(P.LAM_SCHEDULE),
                    help="one temperature per annealing level")
    ap.add_argument("--sigma", type=float, nargs="+",
                    default=list(P.SIGMA_SCHEDULE))
    # Defaults from the plant, not hand-typed.  The previous hard-coded
    # [0.35, 0.30, 0.55] was the goal that asked the arm to *retract* -- it
    # drops the payload's radius from 0.48 m to 0.14 m -- and the oracle was
    # measured moving away from it.  Likewise q0: the XML keyframe is the old
    # carry pose that holds the payload 82 degrees off level.
    ap.add_argument("--goal", type=float, nargs=3, default=list(map(float, P.GOAL)))
    # Three radii, and the verdict does not use any of them.  A single success
    # threshold turned out to be measuring the task rather than `M`: the oracle
    # asymptotes near 0.09 m of the 0.329 m transport and does not converge
    # past it -- at 512 plans, going from 30 to 40 control steps moved the mean
    # gap 0.085 -> 0.093, i.e. it settles and mills around rather than closing.
    # A threshold below that reads ~0 for every M, one above it reads ~1, and
    # picking the one in between means picking the answer.  So the gate reads
    # the continuous distance and the rates are printed for context.
    ap.add_argument("--reach", type=float, nargs="+",
                    default=[0.05, 0.075, 0.10],
                    help="metres; success radii reported side by side")
    args = ap.parse_args(argv)

    model, bare, ids, mj = P.load()
    goal = jnp.asarray(args.goal)
    q0 = jnp.asarray(P.Q_CARRY)
    init, update, world = make_e0(model, bare, ids, args.episodes, args.samples)

    # The same true parameters for every M and every seed: the comparison is
    # paired, so a difference between two rows cannot be a different task.
    theta_true = P.sample_theta(jax.random.PRNGKey(7), args.episodes)
    d0 = float(jnp.linalg.norm(P.P_START - goal))

    print(f"E0: oracle robust MPPI.  {args.episodes} episodes x {args.steps} "
          f"control steps ({args.steps * P.DT:.1f} s), {args.samples} plans, "
          f"{args.belief_seeds} belief draws per M")
    print(f"transport {d0:.3f} m, reach {args.reach} m, sigma {args.sigma}, "
          f"lam {args.lam}, planner substeps {P.PLAN_SUBSTEPS} / world "
          f"{P.SUBSTEPS}\n")
    rn = "".join(f"{'<' + format(r, '.3g'):>8}" for r in args.reach)
    hdr = (f"{'M':>4}{'gap m':>8}{'+- seed':>9}{'worst':>8}{rn}"
           f"{'slip':>7}{'tilt':>7}{'tau':>7}{'ESS%':>7}{'s/step':>9}"
           f"{'s/ep/step':>11}")
    print(hdr); print("-" * len(hdr))

    out = {}
    for M in args.particles:
        gaps_s, profs_s, ess_s, t_s, seed_mean = [], [], [], [], []
        for s in range(args.belief_seeds):
            th_b = P.sample_theta(jax.random.fold_in(jax.random.PRNGKey(11), s), M)
            bel = jnp.full((M,), 1.0 / M)
            gaps, prof, ess, s_step = run_e0(
                ids, init, update, world, theta_true, th_b, bel, goal, q0,
                args.steps, args.sigma, args.lam,
                jax.random.fold_in(jax.random.PRNGKey(0), s))
            gaps_s.append(gaps); seed_mean.append(float(gaps.mean()))
            profs_s.append(prof); ess_s.append(ess); t_s.append(s_step)
        allprof = np.concatenate(profs_s)          # (seeds * episodes, 7)
        pr = allprof.mean(0)
        g = np.concatenate(gaps_s)
        # a success has to be a *clean* success: reached, and no constraint row
        # accumulated anything over the episode
        ok = ((allprof[:, 1] < 1.0) & (allprof[:, 2] < 1.0)
              & (allprof[:, 3] < 1.0))
        rates = "".join(f"{float(((g < r) & ok).mean()):>8.2f}"
                        for r in args.reach)
        out[M] = (float(g.mean()), float(np.std(seed_mean)), np.mean(t_s))
        print(f"{M:4d}{g.mean():8.3f}{np.std(seed_mean):9.3f}{g.max():8.3f}"
              f"{rates}{pr[2]:7.2f}{pr[3]:7.2f}{pr[1]:7.2f}"
              f"{100*np.mean(ess_s):7.1f}{np.mean(t_s):9.3f}"
              f"{np.mean(t_s)/args.episodes:11.4f}")

    # The verdict is on distance covered, which needs no threshold.
    prog = {m: d0 - out[m][0] for m in out}
    best = max(prog, key=prog.get)
    sat = min((m for m in prog if prog[m] >= 0.95 * prog[best]), default=best)
    print(f"\ndistance covered peaks at M = {best} "
          f"({prog[best]:.3f} m of {d0:.3f}, gap {out[best][0]:.3f}), and is "
          f"within 5% of that already at M = {sat}.")
    print(f"seed spread at the peak is +-{out[best][1]:.3f} m, so a difference "
          f"smaller than\nabout {2*out[best][1]:.3f} m between two rows is not "
          f"a difference.")
    print(f"cost of that: {out[sat][2]:.3f} s -> {out[best][2]:.3f} s per "
          f"control step ({out[best][2] / max(out[sat][2], 1e-9):.1f}x).")
    if sat <= 8:
        print("\nGATE FAILED as the spec states it: saturation at M <= 8 means "
              "hedging over a\nwide belief is not what makes this task work, so "
              "a learned stand-in for a\nlarge-M planner has little to buy.  "
              "Redesign the task before training.")
    else:
        print("\nGATE PASSED: success keeps climbing past M = 8, so a large "
              "belief is doing\nreal work and a learned stand-in for it has "
              "something to buy.")
    print("\n`s/step` is the batched dispatch, which is what this script pays; "
          "`s/ep/step` is\nit divided by the episode count.  Neither is a "
          "deployment number -- one robot\nplans for one belief, and below "
          "about 512 rollouts the cost is the rollout's\nsequential depth "
          "rather than M.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
