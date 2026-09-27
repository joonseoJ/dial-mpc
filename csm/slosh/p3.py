"""P3: a learned, theta-conditioned update that stands in for oracle MPPI.

The oracle planner (true theta, the world's liquid restricted to the PLAN grid,
2-level annealing, elite MPPI at ESS 25%) costs K = 512 liquid rollouts per
level per control step.  P3 asks whether one network call can replace them:

    f(o, U, level) ~ Delta U = U_mppi(o, U, level) - U
    o = (carrier state s, liquid field on PLAN, theta)

Deployment runs the same two levels, each `U <- clip(U + f(o, U, level))`,
then shifts the plan exactly as MPPI does.  No rollouts at all.

  collect   run a driver in closed loop on the world and, at every step and
            level, label the plan the driver is holding with the oracle's own
            MPPI update from that plan.  Driver = oracle for round 0, the
            student for DAgger rounds (labels are always the oracle's).
  fit       MSE on Delta U, inputs standardised on round-0 data only.
  eval      paired closed-loop cost against oracle MPPI on the same thetas,
            episodes the student never saw, plus wall time per control step.

Labels are single-cloud elite-MPPI updates, so they carry sampling noise; the
regression recovers their mean, which is what the planner does in expectation.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from csm.slosh import plant as P
from csm.slosh import closed_loop as CL

OUT = "csm_runs/slosh_p3"
N_LEVEL = len(P.SIGMA_SCHEDULE)
# The update is local: a plan is only as good as the state it was made for,
# and the network sees both.  Liquid on the PLAN grid, all three channels.
DIM_OBS = P.DIM_STAGE + P.PLAN.nx * P.PLAN.ny * 3 + P.DIM_THETA


def obs_of(states, fluids_plan, th):
    return jnp.concatenate([states, fluids_plan.reshape(states.shape[0], -1), th], -1)


# ---- network: plain pytrees, saved as numpy, so a fit never depends on classes
def mlp_init(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)
        if i == len(sizes) - 2:
            w = w * 0.01
        ps.append({"w": w, "b": jnp.zeros(b)})
    return ps


def mlp_apply(ps, x):
    for l in ps[:-1]:
        x = jax.nn.gelu(x @ l["w"] + l["b"])
    return x @ ps[-1]["w"] + ps[-1]["b"]


def features(norm, o, U, level):
    lv = jax.nn.one_hot(level, N_LEVEL)
    return jnp.concatenate([(o - norm["mo"]) / norm["so"], U, lv], -1)


def student_update(params, norm, o, U, level):
    d = mlp_apply(params, features(norm, o, U, level)) * norm["sd"]
    return jnp.clip(U + d, -1.0, 1.0)


# ---- collection -------------------------------------------------------------
def collect(th_true, *, seeds, steps, K=512, student=None, w=None, log=True):
    """Closed loop on the world; returns (dataset, per-episode weighted cost).

    `student` = (params, norm) drives the world; otherwise the oracle does.
    Either way every stored label is the oracle's elite-MPPI update from the
    plan the driver actually held.
    """
    w = P.Costs() if w is None else w
    E = th_true.shape[0]
    cloud, apply = CL.make(w, P.PLAN, E, K, P.ELITE)
    th_b, bel = th_true[:, None], jnp.ones((1,))
    wv = np.asarray(P.cost_weights(w))
    stu = None
    if student is not None:
        params, norm = student
        stu = jax.jit(jax.vmap(lambda o, U, l: student_update(params, norm, o, U, l),
                               in_axes=(0, 0, None)))
    D = {k: [] for k in ("o", "U", "level", "dU", "lam")}
    totals = []
    for sd in range(seeds):
        states = jnp.tile(P.stage_init(), (E, 1))
        fluids = jax.vmap(lambda t: P.fluid_init(t, P.WORLD))(th_true)
        Us = jnp.zeros((E, P.DIM_U)); prof = jnp.zeros((E, len(P.ROW_NAMES)))
        key = jax.random.PRNGKey(1000 + sd)
        t0 = time.time()
        for t in range(steps):
            pf1 = CL._restrict(fluids)
            pf = pf1[:, None]
            o = obs_of(states, pf1, th_true)
            for li, sg in enumerate(P.SIGMA_SCHEDULE):
                key, k = jax.random.split(key)
                C, eps = cloud(states, pf, Us, th_b, bel, sg, k)
                Cn = np.asarray(C)
                lv = np.array([CL.lam_for_ess(Cn[e], P.ESS_TARGET) for e in range(E)])
                Uo, *_ = apply(states, pf, Us, th_b, bel, C, eps, jnp.asarray(lv))
                D["o"].append(np.asarray(o)); D["U"].append(np.asarray(Us))
                D["level"].append(np.full(E, li, np.int32))
                D["dU"].append(np.asarray(Uo - Us)); D["lam"].append(lv)
                Us = Uo if stu is None else stu(o, Us, li)
            states, fluids, rows = CL.world_step(states, fluids, Us[:, :P.NU], th_true)
            prof = prof + rows
            Us = jax.vmap(P.shift_nodes)(Us)
        totals.append((np.array(prof) * wv).sum(-1))
        if log:
            print(f"    seed {sd}: {E} episodes x {steps} steps in {time.time() - t0:.0f} s, "
                  f"cost {totals[-1].mean():.2f}", flush=True)
    D = {k: np.concatenate(v) for k, v in D.items()}
    return D, np.stack(totals)


def rollout_student(student, th_true, *, seeds, steps, w=None):
    """Closed loop with the student alone -- no rollouts.  Returns per-episode
    costs, spill fraction, final gap and wall time per control step."""
    w = P.Costs() if w is None else w
    params, norm = student
    E = th_true.shape[0]
    wv = np.asarray(P.cost_weights(w))

    @jax.jit
    def control(states, fluids, Us):
        pf1 = CL._restrict(fluids)
        o = obs_of(states, pf1, th_true)
        for li in range(N_LEVEL):
            Us = jax.vmap(lambda o_, U_: student_update(params, norm, o_, U_, li))(o, Us)
        return Us

    totals, spill, gap, dt = [], [], [], []
    for sd in range(seeds):
        states = jnp.tile(P.stage_init(), (E, 1))
        fluids = jax.vmap(lambda t: P.fluid_init(t, P.WORLD))(th_true)
        Us = jnp.zeros((E, P.DIM_U)); prof = jnp.zeros((E, len(P.ROW_NAMES)))
        for t in range(steps):
            t0 = time.perf_counter()
            Us = control(states, fluids, Us).block_until_ready()
            dt.append(time.perf_counter() - t0)
            states, fluids, rows = CL.world_step(states, fluids, Us[:, :P.NU], th_true)
            prof = prof + rows
            Us = jax.vmap(P.shift_nodes)(Us)
        prof = np.array(prof)
        totals.append((prof * wv).sum(-1)); spill.append(prof[:, 1])
        gap.append(np.asarray(jnp.linalg.norm(states[:, :2] - P.GOAL, axis=-1)))
    return np.stack(totals), np.stack(spill), np.stack(gap), float(np.median(dt[min(5, len(dt) - 1):]))


def time_oracle_step(th, K=512, reps=10):
    """Wall time of one oracle control step for a single tray (E = 1): both
    levels, the temperature solve and the elite update, after compilation."""
    cloud, apply = CL.make(P.Costs(), P.PLAN, 1, K, P.ELITE)
    th_true = th[:1]
    th_b, bel = th_true[:, None], jnp.ones((1,))
    states = jnp.tile(P.stage_init(), (1, 1))
    pf = CL._restrict(jax.vmap(lambda t: P.fluid_init(t, P.WORLD))(th_true))[:, None]
    key = jax.random.PRNGKey(0)
    out = []
    for r in range(reps + 2):
        t0 = time.perf_counter()
        Us = jnp.zeros((1, P.DIM_U))
        for sg in P.SIGMA_SCHEDULE:
            key, k = jax.random.split(key)
            C, eps = cloud(states, pf, Us, th_b, bel, sg, k)
            lv = np.array([CL.lam_for_ess(np.asarray(C)[0], P.ESS_TARGET)])
            Us, *_ = apply(states, pf, Us, th_b, bel, C, eps, jnp.asarray(lv))
        Us.block_until_ready()
        out.append(time.perf_counter() - t0)
    return float(np.median(out[2:]))


# ---- fitting ----------------------------------------------------------------
def fit_norm(D):
    so = D["o"].std(0) + 1e-6
    return {"mo": jnp.asarray(D["o"].mean(0)), "so": jnp.asarray(so),
            "sd": jnp.asarray(D["dU"].std(0) + 1e-6)}


def fit(D, norm, *, steps=40000, batch=1024, lr=1e-3, width=512, depth=3, seed=0,
        init=None, val_frac=0.1, log_every=5000):
    n = D["o"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); nv = int(n * val_frac)
    iv, it = perm[:nv], perm[nv:]
    X = {k: jnp.asarray(v) for k, v in D.items() if k in ("o", "U", "level", "dU")}
    din = DIM_OBS + P.DIM_U + N_LEVEL
    params = init if init is not None else mlp_init(jax.random.PRNGKey(seed),
                                                    [din] + [width] * depth + [P.DIM_U])
    sched = optax.cosine_decay_schedule(lr, steps, 0.05)
    opt = optax.adamw(sched, weight_decay=1e-5)
    st = opt.init(params)

    def loss_fn(ps, idx):
        o, U, l, y = X["o"][idx], X["U"][idx], X["level"][idx], X["dU"][idx]
        pred = mlp_apply(ps, features(norm, o, U, l))
        return jnp.mean((pred - y / norm["sd"]) ** 2)

    @jax.jit
    def train_step(ps, st, idx):
        l, g = jax.value_and_grad(loss_fn)(ps, idx)
        up, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, up), st, l

    @jax.jit
    def metrics(ps, idx):
        o, U, l, y = X["o"][idx], X["U"][idx], X["level"][idx], X["dU"][idx]
        pred = mlp_apply(ps, features(norm, o, U, l)) * norm["sd"]
        out = []
        for li in range(N_LEVEL):
            m = (l == li)[:, None]
            num = jnp.sum(m * (pred - y) ** 2); den = jnp.sum(m * y ** 2)
            cos = jnp.sum(m[:, 0] * jnp.sum(pred * y, -1) /
                          (jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(y, axis=-1) + 1e-9)) \
                / jnp.sum(m)
            out.append(jnp.stack([jnp.sqrt(num / den), cos]))
        return jnp.stack(out)

    ivj = jnp.asarray(iv)
    best, best_ps = np.inf, params
    for i in range(steps):
        idx = jnp.asarray(it[rng.integers(0, len(it), batch)])
        params, st, l = train_step(params, st, idx)
        if (i + 1) % log_every == 0 or i == steps - 1:
            m = np.asarray(metrics(params, ivj))
            score = m[:, 0].mean()
            if score < best:
                best, best_ps = score, params
            print(f"    step {i + 1}: train {float(l):.4f}  val rel-rms / cos per level " +
                  "  ".join(f"L{li} {m[li, 0]:.3f}/{m[li, 1]:.3f}" for li in range(N_LEVEL)),
                  flush=True)
    return best_ps


def save(path, params, norm):
    with open(path, "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, params),
                     "norm": jax.tree.map(np.asarray, norm)}, f)


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return jax.tree.map(jnp.asarray, d["params"]), jax.tree.map(jnp.asarray, d["norm"])


# ---- driver -----------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=256)
    ap.add_argument("--train-seeds", type=int, default=2)
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--dagger", type=int, default=2)
    ap.add_argument("--fit-steps", type=int, default=40000)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)

    th_eval = P.sample_theta(jax.random.PRNGKey(7), a.eval_episodes)     # the gate's thetas
    print(f"=== oracle MPPI on the {a.eval_episodes} evaluation thetas (the reference) ===")
    r_or = CL.run("oracle", th_eval, seeds=a.eval_seeds, steps=a.steps, sigma=P.SIGMA_SCHEDULE,
                  ess_target=P.ESS_TARGET, elite=P.ELITE)
    print(f"  oracle cost {r_or.total.mean():.2f}, spilled {r_or.spilled.mean():.4f}, "
          f"gap {r_or.gap.mean():.3f}", flush=True)

    t_or = time_oracle_step(th_eval)
    print(f"  one tray, one oracle control step: {1e3 * t_or:.1f} ms", flush=True)
    Ds = []
    student = None
    for rnd in range(a.dagger + 1):
        th_tr = P.sample_theta(jax.random.PRNGKey(100 + rnd), a.train_episodes)
        who = "oracle" if student is None else "student"
        print(f"\n=== round {rnd}: collect, {who} driving, {a.train_episodes} thetas x "
              f"{a.train_seeds} seeds ===", flush=True)
        D, tot = collect(th_tr, seeds=a.train_seeds, steps=a.steps, student=student)
        np.savez_compressed(f"{OUT}/round{rnd}.npz", **D)
        Ds.append(D)
        print(f"  {D['o'].shape[0]} labels; driver cost {tot.mean():.2f}")
        if rnd == 0:
            norm = fit_norm(D)
        Dall = {k: np.concatenate([d[k] for d in Ds]) for k in Ds[0]}
        print(f"  fit on {Dall['o'].shape[0]} labels", flush=True)
        params = fit(Dall, norm, steps=a.fit_steps,
                     init=None if student is None else student[0])
        student = (params, norm)
        save(f"{OUT}/student_r{rnd}.pkl", params, norm)

        tot, sp, gp, _ = rollout_student(student, th_eval, seeds=a.eval_seeds, steps=a.steps)
        _, _, _, dt = rollout_student(student, th_eval[:1], seeds=1, steps=12)
        d = (tot - r_or.total).ravel()
        print(f"  student r{rnd}: cost {tot.mean():.2f}  vs oracle {d.mean():+.2f} +- "
              f"{d.std() / np.sqrt(d.size):.2f}  (median {np.median(d):+.2f}), spilled "
              f"{sp.mean():.4f}, gap {gp.mean():.3f};  one tray {1e3 * dt:.2f} ms per control "
              f"step ({t_or / dt:.0f}x faster than the oracle)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
