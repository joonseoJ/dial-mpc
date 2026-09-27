"""P3 with denoised labels: same states, averaged oracle updates.

Two single oracle labels at one state agree only at cosine 0.08-0.15, so most
of each P3 target was sampling noise.  Every stored observation carries what
the oracle needs to label it again -- carrier state, PLAN-grid liquid, theta,
the plan and the level -- so no world simulation is repeated: each state gets
`--repeats` fresh elite-MPPI clouds, averaged with the stored label.

The comparison is controlled: the student fitted here sees exactly the states
the single-label r2 student saw, from the same initialisation schedule, and
differs only in its targets.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P
from csm.slosh import closed_loop as CL
from csm.slosh import p3

S0, SF = P.DIM_STAGE, P.DIM_STAGE + P.PLAN.nx * P.PLAN.ny * 3


def relabel(D, repeats, chunk=512, seed=0):
    """(repeats, n, DIM_U) fresh elite-MPPI updates.  States are grouped by
    annealing level so every cloud runs at one sigma."""
    n = D["o"].shape[0]
    out = np.zeros((repeats,) + D["dU"].shape, np.float32)
    cloud, apply = CL.make(P.Costs(), P.PLAN, chunk, 512, P.ELITE)
    key = jax.random.PRNGKey(10_000 + seed)
    t0, done = time.time(), 0
    for li, sg in enumerate(P.SIGMA_SCHEDULE):
        ids = np.where(D["level"] == li)[0]
        for c0 in range(0, len(ids), chunk):
            sel = ids[c0:c0 + chunk]; m = len(sel)
            idx = np.concatenate([sel, np.repeat(sel[-1:], chunk - m)])
            o = D["o"][idx]
            s = jnp.asarray(o[:, :S0]); th = jnp.asarray(o[:, SF:])
            pf = jnp.asarray(o[:, S0:SF]).reshape(chunk, 1, P.PLAN.nx, P.PLAN.ny, 3)
            Uj = jnp.asarray(D["U"][idx])
            for r in range(repeats):
                key, k = jax.random.split(key)
                C, eps = cloud(s, pf, Uj, th[:, None], jnp.ones((1,)), sg, k)
                Cn = np.asarray(C)
                lv = np.array([CL.lam_for_ess(Cn[e], P.ESS_TARGET) for e in range(chunk)])
                Un, *_ = apply(s, pf, Uj, th[:, None], jnp.ones((1,)), C, eps, jnp.asarray(lv))
                out[r, sel] = np.asarray(Un - Uj)[:m]
            done += m
            if (c0 // chunk) % 20 == 0:
                el = time.time() - t0
                print(f"    {done}/{n} states, {el:.0f} s elapsed, ~{el / done * (n - done):.0f} s left",
                      flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fit-steps", type=int, default=40000)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=36)
    a = ap.parse_args(argv)

    rounds = sorted(glob.glob(f"{p3.OUT}/round[0-9].npz"))
    Ds = [dict(np.load(f)) for f in rounds]
    for f, D in zip(rounds, Ds):
        avg = f.replace(".npz", f"_avg{a.repeats + 1}.npz")
        try:
            D["dU_avg"] = np.load(avg)["dU_avg"]
            print(f"  {avg}: reused")
        except FileNotFoundError:
            print(f"  relabelling {f}: {D['o'].shape[0]} states x {a.repeats} clouds", flush=True)
            extra = relabel(D, a.repeats, seed=len(avg))
            D["dU_avg"] = (D["dU"] + extra.sum(0)) / (a.repeats + 1)
            np.savez_compressed(avg, dU_avg=D["dU_avg"])
        c = np.sum(D["dU"] * D["dU_avg"], -1) / (np.linalg.norm(D["dU"], axis=-1)
                                                  * np.linalg.norm(D["dU_avg"], axis=-1) + 1e-12)
        print(f"    single label vs {a.repeats + 1}-average: mean cosine {c.mean():.3f}; "
              f"rms ratio {np.sqrt((D['dU_avg'] ** 2).mean() / (D['dU'] ** 2).mean()):.2f}")

    th_eval = P.sample_theta(jax.random.PRNGKey(7), 16)
    r_or = CL.run("oracle", th_eval, seeds=a.eval_seeds, steps=a.steps, sigma=P.SIGMA_SCHEDULE,
                  ess_target=P.ESS_TARGET, elite=P.ELITE)
    print(f"  oracle {r_or.total.mean():.2f}", flush=True)

    # Replay the r0 -> r2 schedule of p3.main on the averaged targets:
    # norm from round 0, fit on growing unions, warm start from the last fit.
    student = None
    for rnd in range(len(Ds)):
        Dall = {k: np.concatenate([d[k] for d in Ds[:rnd + 1]]) for k in ("o", "U", "level", "dU_avg")}
        Dall["dU"] = Dall.pop("dU_avg")
        if rnd == 0:
            norm = p3.fit_norm(Dall)
        print(f"  fit r{rnd} on {Dall['o'].shape[0]} averaged labels", flush=True)
        params = p3.fit(Dall, norm, steps=a.fit_steps, init=None if student is None else student[0])
        student = (params, norm)
        p3.save(f"{p3.OUT}/student_avg{a.repeats + 1}_r{rnd}.pkl", params, norm)
        tot, sp, gp, _ = p3.rollout_student(student, th_eval, seeds=a.eval_seeds, steps=a.steps)
        d = (tot - r_or.total).ravel()
        print(f"  averaged-label student r{rnd}: cost {tot.mean():.2f}  vs oracle {d.mean():+.2f} +- "
              f"{d.std() / np.sqrt(d.size):.2f}, spilled {sp.mean():.4f}, gap {gp.mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
