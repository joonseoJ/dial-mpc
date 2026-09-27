"""Is the coarse planner a correct, and a faithful, model of the world?

The world stays on the validated 24x24 grid; the planner's rollouts move to a
coarser one.  That split is only safe if two different things hold, and they
are checked separately because they fail for different reasons:

  correct   the rollout code has no hidden inconsistency with the world.  The
            planner rolled out on the *world's own* grid from the world's
            current liquid must reproduce the world step for step, bit for bit.
            The last closed-loop study ran for hours on a planner that reset the
            liquid on every call; this is the check that would have caught it
            in seconds.
  faithful  the coarse grid ranks plans the way the fine one does.  MPPI never
            uses a cost's value, only which plans are cheaper than which, so the
            criterion is rank agreement across a proposal cloud, measured from
            realistic mid-carry states rather than from rest.

Plus the numerical checks every grid needs before it is trusted: mode
frequencies, conservation, and survival under the most violent command the
sampler can produce, which on this plant is what decides the substep count.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from csm.slosh import plant as P
from csm.slosh import swe, swe_fast

G0 = 9.81


def modes(g: P.Grid):
    """Linear standing-wave frequencies on grid `g` against the analytic value."""
    out = []
    for h0 in (0.020, 0.050):
        for nx, ny in ((1, 0), (1, 1), (2, 0)):
            x = (jnp.arange(g.nx) + 0.5) * g.dx
            y = (jnp.arange(g.ny) + 0.5) * g.dy
            sh = jnp.cos(nx * jnp.pi * x / P.LX)[:, None] * jnp.cos(ny * jnp.pi * y / P.LY)[None]
            h = h0 + 0.01 * h0 * sh
            k = np.sqrt((nx * np.pi / P.LX) ** 2 + (ny * np.pi / P.LY) ** 2)
            w_an = k * np.sqrt(G0 * h0)
            dt = 0.4 * g.dx / np.sqrt(G0 * h0)
            n = int(6 * 2 * np.pi / w_an / dt)

            def body(c, _):
                c = swe_fast.step(*c, dt, g.dx, g.dy, G0, 0.0, 0.0)
                return c, c[0]
            (hf, _, _), tr = jax.lax.scan(body, (h, jnp.zeros_like(h), jnp.zeros_like(h)),
                                          None, length=n)
            a = np.tensordot(np.asarray(tr), np.asarray(sh) / float(jnp.sum(sh * sh)),
                             axes=([1, 2], [0, 1]))
            s = a - a.mean()
            cr = np.where(np.diff(np.sign(s)) != 0)[0]
            t = np.array([(i + s[i] / (s[i] - s[i + 1])) * dt for i in cr])
            w = np.pi / np.diff(t).mean()
            out.append((h0, (nx, ny), w, w_an, float(hf.sum() / h.sum() - 1)))
    return out


def adversarial(g: P.Grid, depth, seeds=32):
    """Peak CFL and NaN count under full-amplitude random-bang commands."""
    th = jnp.array([depth, 1400.0, 0.005, 0.25, 1.0, 0.10, 2.0])

    def run(u):
        s, U = P.stage_init(), P.fluid_init(th, g)

        def body(c, uu):
            s, U = c
            nU, ns, _ = P.fluid_step(U, s, uu, th, g)
            return (ns, nU), swe.wave_speed(nU, 9.81)
        (_, U), sp = jax.lax.scan(body, (s, U), u)
        return jnp.max(sp), jnp.any(~jnp.isfinite(U))

    us = jnp.stack([jnp.sign(jax.random.normal(jax.random.PRNGKey(i), (P.H, P.NU)))
                    for i in range(seeds)])
    sp, bad = jax.jit(jax.vmap(run))(us)
    sp = float(jnp.nanmax(jnp.where(jnp.isfinite(sp), sp, jnp.nan)))
    return sp * g.dt_sub / g.dx, int(jnp.sum(bad))


def midcarry_states(n, key):
    """Realistic states to test from: a moderate carry run on the world."""
    th = P.sample_theta(key, n)
    U = jax.vmap(P.fluid_init)(th)
    s = jnp.tile(P.stage_init(), (n, 1))
    cmd = jnp.array([0.6, 0.1])
    step = jax.jit(jax.vmap(lambda s, U, t: P.fluid_step(U, s, cmd, t)[:2][::-1]))
    for _ in range(8):                           # 0.4 s into the carry
        s, U = step(s, U, th)
    return s, U, th


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-n", type=int, default=16)
    ap.add_argument("--substeps", type=int, nargs="+", default=[6, 8, 10, 12, 16, 20])
    args = ap.parse_args(argv)

    # ---- 1. numerics on the planner grid -----------------------------------
    g0 = P.Grid(args.plan_n, args.plan_n, 20)
    print(f"=== 1. planner grid {g0.nx}x{g0.ny}: linear modes and conservation ===")
    for h0, m, w, wa, dm in modes(g0):
        print(f"  h0 {h0:.3f} mode {m}: omega {w:7.3f} analytic {wa:7.3f} "
              f"err {100*(w/wa-1):+5.2f}%  mass drift {dm:+.1e}")

    # ---- 2. substeps from the adversarial CFL ------------------------------
    print(f"\n=== 2. survival under full-amplitude random bang (32 seeds) ===")
    print(f"  {'substeps':>9}{'CFL shallow':>13}{'NaN':>6}{'CFL deep':>11}{'NaN':>6}")
    chosen = None
    for ss in args.substeps:
        g = P.Grid(args.plan_n, args.plan_n, ss)
        c1, b1 = adversarial(g, 0.020)
        c2, b2 = adversarial(g, 0.050)
        ok = b1 == 0 and b2 == 0 and max(c1, c2) <= 0.62
        print(f"  {ss:>9}{c1:>13.2f}{b1:>6}{c2:>11.2f}{b2:>6}   {'ok' if ok else ''}")
        if ok and chosen is None:
            chosen = ss
    if chosen is None:
        print("  no substep count passed; stopping")
        return 1
    PLAN = P.Grid(args.plan_n, args.plan_n, chosen)
    print(f"  -> planner grid {PLAN}  (world {P.WORLD})")

    # ---- 3. restriction ------------------------------------------------------
    print("\n=== 3. restriction world -> planner grid ===")
    s, U, th = midcarry_states(16, jax.random.PRNGKey(3))
    Uc = jax.vmap(lambda u: P.restrict(u, P.WORLD, PLAN))(U)
    for c, nm in ((0, "volume"), (1, "x momentum"), (2, "y momentum")):
        fw = np.asarray(U[..., c].sum((1, 2)) * P.WORLD.dx * P.WORLD.dy)
        fp = np.asarray(Uc[..., c].sum((1, 2)) * PLAN.dx * PLAN.dy)
        rel = np.max(np.abs(fp - fw)) / (np.max(np.abs(fw)) + 1e-12)
        print(f"  {nm:<11} max relative change {rel:.1e}")
    const = P.restrict(jnp.full((24, 24, 3), 0.035), P.WORLD, PLAN)
    print(f"  a constant field stays constant: max deviation {float(jnp.max(jnp.abs(const-0.035))):.1e}")

    # ---- 4. the rollout reproduces the world ---------------------------------
    print("\n=== 4. planner rollout on the WORLD grid vs the world itself ===")
    W = P.Costs()
    rows_w, _ = P.make_rollout_state(W, P.WORLD)
    Un = jnp.clip(0.4 * jax.random.normal(jax.random.PRNGKey(9), (16, P.DIM_U)), -1, 1)
    pred = jax.jit(jax.vmap(rows_w, in_axes=(0, 0, 0, 0, None)))(s, U, Un, th, P.GOAL)

    def world_exec(s, U, un, t):
        seq = P.node2u(un)

        def body(c, u):
            s, U = c
            nU, ns, lost = P.fluid_step(U, s, u, t)
            return (ns, nU), P.stage_cost(nU, ns, u, t, P.GOAL, lost, 0.0)
        (_, _), r = jax.lax.scan(body, (s, U), seq)
        return r.sum(0)
    real = jax.jit(jax.vmap(world_exec))(s, U, Un, th)
    print(f"  16 mid-carry states x their own plans: max |predicted - realised| "
          f"= {float(jnp.max(jnp.abs(pred - real))):.2e}  (must be 0)")

    # ---- 5. faithfulness of the coarse model --------------------------------
    print(f"\n=== 5. does the {PLAN.nx}x{PLAN.ny} planner rank plans like the world? ===")
    rows_p, cost_p = P.make_rollout_state(W, PLAN)
    _, cost_w = P.make_rollout_state(W, P.WORLD)
    K = 256
    rhos, rels, top = [], [], []
    for i in range(16):
        base = jnp.clip(0.3 * jax.random.normal(jax.random.PRNGKey(100 + i), (P.DIM_U,)), -1, 1)
        for sg in (0.6, 0.25):
            V = jnp.clip(base + sg * jax.random.normal(jax.random.PRNGKey(200 + i), (K, P.DIM_U)), -1, 1)
            cw = jax.jit(jax.vmap(cost_w, in_axes=(None, None, 0, None, None)))(s[i], U[i], V, th[i], P.GOAL)
            cp = jax.jit(jax.vmap(cost_p, in_axes=(None, None, 0, None, None)))(s[i], Uc[i], V, th[i], P.GOAL)
            cw, cp = np.asarray(cw), np.asarray(cp)
            rw, rp = np.argsort(np.argsort(cw)), np.argsort(np.argsort(cp))
            rhos.append(np.corrcoef(rw, rp)[0, 1])
            rels.append(np.median(np.abs(cp - cw) / (np.abs(cw) + 1e-9)))
            # of the world's best 10%, how many does the planner also put in its top 10%
            k = K // 10
            top.append(len(set(np.argsort(cw)[:k]) & set(np.argsort(cp)[:k])) / k)
    print(f"  Spearman rank correlation, world vs planner: mean {np.mean(rhos):.3f}, "
          f"worst {np.min(rhos):.3f}")
    print(f"  top-10% overlap: mean {np.mean(top):.2f}, worst {np.min(top):.2f}")
    print(f"  median relative cost error: {np.median(rels):.3f}")
    print(f"\nPLAN_GRID = Grid({PLAN.nx}, {PLAN.ny}, {PLAN.substeps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
