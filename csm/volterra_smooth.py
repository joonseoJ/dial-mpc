"""Is `K1(U; s)` smooth enough in `s` to evaluate at index points never trained on?

That capability is the whole case for the integral formulation.  A finite basis
gives one field per row, so a new row costs a new collection; an `s`-conditioned
kernel gives a new row for the price of an evaluation -- and that is the one
thing successor features provably cannot do, because a new feature `phi_new`
needs a new `psi_new = E[sum gamma^t phi_new]`, i.e. new data.

But it only works if `K1` is genuinely a smooth function of `s`.  This measures
that **before** any network is trained, because a network cannot interpolate
what is not there.  The test is deliberately generous to the idea: it uses a
high-accuracy reference kernel, holds out index points, interpolates the rest,
and asks how wrong the result is.

Three things are separated, because conflating them would make the answer
meaningless:

  interpolation error   from a coarse `s` grid to the dense one
  estimation noise      the floor from a finite rollout batch, measured by
                        comparing two independent batches.  An interpolation
                        error below this floor is not a limitation at all.
  assembled error       what actually matters: the error in `int w K1 ds`, for
                        weights of the kinds the runtime will use (smooth, a
                        narrow peak, partly zero).  A kernel can be locally
                        rough and still assemble correctly under a smooth `w`.

The default toy cost is `(U-s)^2`, whose kernel is `-2(U-s)/lam` -- *exactly
linear in s*, so it would pass any smoothness test trivially and prove nothing.
Cost families with real `s`-structure are therefore the point, and the rod robot
is the honest case: there `s` indexes body points and the structure comes from
geometry, so the length scale is set by the obstacle margin rather than chosen.
"""
from __future__ import annotations

import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Cost families on a scalar control, with real s-structure
# ---------------------------------------------------------------------------


def toy_cost(name: str):
    """`c(V, s)` for `V` of shape (K,) and `s` of shape (M,) -> (K, M)."""
    if name == "quadratic":                       # K1 linear in s: a null test
        return lambda V, s: (V[:, None] - s[None, :]) ** 2
    if name == "hinge":                           # a kink in s, smoothed by sigma
        return lambda V, s: np.maximum(0.0, s[None, :] - V[:, None]) ** 2
    if name == "barrier":                         # localised in s around a moving centre
        def f(V, s):
            o = 0.3 + 0.4 * s                     # obstacle position depends on s
            return np.maximum(0.0, 0.15 - np.abs(V[:, None] - o[None, :])) ** 2
        return f
    if name == "narrow":                          # a deliberately sharp s-feature
        def f(V, s, ell=0.03):
            gate = np.exp(-0.5 * ((s - 0.5) / ell) ** 2)
            return gate[None, :] * (V[:, None] - 0.8) ** 2
        return f
    raise KeyError(f"unknown toy cost {name!r}; "
                   f"available: quadratic, hinge, barrier, narrow")


def toy_kernel(cost, U: float, s: np.ndarray, sigma2: float, lam: float,
               batch: int, rng: np.random.Generator) -> np.ndarray:
    """`K1[m] = -(1/lam) Sinv E[eps c(U+eps, s_m)]`, the Stein estimator."""
    eps = rng.normal(0.0, np.sqrt(sigma2), size=batch)
    c = cost(U + eps, s)
    return -(1.0 / lam) * (eps[:, None] * c).mean(0) / sigma2


# ---------------------------------------------------------------------------
# The rod robot of the document's Appendix C
# ---------------------------------------------------------------------------


class Rod:
    """A free-floating rod on a plane; `s` indexes a point along its body.

    x = (px, py, theta),  u = (vx, vy, omega),  x_{t+1} = x_t + dt u_t
    q(s; x) = (px + (s - 1/2) L cos theta,  py + (s - 1/2) L sin theta)
    c(U, s) = sum_t max(0, delta - d(q(s; x_t)))^2,  d = min_j (|q - o_j| - r_j)

    Nothing here is a physics engine: the dynamics are the kinematic integrator
    the document specifies, so the whole family runs in numpy and the only
    `s`-structure is the geometric one.
    """

    def __init__(self, L=0.6, dt=0.1, H=20, delta=0.1,
                 obstacles=((0.55, 0.10, 0.18), (0.95, -0.28, 0.14),
                            (1.35, 0.22, 0.16))):
        self.L, self.dt, self.H, self.delta = L, dt, H, delta
        self.obs = np.asarray(obstacles, float)        # (J, 3): ox, oy, r
        self.dim = 3 * H

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """`U` of shape (K, 3H) -> states of shape (K, H, 3)."""
        u = U.reshape(U.shape[0], self.H, 3)
        return x0[None, None, :] + np.cumsum(u * self.dt, axis=1)

    def cost(self, x0: np.ndarray, U: np.ndarray, s: np.ndarray) -> np.ndarray:
        """-> (K, M).  Clearance violation of body point `s`, summed over the horizon."""
        X = self.rollout(x0, U)                                   # (K, H, 3)
        off = (s - 0.5) * self.L                                  # (M,)
        cos, sin = np.cos(X[..., 2]), np.sin(X[..., 2])           # (K, H)
        qx = X[..., 0][..., None] + off[None, None, :] * cos[..., None]
        qy = X[..., 1][..., None] + off[None, None, :] * sin[..., None]
        # signed distance to the nearest disc
        dx = qx[..., None] - self.obs[:, 0]
        dy = qy[..., None] - self.obs[:, 1]
        d = np.sqrt(dx ** 2 + dy ** 2) - self.obs[:, 2]           # (K, H, M, J)
        d = d.min(axis=-1)
        return (np.maximum(0.0, self.delta - d) ** 2).sum(axis=1)  # (K, M)

    def kernel(self, x0, U, s, sigma2, lam, batch, rng) -> np.ndarray:
        """`K1[m, :]` of shape (M, 3H) by the Stein identity."""
        eps = rng.normal(0.0, np.sqrt(sigma2), size=(batch, self.dim))
        c = self.cost(x0, U[None, :] + eps, s)                     # (K, M)
        return -(1.0 / lam) * np.einsum("km,kd->md", c, eps) / batch / sigma2


# ---------------------------------------------------------------------------
# Weights the runtime would actually ask for
# ---------------------------------------------------------------------------


def weights(s: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "uniform": np.ones_like(s),
        "smooth": 1.0 + np.sin(2.0 * np.pi * s),
        "protect the tip": 1.0 + 5.0 * np.exp(-50.0 * (s - 1.0) ** 2),
        "narrow peak": np.exp(-0.5 * ((s - 0.37) / 0.02) ** 2),
        "partly zero": (s > 0.6).astype(float),
    }


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def interp_to(dense_s: np.ndarray, coarse_s: np.ndarray, coarse_K: np.ndarray,
              kind: str = "cubic") -> np.ndarray:
    """Interpolate a kernel sampled on `coarse_s` onto `dense_s`."""
    K = np.atleast_2d(coarse_K.T)                         # (dim, M_coarse)
    out = np.empty((K.shape[0], dense_s.size))
    for i in range(K.shape[0]):
        if kind == "linear" or coarse_s.size < 4:
            out[i] = np.interp(dense_s, coarse_s, K[i])
        else:
            from numpy.polynomial import polynomial as _p  # noqa: F401
            # natural cubic through the coarse samples, via a spline-free
            # piecewise cubic Hermite (monotone-safe, no scipy dependency)
            out[i] = _pchip(coarse_s, K[i], dense_s)
    return out.T if coarse_K.ndim > 1 else out[0]


def _pchip(x, y, xi):
    """Piecewise cubic Hermite with one-sided finite-difference slopes."""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.empty_like(y)
    m[1:-1] = 0.5 * (d[:-1] + d[1:])
    m[0], m[-1] = d[0], d[-1]
    j = np.clip(np.searchsorted(x, xi) - 1, 0, len(h) - 1)
    t = (xi - x[j]) / h[j]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * y[j] + h10 * h[j] * m[j] + h01 * y[j + 1] + h11 * h[j] * m[j + 1]


def lengthscale(s: np.ndarray, K: np.ndarray) -> float:
    """A length scale for `K(s)`, from the energy-weighted mean frequency.

    `ell = 1 / (2 f_rms)` is the Nyquist spacing for the kernel's own structure.
    **Measured, it is far too optimistic as a grid criterion**: the `narrow`
    toy cost reports `ell = 0.126` and yet a grid of spacing 0.125 reconstructs
    it with 63% error, because one sample inside a feature fixes its height and
    not its width.  Treat `ell` as a scale and require a spacing of roughly
    `ell/4` to `ell/8` -- for that cost, error falls below the noise floor only
    at spacing 0.016, i.e. `ell/8`.
    """
    K = np.atleast_2d(K.T)
    K = K - K.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(K, axis=1)) ** 2
    f = np.fft.rfftfreq(s.size, d=float(s[1] - s[0]))
    tot = P.sum()
    if tot <= 0:
        return float("inf")
    f_rms = float(np.sqrt((P * f[None, :] ** 2).sum() / tot))
    return float("inf") if f_rms <= 0 else 1.0 / (2.0 * f_rms)


def report(name, dense_s, K_ref, K_ref2, grids, ws, beta_dense):
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
    floor = rel(K_ref2, K_ref)
    ell = lengthscale(dense_s, K_ref)
    print(f"\n--- {name} ---")
    print(f"  kernel length scale in s: {ell:.4f}   "
          f"(Nyquist; measured, a grid needs ~ell/4 to ell/8, not ell)")
    print(f"  estimation noise floor, two independent batches: {100 * floor:.2f}%")
    print(f"  {'M train':>8}{'spacing':>9}{'K1 interp':>11}" +
          "".join(f"{k:>16}" for k in ws))
    for M in grids:
        idx = np.linspace(0, dense_s.size - 1, M).astype(int)
        cs, ck = dense_s[idx], K_ref[idx]
        Ki = interp_to(dense_s, cs, ck)
        row = f"  {M:8d}{float(dense_s[1] - dense_s[0]) * (dense_s.size - 1) / (M - 1):9.4f}" \
              f"{100 * rel(Ki, K_ref):10.2f}%"
        for k, w in ws.items():
            g_ref = np.tensordot(beta_dense * w, K_ref, axes=(0, 0))
            g_int = np.tensordot(beta_dense * w, Ki, axes=(0, 0))
            row += f"{100 * rel(g_int, g_ref):15.2f}%"
        print(row)
    print(f"  (an interpolation error below the {100 * floor:.2f}% noise floor is "
          f"not a limitation:\n   the network could not see the difference anyway.)")


def cmd_toy(a) -> None:
    dense_s = np.linspace(0.0, 1.0, a.dense)
    beta = np.full(a.dense, 1.0 / a.dense)
    ws = weights(dense_s)
    print(f"Toy: scalar control U = {a.u}, sigma^2 = {a.sigma2}, lam = {a.lam}, "
          f"batch {a.batch}")
    for name in a.costs:
        cost = toy_cost(name)
        K1 = toy_kernel(cost, a.u, dense_s, a.sigma2, a.lam, a.batch,
                        np.random.default_rng(0))
        K2 = toy_kernel(cost, a.u, dense_s, a.sigma2, a.lam, a.batch,
                        np.random.default_rng(1))
        report(f"toy cost: {name}", dense_s, K1, K2, a.grids, ws, beta)


def cmd_rod(a) -> None:
    rod = Rod()
    dense_s = np.linspace(0.0, 1.0, a.dense)
    beta = np.full(a.dense, 1.0 / a.dense)
    ws = weights(dense_s)
    rng = np.random.default_rng(0)
    x0 = np.array([0.0, 0.0, 0.25])
    # a plan that actually approaches the obstacles, so the cost is not zero
    U = np.tile(np.array([0.9, 0.05, 0.02]), rod.H) + 0.05 * rng.normal(size=rod.dim)
    X = rod.rollout(x0, U[None, :])[0]
    c0 = rod.cost(x0, U[None, :], dense_s)[0]
    print(f"Rod robot: L = {rod.L}, dt = {rod.dt}, H = {rod.H}, "
          f"U in R^{rod.dim}, margin {rod.delta}")
    print(f"  {len(rod.obs)} discs at "
          f"{[tuple(np.round(o, 2)) for o in rod.obs]}")
    print(f"  nominal plan ends at ({X[-1, 0]:.2f}, {X[-1, 1]:.2f}, "
          f"{X[-1, 2]:.2f} rad)")
    print(f"  cost profile over s: min {c0.min():.4f}, max {c0.max():.4f}, "
          f"nonzero on {100 * (c0 > 1e-9).mean():.0f}% of the body")
    if c0.max() < 1e-9:
        raise SystemExit("the nominal plan never violates the margin, so every "
                         "kernel is zero and the test is vacuous -- move the "
                         "obstacles or the plan closer.")
    K1 = rod.kernel(x0, U, dense_s, a.sigma2, a.lam, a.batch,
                    np.random.default_rng(10))
    K2 = rod.kernel(x0, U, dense_s, a.sigma2, a.lam, a.batch,
                    np.random.default_rng(11))
    report("rod robot, body-point kernel", dense_s, K1, K2, a.grids, ws, beta)


# ---------------------------------------------------------------------------
# Does the sum-to-one normalisation still help when the cost is not quadratic?
# ---------------------------------------------------------------------------
#
# The exactness measured earlier (10.3% -> 0.000%) relies on the closed form
#
#     G[w](U) = -2 (W0 U - W1) / (lam + 2 sigma^2 W0)
#
# which exists only because `c(U, s)` is quadratic in the *control*: the exponent
# of `exp(-C(U+eps)/lam)` is then quadratic in `eps`, the Gaussian integral
# closes, and the result is linear-in-w over linear-in-w.  Fixing the total mass
# `W0` freezes the denominator and the composition becomes exact.
#
# When `c` is not quadratic in `U` there is no such form.  Expanding instead,
#
#     log J_w = -kappa1/lam + kappa2/(2 lam^2) - ...
#     kappa1 = int w E[c]                      linear in w
#     kappa2 = int int w w' Cov(c_s, c_s')     quadratic in w, and it depends on
#                                              the *shape* of w, not only on W0
#
# so fixing the mass removes the dominant non-linearity but not all of it.  That
# says the exactness will not survive; it does *not* say whether sum-to-one is
# still the better parameterisation.  This measures that.
#
# The true score is computed without any closed form, as
#
#     G[w](U) = E[eps exp(-C_w(U+eps)/lam)] / (sigma^2 E[exp(-C_w(U+eps)/lam)])
#
# and every weight -- the target and all basis rows -- is evaluated on the *same*
# noise batch, so the comparison is paired and the estimator noise largely
# cancels instead of being mistaken for a composition error.


def _true_and_combo_1d(cost, U, s, beta, sigma2, lam, phis_w, eps):
    """True score for arbitrary `w`, and each basis row's true score.

    `phis_w` is (k, M): the basis weight functions sampled on the `s` grid.
    Returns a closure for the target and an array of basis scores, both from the
    shared `eps`.
    """
    c = cost(U + eps, s)                                  # (K, M)

    def true_score(w_on_s):
        C = c @ (beta * w_on_s)                           # (K,)
        z = -C / lam
        z -= z.max()
        e = np.exp(z)
        return float((eps * e).sum() / e.sum() / sigma2)

    basis = np.array([true_score(p) for p in phis_w])
    return true_score, basis


def _rod_scores(rod, x0, U, s, beta, sigma2, lam, phis_w, eps, chunk=4096):
    """Same, for the rod: `U` is 60-dimensional and the cost needs chunking."""
    K = eps.shape[0]
    c = np.empty((K, s.size))
    for a in range(0, K, chunk):
        b = min(a + chunk, K)
        c[a:b] = rod.cost(x0, U[None, :] + eps[a:b], s)

    def true_score(w_on_s):
        C = c @ (beta * w_on_s)
        z = -C / lam
        z -= z.max()
        e = np.exp(z)
        return (eps * e[:, None]).sum(0) / e.sum() / sigma2     # (60,)

    basis = np.stack([true_score(p) for p in phis_w])
    return true_score, basis


def cmd_normcheck(a) -> None:
    k = a.k
    dense = a.dense
    s = np.linspace(0.0, 1.0, dense)
    beta = np.full(dense, 1.0 / dense)
    # equal-mass basis rows: disjoint indicators of height k, so each integrates to 1
    edges = np.linspace(0.0, 1.0, k + 1)
    phis_w = np.stack([np.where((s >= edges[i]) & (s <= edges[i + 1]), float(k), 0.0)
                       for i in range(k)])
    masses = (beta * phis_w).sum(1)
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(k), size=a.trials)

    print(f"{k} equal-mass basis rows (each int phi = {masses[0]:.3f}), "
          f"{a.trials} random targets")
    print(f"sigma^2 = {a.sigma2}, lam = {a.lam}\n")
    print(f"  {'cost c(U,s)':<22}{'quadratic in U?':>16}{'x range':>16}"
          f"{'L2 mean':>10}{'L2 max':>9}{'L1 mean':>10}{'L1 max':>9}")

    def run(name, is_quad, true_fn_factory):
        out = {}
        for tag, norm in (("L2", lambda o: o / np.linalg.norm(o)),
                          ("L1", lambda o: o / o.sum())):
            errs, xs = [], []
            true_score, basis = true_fn_factory()
            for i in range(a.trials):
                om = norm(raw[i])
                w = om @ phis_w
                W0 = float(om @ masses)
                tr = true_score(w)
                cb = np.tensordot(om, basis, axes=(0, 0)) if basis.ndim > 1 \
                    else float(om @ basis)
                nrm = np.linalg.norm(tr) if np.ndim(tr) else abs(tr)
                if nrm < 1e-12:
                    continue
                errs.append(float(np.linalg.norm(np.subtract(cb, tr)) / nrm))
                xs.append(2.0 * a.sigma2 * W0 / a.lam)
            out[tag] = (np.mean(errs), np.max(errs), min(xs), max(xs))
        print(f"  {name:<22}{'yes' if is_quad else 'no':>16}"
              f"{out['L2'][2]:6.2f}..{out['L2'][3]:<9.2f}"
              f"{100*out['L2'][0]:9.2f}%{100*out['L2'][1]:8.2f}%"
              f"{100*out['L1'][0]:9.2f}%{100*out['L1'][1]:8.2f}%")

    eps1 = rng.normal(0.0, np.sqrt(a.sigma2), size=a.batch1d)
    for name, is_quad in (("(U-s)^2", True), ("hinge max(0,s-U)^2", False),
                          ("(U-s)^4", False), ("barrier", False),
                          ("1-cos(U-s)", False)):
        cost = {"(U-s)^2": toy_cost("quadratic"),
                "hinge max(0,s-U)^2": toy_cost("hinge"),
                "(U-s)^4": (lambda V, sv: (V[:, None] - sv[None, :]) ** 4),
                "barrier": toy_cost("barrier"),
                "1-cos(U-s)": (lambda V, sv: 1.0 - np.cos(V[:, None] - sv[None, :]))}[name]
        run(name, is_quad,
            lambda cost=cost: _true_and_combo_1d(cost, a.u, s, beta, a.sigma2,
                                                 a.lam, phis_w, eps1))

    if not a.skip_rod:
        rod = Rod()
        x0 = np.array([0.0, 0.0, 0.25])
        rr = np.random.default_rng(1)
        U = np.tile(np.array([0.9, 0.05, 0.02]), rod.H) + 0.05 * rr.normal(size=rod.dim)
        epsr = rr.normal(0.0, np.sqrt(a.sigma2), size=(a.batchrod, rod.dim))
        s_r = np.linspace(0.0, 1.0, a.rod_quad)
        beta_r = np.full(a.rod_quad, 1.0 / a.rod_quad)
        phis_r = np.stack([np.where((s_r >= edges[i]) & (s_r <= edges[i + 1]),
                                    float(k), 0.0) for i in range(k)])
        masses_r = (beta_r * phis_r).sum(1)
        def factory():
            return _rod_scores(rod, x0, U, s_r, beta_r, a.sigma2, a.lam,
                               phis_r, epsr)
        out = {}
        for tag, norm in (("L2", lambda o: o / np.linalg.norm(o)),
                          ("L1", lambda o: o / o.sum())):
            true_score, basis = factory()
            errs, xs = [], []
            for i in range(a.trials_rod):
                om = norm(raw[i])
                tr = true_score(om @ phis_r)
                cb = np.tensordot(om, basis, axes=(0, 0))
                nrm = np.linalg.norm(tr)
                if nrm < 1e-12:
                    continue
                errs.append(float(np.linalg.norm(cb - tr) / nrm))
                xs.append(2.0 * a.sigma2 * float(om @ masses_r) / a.lam)
            out[tag] = (np.mean(errs), np.max(errs), min(xs), max(xs))
        print(f"  {'rod: SDF hinge^2':<22}{'no':>16}"
              f"{out['L2'][2]:6.2f}..{out['L2'][3]:<9.2f}"
              f"{100*out['L2'][0]:9.2f}%{100*out['L2'][1]:8.2f}%"
              f"{100*out['L1'][0]:9.2f}%{100*out['L1'][1]:8.2f}%")

    print("\nL2 is the project's current convention; L1 is sum-to-one.  The")
    print("exactness only ever applied to the quadratic row; what matters for")
    print("adoption is whether L1 still wins where the cost is not quadratic.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("toy", "rod", "normcheck", "all"))
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--trials-rod", type=int, default=60)
    p.add_argument("--batch1d", type=int, default=200000)
    p.add_argument("--batchrod", type=int, default=30000)
    p.add_argument("--rod-quad", type=int, default=65)
    p.add_argument("--skip-rod", action="store_true")
    p.add_argument("--u", type=float, default=0.35)
    p.add_argument("--sigma2", type=float, default=0.02)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=32768)
    p.add_argument("--dense", type=int, default=257)
    p.add_argument("--grids", type=int, nargs="+", default=[5, 9, 17, 33, 65])
    p.add_argument("--costs", nargs="+",
                   default=["quadratic", "hinge", "barrier", "narrow"])
    a = p.parse_args(argv)
    order = ("toy", "rod", "normcheck") if a.command == "all" else (a.command,)
    for name in order:
        print(f"\n{'=' * 78}\n### {name}\n{'=' * 78}")
        {"toy": cmd_toy, "rod": cmd_rod, "normcheck": cmd_normcheck}[name](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
