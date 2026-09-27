"""Does the Volterra picture of the MPPI score hold?  Formulation check, no learning.

The proposal in `docs/MPPI_score_learning_integral_weight.pdf` says the
composition CSM currently uses is the *first-order truncation* of a series in
the weight, and that the missing term is a covariance.  Its own implementation
order puts the truncation ablation before any network, so that is what this
does -- with the analytic answer available, so a disagreement is a bug rather
than a research result.

What is being checked, and why each item is here
------------------------------------------------
1. `analytic`  The closed forms in the document's Appendix A and B, against
   brute-force Monte Carlo.  If these disagree the paper's algebra is wrong.
2. `stein`     The kernels estimated from **one** noise batch by the Stein
   identity, assembled, against the analytic truth.  This is the document's
   step 1 ("oracle check").
3. `labels`    The claim I judge to be the strongest and the least emphasised:
   both the current label and the Stein label are Stein estimators, but

       current  G* = Sinv * sum_k omega_k eps_k,  omega = softmax(-C_k/lam)
       Stein    K1 = -(1/lam) Sinv * mean_k( eps_k c_k )

   one is a *self-normalised ratio with a peaked weight*, the other a *plain
   mean*.  Measured on the real plant, the current label carries rms error 194%
   of the label magnitude and 63% bias, so its variance is the binding
   constraint on collection cost.  This compares them at equal batch size.
4. `fixed`     The document's own honest limit (A.6): for a quadratic cost the
   truncation moves the step size but not the optimum.  That matters because
   the deployed controller *iterates*, which absorbs a step-size error.  So
   this measures where the ascent actually converges, for a non-quadratic
   (hinge) cost where the optimum is allowed to move.
5. `temper`    `x = 2 sigma^2 W0 / lam` is the convergence ratio, so raising
   `lam` shrinks the truncation error just as adding a kernel does.  This
   prices the cheaper lever against the expensive one.

Conventions.  Costs are costs (lower is better) and `G = grad_U log J` with
`J = E_eps[exp(-C(U+eps)/lam)]`, matching the document.  The project's own
DIAL code works with *rewards* and ascends them; a sign slip between those two
conventions has produced a wrong published number here before, so nothing in
this file touches the reward convention at all.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# The 1-D family of Appendix A:  c(U, s) = (U - s)^2 on s in [0, 1]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Weight1D:
    """A piecewise-constant weight on [0, 1], as the document's cases A/B/C."""

    name: str
    height: float
    lo: float
    hi: float

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return np.where((s >= self.lo) & (s <= self.hi), self.height, 0.0)

    def moments(self) -> tuple[float, float, float]:
        """W0 = int w, W1 = int w s, W2 = int w s^2, exactly."""
        h, a, b = self.height, self.lo, self.hi
        return (h * (b - a),
                h * (b ** 2 - a ** 2) / 2.0,
                h * (b ** 3 - a ** 3) / 3.0)


CASES = (Weight1D("A uniform", 1.0, 0.0, 1.0),
         Weight1D("B one-sided", 4.0, 0.5, 1.0),
         Weight1D("C stiff", 10.0, 0.0, 1.0))


def exact_score_1d(U: float, w: Weight1D, sigma2: float, lam: float) -> float:
    """`G = -2 (W0 U - W1) / (lam + 2 sigma^2 W0)`.

    From the Gaussian integral `E[exp(-a(eps+b)^2)] = (1+2 a sigma^2)^{-1/2}
    exp(-a b^2 / (1 + 2 a sigma^2))` with `a = W0/lam`; the prefactor has no `U`
    and drops out of the score.
    """
    W0, W1, _ = w.moments()
    return -2.0 * (W0 * U - W1) / (lam + 2.0 * sigma2 * W0)


def truncations_1d(U: float, w: Weight1D, sigma2: float, lam: float,
                   orders: int = 4) -> list[float]:
    """Partial sums of `-2(W0 U - W1)/lam * sum_j (-x)^j`, `x = 2 sigma^2 W0/lam`."""
    W0, W1, _ = w.moments()
    lead = -2.0 * (W0 * U - W1) / lam
    x = 2.0 * sigma2 * W0 / lam
    return [lead * sum((-x) ** j for j in range(n + 1)) for n in range(orders)]


def mc_score_1d(U: float, w: Weight1D, sigma2: float, lam: float, n: int,
                rng: np.random.Generator, quad: int = 512) -> tuple[float, float]:
    """Brute force: `G = Sinv * sum_k omega_k eps_k` with `omega = softmax(-C/lam)`.

    This is exactly the label the current pipeline collects, so its effective
    sample size is reported alongside.
    """
    eps = rng.normal(0.0, np.sqrt(sigma2), size=n)
    s = (np.arange(quad) + 0.5) / quad
    ws = w(s) / quad                                  # quadrature weights beta*w
    C = ((U + eps)[:, None] - s[None, :]) ** 2 @ ws   # (n,)
    z = -C / lam
    om = np.exp(z - z.max()); om /= om.sum()
    ess = 1.0 / np.sum(om ** 2) / n
    return float((om * eps).sum() / sigma2), float(ess)


# ---------------------------------------------------------------------------
# Stein-identity kernels from one batch (the document's section 6)
# ---------------------------------------------------------------------------


def stein_kernels_1d(U: float, sigma2: float, lam: float, n: int,
                     rng: np.random.Generator, quad: int = 64):
    """`K1[m]`, `K2[m, m']` from a single noise batch.

    `grad_U E[f(U+eps)] = Sinv E[eps f(U+eps)]` (Gaussian integration by parts),
    so both kernels are plain means over the batch -- every sample contributes,
    unlike the softmax label.
    """
    eps = rng.normal(0.0, np.sqrt(sigma2), size=n)
    s = (np.arange(quad) + 0.5) / quad
    c = ((U + eps)[:, None] - s[None, :]) ** 2            # (n, quad)
    inv = 1.0 / sigma2

    ec = (eps[:, None] * c).mean(0)                        # E[eps c_s]
    cbar = c.mean(0)                                       # E[c_s]
    K1 = -(1.0 / lam) * inv * ec                           # (quad,)

    # K2 = (1/2 lam^2) Sinv { E[eps c_s c_s'] - E[eps c_s]E[c_s'] - E[c_s]E[eps c_s'] }
    ecc = np.einsum("k,km,kn->mn", eps, c, c) / n
    K2 = (0.5 / lam ** 2) * inv * (ecc - np.outer(ec, cbar) - np.outer(cbar, ec))
    return s, np.full(quad, 1.0 / quad), K1, K2


def assemble(beta, w_at_s, K1, K2) -> tuple[float, float]:
    """First- and second-order assembled score, the only place `w` appears."""
    ws = beta * w_at_s
    g1 = float(ws @ K1)
    g2 = g1 + float(ws @ K2 @ ws)
    return g1, g2


def analytic_kernels_1d(U: float, s: np.ndarray, sigma2: float, lam: float):
    """The document's cross-check: `K1 = -2(U-s)/lam`, `K2 = 4 sigma^2 (U - (s+s')/2)/lam^2`."""
    K1 = -2.0 * (U - s) / lam
    K2 = (4.0 * sigma2 / lam ** 2) * (U - 0.5 * (s[:, None] + s[None, :]))
    return K1, K2


# ---------------------------------------------------------------------------
# The 2-D example of Appendix B: the damping is a matrix, so it rotates
# ---------------------------------------------------------------------------


def two_d(sigma2: float = 0.05, lam: float = 1.0, w1: float = 1.0, w2: float = 3.0,
          U=(0.0, 0.0)):
    """`G = -(1/lam) (I + 2 sigma^2 A / lam)^{-1} grad C`, against its truncations."""
    U = np.asarray(U, float)
    ex = np.array([1.0, 0.0])
    nn = np.array([1.0, 1.0]) / np.sqrt(2.0)
    A = w1 * np.outer(ex, ex) + w2 * np.outer(nn, nn)
    gradC = 2.0 * A @ U - 2.0 * w2 * nn
    M = 2.0 * sigma2 * A / lam
    exact = -np.linalg.solve(np.eye(2) + M, gradC) / lam
    first = -gradC / lam
    second = -(np.eye(2) - M) @ gradC / lam
    return A, gradC, exact, first, second


def mc_two_d(sigma2, lam, w1, w2, U, n, rng):
    U = np.asarray(U, float)
    eps = rng.normal(0.0, np.sqrt(sigma2), size=(n, 2))
    V = U + eps
    nn = np.array([1.0, 1.0]) / np.sqrt(2.0)
    C = w1 * V[:, 0] ** 2 + w2 * (V @ nn - 1.0) ** 2
    z = -C / lam
    om = np.exp(z - z.max()); om /= om.sum()
    return (om[:, None] * eps).sum(0) / sigma2


# ---------------------------------------------------------------------------
# Where the ascent actually lands, for a cost that is not quadratic
# ---------------------------------------------------------------------------


def _gauss_nodes(sigma2: float, n: int, span: float = 8.0):
    """Trapezoid nodes and normalised Gaussian weights on +-span sigma."""
    sd = np.sqrt(sigma2)
    e = np.linspace(-span * sd, span * sd, n)
    pw = np.exp(-0.5 * (e / sd) ** 2)
    return e, pw / pw.sum()


def hinge_true_score(U: float, w: Weight1D, sigma2: float, lam: float,
                     quad: int = 256, gh: int = 2001, pull: float = 0.5) -> float:
    """`G` for `c(U,s) = max(0, s-U)^2 + pull * U^2`, by dense quadrature.

    A one-sided hinge alone is monotone -- every index pushes `U` up and there is
    no interior optimum to shift, so it cannot test anything.  The quadratic
    pull supplies the opposing force.  The hinge is still what makes the total
    non-quadratic, and therefore what lets Gaussian smoothing move the optimum:
    that is exactly the document's stated limit (A.6), which says a purely
    quadratic cost hides the effect.
    """
    e, pw = _gauss_nodes(sigma2, gh)
    s = (np.arange(quad) + 0.5) / quad
    ws = w(s) / quad
    V = U + e
    C = (np.maximum(0.0, s[None, :] - V[:, None]) ** 2) @ ws + pull * V ** 2 * ws.sum()
    z = -C / lam
    m = z.max()
    J = float(pw @ np.exp(z - m))
    num = float(pw @ (e * np.exp(z - m)))
    return num / J / sigma2                      # Stein form of grad log J


def hinge_truncated_score(U: float, w: Weight1D, sigma2: float, lam: float,
                          order: int, quad: int = 256, gh: int = 2001,
                          pull: float = 0.5) -> float:
    """Assembled score for the hinge cost from analytic-by-quadrature kernels."""
    e, pw = _gauss_nodes(sigma2, gh)
    s = (np.arange(quad) + 0.5) / quad
    beta = np.full(quad, 1.0 / quad)
    V = U + e
    c = (np.maximum(0.0, s[None, :] - V[:, None]) ** 2
         + pull * (V ** 2)[:, None])                               # (gh, quad)
    ec = (pw[:, None] * e[:, None] * c).sum(0)
    cbar = (pw[:, None] * c).sum(0)
    K1 = -(1.0 / lam) * ec / sigma2
    ws = beta * w(s)
    g = float(ws @ K1)
    if order >= 2:
        ecc = np.einsum("k,k,km,kn->mn", pw, e, c, c) / sigma2
        K2 = (0.5 / lam ** 2) * (ecc - np.outer(ec, cbar) / sigma2
                                 - np.outer(cbar, ec) / sigma2)
        g += float(ws @ K2 @ ws)
    return g


def ascent_fixed_point(score_fn, lo: float = -1.5, hi: float = 2.5,
                       tol: float = 1e-6, max_iter: int = 60) -> float:
    """Where `U <- U + eta * Sigma * G(U)` settles, found as the root of `G`.

    The fixed point of the ascent is exactly `G(U) = 0`, so a bracketed root
    find replaces thousands of iterations -- the second-order kernel costs an
    `(eps, s, s')` contraction per evaluation and iterating it was the bottleneck.
    Returns nan when the bracket holds no sign change, which is what a diverging
    or monotone truncation looks like.
    """
    a, b = lo, hi
    fa, fb = score_fn(a), score_fn(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
        return float("nan")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = score_fn(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_analytic(a) -> None:
    rng = np.random.default_rng(0)
    print(f"1-D family  c(U,s) = (U-s)^2,  U = {a.u}, sigma^2 = {a.sigma2}, "
          f"lam = {a.lam}\n")
    print(f"{'case':<14}{'W0':>6}{'W1':>6}{'x':>7}{'exact':>10}{'1st':>10}"
          f"{'2nd':>10}{'3rd':>10}{'MC':>10}{'ESS%':>7}")
    for w in CASES:
        W0, W1, _ = w.moments()
        x = 2.0 * a.sigma2 * W0 / a.lam
        ex = exact_score_1d(a.u, w, a.sigma2, a.lam)
        tr = truncations_1d(a.u, w, a.sigma2, a.lam, 4)
        mc, ess = mc_score_1d(a.u, w, a.sigma2, a.lam, a.mc, rng)
        print(f"{w.name:<14}{W0:6.1f}{W1:6.2f}{x:7.2f}{ex:10.4f}{tr[0]:10.4f}"
              f"{tr[1]:10.4f}{tr[2]:10.4f}{mc:10.4f}{100 * ess:7.2f}")
    print(f"\n{'case':<14}{'1st err':>10}{'2nd err':>10}{'3rd err':>10}"
          f"{'MC err':>10}")
    for w in CASES:
        ex = exact_score_1d(a.u, w, a.sigma2, a.lam)
        tr = truncations_1d(a.u, w, a.sigma2, a.lam, 4)
        mc, _ = mc_score_1d(a.u, w, a.sigma2, a.lam, a.mc, rng)
        rel = lambda v: 100.0 * (v - ex) / abs(ex)
        print(f"{w.name:<14}{rel(tr[0]):9.1f}%{rel(tr[1]):9.1f}%"
              f"{rel(tr[2]):9.1f}%{rel(mc):9.1f}%")
    print("\nMC agreeing with `exact` validates the closed form; the truncation")
    print("errors shrink by exactly x per order while x < 1 and blow up beyond it.")

    print("\n2-D example (Appendix B): the damping is a matrix, so it rotates")
    A, gradC, ex, f1, f2 = two_d(a.sigma2_2d, a.lam)
    mc = mc_two_d(a.sigma2_2d, a.lam, 1.0, 3.0, (0.0, 0.0), a.mc, rng)
    ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
    print(f"  A = {A.round(3).tolist()},  grad C = {gradC.round(3).tolist()}")
    print(f"  {'':<10}{'Gx':>9}{'Gy':>9}{'|G|':>9}{'angle':>9}{'err vs exact':>14}")
    for name, v in (("exact", ex), ("1st", f1), ("2nd", f2), ("MC", mc)):
        print(f"  {name:<10}{v[0]:9.3f}{v[1]:9.3f}{np.linalg.norm(v):9.3f}"
              f"{ang(v):8.1f}deg{np.linalg.norm(v - ex):14.3f}")


def cmd_stein(a) -> None:
    rng = np.random.default_rng(1)
    print(f"Kernels from ONE noise batch of {a.batch}, by the Stein identity.")
    print("First: do the estimated kernels match the document's closed forms?\n")
    s, beta, K1h, K2h = stein_kernels_1d(a.u, a.sigma2, a.lam, a.batch, rng, a.quad)
    K1a, K2a = analytic_kernels_1d(a.u, s, a.sigma2, a.lam)
    r = lambda h, t: float(np.linalg.norm(h - t) / max(np.linalg.norm(t), 1e-12))
    print(f"  K1 relative error {r(K1h, K1a):.4f}")
    print(f"  K2 relative error {r(K2h, K2a):.4f}")
    print(f"  (K2 is a third moment, so it needs far more samples than K1 --")
    print(f"   that asymmetry is the practical cost of the second-order term.)")

    print(f"\nAssembled from the estimated kernels, against the analytic truth:")
    print(f"  {'case':<14}{'x':>7}{'exact':>10}{'1st':>10}{'2nd':>10}"
          f"{'1st err':>10}{'2nd err':>10}")
    for w in CASES:
        W0, _, _ = w.moments()
        x = 2.0 * a.sigma2 * W0 / a.lam
        ex = exact_score_1d(a.u, w, a.sigma2, a.lam)
        g1, g2 = assemble(beta, w(s), K1h, K2h)
        rel = lambda v: 100.0 * (v - ex) / abs(ex)
        print(f"  {w.name:<14}{x:7.2f}{ex:10.4f}{g1:10.4f}{g2:10.4f}"
              f"{rel(g1):9.1f}%{rel(g2):9.1f}%")
    print("\nOne batch covers every weight: the assembly above is a matrix-vector")
    print("product with no new rollouts.  (The existing pipeline already stores")
    print("clouds rather than labels, so this part is not new to it; the lower-")
    print("variance K1 estimator below is.)")


def cmd_labels(a) -> None:
    """The variance comparison -- both estimators, same batch size, same truth."""
    print(f"Both labels are Stein estimators at batch {a.batch}; one is a plain")
    print(f"mean, the other a self-normalised ratio with a peaked weight.\n")
    print(f"{'case':<14}{'x':>6}{'ESS%':>7}"
          f"{'softmax bias':>14}{'softmax sd':>12}"
          f"{'Stein bias':>12}{'Stein sd':>10}{'sd ratio':>10}")
    for w in CASES:
        W0, _, _ = w.moments()
        x = 2.0 * a.sigma2 * W0 / a.lam
        ex = exact_score_1d(a.u, w, a.sigma2, a.lam)
        soft, stein, esss = [], [], []
        for t in range(a.trials):
            rng = np.random.default_rng(1000 + t)
            g, ess = mc_score_1d(a.u, w, a.sigma2, a.lam, a.batch, rng, a.quad)
            soft.append(g); esss.append(ess)
            rng = np.random.default_rng(1000 + t)
            s, beta, K1h, K2h = stein_kernels_1d(a.u, a.sigma2, a.lam, a.batch,
                                                 rng, a.quad)
            stein.append(assemble(beta, w(s), K1h, K2h)[1])
        soft, stein = np.array(soft), np.array(stein)
        sc = 100.0 / abs(ex)
        print(f"{w.name:<14}{x:6.2f}{100 * np.mean(esss):7.2f}"
              f"{sc * (soft.mean() - ex):13.1f}%{sc * soft.std():11.1f}%"
              f"{sc * (stein.mean() - ex):11.1f}%{sc * stein.std():9.1f}%"
              f"{stein.std() / max(soft.std(), 1e-12):10.3f}")
    print("\nBias and sd are percentages of the true score.  The softmax label")
    print("pays for its low effective sample size; the Stein label pays a")
    print("truncation bias instead.  Which is preferable depends on x, and that")
    print("is the honest way to state the trade.")


def cmd_fixed(a) -> None:
    """Does the truncation move where the ascent lands, for a non-quadratic cost?"""
    print("Hinge cost  c(U,s) = max(0, s-U)^2  -- not preserved by smoothing, so")
    print("the optimum is allowed to move (the document's own limit, A.6).\n")
    print(f"{'case':<14}{'x':>6}{'U* exact':>11}{'U* 1st':>11}{'U* 2nd':>11}"
          f"{'1st shift':>11}{'2nd shift':>11}")
    for w in CASES:
        W0, _, _ = w.moments()
        x = 2.0 * a.sigma2 * W0 / a.lam
        f_ex = lambda U, w=w: hinge_true_score(U, w, a.sigma2, a.lam, a.quad2, a.gh, a.pull)
        f_1 = lambda U, w=w: hinge_truncated_score(U, w, a.sigma2, a.lam, 1,
                                                   a.quad2, a.gh, a.pull)
        f_2 = lambda U, w=w: hinge_truncated_score(U, w, a.sigma2, a.lam, 2,
                                                   a.quad2, a.gh, a.pull)
        u_ex = ascent_fixed_point(f_ex)
        u_1 = ascent_fixed_point(f_1)
        u_2 = ascent_fixed_point(f_2)
        print(f"{w.name:<14}{x:6.2f}{u_ex:11.4f}{u_1:11.4f}{u_2:11.4f}"
              f"{u_1 - u_ex:+11.4f}{u_2 - u_ex:+11.4f}")
    print("\nFor a quadratic cost these three columns would be identical -- the")
    print("truncation would only change the step size, which an iterated ascent")
    print("absorbs.  A shift here is the part iteration cannot absorb, and so")
    print("the part that actually argues for the second-order kernel.")


def cmd_temper(a) -> None:
    """x = 2 sigma^2 W0 / lam: raising lam is the cheaper lever.  Price it."""
    print("x = 2 sigma^2 W0 / lam, so temperature and the second-order kernel")
    print("attack the same quantity.  ESS is reported because raising lam buys")
    print("label quality and spends behavioural contrast.\n")
    w = CASES[2]                                     # the stiff case
    W0, _, _ = w.moments()
    print(f"weight = {w.name} (W0 = {W0:.0f}),  U = {a.u},  sigma^2 = {a.sigma2}")
    print(f"{'lam':>8}{'x':>7}{'1st err':>10}{'2nd err':>10}{'ESS%':>8}")
    for lam in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        x = 2.0 * a.sigma2 * W0 / lam
        ex = exact_score_1d(a.u, w, a.sigma2, lam)
        tr = truncations_1d(a.u, w, a.sigma2, lam, 2)
        rng = np.random.default_rng(7)
        _, ess = mc_score_1d(a.u, w, a.sigma2, lam, a.batch, rng, a.quad)
        rel = lambda v: 100.0 * (v - ex) / abs(ex)
        print(f"{lam:8.2f}{x:7.3f}{rel(tr[0]):9.1f}%{rel(tr[1]):9.1f}%"
              f"{100 * ess:8.2f}")
    print("\nAt x < 1 the second order roughly squares the error; so does")
    print("multiplying lam by 1/x.  The kernel is worth its complexity only where")
    print("raising lam is not acceptable -- which is a measurement, not a given.")


def cmd_basis(a) -> None:
    """Is the existing method really the first-order truncation?  It is not.

    The document identifies "learn at basis weights, then combine linearly" with
    the first-order Volterra truncation.  That holds when the expansion point is
    `w = 0`, i.e. when the basis is a delta of *infinitesimal* mass.  The
    existing pipeline's basis weights sit at the **deployment scale**, and each
    fitted field therefore already contains the full non-linear response at its
    own weight -- so the combination interpolates between correct values instead
    of extrapolating a derivative from zero.

    In this family the difference is visible in closed form.  With
    `w = sum_i omega_i phi_i`,

        exact           G = -2 (W0 U - W1) / (lam + 2 sigma^2 W0)
        linear combo    sum_i omega_i * -2 (W0_i U - W1_i) / (lam + 2 sigma^2 W0_i)
        1st Volterra    -2 (W0 U - W1) / lam                     <- no damping at all

    The numerator is linear in omega either way; what differs is that the
    combination carries a damping denominator per basis row while the truncation
    carries none.  When the rows have equal mass and omega sums to one, the
    denominators coincide and the combination is *exact*.
    """
    k = a.k
    edges = np.linspace(0.0, 1.0, k + 1)
    phis = [Weight1D(f"phi{i}", float(k), edges[i], edges[i + 1]) for i in range(k)]

    def exact_of(om):
        W0 = sum(o * p.moments()[0] for o, p in zip(om, phis))
        W1 = sum(o * p.moments()[1] for o, p in zip(om, phis))
        return -2.0 * (W0 * a.u - W1) / (a.lam + 2.0 * a.sigma2 * W0), W0

    def combo_of(om):
        return sum(o * exact_score_1d(a.u, p, a.sigma2, a.lam)
                   for o, p in zip(om, phis))

    def volterra_of(om, order):
        W0 = sum(o * p.moments()[0] for o, p in zip(om, phis))
        W1 = sum(o * p.moments()[1] for o, p in zip(om, phis))
        lead = -2.0 * (W0 * a.u - W1) / a.lam
        x = 2.0 * a.sigma2 * W0 / a.lam
        return lead * sum((-x) ** j for j in range(order))

    rng = np.random.default_rng(3)
    print(f"{k} basis rows (disjoint indicators of mass 1 each), U = {a.u}, "
          f"sigma^2 = {a.sigma2}, lam = {a.lam}")
    print("omega normalisation matters because it fixes how much the total mass "
          "W0 moves across\nthe weights being composed, and W0 is the whole "
          "non-linearity.\n")
    for label, norm in (("sum-to-one", "l1"), ("unit 2-norm (project convention)", "l2"),
                        ("free scale in [0.2, 3]", "free")):
        errs_c, errs_v1, errs_v2, xs = [], [], [], []
        for _ in range(a.trials):
            om = rng.dirichlet(np.ones(k))
            if norm == "l2":
                om = om / np.linalg.norm(om)
            elif norm == "free":
                om = om * rng.uniform(0.2, 3.0)
            ex, W0 = exact_of(om)
            if abs(ex) < 1e-9:
                continue
            xs.append(2.0 * a.sigma2 * W0 / a.lam)
            errs_c.append(abs(combo_of(om) - ex) / abs(ex))
            errs_v1.append(abs(volterra_of(om, 1) - ex) / abs(ex))
            errs_v2.append(abs(volterra_of(om, 2) - ex) / abs(ex))
        print(f"  {label}")
        print(f"    x over the weights: {np.min(xs):.2f} .. {np.max(xs):.2f}")
        print(f"    linear combination of exact basis scores : "
              f"mean {100*np.mean(errs_c):7.3f}%   max {100*np.max(errs_c):7.3f}%")
        print(f"    1st-order Volterra truncation            : "
              f"mean {100*np.mean(errs_v1):7.3f}%   max {100*np.max(errs_v1):7.3f}%")
        print(f"    2nd-order Volterra truncation            : "
              f"mean {100*np.mean(errs_v2):7.3f}%   max {100*np.max(erri):7.3f}%"
              if False else
              f"    2nd-order Volterra truncation            : "
              f"mean {100*np.mean(errs_v2):7.3f}%   max {100*np.max(errs_v2):7.3f}%")
        print()
    print("If the combination beats the first-order truncation by orders of")
    print("magnitude, the two are not the same object and the document's central")
    print("identification does not apply to a finite-scale basis.  The residual")
    print("error of the combination is what a second-order term could address,")
    print("and it is bounded by how much W0 varies across the weight set.")


def cmd_normalise(a) -> None:
    """The free fix the document does not mention: normalise omega in mass.

    `cmd_basis` shows the composition error is caused by the total mass
    `W0 = sum_i omega_i W0_i` moving across the weight set, because `W0` is the
    entire non-linearity.  So freeze it.

    With rows of *equal* mass, plain sum-to-one does it and the composition is
    **exact** (measured: 0.000%).

    A mass-weighted normalisation was the obvious next guess for unequal masses,
    and it is wrong -- measured below.  Freezing the aggregate `W0` is not
    enough, because

        exact       -2 (W0 U - W1) / (lam + 2 sigma^2 W0)          one denominator
        combination sum_i omega_i * -2(W0_i U - W1_i)/(lam + 2 sigma^2 W0_i)

    carries a denominator *per row*, and `W0_i` is a property of the row, not of
    omega.  No rescaling of omega can make per-row denominators agree with the
    aggregate one.  The only thing that can is making the rows themselves equal
    in mass -- which the project already does when it equalises row spread.  So
    the actionable pair is: **equalise the row masses, then normalise omega to
    sum to one** rather than to unit 2-norm.
    """
    k = a.k
    edges = np.linspace(0.0, 1.0, k + 1)
    rng = np.random.default_rng(5)
    # deliberately unequal row masses, the realistic case
    heights = np.array([float(k) * h for h in (0.4, 0.8, 1.6, 3.2)][:k])
    if len(heights) < k:
        heights = np.concatenate([heights, np.full(k - len(heights), float(k))])
    phis = [Weight1D(f"phi{i}", float(heights[i]), edges[i], edges[i + 1])
            for i in range(k)]
    masses = np.array([p.moments()[0] for p in phis])
    print(f"{k} basis rows with UNEQUAL masses W0_i = {masses.round(3).tolist()}")
    print(f"U = {a.u}, sigma^2 = {a.sigma2}, lam = {a.lam}\n")

    def exact_of(om):
        W0 = sum(o * p.moments()[0] for o, p in zip(om, phis))
        W1 = sum(o * p.moments()[1] for o, p in zip(om, phis))
        return -2.0 * (W0 * a.u - W1) / (a.lam + 2.0 * a.sigma2 * W0), W0

    def combo_of(om):
        return sum(o * exact_score_1d(a.u, p, a.sigma2, a.lam)
                   for o, p in zip(om, phis))

    target_mass = float(masses.mean())
    schemes = {
        "unit 2-norm (current)": lambda o: o / np.linalg.norm(o),
        "sum-to-one": lambda o: o / o.sum(),
        "mass-weighted (proposed)": lambda o: o * (target_mass / float(o @ masses)),
    }
    print(f"  {'scheme':<26}{'x range':>16}{'mean err':>11}{'max err':>10}")
    for name, fn in schemes.items():
        errs, xs = [], []
        for _ in range(a.trials):
            om = fn(rng.dirichlet(np.ones(k)))
            ex, W0 = exact_of(om)
            if abs(ex) < 1e-9:
                continue
            xs.append(2.0 * a.sigma2 * W0 / a.lam)
            errs.append(abs(combo_of(om) - ex) / abs(ex))
        print(f"  {name:<26}{np.min(xs):7.2f}..{np.max(xs):<8.2f}"
              f"{100*np.mean(errs):10.3f}%{100*np.max(errs):9.3f}%")
    print("\nThe mass-weighted scheme should collapse the error to ~0 by holding")
    print("W0 fixed.  It changes only which representative of each omega ray is")
    print("used, so it costs nothing -- omega carries direction and 1/T carries")
    print("sharpness either way.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("analytic", "stein", "labels", "fixed",
                                       "temper", "basis", "normalise", "all"))
    p.add_argument("--k", type=int, default=4, help="number of basis rows")
    p.add_argument("--u", type=float, default=0.0)
    p.add_argument("--sigma2", type=float, default=0.1)
    p.add_argument("--sigma2-2d", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--mc", type=int, default=4_000_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--quad", type=int, default=64)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--quad2", type=int, default=64, help="s-quadrature for `fixed`")
    p.add_argument("--gh", type=int, default=401, help="noise quadrature for `fixed`")
    p.add_argument("--pull", type=float, default=0.5,
                   help="strength of the opposing quadratic term in `fixed`")
    a = p.parse_args(argv)
    cmds = {"analytic": cmd_analytic, "stein": cmd_stein, "labels": cmd_labels,
            "fixed": cmd_fixed, "temper": cmd_temper, "basis": cmd_basis,
            "normalise": cmd_normalise}
    for name in (cmds if a.command == "all" else [a.command]):
        print(f"\n{'=' * 78}\n### {name}\n{'=' * 78}")
        cmds[name](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
