"""Checks of the omega-space matrix elements and the frontier solver (CPU, ~1 min).

    python test_krylovchain.py          (or python -m pytest -q test_krylovchain.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

import krylovchain as kc


def test_orthonormality_and_jacobi_matrix():
    """int p_m p_n rho = delta_mn and int omega p_m p_n rho = calL_mn, up to K = 200."""
    for K in (60, 200):
        nodes, weights = kc.quadrature_grid(0.0, K)
        P = kc.sqrt_rho_p(K, nodes)
        G = kc.function_of_L(P, nodes, weights, lambda w: np.ones_like(w))
        assert np.max(np.abs(G - np.eye(K))) < 1e-11
        full = np.concatenate([-nodes[::-1], nodes])
        wfull = np.concatenate([weights[::-1], weights])
        Pf = kc.sqrt_rho_p(K, full)
        J = (Pf * (wfull * full)[None, :]) @ Pf.T
        Lref = np.diag(kc.b(np.arange(1, K)), 1) + np.diag(kc.b(np.arange(1, K)), -1)
        assert np.max(np.abs(J - Lref)) < 1e-9


def test_theta_zero_is_bare_operator():
    """theta = 0: calR = 1, c = e_0, lambda = nu = e^{-mu}, tau = 1/b_1 = 1/sqrt2."""
    for mode in ("truncated", "exact"):
        r = kc.solve_frontier(kc.MU_EVN, 0.0, 12, mu_norm=mode)
        assert np.isclose(r["lam"], 1 / 3) and np.isclose(r["nu"], 1 / 3)
        assert np.isclose(r["tau"], 1 / np.sqrt(2))


def _mp_moment(theta, power, w2):
    f = lambda w: (w / (2 * mp.sinh(mp.pi * w / 2)) if w != 0 else 1 / mp.pi) \
        * (w * w if w2 else 1) / (1 + theta * w * w) ** power
    return 2 * mp.quad(f, [0, 1 / mp.sqrt(theta), 1, 10, mp.inf])


def test_mu_infinity_closed_form_against_mpmath_and_solver():
    """closed_form_mu_inf vs 30-digit mpmath, and solve_frontier(mu=inf, N) for several N."""
    mp.mp.dps = 30
    for theta in (0.1, 3.0, 1e2, 1e4, 1e6):
        cf = kc.closed_form_mu_inf(theta)
        a, b2, c = _mp_moment(theta, 1, False), _mp_moment(theta, 2, False), _mp_moment(theta, 2, True)
        assert np.isclose(cf["nu"], float(a * a / b2), rtol=1e-11, atol=0)
        assert np.isclose(cf["tau"], float(mp.sqrt(b2 / c)), rtol=1e-11, atol=0)
        for N in (1, 2, 5, 20):
            r = kc.solve_frontier(np.inf, theta, N)
            assert np.isclose(r["nu"], cf["nu"], rtol=1e-12, atol=0)
            assert np.isclose(r["tau"], cf["tau"], rtol=1e-12, atol=0)


def test_quadrature_converged():
    """Doubling the Gauss order or the cutoff margin changes nothing."""
    for theta in (0.5, 50.0, 5e4):
        a = kc.solve_frontier(kc.MU_EVN, theta, 30)
        b_ = kc.solve_frontier(kc.MU_EVN, theta, 30, order=48, margin=120.0)
        for k in ("lam", "nu", "tau"):
            assert np.isclose(a[k], b_[k], rtol=1e-12, atol=0)


def test_truncation_convergence_and_mu_norm_tail():
    """mu = log 3: both mu-norm forms converge in N; the dropped tail is positive and
    <~ 0.4 * 3^{-N} for any theta."""
    for theta in (1.0, 1e3, 1e6):
        ref = kc.solve_frontier(kc.MU_EVN, theta, 60, mu_norm="exact")
        for mode in ("truncated", "exact"):
            r = kc.solve_frontier(kc.MU_EVN, theta, 40, mu_norm=mode)
            for k in ("lam", "nu", "tau"):
                assert np.isclose(r[k], ref[k], rtol=1e-12, atol=0)
        for N in (2, 6, 12):
            e = kc.solve_frontier(kc.MU_EVN, theta, N, mu_norm="exact")
            tail = 1 - e["mu_norm2_trunc"] / e["mu_norm2"]
            assert 0 < tail < 0.4 * 3.0 ** -N


def test_truncated_solutions_lie_below_frontier():
    """With mu_norm="exact" each truncated c~_N gives an actual operator with exact (tau, nu):
    it lies on or below the converged frontier at the same tau."""
    def front_at(tau):
        f = lambda x: np.log(kc.solve_frontier(kc.MU_EVN, np.exp(x), 40)["tau"] / tau)
        lt = brentq(f, np.log(1e-6), np.log(1e10), xtol=1e-15)
        return kc.solve_frontier(kc.MU_EVN, np.exp(lt), 40)["nu"]

    for theta in (1.0, 1e2, 1e4):
        for N in (1, 3, 8):
            r = kc.solve_frontier(kc.MU_EVN, theta, N, mu_norm="exact")
            assert r["nu"] <= front_at(r["tau"]) * (1 + 1e-12)


def test_large_tau_limit():
    """nu tau -> 2 sum_j e^{-mu(2j+1)} p_2j(0)^2 = 2 artanh(e^{-mu}) (p_2j(0)^2 = 1/(2j+1))."""
    f = kc.solve_frontier(kc.MU_EVN, 1e12, 40)
    assert abs(f["nu"] * f["tau"] / np.log(2.0) - 1) < 2e-6
    i = kc.closed_form_mu_inf(1e12)
    assert abs(i["nu"] * i["tau"] / 2.0 - 1) < 2e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
