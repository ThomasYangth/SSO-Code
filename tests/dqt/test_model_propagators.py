"""Model builder and propagator engines (CPU, small L)."""
import numpy as np
import pytest
from scipy.linalg import expm

from sso.dqt.model import mfim_hk, random_product_state, spectral_bracket
from sso.dqt.propagators import (Propagator, adaptive_krylov_expm_multiply,
                                 chebyshev_expm_multiply)

L, J, G, H = 7, 1.0, -1.05, 0.5


def _vec(D, seed):
    r = np.random.default_rng(seed)
    v = r.standard_normal(D) + 1j * r.standard_normal(D)
    return v / np.linalg.norm(v)


def test_hk_structure():
    H0 = mfim_hk(L, J, G, H, 0).toarray()
    Hk = mfim_hk(L, J, G, H, 1).toarray()
    Hmk = mfim_hk(L, J, G, H, -1).toarray()
    assert np.allclose(H0, H0.conj().T)
    assert np.allclose(Hk.conj().T, Hmk)                    # H_k^dag = H_{-k}
    # sum over k of H_k = L h_0 (site-0 density): check against explicit h_0 terms
    tot = sum(mfim_hk(L, J, G, H, n).toarray() for n in range(L)) / L
    Z = np.diag([1, -1]); X = np.array([[0, 1], [1, 0]])
    def site(op, x):
        mats = [np.eye(2)] * L; mats[x] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out
    h0 = G * site(X, 0) + H * site(Z, 0) + 0.5 * J * (site(Z, L - 1) @ site(Z, 0) + site(Z, 0) @ site(Z, 1))
    assert np.allclose(tot, h0)


def test_matches_quspin():
    pytest.importorskip("quspin")
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian
    basis = spin_basis_1d(L=L, S="1/2", pauli=1)
    ph = lambda x: np.exp(2j * np.pi * x / L)
    zz = [[0.5 * J * ph(x), x, (x + 1) % L] for x in range(L)] + \
         [[0.5 * J * ph(x), (x - 1) % L, x] for x in range(L)]
    ref = hamiltonian([["zz", zz], ["x", [[G * ph(x), x] for x in range(L)]],
                       ["z", [[H * ph(x), x] for x in range(L)]]], [], basis=basis,
                      dtype=np.complex128, check_herm=False, check_symm=False,
                      check_pcon=False).toarray()
    assert np.abs(ref - mfim_hk(L, J, G, H, 1).toarray()).max() < 1e-13


def test_bracket_contains_spectrum():
    Hs = mfim_hk(L, J, G, H, 0)
    E = np.linalg.eigvalsh(Hs.toarray())
    lo, hi = spectral_bracket(Hs)
    assert lo < E[0] and hi > E[-1]


def test_product_state_normalized():
    psi = random_product_state(6, np.random.RandomState(3))
    assert abs(np.linalg.norm(psi) - 1) < 1e-12


@pytest.mark.parametrize("t", [0.1, 0.5, 3.0, 17.0, -4.0])
def test_engines_vs_expm(t):
    Hs = mfim_hk(L, J, G, H, 0)
    v = _vec(2 ** L, 1)
    exact = expm(-1j * t * Hs.toarray()) @ v
    lo, hi = spectral_bracket(Hs)
    cheb = chebyshev_expm_multiply(Hs, v, t, lo, hi)
    kry = adaptive_krylov_expm_multiply(Hs, v, -1j * t)
    assert np.linalg.norm(cheb - exact) < 1e-11
    assert np.linalg.norm(kry - exact) < 1e-10
    assert abs(np.linalg.norm(cheb) - 1) < 1e-12


def test_propagator_step_composition():
    Hs = mfim_hk(L, J, G, H, 0)
    br = spectral_bracket(Hs)
    v = _vec(2 ** L, 2)
    for engine in ("chebyshev", "krylov"):
        P = Propagator(Hs, 0.5, br, engine)
        w = v
        for _ in range(10):
            w = P.step(w)
        assert np.linalg.norm(w - P.evolve(v, 5.0)) < 1e-10
    a = Propagator(Hs, 0.5, br, "chebyshev").evolve(v, 5.0)
    b = Propagator(Hs, 0.5, br, "krylov").evolve(v, 5.0)
    assert np.linalg.norm(a - b) < 1e-10
