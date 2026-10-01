"""Analytic Pauli-algebra Grams and the ED reference (CPU, small L)."""
import numpy as np

from sso.dqt.exact import exact_reference, hk_rps_analytic, rps_norm2
from sso.dqt.model import mfim_hk
from sso.dqt.pauli import hs_gram, rps_gram

J, G, H, N = 1.0, -1.05, 0.5, 1


def _krylov_dense(L, maxmu):
    H0 = mfim_hk(L, J, G, H, 0).toarray()
    B = [mfim_hk(L, J, G, H, N).toarray()]
    for _ in range(maxmu):
        B.append(H0 @ B[-1] - B[-1] @ H0)
    return B


def test_grams_vs_dense():
    L = 9
    B = _krylov_dense(L, 2)
    R = rps_gram(L, J, G, H, N, 2)
    S = hs_gram(L, J, G, H, N, 2)
    for a in range(3):
        assert abs(R[a, a].real - rps_norm2(B[a], L)) < 1e-10 * max(1, abs(R[a, a]))
        for b in range(3):
            hs = np.trace(B[a].conj().T @ B[b]) / 2 ** L
            assert abs(S[a, b] - hs) < 1e-10 * max(1, abs(hs))
    # polarization: off-diagonal RPS entry from ||B0 + B2||^2
    off = 0.5 * (rps_norm2(B[0] + B[2], L) - R[0, 0].real - R[2, 2].real)
    assert abs(R[0, 2].real - off) < 1e-10


def test_gram_reference_values():
    """Values recorded by the original pauli_sym self-check (exact 4^L transform at L=12)."""
    R = rps_gram(12, J, G, H, N, 2).real
    assert np.allclose(np.diag(R), [6.654, 0.788, 2.195], atol=6e-4)


def test_analytic_bare_rps_norm():
    L = 8
    ref = exact_reference(L, J, G, H, N, [1e-6])["M"][0]
    assert abs(ref - hk_rps_analytic(L, J, G, H, N)) < 1e-8
