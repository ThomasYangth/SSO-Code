r"""Exact (full-diagonalization) reference for the windowed functionals (small L only).

In the eigenbasis of H (Bohr frequencies Omega_ab = E_a - E_b) the filter is diagonal,
(H_k(s))_ab = (H_k)_ab e^{-s^2 Omega_ab^2/4}, so with the normalized HS product

    N1(s) = 2^{-L} sum_ab |(H_k)_ab|^2 e^{-s^2 Omega_ab^2/2},
    N2(s) = 2^{-L} sum_ab Omega_ab^2 |(H_k)_ab|^2 e^{-s^2 Omega_ab^2/2},
    M(s)  = ||H_k(s)||^2_RPS = sum_P |c_P|^2 3^{-w(P)},

where c_P are the Pauli-string coefficients of H_k(s) = sum_P c_P P (w = weight).
Cost O(8^L) time / O(4^L) memory: L <= 10 on a laptop, L ~ 12-13 on a GPU.
"""

import numpy as np

from sso import backend
from sso.dqt.model import mfim_hk

# per-site map  2x2 operator (row-major vec)  <->  Pauli coefficients (I, X, Y, Z)
_U_SITE = np.array([[1, 0, 0, 1], [0, 1, -1j, 0], [0, 1, 1j, 0], [1, 0, 0, -1]], complex)


def _rps_weights(L):
    """3^{-w(P)} for all 4^L Pauli strings in base-4 order (digit 0 = identity)."""
    idx = np.arange(4 ** L, dtype=np.int64)
    cnt = np.zeros(4 ** L, dtype=np.int64)
    for _ in range(L):
        cnt += (idx % 4 != 0)
        idx //= 4
    return 3.0 ** (-cnt)


def to_pauli_coeffs(O, L, usegpu=False):
    """Pauli coefficients c_P of a dense 2^L x 2^L operator, O = sum_P c_P P (flat, base-4)."""
    xp = backend.xp(usegpu)
    uinv = xp.asarray(_U_SITE.conj().T / 2.0)
    inter = [i for k in range(L) for i in (k, L + k)]
    w = O.reshape((2,) * (2 * L)).transpose(inter).reshape((4,) * L)
    for i in range(L):
        w = xp.moveaxis(xp.tensordot(uinv, w, axes=([1], [i])), 0, i)
    return w.reshape(-1)


def rps_norm2(O, L, usegpu=False):
    """||O||^2_RPS = E_psi |<psi|O|psi>|^2 over Haar-random product states (exact)."""
    xp = backend.xp(usegpu)
    c = to_pauli_coeffs(O, L, usegpu)
    return float(backend.to_cpu(xp.sum((c.conj() * c).real * xp.asarray(_rps_weights(L)))))


def hk_rps_analytic(L, J, g, h, n):
    """||H_k||^2_RPS = L [g^2/3 + h^2/3 + (J^2/2)(1 + cos k)/9]  (bare mode, k = 2 pi n/L)."""
    k = 2.0 * np.pi * n / L
    return L * (g * g / 3.0 + h * h / 3.0 + 0.5 * J * J * (1.0 + np.cos(k)) / 9.0)


def exact_reference(L, J, g, h, n, svals, usegpu=False):
    """Exact N1(s), N2(s), M(s) (normalized HS; s-convention of :mod:`sso.dqt.kernels`)."""
    xp = backend.xp(usegpu)
    D = 2 ** L
    E, U = xp.linalg.eigh(xp.asarray(mfim_hk(L, J, g, h, 0).toarray()))
    Uh = U.conj().T
    Om2 = (E[:, None] - E[None, :]) ** 2
    Hk_eig = Uh @ xp.asarray(mfim_hk(L, J, g, h, n).toarray()) @ U
    absHk2 = xp.abs(Hk_eig) ** 2
    out = {key: np.empty(len(svals)) for key in ("N1", "N2", "M")}
    for i, s in enumerate(svals):
        W1 = xp.exp(-0.5 * s ** 2 * Om2)
        out["N1"][i] = float(backend.to_cpu(xp.sum(absHk2 * W1))) / D
        out["N2"][i] = float(backend.to_cpu(xp.sum(Om2 * absHk2 * W1))) / D
        Os = U @ (Hk_eig * xp.exp(-0.25 * s ** 2 * Om2)) @ Uh
        out["M"][i] = rps_norm2(Os, L, usegpu)
    return out
