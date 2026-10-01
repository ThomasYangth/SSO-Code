r"""Model B: mixed-field Ising ring and its energy-density Fourier modes.

Site-centred energy density on a periodic chain of L spins (Pauli matrices),

    h_x = (J/2) (Z_{x-1} Z_x + Z_x Z_{x+1}) + g X_x + h Z_x ,      sum_x h_x = H ,

and the momentum-k mode  H_k = sum_x e^{ikx} h_x,  k = 2 pi n / L  (n = 0 gives H).
Production values (paper "Model B"): J = 1, g = -1.05 (transverse), h = 0.5
(longitudinal), n = 1.

Basis: the standard tensor-product (``np.kron``) ordering, site 0 = most significant
bit, local state 0 = Z-up.  This is identical (element by element) to the QuSpin
``spin_basis_1d(L, pauli=1)`` matrices used by the original production code, so
seeded runs reproduce the archived samples.

Random states: drawn from a ``np.random.RandomState(seed)`` in exactly the order of the
production script (all Haar states first, then all product states), so a run with a
given seed reproduces the archived per-sample arrays.
"""

import numpy as np
import scipy.sparse as sp

from sso import backend

MODEL_B = dict(J=1.0, g=-1.05, h=0.5)


def mfim_hk(L, J, g, h, n):
    """Sparse CSR (complex128) matrix of H_k = sum_x e^{2 pi i n x/L} h_x on 2^L states.

    Built row-wise without intermediate COO arrays: every row has exactly L+1
    stored entries (diagonal ZZ/Z part and the L single spin flips of X_x), so peak
    host memory is ~20 (L+1) 2^L bytes.  Indices are int32 (valid for L <= 26).
    """
    D = 2 ** L
    k = 2.0 * np.pi * n / L
    ph = np.exp(1j * k * np.arange(L))
    idx = np.arange(D, dtype=np.int64)
    z = [(1 - 2 * ((idx >> (L - 1 - x)) & 1)).astype(np.int8) for x in range(L)]
    diag = np.zeros(D, complex)
    for x in range(L):
        bond = 0.5 * J * (z[(x - 1) % L] * z[x] + z[x] * z[(x + 1) % L])
        diag += ph[x] * (h * z[x] + bond)
    del z
    cols = np.empty((D, L + 1), dtype=np.int32)
    vals = np.empty((D, L + 1), dtype=np.complex128)
    cols[:, 0] = idx
    vals[:, 0] = diag
    for x in range(L):
        cols[:, x + 1] = idx ^ (1 << (L - 1 - x))
        vals[:, x + 1] = g * ph[x]
    del idx, diag
    order = np.argsort(cols, axis=1, kind="stable")
    cols = np.take_along_axis(cols, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    del order
    indptr = np.arange(0, D * (L + 1) + 1, L + 1, dtype=np.int64)
    return sp.csr_matrix((vals.ravel(), cols.ravel(), indptr), shape=(D, D))


def spectral_bracket(H, pad_frac=0.02):
    """Safe Chebyshev bracket [Emin, Emax] of a Hermitian CSR matrix (CPU).

    Gershgorin discs (always contain the spectrum), widened by ``pad_frac`` of the
    width on each side.  ``pad_frac=0.02`` is the production setting.
    """
    diag = H.diagonal()
    absH = H.copy()
    absH.data = np.abs(absH.data)
    radii = np.asarray(absH.sum(axis=1)).ravel() - np.abs(diag)
    emax = float((diag.real + radii).max())
    emin = float((diag.real - radii).min())
    w = emax - emin
    return emin - pad_frac * w, emax + pad_frac * w


def to_backend(A, usegpu=False):
    """Move a scipy CSR matrix to the chosen backend (cupyx CSR on GPU)."""
    if not usegpu:
        return A
    backend.xp(True)
    import cupyx.scipy.sparse as csp
    return csp.csr_matrix(A)


def haar_state(D, rng, usegpu=False):
    """Normalized complex-Gaussian (Haar) state; draws randn(D) (real) then randn(D) (imag)."""
    xp = backend.xp(usegpu)
    v = xp.asarray(rng.randn(D) + 1j * rng.randn(D))
    return v / xp.linalg.norm(v)


def random_product_state(L, rng, usegpu=False):
    """Haar-random product state  psi = (x)_x (cos(th_x/2), e^{i ph_x} sin(th_x/2)).

    cos(th_x) uniform in [-1,1] and ph_x uniform in [0, 2 pi): uniform on each Bloch
    sphere, i.e. the random-product-state (RPS) ensemble.
    """
    ct = 2 * rng.rand(L) - 1
    th = np.arccos(ct)
    pph = 2 * np.pi * rng.rand(L)
    aa = np.cos(th / 2)
    bb = np.exp(1j * pph) * np.sin(th / 2)
    psi = np.array([1.0 + 0j])
    for x in range(L):
        psi = np.kron(psi, np.array([aa[x], bb[x]], complex))
    return backend.xp(usegpu).asarray(psi / np.linalg.norm(psi))
