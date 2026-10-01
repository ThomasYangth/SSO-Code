"""Exact eigendecomposition of a static Hamiltonian or a Floquet unitary.

One routine covers both cases and both bases:

* ``kind='static'``   H = V diag(E) V^dag  (dense ``eigh``)
* ``kind='floquet'``  U = V diag(e^{i phi}) V^dag  (complex Schur, which returns
  an orthonormal V even for degenerate quasi-energies; for a unitary the Schur
  form is diagonal up to roundoff)
* ``momentum=True``   (PBC only) diagonalize each QuSpin momentum block
  k = 0..L-1 and embed its eigenvectors into the full 2^L basis, so every
  column of V has a definite momentum k_m.  Valid for Floquet because every
  layer is translation invariant, hence block diagonal in k.
* ``momentum=False``  diagonalize the full 2^L matrix (also the OBC path);
  ``k`` is then None.

The doubled-space translation acts on the E-basis pair (m, n) with eigenvalue
exp(2 pi i (k_m - k_n)/L), so K = (k_m - k_n) mod L labels the doubled-momentum
sector of an operator.  K and L-K are related by O -> O^dag and give identical
(tau, nu).
"""

import numpy as np
from scipy.linalg import eigh, expm, schur
from quspin.basis import spin_basis_1d

from sso.backend import xp as get_xp, to_cpu, free_gpu_memory
from sso.nutau.conventions import tau_of
from sso.nutau.models import operator_matrix


class Spectrum:
    """Eigenbasis of H (static) or U (Floquet).

    Attributes
    ----------
    kind     'static' or 'floquet'
    L, pbc   chain length and boundary condition
    values   E_m (static) or phi_m = arg(u_m) in (-pi, pi] (Floquet), numpy
    V        eigenvector matrix (numpy, or cupy for the GPU full-space eigh)
    k        momentum label of each column of V, or None
    """

    def __init__(self, kind, L, pbc, values, V, k=None):
        self.kind, self.L, self.pbc = kind, L, pbc
        self.values = np.asarray(values, dtype=float)
        self.V = V
        self.k = None if k is None else np.asarray(k, dtype=np.int32)
        self.n = len(self.values)

    # -- frequencies ---------------------------------------------------------
    def detuning_sq_pairs(self, m, n, lam=0.0):
        """Delta^2 for the pairs (m[i], n[i]): (E_m - E_n - lam)^2 or 4 sin^2((phi_m - phi_n - lam)/2).

        The Floquet half-angle form keeps full relative accuracy for small
        detunings (2 - 2 cos x would cancel to 0 below |x| ~ 1.5e-8).
        """
        d = self.values[m] - self.values[n] - lam
        if self.kind == "static":
            return d * d
        s = np.sin(0.5 * d)
        return 4.0 * s * s

    def detuning_sq(self, lam=0.0):
        """Delta^2 as an (n, n) array."""
        m = np.arange(self.n)
        return self.detuning_sq_pairs(m[:, None], m[None, :], lam)

    def omega(self):
        """Signed (n, n) frequency matrix: E_m - E_n, or phi_m - phi_n wrapped into (-pi, pi]."""
        w = self.values[:, None] - self.values[None, :]
        if self.kind == "floquet":
            w = (w + np.pi) % (2 * np.pi) - np.pi
        return w

    def tau(self, As):
        """tau(As) in this spectrum's convention (static 1/sqrt, Floquet chord)."""
        return tau_of(As, self.kind)

    # -- trivially conserved operators --------------------------------------
    def diag_ortho(self, o):
        """Orthonormal columns (n, o) spanning the trivially slow operators, or None.

        Static: span{I, H, ..., H^{o-1}} (QR of the Vandermonde matrix in E);
        Floquet: only the identity is removed (o <= 1).  All of these are
        diagonal in the eigenbasis, i.e. supported on the (m, m) entries.
        """
        if o <= 0:
            return None
        if self.kind == "floquet" and o > 1:
            raise ValueError("Floquet frontier projects out the identity only (o <= 1)")
        q, _ = np.linalg.qr(np.column_stack([self.values ** p for p in range(o)]))
        return q

    # -- index sets -----------------------------------------------------------
    def sector_indices(self, K):
        """Sorted flat indices m*n + n' with (k_m - k_n') mod L == K (None for K=None)."""
        if K is None:
            return None
        if self.k is None:
            raise ValueError("momentum sector requested but the spectrum is not momentum resolved")
        diff = (self.k[:, None] - self.k[None, :]) % self.L
        return np.flatnonzero(diff.reshape(-1) == K)

    def multiplets(self, tol=1e-10):
        """Groups of eigenvalue indices whose values are degenerate within ``tol``.

        Values are sorted and each group collects the values within ``tol`` of
        its smallest member; for Floquet the phases live on a circle, so the
        first and last groups merge when they are within tol across the branch
        cut.
        """
        order = np.argsort(self.values)
        v = self.values[order]
        groups, cur, start = [], [order[0]], v[0]
        for i in range(1, self.n):
            if v[i] - start <= tol:
                cur.append(order[i])
            else:
                groups.append(cur)
                cur, start = [order[i]], v[i]
        groups.append(cur)
        if (self.kind == "floquet" and len(groups) > 1
                and v[0] + 2 * np.pi - v[-1] <= tol):
            groups[0] = groups.pop() + groups[0]
        return [np.sort(np.asarray(g)) for g in groups]

    def resonant_indices(self, K=None, lam=0.0, tol=1e-10):
        """Sorted flat indices of the exact commutant (Delta^2 <= tol^2), intersected with sector K.

        lam = 0: pairs inside degenerate multiplets (block-diagonal operators).
        lam != 0: pairs with |Delta| <= tol, evaluated pairwise.
        """
        n = self.n
        if lam == 0.0:
            idx = np.concatenate([(g[:, None] * n + g[None, :]).reshape(-1)
                                  for g in self.multiplets(tol)])
        else:
            idx = np.flatnonzero(self.detuning_sq(lam).reshape(-1) <= tol ** 2)
        idx = np.sort(idx)
        if K is not None:
            m, nn = idx // n, idx % n
            idx = idx[(self.k[m] - self.k[nn]) % self.L == K]
        return idx


# -- diagonalization -----------------------------------------------------------

def _static_block(op, L, pbc):
    def diag(basis):
        H = operator_matrix(op, L, basis, pbc)
        H = 0.5 * (H + H.conj().T)
        if float(np.max(np.abs(H.imag))) < 1e-10:
            H = H.real
        return eigh(H)
    return diag


def _floquet_block(layers, L, pbc):
    def diag(basis):
        U = np.eye(basis.Ns, dtype=np.complex128)
        for op in layers:
            H = operator_matrix(op, L, basis, pbc)
            U = U @ expm(-1j * 0.5 * (H + H.conj().T))
        T, W = schur(U, output="complex")
        u = np.diag(T).copy()
        off = float(np.linalg.norm(np.triu(T, 1)))
        if off > 1e-9:
            raise RuntimeError(f"Schur form of U not diagonal (residual {off:.2e})")
        return np.angle(u / np.abs(u)), W
    return diag


def _diagonalize(kind, block_diag, L, pbc, momentum):
    if not momentum:
        vals, V = block_diag(spin_basis_1d(L, pauli=1))
        return Spectrum(kind, L, pbc, vals, V)
    if not pbc:
        raise ValueError("momentum sectors require periodic boundary conditions")
    vals, cols, ks = [], [], []
    for k in range(L):
        basis = spin_basis_1d(L, kblock=k, pauli=1)
        if basis.Ns == 0:
            continue
        v, W = block_diag(basis)
        cols.append(np.asarray(basis.get_proj(dtype=np.complex128) @ W))
        vals.append(v)
        ks.append(np.full(len(v), k, dtype=np.int32))
    V = np.hstack(cols).astype(np.complex128, copy=False)
    if V.shape[1] != 2 ** L:
        raise RuntimeError(f"momentum blocks sum to {V.shape[1]}, expected {2 ** L}")
    return Spectrum(kind, L, pbc, np.concatenate(vals), V, np.concatenate(ks))


def diagonalize_static(op, L, pbc=True, momentum=True, usegpu=False):
    """Spectrum of the static Hamiltonian ``op`` (an ``sso.operators.Operator``).

    ``usegpu`` only affects the full-space (momentum=False) eigh, which then
    runs in cupy and returns V on the device (needed at L >= 14).  A real
    symmetric H (no Y terms) is diagonalized in float64.
    """
    if momentum or not usegpu:
        return _diagonalize("static", _static_block(op, L, pbc), L, pbc, momentum)
    xp = get_xp(True)
    H = operator_matrix(op, L, spin_basis_1d(L, pauli=1), pbc)
    H = 0.5 * (H + H.conj().T)
    if float(np.max(np.abs(H.imag))) < 1e-10:
        H = H.real.astype(np.float64)
    E, V = xp.linalg.eigh(xp.asarray(H))
    del H
    free_gpu_memory()
    return Spectrum("static", L, pbc, to_cpu(E), V)


def diagonalize_floquet(layers, L, pbc=True, momentum=True):
    """Spectrum of U = prod_i exp(-i H_i) for the layer Operators ``layers``."""
    return _diagonalize("floquet", _floquet_block(layers, L, pbc), L, pbc, momentum)


def diagonalize(kind, model, L, pbc=True, momentum=True, usegpu=False):
    """Dispatch to :func:`diagonalize_static` or :func:`diagonalize_floquet`."""
    if kind == "static":
        return diagonalize_static(model, L, pbc, momentum, usegpu)
    return diagonalize_floquet(model, L, pbc, momentum)
