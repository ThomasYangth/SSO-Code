"""Pauli-string basis, the RPS kernel B, and the energy-basis <-> Pauli maps.

An operator O on L qubits (a 2^L x 2^L matrix in the computational basis) is a
vector in the doubled space.  Grouping the row and column bit of each site into
one 4-dim index |r c>, the per-site change of basis to Pauli strings is

    columns of U_SITE:  I -> |00>+|11>,  X -> |01>+|10>,  Y -> -i|01>+i|10>,
                        Z -> |00>-|11>,          U^dag U = 2,  U^-1 = U^dag / 2,

so ``matrix_to_pauli`` returns the Pauli coefficients c_sigma of
O = sum_sigma c_sigma sigma.

RPS kernel.  B is diagonal in the Pauli basis, B_sigma = 3^{-|sigma|} with
|sigma| the number of non-identity sites.  The RPS weight of an operator is

    nu = <O|B|O> / <O|O> = sum |c|^2 3^{-|sigma|} / sum |c|^2.

Energy basis.  With V the eigenvector matrix (columns |m>), an operator has
E-basis matrix M = V^dag O V; every solver works on M (flattened row-major,
index m*n + n') because the detuning and filters are diagonal there.  B acts on
M through  M -> V M V^dag -> Pauli -> x B -> back -> V^dag (.) V.
"""

import numpy as np

from sso.backend import xp as get_xp

U_SITE = np.array([
    [1, 0, 0, 1],      # |00>
    [0, 1, -1j, 0],    # |01>
    [0, 1, 1j, 0],     # |10>
    [1, 0, 0, -1],     # |11>
], dtype=complex)
UINV_SITE = U_SITE.conj().T / 2.0


def _interleave(L):
    """Axis permutation (r_0..r_{L-1}, c_0..c_{L-1}) -> (r_0, c_0, r_1, c_1, ...)."""
    order = []
    for i in range(L):
        order.extend([i, L + i])
    inverse = [0] * (2 * L)
    for new, old in enumerate(order):
        inverse[old] = new
    return order, inverse


def rps_kernel(L):
    """Diagonal of B in the Pauli basis: 3^{-#non-identity sites} (site code I=0,X=1,Y=2,Z=3)."""
    temp = np.arange(4 ** L, dtype=np.int64)
    count = np.zeros(4 ** L, dtype=np.int32)
    for _ in range(L):
        count += (temp % 4 != 0).astype(np.int32)
        temp //= 4
    return 3.0 ** (-count.astype(np.float64))


def _per_site(w, mat, L, xp):
    for i in range(L):
        w = xp.moveaxis(xp.tensordot(mat, w, axes=([1], [i])), 0, i)
    return w


def matrix_to_pauli(O, L, usegpu=False):
    """Pauli coefficients (length 4^L) of the computational-basis matrix O."""
    xp = get_xp(usegpu)
    order, _ = _interleave(L)
    w = xp.asarray(O).reshape((2,) * (2 * L)).transpose(order).reshape((4,) * L)
    return _per_site(w, xp.asarray(UINV_SITE), L, xp).reshape(-1)


def pauli_to_matrix(c, L, usegpu=False):
    """Inverse of :func:`matrix_to_pauli`: the 2^L x 2^L matrix sum_sigma c_sigma sigma."""
    xp = get_xp(usegpu)
    _, inverse = _interleave(L)
    w = _per_site(xp.asarray(c).reshape((4,) * L), xp.asarray(U_SITE), L, xp)
    n = 2 ** L
    return w.reshape((2,) * (2 * L)).transpose(inverse).reshape(n, n)


class EnergyBasisB:
    """The RPS kernel B acting on energy-basis matrices M (n x n).

    ``apply(M)`` returns V^dag B(V M V^dag) V.  Two n x n scratch buffers are
    preallocated and reused (``matmul(out=...)``) so that the largest systems
    (L = 14, n^2 = 2.7e8) fit on one GPU.  V may be real (static full-space
    eigh) or complex.
    """

    def __init__(self, V, L, usegpu=False):
        xp = get_xp(usegpu)
        self.xp, self.L, self.usegpu = xp, L, usegpu
        self.V = xp.asarray(V)
        self.Vh = self.V.conj().T
        self.n = self.V.shape[0]
        self.B = xp.asarray(rps_kernel(L))
        self.buf_a = xp.zeros((self.n, self.n), dtype=complex)
        self.buf_b = xp.zeros((self.n, self.n), dtype=complex)

    def to_pauli(self, M):
        """Pauli coefficients of the operator with E-basis matrix M."""
        xp = self.xp
        xp.matmul(self.V, xp.asarray(M).reshape(self.n, self.n), out=self.buf_b)
        xp.matmul(self.buf_b, self.Vh, out=self.buf_a)
        return matrix_to_pauli(self.buf_a, self.L, self.usegpu)

    def from_pauli(self, c):
        """E-basis matrix of the operator with Pauli coefficients c."""
        O = pauli_to_matrix(c, self.L, self.usegpu)
        return self.Vh @ O @ self.V

    def apply(self, M):
        """V^dag B(V M V^dag) V for an n x n (or flat n^2) E-basis matrix M.

        Returns an internal buffer, valid until the next call.
        """
        xp = self.xp
        c = self.to_pauli(M) * self.B
        self.buf_a[:] = pauli_to_matrix(c, self.L, self.usegpu)
        del c
        xp.matmul(self.Vh, self.buf_a, out=self.buf_b)
        xp.matmul(self.buf_b, self.V, out=self.buf_a)
        return self.buf_a

    def nu(self, M):
        """RPS weight nu = <O|B|O>/<O|O> of the operator with E-basis matrix M."""
        c = self.to_pauli(M)
        w2 = self.xp.real(c.conj() * c)
        denom = float(w2.sum())
        return float((w2 * self.B).sum()) / denom if denom > 0 else 0.0
