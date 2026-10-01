r"""Chebyshev-moment DQT estimator (validation tool; not used for the paper figures).

Expanding e^{-iH tau} = sum_n c_n(tau) T_n(Htil),
c_n(tau) = e^{-ib tau}(2 - delta_n0)(-i)^n J_n(a tau), every functional of
H_k(s) becomes an s-independent moment block contracted with state-independent kernels:

    M:      K_nm = <phi_n|H_k|phi_m>,  phi_m = T_m(Htil) psi        (product state psi)
            a(psi; s) = sum_nm K_nm W_nm(s),   W_nm = int g_s c_n^* c_m
    N1,N2:  P_nm = <xi_n|zeta_m>,  xi_n = T_n(Htil) H_k phi,  zeta_m = H_k T_m(Htil) phi
            N1 = E Re sum P V1(s),  V1 = int G_{s^2} c_n^* c_m ;   N2 likewise with -G''.

No time grid is formed, so there is no dt-vs-s resolution issue (contrast
:mod:`sso.dqt.fourier`), but memory is N_cheb 2^L per sample.  Kept as an
independent cross-check of the Fourier-correlator path at small L.
"""

import numpy as np
from scipy.special import jv

from sso import backend
from sso.dqt.kernels import gauss, gwin, neg_gpp
from sso.dqt.model import haar_state, random_product_state, spectral_bracket


def cheb_vectors(matvec, seed, N, a, b, usegpu=False):
    """Phi[:, n] = T_n(Htil) seed for n = 0..N (three-term recurrence); shape (D, N+1)."""
    xp = backend.xp(usegpu)
    Phi = xp.empty((seed.shape[0], N + 1), dtype=xp.complex128)
    Phi[:, 0] = seed
    Phi[:, 1] = (matvec(seed) - b * seed) / a
    for n in range(2, N + 1):
        Phi[:, n] = 2.0 * ((matvec(Phi[:, n - 1]) - b * Phi[:, n - 1]) / a) - Phi[:, n - 2]
    return Phi


def moment_K(H, Hk, psi, N, a, b, usegpu=False):
    """K_nm = <phi_n|H_k|phi_m>, phi_m = T_m(Htil) psi;  (N+1, N+1)."""
    Phi = cheb_vectors(H.dot, psi, N, a, b, usegpu)
    return Phi.conj().T @ Hk.dot(Phi)


def moment_P(H, Hk, phi, N, a, b, usegpu=False):
    """P_nm = <xi_n|zeta_m>, xi_n = T_n(Htil)(H_k phi), zeta_m = H_k T_m(Htil) phi."""
    Phi = cheb_vectors(H.dot, phi, N, a, b, usegpu)
    Xi = cheb_vectors(H.dot, Hk.dot(phi), N, a, b, usegpu)
    return Xi.conj().T @ Hk.dot(Phi)


def cheb_coeff_matrix(a, b, N, tau_grid):
    """cmat[n, i] = c_n(tau_i)  (numpy, (N+1, ntau))."""
    ns = np.arange(N + 1)
    pref = (2.0 - (ns == 0)).astype(np.complex128) * (-1j) ** ns
    return pref[:, None] * jv(ns[:, None], a * tau_grid[None, :]) * np.exp(-1j * b * tau_grid)[None, :]


def kernel_matrix(cmat, dtau, weight):
    """V_nm = int weight(tau) conj(c_n(tau)) c_m(tau) dtau on a uniform grid."""
    return (cmat.conj() * (weight * dtau)[None, :]) @ cmat.T


def make_tau_grid(a, s_max, nsigma, oversample=6):
    """Symmetric fine grid |tau| <= nsigma s_max, spacing 1/(oversample max(a,1))."""
    dtau = 1.0 / (oversample * max(a, 1.0))
    nhalf = int(np.ceil(nsigma * s_max / dtau))
    return np.arange(-nhalf, nhalf + 1) * dtau, dtau


def cheb_order_for(a, tau_max, buffer=30, tol=1e-14):
    """Smallest N with |J_n(a tau_max)| < tol for n > N, plus ``buffer``."""
    alpha = a * tau_max
    nmax = int(np.ceil(alpha)) + max(60, int(12.0 * alpha ** (1.0 / 3.0)) + 60)
    sig = np.nonzero(np.abs(jv(np.arange(nmax + 1), alpha)) > tol)[0]
    return (int(sig[-1]) if sig.size else 1) + buffer


def chebmoment_estimate(H, Hk, L, svals, nhaar, nprod, seed=0, nsigma=6.0, usegpu=False):
    """N1, N2 (Haar DQT) and M (product states, plain mean) for all s from moment blocks.

    ``H``, ``Hk`` are CSR matrices on the chosen backend; the bracket is taken from the
    CPU copy of H.  Returns dict(N1, N1_err, N2, N2_err, M, M_err, Ncheb).
    """
    Emin, Emax = spectral_bracket(backend.to_cpu(H))
    a, b = 0.5 * (Emax - Emin), 0.5 * (Emax + Emin)
    smax = float(np.max(svals))
    N = cheb_order_for(a, nsigma * smax)
    tau, dtau = make_tau_grid(a, smax, nsigma)
    cmat = cheb_coeff_matrix(a, b, N, tau)
    V1 = np.stack([kernel_matrix(cmat, dtau, gauss(s ** 2, tau)) for s in svals])
    V2 = np.stack([kernel_matrix(cmat, dtau, neg_gpp(s ** 2, tau)) for s in svals])
    W = np.stack([kernel_matrix(cmat, dtau, gwin(s, tau)) for s in svals])
    rng = np.random.RandomState(seed)
    n1 = np.empty((nhaar, len(svals)))
    n2 = np.empty((nhaar, len(svals)))
    for r in range(nhaar):
        P = backend.to_cpu(moment_P(H, Hk, haar_state(2 ** L, rng, usegpu), N, a, b, usegpu))
        n1[r] = np.einsum("nm,snm->s", P, V1).real
        n2[r] = np.einsum("nm,snm->s", P, V2).real
    absa2 = np.empty((nprod, len(svals)))
    for r in range(nprod):
        K = backend.to_cpu(moment_K(H, Hk, random_product_state(L, rng, usegpu), N, a, b, usegpu))
        absa2[r] = np.abs(np.einsum("nm,snm->s", K, W)) ** 2
    sem = lambda x: x.std(0, ddof=1) / np.sqrt(len(x))
    return dict(N1=n1.mean(0), N1_err=sem(n1), N2=n2.mean(0), N2_err=sem(n2),
                M=absa2.mean(0), M_err=sem(absa2), Ncheb=N)
