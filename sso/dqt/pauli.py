r"""Symbolic Pauli-string algebra: exact Gram matrices of the Krylov operators
B_mu = sum_x e^{ikx} ad_H^mu h_x  for any L (no 2^L or 4^L objects).

Operators are dicts {key: coeff}, key = tuple(sorted((site, op))), op in {1,2,3} = {X,Y,Z};
sigma_a sigma_b = delta_ab I + i eps_abc sigma_c.  h_x is the site-centred MFIM density
(see :mod:`sso.dqt.model`).  ad_H^mu h_0 is local, so by translation covariance

    <B_a|B_b> = L sum_d e^{ikd} <ad^a h_0, ad^b h_d>,     k = 2 pi n / L,

with the RPS inner product  <A,B>_RPS = sum_P conj(a_P) b_P 3^{-w(P)}  (w = number of
non-identity sites; equals E_psi conj<psi|A|psi> <psi|B|psi> over random product states)
or the normalized Hilbert-Schmidt one (weight 1; <<B_a|B_b>> = <<H_k|ad^{a+b} H_k>>).
Used for the {H_k, ad^2 H_k} control variate of the RPS norm (see :mod:`sso.dqt.fourier`).
"""

import numpy as np

_MUL = {(1, 1): (1, 0), (2, 2): (1, 0), (3, 3): (1, 0),
        (1, 2): (1j, 3), (2, 1): (-1j, 3), (2, 3): (1j, 1), (3, 2): (-1j, 1),
        (3, 1): (1j, 2), (1, 3): (-1j, 2)}


def _str_mul(A, B):
    """Product of Pauli strings A, B (dict site->op) -> (phase, dict site->op)."""
    ph = 1.0 + 0j
    out = dict(A)
    for site, ob in B.items():
        if site in out:
            p, oc = _MUL[(out[site], ob)]
            ph *= p
            if oc == 0:
                del out[site]
            else:
                out[site] = oc
        else:
            out[site] = ob
    return ph, out


def _key(d):
    return tuple(sorted(d.items()))


def op_mul(P, Q):
    """Operator product P Q."""
    R = {}
    for kp, cp in P.items():
        dp = dict(kp)
        for kq, cq in Q.items():
            ph, dd = _str_mul(dp, dict(kq))
            k = _key(dd)
            R[k] = R.get(k, 0j) + cp * cq * ph
    return {k: v for k, v in R.items() if abs(v) > 1e-14}


def comm(P, Q):
    """Commutator [P, Q]."""
    R = dict(op_mul(P, Q))
    for k, v in op_mul(Q, P).items():
        R[k] = R.get(k, 0j) - v
    return {k: v for k, v in R.items() if abs(v) > 1e-14}


def h_term(x, J, g, h):
    """Site-centred h_x = g X_x + h Z_x + (J/2)(Z_{x-1}Z_x + Z_x Z_{x+1})."""
    return {_key({x: 1}): g + 0j, _key({x: 3}): h + 0j,
            _key({x - 1: 3, x: 3}): 0.5 * J + 0j, _key({x: 3, x + 1: 3}): 0.5 * J + 0j}


def _support(P):
    return {site for k in P for site, _ in k}


def ad_H(P, J, g, h):
    """ad_H P = sum_y [h_y, P] over the terms overlapping supp(P)."""
    if not P:
        return {}
    sup = _support(P)
    out = {}
    for y in range(min(sup) - 2, max(sup) + 3):
        for k, v in comm(h_term(y, J, g, h), P).items():
            out[k] = out.get(k, 0j) + v
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


def krylov_ops(J, g, h, maxmu):
    """[ad^0 h_0, ..., ad^maxmu h_0]."""
    ops = [h_term(0, J, g, h)]
    for _ in range(maxmu):
        ops.append(ad_H(ops[-1], J, g, h))
    return ops


def _translate(P, d):
    return {_key({s + d: o for s, o in dict(k).items()}): v for k, v in P.items()}


def _ip(A, B, rps):
    tot = 0j
    for k, a in A.items():
        if k in B:
            tot += np.conj(a) * B[k] * (3.0 ** (-len(k)) if rps else 1.0)
    return tot


def _gram(L, J, g, h, n, maxmu, rps):
    k = 2.0 * np.pi * n / L
    ops = krylov_ops(J, g, h, maxmu)
    G = np.zeros((maxmu + 1, maxmu + 1), complex)
    for a in range(maxmu + 1):
        for b in range(maxmu + 1):
            supa, supb = _support(ops[a]), _support(ops[b])
            tot = 0j
            for d in range(min(supa) - max(supb) - 1, max(supa) - min(supb) + 2):
                tot += np.exp(1j * k * d) * _ip(ops[a], _translate(ops[b], d), rps)
            G[a, b] = L * tot
    return G


def rps_gram(L, J, g, h, n, maxmu=2):
    """Exact RPS Gram  E_psi conj<psi|B_a|psi> <psi|B_b|psi>,  a,b = 0..maxmu.

    Valid when the operator range is shorter than the ring (L > ~2 maxmu + 3)."""
    return _gram(L, J, g, h, n, maxmu, rps=True)


def hs_gram(L, J, g, h, n, maxmu=2):
    """Exact normalized-HS Gram <<B_a|B_b>> = <<H_k|ad^{a+b} H_k>>,  a,b = 0..maxmu."""
    return _gram(L, J, g, h, n, maxmu, rps=False)
