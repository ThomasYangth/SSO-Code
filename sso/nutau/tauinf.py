r"""tau = infinity: the most local exactly conserved operator, and the H^n baseline.

nu_o(inf) is the top eigenvalue of Pi_o B Pi_o on the exact commutant of H, i.e.
on operators block diagonal in the degenerate multiplets of H (full-space
spectrum, no momentum restriction), with Pi_o removing span{I, H, ..., H^{o-1}}.
This is the theta = infinity limit of the frontier (:mod:`sso.nutau.frontier`
with K = None), so the same Support / solve machinery is used.  In the figure
n = o - 1 labels the first power of H that is NOT projected out.

Baseline: the specific polynomial direction q_p = H^p Gram-Schmidt-orthogonalised
against {I, ..., H^{p-1}} (column p of the QR of the Vandermonde matrix in E),
with nu(q_p) = <q_p|B|q_p>/<q_p|q_p>.  In the thermodynamic limit
nu(q_d) -> r^d with r = nu(H) = (J^2/9 + (hx^2 + hz^2)/3)/(J^2 + hx^2 + hz^2).
"""

import numpy as np

from sso.nutau.frontier import Support, solve


def tauinf_nu(spec, Bop, o, degen_tol=1e-10, usegpu=False):
    """nu_o(inf) and multiplet statistics for one projector rank o.

    Returns dict(nu, W, n_mult, max_d, packed) with packed = sum_k d_k^2 the
    dimension of the commutant.
    """
    sup = Support(spec, None, o, 0.0, resonant=True, degen_tol=degen_tol,
                  usegpu=usegpu)
    res = solve(sup, Bop, np.inf, k=1, seed=999)
    mult = spec.multiplets(degen_tol)
    return dict(nu=float(res["Bs"][0]), W=float(res["w"][0]), n_mult=len(mult),
                max_d=max(len(g) for g in mult), packed=sup.dim)


def power_baseline_nu(spec, Bop, n_max=5):
    """nu of q_p = H^p orthogonalised against lower powers, p = 0..n_max."""
    xp = Bop.xp
    q = spec.diag_ortho(n_max + 1)
    return [Bop.nu(xp.diag(xp.asarray(q[:, p].astype(complex)))) for p in range(n_max + 1)]


def ising_r(J, hx, hz):
    """r = nu(H) for the mixed-field Ising chain (thermodynamic-limit baseline)."""
    return (J ** 2 / 9.0 + (hx ** 2 + hz ** 2) / 3.0) / (J ** 2 + hx ** 2 + hz ** 2)
