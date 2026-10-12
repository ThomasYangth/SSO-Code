"""Single-site Pauli direction n.sigma whose tau = inf projection has the largest nu.

    python scripts/compute/nutau/optimal_onesite.py --model=Ising --J=1.0 --hx=0.905 --hz=0.809 \
        --L=12 [--pows_max=3] [--usegpu=True]

For the families  sum:  O(n) = sum_x (n_x X_x + n_y Y_x + n_z Z_x)   (doubled momentum K = 0)
             and  site: O(n) = n_x X_0 + n_y Y_0 + n_z Z_0             (all momenta)
each component is projected off span{I, H, ..., H^pows_max} (as filter_static.py) and
then onto its conserved part P0 (E-basis elements with |E_m - E_n| <= 1e-9 max|dE|),
i.e. the tau -> inf limit of every filter.  With c_i the Pauli coefficients of P0 O_i,
    A_ij = sum_P conj(c_i,P) c_j,P 3^{-|P|},   G_ij = sum_P conj(c_i,P) c_j,P,
nu_inf(n) = n^T A n / n^T G n (real parts; the cross terms are real for real n) is
maximised by the top generalised eigenvector on the range of G.  Prints, per family,
nu_inf of X, Y, Z alone and the optimum (n normalised, sign fixed by the largest entry).
Writes $SSO_OUTPUT/<model>_StaticFilter_Data/onesite_opt_L{L}_o{o}.npz.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np

from sso.backend import to_cpu, xp as get_xp
from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.filters import to_energy_basis
from sso.nutau.models import base_operator, full_basis, model_from_args, operator_matrix
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize

FAMILIES = {"sum": ("Xsum", "Ysum", "Zsum"), "site": ("X0", "Y0", "Z0")}


def main():
    args = parse_argv()
    kind, model, name = model_from_args(args)
    usegpu = find_value(args, "usegpu", bool, False)
    xp = get_xp(usegpu)
    L = find_value(args, "L", int, 8)
    o = find_value(args, "pows_max", int, 3) + 1
    spec = diagonalize(kind, model, L)
    n = spec.n
    dE = spec.omega()
    zero = np.abs(dE) <= 1e-9 * float(np.abs(dE).max())
    zero_d = xp.asarray(zero)
    q = spec.diag_ortho(o)
    q_d = xp.asarray(q.astype(complex)) if q is not None else None
    diag = xp.arange(n)
    Bop = EnergyBasisB(spec.V, L, usegpu)
    basis = full_basis(L)
    out = {}
    for fam, ops in FAMILIES.items():
        C = []
        for op in ops:
            M = xp.asarray(to_energy_basis(operator_matrix(base_operator(op), L, basis), spec.V))
            if q_d is not None:
                M[diag, diag] -= q_d @ (q_d.conj().T @ M[diag, diag])
            C.append(Bop.to_pauli(xp.where(zero_d, M, 0.0)).reshape(-1).copy())
        C = xp.stack(C)
        A = to_cpu((C.conj() * Bop.B.reshape(1, -1)) @ C.T).real
        G = to_cpu(C.conj() @ C.T).real
        A, G = 0.5 * (A + A.T), 0.5 * (G + G.T)
        single = [A[i, i] / G[i, i] if G[i, i] > 1e-14 * G.max() else np.nan for i in range(3)]
        g, U = np.linalg.eigh(G)
        keep = g > 1e-12 * g.max()
        W = U[:, keep] / np.sqrt(g[keep])
        a, Y = np.linalg.eigh(W.T @ A @ W)
        nvec = W @ Y[:, -1]
        nvec /= np.linalg.norm(nvec)
        nvec *= np.sign(nvec[np.argmax(np.abs(nvec))])
        nu_opt = float(nvec @ A @ nvec / (nvec @ G @ nvec))
        fprint(f"L={L} o={o} family={fam}: nu_inf(X,Y,Z) = " + ", ".join(f"{v:.6g}" for v in single)
               + f"; optimum n = ({nvec[0]:+.6f}, {nvec[1]:+.6f}, {nvec[2]:+.6f}), nu_inf = {nu_opt:.6g}")
        out.update({f"{fam}_A": A, f"{fam}_G": G, f"{fam}_single": np.array(single),
                    f"{fam}_n": nvec, f"{fam}_nu": nu_opt})
    path = os.path.join(output_dir(f"{name}_StaticFilter_Data"), f"onesite_opt_L{L}_o{o}.npz")
    np.savez(path, L=L, o=o, **out)
    fprint(f"saved {path}")


if __name__ == "__main__":
    main()
