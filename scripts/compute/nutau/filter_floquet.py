"""Gaussian / hard / N-period-average filters of a fixed operator under a Floquet U.

Replaces NuTauHardCutoffFloquet.py (--filters=gauss|hard) and
NuTauTrajectoryFloquet.py (--filters=avg).  The base operator must be
traceless (orthogonal to the identity, the only projected-out operator).
--K=<int> keeps the doubled-momentum sector K of the operator, --K=none the
full space (needed for a strictly local operator such as Z0).  Chord
convention tau = 1/(2 arcsin(sqrt(As)/2)).
Writes $SSO_OUTPUT/<model>_FloquetHardCut_Data/{gausscut,hardcut}_L{L}_K{K}_{op}_l0.0.npz
       (keys width/omega_star, As, nu, tau, kept)
   and $SSO_OUTPUT/<model>_FloquetTrajectory_Data/ftraj_L{L}_K{K}_{op}_N{N_max:g}l0.0.npz
       (keys N, As_avg, nu_avg, tau_avg).

Example:
    python scripts/compute/nutau/filter_floquet.py --model=KickedIsing --L=12 --K=0 \
        --init_op=Zsum --filters=gauss --n_width=50 --usegpu=True
"""

import os
import sys
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np

from sso.backend import xp as get_xp
from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.conventions import tau_floquet
from sso.nutau.filters import filter_sweep, sample_periods, to_energy_basis, weighted_mean, width_grid
from sso.nutau.models import base_operator, full_basis, model_from_args, operator_matrix
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize


def main():
    args = parse_argv()
    fprint("arguments:", args)
    kind, model, name = model_from_args(args)
    if kind != "floquet":
        raise ValueError("use filter_static.py for static models")
    usegpu = find_value(args, "usegpu", bool, False)
    xp = get_xp(usegpu)
    L = find_value(args, "L", int, 12)
    K_arg = str(args.get("K", "0"))
    K = None if K_arg == "none" else int(K_arg)
    init_op = find_value(args, "init_op", str, "Zsum")
    filters = str(args.get("filters", "gauss")).split(",")
    n_width = find_value(args, "n_width", int, 45)
    N_max = find_value(args, "N_max", float, 1e6)
    override = find_value(args, "override", bool, False)

    t0 = time()
    spec = diagonalize(kind, model, L)
    fprint(f"diagonalization {time() - t0:.1f}s")
    O = operator_matrix(base_operator(init_op), L, full_basis(L))
    if abs(np.trace(O)) > 1e-10 * spec.n:
        raise ValueError(f"{init_op} is not traceless")
    M0 = to_energy_basis(O, spec.V)
    if K is not None:
        keep = ((spec.k[:, None] - spec.k[None, :]) % L) == K
        frac = float(np.sum(np.abs(M0[keep]) ** 2) / np.sum(np.abs(M0) ** 2))
        fprint(f"sector K={K} holds {frac:.6f} of the operator weight")
        M0 = M0 * keep
    Bop = EnergyBasisB(spec.V, L, usegpu)
    M0_d = xp.asarray(M0)
    dsq_d = xp.asarray(spec.detuning_sq(0.0))
    omega = spec.omega()
    As0, nu0 = weighted_mean(M0_d, dsq_d, xp), Bop.nu(M0_d)
    fprint(f"base operator: As={As0:.6e} (tau={tau_floquet(As0):.6g}), nu={nu0:.6e}")

    for kind_f in filters:
        t0 = time()
        if kind_f == "avg":
            path = os.path.join(output_dir(f"{name}_FloquetTrajectory_Data"),
                                f"ftraj_L{L}_K{K_arg}_{init_op}_N{N_max:g}l0.0.npz")
            grid = sample_periods(N_max, find_value(args, "n_sample", int, 60))
            raw = spec.values[:, None] - spec.values[None, :]
            omega_f = xp.asarray(raw)
        else:
            tag = {"gauss": "gausscut", "hard": "hardcut"}[kind_f]
            path = os.path.join(output_dir(f"{name}_FloquetHardCut_Data"),
                                f"{tag}_L{L}_K{K_arg}_{init_op}_l0.0.npz")
            grid = width_grid(np.abs(omega), n_width, kind_f, top=np.pi,
                              gauss_margin=2.0, zero_tol=1e-12)
            omega_f = xp.asarray(omega)
        if os.path.exists(path) and not override:
            fprint(f"{path} exists; use --override=True")
            continue
        As, nu, kept = filter_sweep(M0_d, Bop, omega_f, dsq_d, kind_f, grid, floquet=True)
        del omega_f
        tau = tau_floquet(As)
        fprint(f"'{kind_f}': {len(grid)} widths in [{grid[0]:.4g}, {grid[-1]:.4g}], "
               f"nu {nu[0]:.6e} .. {nu[-1]:.6e}  ({time() - t0:.1f}s)")
        if kind_f == "avg":
            np.savez(path, L=L, K=-1 if K is None else K, K_arg=K_arg, model_name=name,
                     lam=0.0, init_op=init_op, phi=spec.values, k_m=spec.k, dim=spec.n,
                     As0=As0, nu0=nu0, tau0=tau_floquet(As0),
                     N=grid, As_avg=As, nu_avg=nu, tau_avg=tau)
        else:
            np.savez(path, L=L, K=K_arg, lam=0.0, model_name=name, init_op=init_op,
                     n_dim=spec.n, phi=spec.values, filter=kind_f, omega_star=grid,
                     width=grid, As=As, nu=nu, tau=tau, kept=kept)
        fprint(f"saved {path}")


if __name__ == "__main__":
    main()
