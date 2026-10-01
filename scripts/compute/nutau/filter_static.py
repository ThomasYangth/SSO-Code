"""Gaussian / hard / time-average filters of a fixed operator under a static H.

Replaces NuTauFilterStatic.py.  Base operator (--init_op): Zsum, Xsum, Z0, YZcur,
or Hdens = the modulated energy density H_q = sum_x e^{-i q x} h_x,
q = 2 pi K / L; or a real mixture --mix=Zsum:0.78,Xsum:-0.62 (file tag
--mix_tag, default MIX).  Static convention tau = 1/sqrt(As).

* K = 0: the operator is projected off span{I, H, ..., H^pows_max}; this
  commutes with every filter at lam = 0 (the span sits on omega = 0, where
  every filter equals 1).  K != 0 forces pows_max = -1 (nothing to remove).
* The operator must lie in one doubled-momentum sector (asserted); H_q may land
  in L - K instead of K (the two conventions differ by a sign; identical
  tau, nu), the file keeps K.
* Conserved component.  If the operator has weight at omega = 0 (always for
  K = 0; for H_q at K = 1 only at odd L, from the degenerate k <-> -k pairs in
  sector 2k = 1 mod L) every filter must converge to that projection nu_diag.
  Otherwise there is no tau -> inf limit: the narrow end of the sweep keeps
  exactly the slowest occupied element (keep_slowest) and the checks are
  tau <= 1/|omega|_min on the support and monotonic tau along the hard sweep.
* The sweep range is set by |omega| on the occupied support of the operator.

Writes $SSO_OUTPUT/<model>_StaticFilter_Data/filt_L{L}_K{K}_{op|mix_tag}_l0.0o{o}.npz
with keys {avg,hard,gauss}_{width,As,nu,tau,kept} plus nu_bare, nu_ortho,
nu_diag, has_diag, w_zero_frac, As0, tau0, omega_min_sup, K_actual, E, k_m.

Example (hx=0.9 current):
    python scripts/compute/nutau/filter_static.py --J=1.0 --hx=0.9 --hz=0.03 \
        --L=12 --K=0 --init_op=YZcur --pows_max=3 --n_width=60 --usegpu=True
"""

import os
import sys
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np

from sso.backend import to_cpu, xp as get_xp
from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.conventions import tau_static
from sso.nutau.filters import filter_sweep, to_energy_basis, weighted_mean, width_grid
from sso.nutau.models import (base_operator, full_basis, mixture_matrix, model_from_args,
                              modulated_density, operator_matrix)
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize


def main():
    args = parse_argv()
    fprint("arguments:", args)
    kind, model, name = model_from_args(args)
    if kind != "static":
        raise ValueError("use filter_floquet.py for Floquet models")
    usegpu = find_value(args, "usegpu", bool, False)
    xp = get_xp(usegpu)
    L = find_value(args, "L", int, 12)
    K = find_value(args, "K", int, 0)
    init_op = find_value(args, "init_op", str, "Zsum")
    mix = find_value(args, "mix", str, "")
    tag = find_value(args, "mix_tag", str, "MIX") if mix else init_op
    pows_max = find_value(args, "pows_max", int, 3) if K == 0 else -1
    n_width = find_value(args, "n_width", int, 40)
    o = pows_max + 1
    path = os.path.join(output_dir(f"{name}_StaticFilter_Data"),
                        f"filt_L{L}_K{K}_{tag}_l0.0o{o}.npz")
    if os.path.exists(path) and not find_value(args, "override", bool, False):
        fprint(f"{path} exists; use --override=True")
        return

    t0 = time()
    spec = diagonalize(kind, model, L)
    fprint(f"diagonalization {time() - t0:.1f}s")
    n = spec.n
    dE = spec.omega()
    dsq = dE ** 2
    basis = full_basis(L)
    if mix:
        O = mixture_matrix(mix, L, basis)
    elif init_op == "Hdens":
        O = operator_matrix(modulated_density(model, K, L), L, basis)
    else:
        O = operator_matrix(base_operator(init_op), L, basis)
    M0 = to_energy_basis(O, spec.V)

    kdiff = (spec.k[:, None] - spec.k[None, :]) % L
    w2 = np.abs(M0) ** 2
    sec = np.array([w2[kdiff == k].sum() for k in range(L)]) / w2.sum()
    K_actual = int(np.argmax(sec))
    if sec[K_actual] < 1 - 1e-9:
        raise RuntimeError(f"{tag} is not confined to one sector (best K={K_actual}: {sec[K_actual]:.3e})")
    if K_actual not in (K, (L - K) % L):
        raise RuntimeError(f"{tag} lies in K={K_actual}, neither K={K} nor its conjugate")
    fprint(f"operator in doubled-momentum sector K={K_actual} (requested {K})")

    Bop = EnergyBasisB(spec.V, L, usegpu)
    M0_d, dE_d, dsq_d = xp.asarray(M0), xp.asarray(dE), xp.asarray(dsq)
    nu_bare = Bop.nu(M0_d)
    q = spec.diag_ortho(o)
    diag = xp.arange(n)
    if q is not None:
        q_d = xp.asarray(q.astype(complex))
        M0_d[diag, diag] -= q_d @ (q_d.conj().T @ M0_d[diag, diag])
    nu_ortho = Bop.nu(M0_d)
    As0 = weighted_mean(M0_d, dsq_d, xp)
    fprint(f"nu: {nu_bare:.6e} (bare) -> {nu_ortho:.6e} (orthogonalised), tau0={tau_static(As0):.6g}")

    w2 = np.abs(to_cpu(M0_d)) ** 2
    support = (kdiff == K_actual) & (w2 > 1e-24 * w2.sum())
    zero_tol = 1e-9 * float(np.abs(dE).max())
    zero = support & (np.abs(dE) <= zero_tol)
    M_zero = xp.where(xp.asarray(zero), M0_d, 0.0)
    w_zero_frac = float(xp.real(xp.vdot(M_zero.reshape(-1), M_zero.reshape(-1)))) / float(w2.sum())
    has_diag = w_zero_frac > 1e-20
    nu_diag = Bop.nu(M_zero) if has_diag else np.nan
    abs_sup = np.abs(dE)[support]
    omega_min_sup = float(abs_sup[abs_sup > zero_tol].min())
    fprint(f"omega=0 weight {w_zero_frac:.6e} over {int(zero.sum())} elements"
           + (f", nu_diag={nu_diag:.6e}" if has_diag else " (no conserved component)")
           + f"; |omega|_min on support {omega_min_sup:.6e}")

    payload = dict(L=L, K=K, K_actual=K_actual, lam=0.0, model_name=name, init_op=tag,
                   mix_spec=mix, pows_max=pows_max, E=spec.values, k_m=spec.k, n_dim=n,
                   omega_min_sup=omega_min_sup, nu_bare=nu_bare, nu_ortho=nu_ortho,
                   nu_diag=nu_diag, has_diag=has_diag, w_zero_frac=w_zero_frac,
                   n_zero=int(zero.sum()), zero_tol=zero_tol, As0=As0, tau0=tau_static(As0))
    for kind_f in ("avg", "hard", "gauss"):
        grid = width_grid(abs_sup, n_width, kind_f, top=float(abs_sup.max()), gauss_margin=20.0,
                          zero_tol=zero_tol, keep_slowest=not has_diag)
        t0 = time()
        As, nu, kept = filter_sweep(M0_d, Bop, dE_d, dsq_d, kind_f, grid)
        fprint(f"'{kind_f}': {len(grid)} widths in [{grid[0]:.4g}, {grid[-1]:.4g}] ({time() - t0:.1f}s)")
        payload.update({f"{kind_f}_width": grid, f"{kind_f}_As": As, f"{kind_f}_nu": nu,
                        f"{kind_f}_tau": tau_static(As), f"{kind_f}_kept": kept})

    if has_diag:
        for kind_f in ("hard", "gauss"):
            rel = abs(payload[f"{kind_f}_nu"][0] / nu_diag - 1)
            fprint(f"'{kind_f}' narrowest nu vs omega=0 projection: rel {rel:.2e}")
            if rel > 1e-3:
                raise RuntimeError(f"'{kind_f}' does not converge to the omega = 0 projection")
    else:
        kept = np.nan_to_num(payload["hard_kept"])
        alive = np.flatnonzero(kept > 1e-12)
        if alive.size == 0:
            raise RuntimeError("the hard sweep never retains any weight")
        tau_end, bound = payload["hard_tau"][alive[0]], 1.0 / omega_min_sup
        fprint(f"'hard' narrowest live tau={tau_end:.6g} <= 1/|omega|_min={bound:.6g}")
        if tau_end > bound * (1 + 1e-6):
            raise RuntimeError("tau exceeds 1/|omega|_min: wrong support")
        live = payload["hard_tau"][alive]
        if np.any(np.diff(live) > 1e-9 * np.maximum(live[:-1], 1.0)):
            raise RuntimeError("tau is not monotonic along the hard sweep")
    np.savez(path, **payload)
    fprint(f"saved {path}")


if __name__ == "__main__":
    main()
