r"""Explicit slow operators: low-pass filters applied to a fixed base operator.

An operator with E-basis matrix M_mn evolves as M_mn(t) = e^{i omega_mn t} M_mn,
omega_mn = E_m - E_n (static) or the quasi-energy difference phi_m - phi_n
(Floquet, per period).  A filter multiplies every element by f(omega_mn):

    'gauss'  exp(-omega^2 / (2 s^2))           Gaussian time window of width 1/s
    'hard'   1[|omega| <= w]                    brick wall in frequency
    'avg'    static:  (1/T) int_0^T e^{i omega t} dt = e^{i omega T/2} sinc(omega T / 2 pi)
             Floquet: (1/N) sum_{t<N} e^{i omega t}  = e^{i x (N-1)} sin(N x)/(N sin x),
                      x = omega/2, evaluated as sinc(N x/pi)/sinc(x/pi) (exact at x -> 0)

For Floquet 'gauss'/'hard' use |omega| wrapped into [0, pi]; 'avg' is periodic in
omega and uses the raw phase difference.  The N-period average is therefore a
closed-form filter: no time stepping, N = 10^6 costs the same as N = 2.

Each filtered operator O_f is reported as (As, nu, kept) with
As = <O_f|Delta^2|O_f>/<O_f|O_f>, nu its RPS weight and kept = |O_f|^2/|O|^2.
"""

import numpy as np

from sso.backend import xp as get_xp


def filter_values(kind, omega, width, xp, floquet=False):
    """f(omega) for one filter width (s, w, T or N)."""
    if kind == "gauss":
        return xp.exp(-(omega ** 2) / (2.0 * width ** 2)).astype(np.complex128)
    if kind == "hard":
        return (xp.abs(omega) <= width).astype(np.complex128)
    if kind == "avg" and floquet:
        x = 0.5 * omega
        return xp.exp(1j * x * (width - 1)) * (xp.sinc(width * x / np.pi) / xp.sinc(x / np.pi))
    if kind == "avg":
        return xp.exp(0.5j * omega * width) * xp.sinc(omega * width / (2.0 * np.pi))
    raise ValueError(f"unknown filter {kind!r}")


def width_grid(abs_omega, n_width, kind, top, gauss_margin, zero_tol, keep_slowest=False):
    """Log-spaced sweep from the narrow (tau -> inf) end to the transparent end.

    ``abs_omega`` are the |omega| the sweep should resolve (the static driver
    passes the occupied support of the operator, the Floquet driver all pairs);
    w_min is their smallest value above ``zero_tol``.  The narrow end w_lo is
      hard, avg   w_min / 2
      gauss       w_min / gauss_margin  (exp(-w^2/2s^2) only kills w_min for s << w_min)
      keep_slowest=True (all kinds): w_min (1 + 1e-6) -- for an operator with no
                  omega = 0 component, keep exactly its slowest element
    and the sweep is
      hard   geomspace(w_lo, top)
      gauss  geomspace(w_lo, 10 top)        (transparent only for s >> top)
      avg    geomspace(0.1/top, 20/w_lo)    (a time; large = narrow)
    """
    off = abs_omega[abs_omega > zero_tol]
    w_min = float(off.min()) if off.size else 2e-6
    if keep_slowest:
        lo = w_min * (1.0 + 1e-6)
    elif kind == "gauss":
        lo = w_min / gauss_margin
    else:
        lo = 0.5 * w_min
    if kind == "hard":
        return np.geomspace(lo, top, n_width)
    if kind == "gauss":
        return np.geomspace(lo, 10.0 * top, n_width)
    if kind == "avg":
        return np.geomspace(0.1 / top, 20.0 / lo, n_width)
    raise ValueError(f"unknown filter {kind!r}")


def sample_periods(N_max, n_sample):
    """Log-spaced distinct integer period counts N in [1, N_max]."""
    N = np.unique(np.round(np.geomspace(1.0, float(N_max), n_sample)).astype(np.int64))
    return N[N >= 1]


def weighted_mean(M, weight, xp):
    """sum |M|^2 weight / sum |M|^2 (0 for M = 0)."""
    w2 = xp.real(M.conj() * M)
    s = float(w2.sum())
    return float((w2 * weight).sum()) / s if s > 0 else 0.0


def filter_sweep(M0, Bop, omega, dsq, kind, grid, floquet=False):
    """(As, nu, kept) of f(omega; w) * M0 for every width w in ``grid``.

    M0, omega, dsq are (n, n) arrays on Bop's device; ``Bop`` is the
    :class:`sso.nutau.pauli.EnergyBasisB` of the same eigenbasis.
    """
    xp = Bop.xp
    As, nu, kept = (np.zeros(len(grid)) for _ in range(3))
    tot = float(xp.real(xp.vdot(M0.reshape(-1), M0.reshape(-1))))
    for i, w in enumerate(grid):
        M = M0 * filter_values(kind, omega, w, xp, floquet=floquet)
        w2 = float(xp.real(xp.vdot(M.reshape(-1), M.reshape(-1))))
        kept[i] = w2 / tot
        if w2 <= 0:
            As[i] = nu[i] = np.nan
            continue
        As[i] = weighted_mean(M, dsq, xp)
        nu[i] = Bop.nu(M)
    return As, nu, kept


def to_energy_basis(O_phys, V, usegpu=False):
    """E-basis matrix V^dag O V of a physical-basis operator (on the device)."""
    xp = get_xp(usegpu)
    V = xp.asarray(V)
    return V.conj().T @ xp.asarray(O_phys) @ V
