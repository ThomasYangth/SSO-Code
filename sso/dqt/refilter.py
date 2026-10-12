r"""Re-filter the stored raw DQT curves with another frequency filter (no new propagation).

Seeds that store raw curves (seeds >= 100; :func:`sso.dqt.combine._with_curves`) keep
every Haar correlator C_phi(tau) on tau in [0, T_c] and every product-state trajectory
m_psi(t) on [-T_p, T_p].  N1, N2, V0, V2 are linear in C and a = <psi|H_k(f)|psi> is
linear in m, so any filter f(Omega) whose time kernels are known gives a new front from
the same samples:

    N1 = int K C,  N2 = int (-K'') C,  V0 = int l C,  V2 = int (-l'') C,  a = int l m,

with l the single window (FT f) and K = l * l (FT f^2); the control variate and pooling
are those of :func:`sso.dqt.combine.pool_samples`.  Filters (kernels in
:mod:`sso.dqt.kernels`):

* 'gauss'    width s, f = e^{-s^2 Omega^2/4}.  A sample covers s if T_c >= nsigma s and
             T_p >= nsigma s/sqrt2 (the production rule): reproduces the stored samples.
* 'lorentz'  width m, f = m^2/(Omega^2 + m^2) = 1/(1 + theta Omega^2), theta = 1/m^2
             (the default filter; KrylovChain convention).  l = (m/2) e^{-m|t|},
             K = (m/4)(1 + m|t|) e^{-m|t|}; V2 = m^2 (C(0) - V0) exactly
             (Omega^2 f = m^2 - m^2 f: -l'' contains m^2 delta(t)).  tau ~ 1/m.
* 'lorentz2' width m, f = m^4/(Omega^2 + m^2)^2 (squared; filter comparisons only).
             tau ~ sqrt5/m.

Coverage.  Exponential tails: a sample covers m only if m T_c >= y_C and m T_p >= y_P
(defaults :data:`COVER`).  The neglected kernel mass (e^{-y_P} for the 'lorentz' window)
multiplies the late-time correlator and trajectories, which have decayed, so the actual
bias is far smaller: cutting the stored L = 16, 20, 23 records to m T_p = 8, 6, 5, 4 at
fixed m (tau = 11..29) changes nu by <= 5e-7, 1e-5, 1.5e-5, 6e-5 relative (all < 0.01 sigma).
'lorentz' uses y_P = 6, y_C = 6 sqrt2 (the T_c/T_p ratio); 'lorentz2' keeps the
conservative y_P = 14, y_C = 20.

Kink correction.  The Lorentzian kernels are smooth except at tau = 0, where odd
derivatives jump (:func:`kernel_jumps`).  The grid has a node there, so the trapezoid
rule on each half-line is corrected by the Euler-Maclaurin endpoint terms

    int F = trapz + sum_j B_2j/(2j)! dt^2j [F^(2j-1)(0+) - F^(2j-1)(0-)],   j = 1, 2, 3,

with the analytic kernel jumps and the derivatives of C or m at 0: C(0) per Haar sample,
C''(0) = -<<ad H_k|ad H_k>> and C''''(0) = <<ad^2 H_k|ad^2 H_k>> from the exact HS
Gram (scaled by the sample's C(0)/<<H_k|H_k>>), m''(0) = -<psi|ad_H^2 H_k|psi> from the
stored b2_samples (central difference if absent); m''''(0) is not available and set to 0.  Without
it, e.g., the 'lorentz2' V2 is overestimated by 71 % at m = 1.6, dt = 0.5 (ED, L = 10).

Each width is pooled over the samples that cover it, so the large-tau end of a
Lorentzian front rests on the seeds with the longest time records only.
"""

import numpy as np

from sso.dqt.combine import _with_curves, pool_samples
from sso.dqt.kernels import (gauss, lor2_K, lor2_win, lor_win, neg_gpp, neg_lor2_Kpp,
                             neg_lor2_win_pp)

COVER = {"lorentz": dict(y_C=8.5, y_P=6.0), "lorentz2": dict(y_C=20.0, y_P=14.0)}
TAU_PER_INV_M = {"lorentz": 1.0, "lorentz2": np.sqrt(5.0)}   # tau_obs * m, flat spectrum

_trapz = getattr(np, "trapezoid", None) or np.trapz


def filter_kernels(kind, w):
    """(K, -K'', l, -l'') as functions of tau for one width w; -l'' is None for 'lorentz'
    (V2 = m^2 (C(0) - V0) is used instead)."""
    if kind == "gauss":
        v = w * w
        return (lambda t: gauss(v, t), lambda t: neg_gpp(v, t),
                lambda t: gauss(0.5 * v, t), lambda t: neg_gpp(0.5 * v, t))
    if kind == "lorentz":
        return (lambda t: lor2_win(w, t), lambda t: neg_lor2_win_pp(w, t),
                lambda t: lor_win(w, t), None)
    if kind == "lorentz2":
        return (lambda t: lor2_K(w, t), lambda t: neg_lor2_Kpp(w, t),
                lambda t: lor2_win(w, t), lambda t: neg_lor2_win_pp(w, t))
    raise ValueError(f"unknown filter {kind!r}")


def covers(kind, w, T_c, T_p, nsigma=6.0, y_C=None, y_P=None):
    """(Haar ok, product ok): does a record of length T_c / T_p resolve width w?"""
    if kind == "gauss":
        eps = 1e-9
        return T_c >= nsigma * w - 0.5 - eps, T_p >= nsigma * w / np.sqrt(2.0) - 0.5 - eps
    y_C = COVER[kind]["y_C"] if y_C is None else y_C
    y_P = COVER[kind]["y_P"] if y_P is None else y_P
    return w * T_c >= y_C, w * T_p >= y_P


_EM = {1: 1.0 / 12.0, 3: -1.0 / 720.0, 5: 1.0 / 30240.0}   # B_2j/(2j)!, keyed by 2j-1


def kernel_jumps(kind, w):
    """Jumps k^(n)(0+) - k^(n)(0-), odd n <= 5, of the kernels (K, -K'', l, -l'')."""
    m = w
    win2 = {3: m ** 4, 5: 2.0 * m ** 6}                         # lor2_win
    win2pp = {1: -m ** 4, 3: -2.0 * m ** 6, 5: -3.0 * m ** 8}    # neg_lor2_win_pp
    if kind == "lorentz":
        return (win2, win2pp, {1: -m ** 2, 3: -m ** 4, 5: -m ** 6}, {})
    if kind == "lorentz2":
        return ({}, {5: -m ** 8}, win2, win2pp)
    return ({}, {}, {}, {})


def _em_correction(jk, dt, f0, f2, f4):
    """sum_j B_2j/(2j)! dt^2j J_{2j-1} for the product of a kernel (odd-derivative jumps
    jk at 0) and a smooth function with value f0, f2 = f''(0), f4 = f''''(0)."""
    j1, j3, j5 = (jk.get(n, 0.0) for n in (1, 3, 5))
    J = {1: j1 * f0, 3: j3 * f0 + 3 * j1 * f2, 5: j5 * f0 + 10 * j3 * f2 + 5 * j1 * f4}
    return sum(_EM[n] * dt ** (n + 1) * J[n] for n in (1, 3, 5))


def seed_samples(d, kind, widths, **cover_kw):
    """Per-sample (N1, N2, V0, V2) of shape (n_haar, n_w) and a of shape (n_prod, n_w).

    ``d`` is a loaded seed NPZ with raw curves; uncovered widths are NaN.
    """
    th, tp = d["tau_h"], d["tau_p"]
    tf = np.concatenate([-th[:0:-1], th])
    Cs, ms = d["C_samples"], d["m_samples"]
    Cf = np.concatenate([np.conj(Cs[:, :0:-1]), Cs], axis=1)
    nw = len(widths)
    dt = float(th[1] - th[0])
    G = d["GHS"]
    C0 = Cs[:, 0].real
    C2, C4 = -G[0, 1] * C0 / G[0, 0], G[1, 1] * C0 / G[0, 0]
    i0 = len(tp) // 2
    m0 = ms[:, i0]
    if "b2_samples" in getattr(d, "files", d):          # m''(0) = -<psi|ad_H^2 H_k|psi> exactly
        m2 = -np.atleast_1d(d["b2_samples"])
    else:
        m2 = (ms[:, i0 + 1] - 2 * m0 + ms[:, i0 - 1]) / dt ** 2
    N1, N2, V0, V2 = (np.full((Cs.shape[0], nw), np.nan) for _ in range(4))
    A = np.full((ms.shape[0], nw), np.nan, complex)
    for i, w in enumerate(widths):
        K, Kpp, l, lpp = filter_kernels(kind, w)
        jumps = kernel_jumps(kind, w)
        okH, okP = covers(kind, w, th[-1], tp[-1], **cover_kw)
        if okH:
            for arr, ker, jk in zip((N1, N2, V0, V2), (K, Kpp, l, lpp), jumps):
                if ker is None:
                    continue
                arr[:, i] = (_trapz(ker(tf)[None, :] * Cf, tf, axis=1).real
                             + _em_correction(jk, dt, C0, C2, C4))
            if lpp is None:                                   # 'lorentz': V2 = m^2 (C(0) - V0)
                V2[:, i] = w * w * (C0 - V0[:, i])
        if okP:
            A[:, i] = (_trapz(l(tp)[None, :] * ms, tp, axis=1)
                       + _em_correction(jumps[2], dt, m0, m2, 0.0))
    return N1, N2, V0, V2, A


def refilter_front(files, kind, widths, **cover_kw):
    """Pooled front (as :func:`sso.dqt.combine.pool_samples`) for one filter and width grid.

    Uses only seeds with raw curves.  Adds ``width``, ``kind``, ``n_seeds``.
    """
    fs = _with_curves(files)
    if not fs:
        raise ValueError("no seed files with raw curves")
    cols = {k: [] for k in ("N1", "N2", "V0", "V2", "a", "b0", "b2")}
    Rg = GHS = None
    for f in fs:
        d = np.load(f, allow_pickle=True)
        if Rg is None:
            Rg, GHS = d["Rg"], d["GHS"]
        for k, arr in zip(("N1", "N2", "V0", "V2", "a"), seed_samples(d, kind, widths, **cover_kw)):
            cols[k].append(arr)
        cols["b0"].append(np.atleast_1d(d["b0_samples"]))
        cols["b2"].append(np.atleast_1d(d["b2_samples"]))
    N1, N2, V0, V2, A = (np.vstack(cols[k]) for k in ("N1", "N2", "V0", "V2", "a"))
    B0, B2 = np.concatenate(cols["b0"]), np.concatenate(cols["b2"])
    out = pool_samples(N1, N2, V0, V2, A, B0, B2, GHS, Rg)
    return dict(width=np.asarray(widths, float), kind=kind, n_seeds=len(fs), **out)


def default_widths(files, kind, tau_min=1.0, n_width=40, **cover_kw):
    """Width grid used by the driver and the extractor for a set of seed files.

    lorentz / lorentz2: m = TAU_PER_INV_M[kind] / tau_target with tau_target on n_width
    log-spaced values from tau_min up to the largest width that the longest records cover
    (m T_p >= y_P, m T_c >= y_C; defaults :data:`COVER`); gauss: the production s-grid up
    to sqrt2 T_p / 6.  Returns (widths, cover kwargs for refilter_front).
    """
    from sso.dqt.fourier import production_svals
    fs = _with_curves(files)
    T_p = max(float(np.load(f)["tau_p"][-1]) for f in fs)
    T_c = max(float(np.load(f)["tau_h"][-1]) for f in fs)
    if kind in COVER:
        cover = {**COVER[kind], **{k: v for k, v in cover_kw.items() if v is not None}}
        m_min = max(cover["y_P"] / T_p, cover["y_C"] / T_c)
        c = TAU_PER_INV_M[kind]
        tt = np.geomspace(tau_min, c / m_min, n_width)
        return c / tt, cover
    return production_svals(np.sqrt(2.0) * T_p / 6.0), {}


def refilter_filename(kind, L, g=-1.05, h=0.5, n=1):
    """modhydro_{kind}_L{L}_n{n}_g{g}_h{h}_REFILTER.npz."""
    return f"modhydro_{kind}_L{L}_n{n}_g{g:+.3g}_h{h:+.3g}_REFILTER.npz"
