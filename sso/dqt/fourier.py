r"""Fourier-correlator DQT estimator of the s-windowed energy-density mode (production).

Target (normalized HS  <<A|B>> = 2^{-L} tr A^dag B,  H_k(s) = exp(-s^2 ad_H^2/4) H_k):

    N1(s) = <<H_k(s)|H_k(s)>>,   N2(s) = <<H_k(s)|ad_H^2|H_k(s)>>,
    M(s)  = E_psi |<psi|H_k(s)|psi>|^2   (random product states: the RPS norm),
    tau_obs = sqrt(N1/N2),   nu = M/N1.

N1, N2 (Haar DQT).  For a normalized Haar state phi,
    C(tau) = <gamma(tau)|H_k|alpha(tau)>,  alpha = e^{-iH tau} phi,  gamma = e^{-iH tau} H_k phi
estimates C(k,tau) = <<H_k|H_k(tau)>> on tau = 0, dt, ..., T_c (T_c = nsigma s_max);
C(-tau) = C(tau)^*.  Then N1 = int G_{s^2} C,  N2 = int (-G''_{s^2}) C (trapezoid on the
symmetric grid; kernels in :mod:`sso.dqt.kernels`).  The same curve also gives the
projections V0 = <<H_k|H_k(s)>> and V2 = <<ad^2 H_k|H_k(s)>> used below.

M (product-state trajectories).  m_psi(t) = <psi(t)|H_k|psi(t)> on t in [-T_p, T_p]
(T_p = nsigma s_max / sqrt2, i.e. nsigma standard deviations of g_s), and
a_s(psi) = int g_s m_psi = <psi|H_k(s)|psi>.

Control variate.  Project H_k(s) on span{H_k, ad_H^2 H_k} in the HS metric,
c(s) = G_HS^{-1} (V0, V2) (the odd rung ad_H H_k has zero overlap by time-reversal
parity), with exact Grams from :mod:`sso.dqt.pauli`.  The projection's expectation
p(psi) = c0 <psi|H_k|psi> + c2 <psi|ad^2 H_k|psi> is computed exactly per state by
sparse mat-vecs and its RPS norm E|p|^2 = c^T R c analytically, so

    M = mean_psi( |a_s|^2 - |p|^2 ) + c^T R c

is unbiased with much lower variance at large s (where H_k(s) -> projection).
(c is itself estimated from the Haar samples; any error in c only changes the variance,
not the mean, because E|p|^2 is exact for whatever c is used.)

Caveat: the N2 kernel has width ~s, so the dt-grid quadrature breaks down for s <~ dt.
At the production dt = 0.5 the quadrature bias of N2 (exact correlator, L = 8) is
3e-5 at s = 0.7 (the production s_min) but 4e-4 at s = 0.5 and 60 % at s = 0.3; use a
smaller dt, ED, or :mod:`sso.dqt.chebmoment` for s below ~dt.
"""

import time

import numpy as np

from sso import backend
from sso.cli import fprint
from sso.dqt import pauli
from sso.dqt.kernels import gauss, gwin, neg_gpp
from sso.dqt.model import (haar_state, mfim_hk, random_product_state, spectral_bracket,
                           to_backend)
from sso.dqt.propagators import Propagator

_trapz = getattr(np, "trapezoid", None) or np.trapz
_CV_RUNGS = [0, 2]           # {H_k, ad^2 H_k}


def production_svals(smax):
    """Production s grid: 0.7, 1.4, 2, 3, 4, 6, 8, 10, 12, 14, then 16, 18, ..., smax (step 2).

    Each campaign's grid is a prefix of the next one's, which :mod:`sso.dqt.combine` relies on.
    """
    head = [0.7, 1.4, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    return np.array(head + [float(s) for s in range(16, int(smax) + 1, 2)])


def time_grids(svals, dt, nsigma):
    """(tau_h, tau_f, tau_p): correlator grid tau>=0, its symmetric extension, product grid."""
    smax = float(np.max(svals))
    nH = int(np.ceil(nsigma * smax / dt))
    tau_h = np.arange(nH + 1) * dt
    tau_f = np.concatenate([-tau_h[:0:-1], tau_h])
    nP = int(np.ceil(nsigma * smax / np.sqrt(2.0) / dt))
    tau_p = np.arange(-nP, nP + 1) * dt
    return tau_h, tau_f, tau_p


def correlator_integrals(C, tau_f, svals):
    """(N1, N2, V0, V2) per s from one correlator curve C(tau >= 0)."""
    Cf = np.concatenate([np.conj(C[:0:-1]), C])
    out = []
    for kern, var in ((gauss, 1.0), (neg_gpp, 1.0), (gauss, 0.5), (neg_gpp, 0.5)):
        out.append(np.array([_trapz(kern(var * s ** 2, tau_f) * Cf, tau_f).real for s in svals]))
    return tuple(out)


def window_amplitudes(m, tau_p, svals):
    """a_s = int g_s(t) m(t) dt for every s (complex)."""
    return np.array([_trapz(gwin(s, tau_p) * m, tau_p) for s in svals])


def cv_grams(L, J, g, h, n):
    """(Rg, GHS): real 2x2 RPS and HS Grams of {H_k, ad^2 H_k}."""
    ix = np.ix_(_CV_RUNGS, _CV_RUNGS)
    return (pauli.rps_gram(L, J, g, h, n, 2)[ix].real,
            pauli.hs_gram(L, J, g, h, n, 2)[ix].real)


def cv_estimate(a, b0, b2, c, Rg):
    """Control-variate M and its standard error for one s.

    a: windowed amplitudes a_s(psi); b0, b2: <psi|H_k|psi>, <psi|ad^2 H_k|psi>;
    c: projection coefficients (c0, c2); Rg: RPS Gram of {H_k, ad^2 H_k}.
    """
    res = np.abs(a) ** 2 - np.abs(c[0] * b0 + c[1] * b2) ** 2
    err = res.std(ddof=1) / np.sqrt(len(res)) if len(res) > 1 else 0.0
    return res.mean() + float(c @ Rg @ c), err


def bare_expectations(H, Hk, psi, usegpu=False):
    """(<psi|H_k|psi>, <psi|ad_H^2 H_k|psi>) with ad^2 H_k = H^2 H_k + H_k H^2 - 2 H H_k H."""
    xp = backend.xp(usegpu)
    Hp = H.dot(psi)
    H2p = H.dot(Hp)
    b0 = complex(backend.to_cpu(xp.vdot(psi, Hk.dot(psi))))
    b2 = complex(backend.to_cpu(xp.vdot(H2p, Hk.dot(psi)) + xp.vdot(psi, Hk.dot(H2p))
                                - 2 * xp.vdot(Hp, Hk.dot(Hp))))
    return b0, b2


def haar_correlator(prop, Hk, phi, ntau, usegpu=False):
    """C(tau_j) = <gamma(tau_j)|H_k|alpha(tau_j)>, j = 0..ntau-1, steps of prop.dt."""
    xp = backend.xp(usegpu)
    alpha, gamma = phi.copy(), Hk.dot(phi)
    C = np.empty(ntau, complex)
    for it in range(ntau):
        C[it] = complex(backend.to_cpu(xp.vdot(gamma, Hk.dot(alpha))))
        if it < ntau - 1:
            alpha, gamma = prop.step(alpha), prop.step(gamma)
    return C


def product_trajectory(prop, Hk, psi0, tau_p, usegpu=False):
    """m(t) = <psi(t)|H_k|psi(t)> on the symmetric grid tau_p (starts at e^{+iH T_p} psi0)."""
    xp = backend.xp(usegpu)
    psi = prop.evolve(psi0, -tau_p[-1])
    m = np.empty(len(tau_p), complex)
    for it in range(len(tau_p)):
        m[it] = complex(backend.to_cpu(xp.vdot(psi, Hk.dot(psi))))
        if it < len(tau_p) - 1:
            psi = prop.step(psi)
    return m


def run_fourier_cv(L, svals, nhaar, nprod, seed, J=1.0, g=-1.05, h=0.5, n=1, dt=0.5,
                   nsigma=6.0, engine="chebyshev", usegpu=False, log=fprint):
    """One seed of the production estimator; returns the dict saved as the seed NPZ.

    Keys (identical to the archived ``modhydro_fourier_cv_*_seed*.npz``): svals, N1,
    N1_err, N2, N2_err, M, M_err (control variate), M_direct, M_direct_err (plain mean,
    diagnostic), cc, Rg, GHS, tau_obs, nu, per-sample N1/N2/V0/V2/a/b0/b2_samples,
    raw curves tau_h, C_samples (Haar correlators) and tau_p, m_samples (trajectories),
    and ``parameters``.
    """
    svals = np.asarray(svals, float)
    ns = len(svals)
    rng = np.random.RandomState(seed)
    D = 2 ** L
    H_cpu = mfim_hk(L, J, g, h, 0)
    bracket = spectral_bracket(H_cpu)
    H = to_backend(H_cpu, usegpu)
    del H_cpu
    Hk = to_backend(mfim_hk(L, J, g, h, n), usegpu)
    prop = Propagator(H, dt, bracket, engine, usegpu)
    tau_h, tau_f, tau_p = time_grids(svals, dt, nsigma)
    Rg, GHS = cv_grams(L, J, g, h, n)
    log(f"[fourier_cv] L={L} n={n} J={J} g={g} h={h} engine={engine} gpu={usegpu} seed={seed}")
    log(f"  bracket=[{bracket[0]:.2f},{bracket[1]:.2f}] dt={dt} order(dt)={prop.order} "
        f"corr pts={len(tau_h)} prod pts={len(tau_p)} nhaar={nhaar} nprod={nprod} "
        f"svals={svals.tolist()}")

    t0 = time.time()
    N1_s, N2_s, V0_s, V2_s = (np.zeros((nhaar, ns)) for _ in range(4))
    C_s = np.zeros((nhaar, len(tau_h)), complex)
    for r in range(nhaar):
        C_s[r] = haar_correlator(prop, Hk, haar_state(D, rng, usegpu), len(tau_h), usegpu)
        N1_s[r], N2_s[r], V0_s[r], V2_s[r] = correlator_integrals(C_s[r], tau_f, svals)
        log(f"  haar {r + 1}/{nhaar}  {(time.time() - t0) / (r + 1):.2f}s/sample")
    t_haar = (time.time() - t0) / max(nhaar, 1)
    V0, V2 = V0_s.mean(0), V2_s.mean(0)
    cc = np.array([np.linalg.solve(GHS, np.array([V0[i], V2[i]])) for i in range(ns)])

    t1 = time.time()
    a_s = np.zeros((nprod, ns), complex)
    b0_s, b2_s = np.zeros(nprod, complex), np.zeros(nprod, complex)
    m_s = np.zeros((nprod, len(tau_p)), complex)
    for r in range(nprod):
        psi0 = random_product_state(L, rng, usegpu)
        b0_s[r], b2_s[r] = bare_expectations(H, Hk, psi0, usegpu)
        m_s[r] = product_trajectory(prop, Hk, psi0, tau_p, usegpu)
        a_s[r] = window_amplitudes(m_s[r], tau_p, svals)
        log(f"  prod {r + 1}/{nprod}  {(time.time() - t1) / (r + 1):.2f}s/state")
    t_prod = (time.time() - t1) / max(nprod, 1)

    def mean_sem(x):
        return x.mean(0), (x.std(0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else x[0] * 0)

    N1, N1e = mean_sem(N1_s)
    N2, N2e = mean_sem(N2_s)
    Mdir, Mdir_e = mean_sem(np.abs(a_s) ** 2)
    M, Me = np.empty(ns), np.empty(ns)
    for i in range(ns):
        M[i], Me[i] = cv_estimate(a_s[:, i], b0_s, b2_s, cc[i], Rg)
    tau_obs, nu = np.sqrt(N1 / N2), M / N1
    for i, s in enumerate(svals):
        log(f"  s={s:6.2f}  N1={N1[i]:.4e} N2={N2[i]:.4e} M={M[i]:.4e}+-{Me[i]:.1e} "
            f"tau={tau_obs[i]:.3f} nu={nu[i]:.4f}")
    params = dict(L=L, J=J, g=g, h=h, n_out=n, D=D, dt=dt, nsigma=nsigma, NHAAR=nhaar,
                  NPROD=nprod, seed=seed, engine=engine,
                  backend="cupy" if usegpu else "numpy", t_haar=t_haar, t_prod=t_prod,
                  method="fourier_cv_kry2", conv="normalizedHS_s")
    return dict(svals=svals, N1=N1, N1_err=N1e, N2=N2, N2_err=N2e, M=M, M_err=Me,
                M_direct=Mdir, M_direct_err=Mdir_e, cc=cc, Rg=Rg, GHS=GHS,
                tau_obs=tau_obs, nu=nu, N1_samples=N1_s, N2_samples=N2_s,
                V0_samples=V0_s, V2_samples=V2_s, a_samples=a_s, b0_samples=b0_s,
                b2_samples=b2_s, tau_h=tau_h, tau_p=tau_p, C_samples=C_s, m_samples=m_s,
                parameters=params)


def seed_filename(L, g, h, seed, n=1):
    """Archived naming: modhydro_fourier_cv_L{L}_n{n}_g{g:+.3g}_h{h:+.3g}_seed{seed}.npz."""
    return f"modhydro_fourier_cv_L{L}_n{n}_g{g:+.3g}_h{h:+.3g}_seed{seed}.npz"
