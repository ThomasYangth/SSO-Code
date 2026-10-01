r"""Pool seed-parallel Fourier-CV runs (:func:`sso.dqt.fourier.run_fourier_cv` outputs).

Seeds may have different s-grids provided each is a prefix of the longest one (the
production campaigns extended s_max in steps); each s is pooled over every seed that
reaches it.  Seeds from the earliest campaign revision lack the raw curves
(C_samples/m_samples/tau_h/tau_p); they enter the nu(tau) front but not the correlator
curves.

* :func:`combine_fourier`   ->  nu(tau) front with errors (the archived ``*_COMBINED.npz``).
* :func:`pooled_correlators` ->  C(t) and R(t) = ||H_k(t)||^2_RPS curves.
* :func:`g_rps_window`       ->  two-time RPS correlator |G_RPS(t1,t2)|.
"""

import glob
import os

import numpy as np

from sso.dqt.fourier import cv_estimate


def seed_files(directory, L, g=-1.05, h=0.5, n=1):
    """Sorted (lexicographic, as in production) list of seed NPZ files for one L."""
    pat = f"modhydro_fourier_cv_L{L}_n{n}_g{g:+.3g}_h{h:+.3g}_seed*.npz"
    return sorted(glob.glob(os.path.join(directory, pat)))


def combined_filename(L, g=-1.05, h=0.5, n=1):
    """Archived naming of the pooled front."""
    return f"modhydro_fourier_L{L}_n{n}_g{g:+.3g}_h{h:+.3g}_COMBINED.npz"


def _pad(A, n):
    A = np.atleast_2d(A)
    r, k = A.shape
    return np.hstack([A, np.full((r, n - k), np.nan)]) if k < n else A


def combine_fourier(files):
    """Pool seeds into one front.

    Per s: N1, N2 = pooled Haar means (SEM errors); control-variate coefficients
    c = G_HS^{-1}(mean V0, mean V2) from all pooled Haar samples; M = CV estimate over
    all pooled product states;  tau_obs = sqrt(N1/N2), nu = M/N1 with first-order error
    propagation  tau_err = tau/2 sqrt((dN1/N1)^2 + (dN2/N2)^2),
    nu_err = nu sqrt((dM/M)^2 + (dN1/N1)^2).
    Returns the dict stored as ``*_COMBINED.npz`` (svals, N1, N1_err, N2, N2_err, M,
    M_err, tau_obs, tau_err, nu, nu_err, cc, n_haar, n_prod).
    """
    if not files:
        raise ValueError("no seed files")
    svs = [np.load(f)["svals"] for f in files]
    sv = max(svs, key=len)
    ns = len(sv)
    cols = {k: [] for k in ("N1", "N2", "V0", "V2", "a", "b0", "b2")}
    Rg = GHS = None
    for f, s in zip(files, svs):
        if not np.allclose(s, sv[:len(s)]):
            raise ValueError(f"svals of {f} is not a prefix of the master grid")
        d = np.load(f, allow_pickle=True)
        if Rg is None:
            Rg, GHS = d["Rg"], d["GHS"]
        for k in ("N1", "N2", "V0", "V2", "a"):
            cols[k].append(_pad(d[f"{k}_samples"], ns))
        for k in ("b0", "b2"):
            cols[k].append(np.atleast_1d(d[f"{k}_samples"]))
    N1, N2, V0, V2, A = (np.vstack(cols[k]) for k in ("N1", "N2", "V0", "V2", "a"))
    B0, B2 = np.concatenate(cols["b0"]), np.concatenate(cols["b2"])
    out = {k: np.zeros(ns) for k in ("N1", "N1_err", "N2", "N2_err", "M", "M_err")}
    cc = np.zeros((ns, 2))
    nH = np.zeros(ns, int)
    nP = np.zeros(ns, int)
    sem = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    for i in range(ns):
        hm, pm = ~np.isnan(N1[:, i]), ~np.isnan(A[:, i])
        nH[i], nP[i] = hm.sum(), pm.sum()
        out["N1"][i], out["N1_err"][i] = N1[hm, i].mean(), sem(N1[hm, i])
        out["N2"][i], out["N2_err"][i] = N2[hm, i].mean(), sem(N2[hm, i])
        cc[i] = np.linalg.solve(GHS, np.array([V0[hm, i].mean(), V2[hm, i].mean()]))
        out["M"][i], out["M_err"][i] = cv_estimate(A[pm, i], B0[pm], B2[pm], cc[i], Rg)
    tau = np.sqrt(out["N1"] / out["N2"])
    tau_e = 0.5 * tau * np.sqrt((out["N1_err"] / out["N1"]) ** 2 + (out["N2_err"] / out["N2"]) ** 2)
    nu = out["M"] / out["N1"]
    nu_e = nu * np.sqrt((out["M_err"] / out["M"]) ** 2 + (out["N1_err"] / out["N1"]) ** 2)
    return dict(svals=sv, **out, tau_obs=tau, tau_err=tau_e, nu=nu, nu_err=nu_e, cc=cc,
                n_haar=nH, n_prod=nP)


def _with_curves(files):
    return [f for f in files if "m_samples" in np.load(f).files]


def pooled_correlators(files):
    """Pooled HS autocorrelator and RPS norm of the bare mode from stored raw curves.

    C(t) = Re mean_phi C_phi(t) on the longest tau_h grid (= Re <<H_k|H_k(t)>>);
    R(t) = mean_psi |m_psi(t)|^2 on the longest tau_p grid (= ||H_k(t)||^2_RPS), shorter
    grids being centred sub-grids; R is symmetrized over +-t and returned for t >= 0.
    Returns dict(t_C, C, t_R, R, nprod, nhaar).
    """
    fs = _with_curves(files)
    if not fs:
        raise ValueError("no seed files with raw curves")
    tp = max((np.load(f)["tau_p"] for f in fs), key=len)
    th = max((np.load(f)["tau_h"] for f in fs), key=len)
    T = len(tp)
    num, cnt = np.zeros(T), np.zeros(T)
    cnum, ccnt = np.zeros(len(th), complex), np.zeros(len(th))
    nprod = nhaar = 0
    for f in fs:
        d = np.load(f)
        m, c = d["m_samples"], d["C_samples"]
        off = (T - m.shape[1]) // 2
        num[off:off + m.shape[1]] += (np.abs(m) ** 2).sum(0)
        cnt[off:off + m.shape[1]] += m.shape[0]
        cnum[:c.shape[1]] += c.sum(0)
        ccnt[:c.shape[1]] += c.shape[0]
        nprod += m.shape[0]
        nhaar += c.shape[0]
    R = num / cnt
    i0 = T // 2
    Rp = 0.5 * (R[i0:] + R[i0::-1][:T - i0])
    return dict(t_R=tp[i0:], R=Rp, t_C=th, C=(cnum / ccnt).real, nprod=nprod, nhaar=nhaar)


def g_rps_window(files, tmax):
    """|G_RPS(t1,t2)| = |E_psi m_psi(t1)^* m_psi(t2)| on |t| <= tmax.

    Each (t1, t2) entry is averaged over the product states whose trajectory covers
    both times.  Returns (t, |G|, number of product states).
    """
    fs = _with_curves(files)
    tp = max((np.load(f)["tau_p"] for f in fs), key=len)
    T = len(tp)
    w = np.abs(tp) <= tmax
    rows, masks = [], []
    for f in fs:
        m = np.load(f)["m_samples"]
        off = (T - m.shape[1]) // 2
        r = np.zeros((m.shape[0], T), complex)
        k = np.zeros((m.shape[0], T))
        r[:, off:off + m.shape[1]] = m
        k[:, off:off + m.shape[1]] = 1.0
        rows.append(r[:, w])
        masks.append(k[:, w])
    Mw, Kw = np.vstack(rows), np.vstack(masks)
    cnt = Kw.T @ Kw
    cnt[cnt == 0] = np.nan
    return tp[w], np.abs((Mw.conj().T @ Mw) / cnt), Mw.shape[0]
