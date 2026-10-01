"""Reader for thermodynamic-limit (translation-invariant ansatz) results.

The Julia package ``julia/TIAnsatz`` writes one whitespace table per
(support span M, momentum q) or per Floquet run::

    # <key=value metadata ...>
    # theta  nu  sigma2  tau  nu_bar  accepted
    <rows>

``theta`` is the Lorentzian penalty, ``nu`` the RPS weight of the optimal
operator, ``sigma2`` its (static: energy; Floquet: 4 sin^2) detuning variance,
``tau`` the Julia-side slowness estimate, ``nu_bar`` the Lanczos error bar and
``accepted`` a convergence flag.
"""

import numpy as np

COLUMNS = ("theta", "nu", "sigma2", "tau", "nu_bar", "accepted")


def read_ti_table(path):
    """Return ``(meta, cols)``: header ``key=value`` dict and a dict of column arrays."""
    meta = {}
    with open(path) as f:
        first = f.readline()
    for tok in first.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            meta[k] = v
    arr = np.atleast_2d(np.loadtxt(path, comments="#"))
    cols = {name: arr[:, i] for i, name in enumerate(COLUMNS[: arr.shape[1]])}
    return meta, cols


# --------------------------------------------------------------------------------------
# Finite-momentum q scans (fig3panel (c)):  Delta nu(q, tau) = nu0_M(tau) - nu_M(q, tau)
# --------------------------------------------------------------------------------------
# Files ``nutau_M{M}_q{q:.5f -> 'p'}.dat`` (Kim-Huse J=1, g=0.905, h=0.809, infinite chain,
# ansatz support span M, momentum q in rad/site) from the julia/TIAnsatz q-scan driver
# (ham_mf_driver_q_gpures.jl).  q = 0 files are the baselines (not energy-projected).

MSTAB_MS = (8, 9, 10, 11)      # spans compared for M-stability; last = representative
MSTAB_TOL = (0.05, 0.40)       # log-ramp of the successive-M change r: opaque <= lo, faint >= hi
MSTAB_ALPHA_MIN = 0.08
MSTAB_MIN_COVER = 3            # spans that must cover a tau before stability is assessed


def q_tag(q):
    """0.002 -> '0p00200' (file names use 5 decimals)."""
    return ("%.5f" % q).replace(".", "p")


def load_q_curve(directory, M, q):
    """(tau, nu) of one q-scan table sorted by tau, or None if missing / < 2 rows."""
    import os
    f = os.path.join(directory, f"nutau_M{M}_q{q_tag(q)}.dat")
    if not os.path.isfile(f):
        return None
    _, c = read_ti_table(f)
    if len(c["tau"]) < 2:
        return None
    o = np.argsort(c["tau"])
    return c["tau"][o], c["nu"][o]


def discover_qs(directory, Ms=MSTAB_MS):
    """Sorted q > 0 present for every M in ``Ms`` (unsigned Kim-Huse files only)."""
    import glob
    import os
    import re
    per = []
    for M in Ms:
        qs = set()
        for f in glob.glob(os.path.join(directory, f"nutau_M{M}_q*.dat")):
            m = re.match(rf"nutau_M{M}_q([0-9p]+)\.dat$", os.path.basename(f))
            if m:
                qs.add(float(m.group(1).replace("p", ".")))
        per.append(qs)
    return sorted(q for q in set.intersection(*per) if q > 0)


def dnu_curve(directory, M, q):
    """(tau, nu(q) - nu0_M(tau)) on the baseline's tau range.

    nu0_M is the q=0 table of the same M interpolated by PCHIP in log(tau) (shape
    preserving; removes ~1e-5 linear-interpolation noise at small tau).
    """
    from scipy.interpolate import PchipInterpolator
    c = load_q_curve(directory, M, q)
    if c is None:
        return None
    t0, n0 = load_q_curve(directory, M, 0.0)
    spl = PchipInterpolator(np.log(t0), n0, extrapolate=False)
    tau, nu = c
    dn = nu - spl(np.log(tau))
    m = np.isfinite(dn) & (tau >= t0.min()) & (tau <= t0.max())
    return tau[m], dn[m]


def _roll_median(x, w=3):
    n, hw = len(x), w // 2
    return np.array([np.median(x[max(0, i - hw):min(n, i + hw + 1)]) for i in range(n)])


def mstab_family(directory, Ms=MSTAB_MS, min_cover=MSTAB_MIN_COVER):
    """Per-q representative curve (M = Ms[-1]) with its M-stability metric.

    For each q: trep, dnrep = dnu_curve at the largest M; r = |dn_M - dn_{M-1}| / |dn_M|
    (M-1 interpolated linearly in log tau, never extrapolated; non-finite -> 10;
    3-point rolling median); valid = at least ``min_cover`` spans cover tau and M-1 does.
    Returns {q: dict(trep, dnrep, r, valid)} for the q's of :func:`discover_qs`.
    """
    Mrep = Ms[-1]
    fam = {}
    for q in discover_qs(directory, Ms):
        perM = {M: dnu_curve(directory, M, q) for M in Ms}
        perM = {M: v for M, v in perM.items() if v is not None}
        if Mrep not in perM or (Mrep - 1) not in perM:
            continue
        trep, dnrep = perM[Mrep]
        ltr = np.log(trep)

        def interp(M):
            t, dn = perM[M]
            di = np.interp(ltr, np.log(t), dn)
            di[(ltr < np.log(t.min())) | (ltr > np.log(t.max()))] = np.nan
            return di

        dprev = interp(Mrep - 1)
        cover = np.sum([np.isfinite(interp(M)) for M in perM], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(dnrep - dprev) / np.maximum(np.abs(dnrep), 1e-9)
        r = _roll_median(np.where(np.isfinite(r), r, 10.0), w=3)
        fam[q] = dict(trep=trep, dnrep=dnrep, r=r, valid=(cover >= min_cover) & np.isfinite(dprev))
    return fam


def log_ramp(x, lo, hi):
    """Brightness in [0,1] linear in log10 x: 1 for x <= lo, 0 for x >= hi."""
    lx = np.log10(np.clip(x, 1e-6, 1e3))
    return np.clip((np.log10(hi) - lx) / (np.log10(hi) - np.log10(lo)), 0.0, 1.0)


def dnu_tq2(F, q, tol=MSTAB_TOL, alpha_min=MSTAB_ALPHA_MIN):
    """Points of Delta nu = nu0 - nu(q) > 0 vs tau q^2 with M-stability opacity.

    Returns (x = tau q^2, y = Delta nu, alpha) on the physical branch (nu(q) < nu0);
    alpha = alpha_min + (1 - alpha_min) b_M, b_M = log_ramp(r) on valid points, else 0.
    """
    t, dn = F["trep"], F["dnrep"]
    x, y = t * q ** 2, -dn
    keep = (dn < 0) & np.isfinite(y)
    bM = np.where(F["valid"][keep], log_ramp(F["r"][keep], *tol), 0.0)
    return x[keep], y[keep], alpha_min + (1.0 - alpha_min) * bM
