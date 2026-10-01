"""Frontier curves for plotting: grid -> (tau, nu), and the sector-envelope assembly.

The global frontier over doubled-momentum sectors is, at each theta, the sector
whose M_K has the larger top eigenvalue W (comparable across K) -- decided by
W, not by nu: at a hand-over the winner can have lower nu but larger tau.  Where
the winner changes, the tau gap is filled by the cross-sector superposition
arc; a within-sector crossing (two eigenoperators of one K exchanging) is
filled by its own arc, drawn in whichever branch of that sector holds theta*.
"""

import numpy as np

from sso.nutau.conventions import tau_of


def grid_curve(grid, kind="static"):
    """(theta, tau, nu, W) of a stored grid: finite theta with As > 0, sorted by theta."""
    th = np.asarray(grid["theta"], float)
    As = np.asarray(grid["As"], float)
    ok = np.isfinite(th) & (As > 0)
    o = np.argsort(th[ok])
    return (th[ok][o], tau_of(As[ok][o], kind), np.asarray(grid["Bs"], float)[ok][o],
            np.asarray(grid["W"], float)[ok][o])


def tau_sorted(tau, nu):
    """(tau, nu) sorted by tau."""
    o = np.argsort(tau)
    return np.asarray(tau)[o], np.asarray(nu)[o]


def break_gaps(t, n, ratio=2.5):
    """Insert NaN between consecutive points whose tau grows by more than ``ratio``."""
    ot, on = [t[0]], [n[0]]
    for i in range(1, len(t)):
        if t[i] > ratio * t[i - 1]:
            ot.append(np.nan)
            on.append(np.nan)
        ot.append(t[i])
        on.append(n[i])
    return np.array(ot), np.array(on)


def _clip_arc(tau_a, nu_a, tau_max):
    """Arc points with tau <= tau_max, plus the arc interpolated at tau_max if it continues."""
    m = tau_a <= tau_max
    ta, na = list(tau_a[m]), list(nu_a[m])
    if (~m).any() and m.any():
        o = np.argsort(tau_a)
        ta.append(tau_max)
        na.append(float(np.exp(np.interp(np.log(tau_max), np.log(tau_a[o]), np.log(nu_a[o])))))
    return ta, na


def sector_frontier(g0, g1, cross_arcs, within_arcs=(), tau_max=np.inf, tol=1e-8):
    """Assemble the two-sector (K = 0, 1) envelope.

    g0, g1       stored grids (dicts theta, As, Bs, W) of K = 0 and K = 1
    cross_arcs   [(tau, nu, theta*)] K0 <-> K1 superposition arcs
    within_arcs  [(tau, nu, theta*)] arcs of an internal K = 0 crossing
    tol          relative W tie tolerance; ties go to K = 0

    Returns dict with
      lines   polylines [dict(x, y, sector=0|1|'mix', solid=bool)]: the winning
              (solid) and losing (faded) branch of each sector, NaN-broken where
              a within-sector arc spans the gap, the arcs bridged to the
              neighbouring winning grid points, clipped at tau_max
      union   (tau, nu) of all winning grid points plus every point of the arcs
              that bridge a hand-over (tau <= tau_max), sorted by tau
      nu_inf0 nu of the largest-theta K = 0 point (the tau -> inf plateau)
      win0, win1  winner masks on the two grids, and th0/t0/n0, th1/t1/n1
    """
    th0, t0, n0, W0 = grid_curve(g0)
    th1, t1, n1, W1 = grid_curve(g1)
    win0 = W0 >= np.interp(np.log(th0), np.log(th1), W1) * (1 - tol)
    win1 = W1 > np.interp(np.log(th1), np.log(th0), W0) * (1 + tol)
    cut0, cut1 = t0 <= tau_max, t1 <= tau_max
    o0, o1 = np.argsort(t0), np.argsort(t1)
    lines = []

    def sector_line(t, n, mask, o, sector, solid):
        lines.append(dict(x=t[o], y=np.where(mask, n, np.nan)[o], sector=sector, solid=solid))

    sector_line(t1, n1, win1 & cut1, o1, 1, True)
    ww_t = np.concatenate([t1[win1 & cut1], t0[win0 & cut0]])
    ww_n = np.concatenate([n1[win1 & cut1], n0[win0 & cut0]])

    def winner_at(theta):
        return 0 if win0[np.argmin(np.abs(np.log(th0) - np.log(theta)))] else 1

    used = []
    for tau_a, nu_a, ath in cross_arcs:
        lo, hi = th0[th0 < ath], th0[th0 > ath]
        if lo.size and hi.size and winner_at(lo.max()) == winner_at(hi.min()):
            continue                                   # no hand-over: a point arc
        used.append((tau_a, nu_a))
        seg_t, seg_n = _clip_arc(tau_a, nu_a, tau_max)
        pre, post = ww_t <= min(seg_t), ww_t >= max(seg_t)
        if pre.any():
            i = np.argmax(ww_t[pre])
            seg_t, seg_n = [ww_t[pre][i]] + seg_t, [ww_n[pre][i]] + seg_n
        if post.any():
            i = np.argmin(ww_t[post])
            seg_t, seg_n = seg_t + [ww_t[post][i]], seg_n + [ww_n[post][i]]
        lines.append(dict(x=np.array(seg_t), y=np.array(seg_n), sector="mix", solid=True))
    sector_line(t1, n1, (~win1) & cut1, o1, 1, False)

    for mask, solid in ((win0, True), (~win0, False)):
        sel = o0[(mask & cut0)[o0]]
        bt, bn, bth = t0[sel], n0[sel], th0[sel]
        xs, ys = [], []
        for j in range(len(sel)):
            xs.append(bt[j])
            ys.append(bn[j])
            if j < len(sel) - 1 and any(min(bth[j], bth[j + 1]) < a[2] < max(bth[j], bth[j + 1])
                                        for a in within_arcs):
                xs.append(np.nan)
                ys.append(np.nan)
        lines.append(dict(x=np.array(xs), y=np.array(ys), sector=0, solid=solid))
        for tau_a, nu_a, ath in within_arcs:
            if not mask[np.argmin(np.abs(np.log(th0) - np.log(ath)))]:
                continue
            ta, na = _clip_arc(tau_a, nu_a, tau_max)
            below, above = bth < ath, bth > ath
            if below.any():
                k = np.argmax(bth[below])
                ta, na = [bt[below][k]] + ta, [bn[below][k]] + na
            if above.any():
                k = np.argmin(bth[above])
                ta, na = ta + [bt[above][k]], na + [bn[above][k]]
            lines.append(dict(x=np.array(ta), y=np.array(na), sector=0, solid=solid))

    ut = np.concatenate([t0[win0], t1[win1]] + [a[0] for a in used])
    un = np.concatenate([n0[win0], n1[win1]] + [a[1] for a in used])
    keep = ut <= tau_max
    union = tau_sorted(ut[keep], un[keep])
    return dict(lines=lines, union=union, nu_inf0=float(n0[np.argmax(th0)]), win0=win0, win1=win1,
                th0=th0, t0=t0, n0=n0, th1=th1, t1=t1, n1=n1)
