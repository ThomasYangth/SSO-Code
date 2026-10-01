r"""Branch crossings of the frontier and the superposition arcs that fill them.

When two branches exchange which one maximises M(theta) at theta*, the stored
theta grid jumps in tau.  At theta* the two top eigenvectors w_a, w_b are
degenerate (eigenvalue W), so every real combination

    u(alpha) = cos(alpha) w_a + sin(alpha) w_b

is optimal for the same objective, and with N_ij = <w_i|L^2|w_j>,
D_ij = <w_i|L^2 Delta^2|w_j> (L = L_theta*)

    N(alpha) = c^2 N_aa + s^2 N_bb + 2 c s Re N_ab,   D(alpha) likewise,
    nu(alpha) = W / N(alpha),   sigma^2(alpha) = D(alpha) / N(alpha),

which sweeps continuously between the two branch endpoints.  w_b is rotated so
that N_ab is real and >= 0 ("gauge").

Two situations:

* across sectors (K_a != K_b): W_Ka - W_Kb is a signed function already
  tabulated on the stored grids; theta* is found by false position in
  log(theta).  The cross terms vanish identically (disjoint supports).
* within one sector, or in the full space (K = None): W_0 - W_1 touches zero
  rather than changing sign, so the branches must be labelled.  mode 'nu':
  nu labels the branch, s(theta) = nu_0(theta) - (nu_lo + nu_hi)/2 changes
  sign at theta* and is bisected in log(theta) with the top pair only.
  mode 'overlap': k_calc eigenpairs per trial theta, grouped in multiplets and
  tracked by subspace overlap with the bracket-end references, bisecting
  g = W_A - W_B.  Brackets come from the largest tau jump (no stored vectors
  needed) or from top-multiplet overlap drops between stored eigenvectors
  (``detect_crossings``).  The branch vectors at theta* are the projections of
  the references onto the degenerate subspace S.  A splitting > 1e-6
  relative marks an avoided crossing (``exact_degeneracy=False``).
"""

import numpy as np

from sso.backend import free_gpu_memory, to_cpu
from sso.cli import fprint
from sso.nutau.conventions import tau_of
from sso.nutau.frontier import solve


# -- detection on stored grids --------------------------------------------------

def sign_change_brackets(theta_a, W_a, theta_b, W_b):
    """Theta intervals where W_a - W_b changes sign, on the common theta grid.

    Returns a list of (theta_lo, theta_hi, g_lo, g_hi).
    """
    common = np.intersect1d(theta_a, theta_b)
    common = common[np.isfinite(common)]
    wa = W_a[np.searchsorted(theta_a, common)]
    wb = W_b[np.searchsorted(theta_b, common)]
    g = wa - wb
    i = np.flatnonzero(np.diff(np.sign(g)) != 0)
    return [(float(common[j]), float(common[j + 1]), float(g[j]), float(g[j + 1]))
            for j in i]


def largest_tau_jump(theta, tau, nu):
    """Bracket of the largest upward jump of tau between adjacent finite theta.

    Returns dict(theta_lo, theta_hi, tau_lo, tau_hi, nu_lo, nu_hi, rel_tau_jump).
    """
    order = np.argsort(theta)
    th, ta, nn = theta[order], tau[order], nu[order]
    ok = np.isfinite(th) & np.isfinite(ta)
    dlog = np.full(len(th) - 1, -np.inf)
    for i in range(len(th) - 1):
        if ok[i] and ok[i + 1] and ta[i + 1] > ta[i]:
            dlog[i] = np.log(ta[i + 1] / ta[i])
    i = int(np.argmax(dlog))
    return dict(theta_lo=th[i], theta_hi=th[i + 1], tau_lo=ta[i], tau_hi=ta[i + 1],
                nu_lo=nn[i], nu_hi=nn[i + 1], rel_tau_jump=dlog[i])


# -- the arc ----------------------------------------------------------------------

def degenerate_pair_arc(W, w_a, w_b, mask, dsq, tau_of, alpha):
    """Arc through real combinations of two orthonormal degenerate vectors.

    ``w_a``, ``w_b`` may live on different supports (cross-sector, then pass
    the per-sector (mask, dsq) as 2-tuples) or on one support.
    Returns dict(alpha, tau, nu, N_aa, N_bb, N_ab, D_aa, D_bb, D_ab_re,
    D_ab_im, cross_frac_N, cross_frac_D).
    """
    def form(m, d, x, y):
        w = m * m if d is None else m * m * d
        return complex(to_cpu((x.conj() * w * y).sum()))

    if isinstance(mask, tuple):
        (ma, mb), (da, db) = mask, dsq
        N_aa, N_bb = form(ma, None, w_a, w_a).real, form(mb, None, w_b, w_b).real
        D_aa, D_bb = form(ma, da, w_a, w_a).real, form(mb, db, w_b, w_b).real
        N_ab = D_ab = 0j
    else:
        N_aa, N_bb = form(mask, None, w_a, w_a).real, form(mask, None, w_b, w_b).real
        D_aa, D_bb = form(mask, dsq, w_a, w_a).real, form(mask, dsq, w_b, w_b).real
        N_ab, D_ab = form(mask, None, w_a, w_b), form(mask, dsq, w_a, w_b)
        if abs(N_ab) > 0:
            phase = np.exp(-1j * np.angle(N_ab))
            N_ab, D_ab = N_ab * phase, D_ab * phase
    c, s = np.cos(alpha), np.sin(alpha)
    N = c * c * N_aa + s * s * N_bb + 2 * c * s * N_ab.real
    D = c * c * D_aa + s * s * D_bb + 2 * c * s * D_ab.real
    keep = (N > 0) & (D > 0)
    return dict(alpha=alpha[keep], tau=tau_of(D[keep] / N[keep]), nu=W / N[keep],
                N_aa=N_aa, N_bb=N_bb, N_ab=N_ab.real, D_aa=D_aa, D_bb=D_bb,
                D_ab_re=D_ab.real, D_ab_im=D_ab.imag,
                cross_frac_N=abs(N_ab) / np.sqrt(max(N_aa * N_bb, 1e-300)),
                cross_frac_D=abs(D_ab) / np.sqrt(max(D_aa * D_bb, 1e-300)))


# -- across sectors ---------------------------------------------------------------

def cross_sector_arc(sup_a, sup_b, Bop, bracket, tol=1e-10, max_iter=12,
                     n_alpha=721, ncv=50):
    """Locate theta* in a W_a - W_b sign-change bracket and build the arc.

    False position in log(theta), step clamped to [0.05, 0.95] of the bracket,
    converged when |W_a - W_b| / max|W| < tol.  Returns a dict with the arc and
    theta_star, W (mean), W_a, W_b, rel_degeneracy, n_solves.
    """
    lo, hi, g_lo, g_hi = bracket
    n_solves = 0
    for it in range(max_iter):
        f = min(max(g_lo / (g_lo - g_hi), 0.05), 0.95)
        mid = float(np.exp(np.log(lo) + f * (np.log(hi) - np.log(lo))))
        ra = solve(sup_a, Bop, mid, k=1, ncv=ncv, seed=12345 + sup_a.K)
        rb = solve(sup_b, Bop, mid, k=1, ncv=ncv, seed=12345 + sup_b.K)
        n_solves += 2
        Wa, Wb = float(ra["w"][0]), float(rb["w"][0])
        g = Wa - Wb
        rel = abs(g) / max(abs(Wa), abs(Wb), 1e-300)
        fprint(f"    iter {it}: theta={mid:.10g}  g={g:+.4e}  |g|/W={rel:.3e}")
        if rel < tol:
            break
        if np.sign(g) == np.sign(g_lo):
            lo, g_lo = mid, g
        else:
            hi, g_hi = mid, g
    else:
        fprint(f"    WARNING: |g|/W = {rel:.3e} after {max_iter} iterations; arc approximate")
    W = 0.5 * (Wa + Wb)
    alpha = np.linspace(0.0, 0.5 * np.pi, n_alpha)
    arc = degenerate_pair_arc(W, ra["vecs"][:, 0], rb["vecs"][:, 0],
                              (ra["mask"], rb["mask"]), (sup_a.dsq, sup_b.dsq),
                              sup_a.spec.tau, alpha)
    arc.update(theta_star=mid, W=W, W_a=Wa, W_b=Wb, n_solves=n_solves,
               rel_degeneracy=abs(Wa - Wb) / max(abs(W), 1e-300))
    return arc


# -- within one sector --------------------------------------------------------------

def eig_multiplets(w, rel_tol=1e-8):
    """Groups of a descending eigenvalue list within rel_tol |w_0| of each group's first member."""
    w = np.asarray(w).real
    scale = max(abs(float(w[0])), 1e-300)
    groups, cur = [], [0]
    for i in range(1, len(w)):
        if abs(w[i] - w[cur[0]]) <= rel_tol * scale:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def top_multiplet(w, rel_tol=1e-8):
    """Indices of the leading (near-)degenerate group of a descending eigenvalue list."""
    return eig_multiplets(w, rel_tol)[0]


def _orth(cols, xp):
    q, _ = xp.linalg.qr(cols)
    return q


def subspace_overlap(Qa, Qb, xp):
    """||Qa^dag Qb||_F^2 / min(dim): 1 iff one span contains the other, 0 iff orthogonal."""
    s = xp.linalg.svd(Qa.conj().T @ Qb, compute_uv=False)
    return float(to_cpu((s ** 2).sum())) / min(Qa.shape[1], Qb.shape[1])


def stored_vectors(sup, Bop, v2_cols):
    """Stored Pauli-basis eigenvectors (columns) -> orthonormal columns on ``sup``."""
    xp = sup.xp
    cols = [sup.gather(Bop.from_pauli(xp.asarray(v2_cols[:, j]))) for j in range(v2_cols.shape[1])]
    return _orth(xp.stack(cols, axis=1), xp)


def detect_crossings(files, kind="static", overlap_tol=0.9, mult_rel_tol=1e-8):
    """Grid intervals where the TOP MULTIPLET changes identity (needs stored v2).

    ``files`` = [(theta, path)] of one grid (finite theta).  The test is the
    subspace overlap of the stored top multiplets at adjacent theta (computed in
    the Pauli basis, where the conversion is a uniform rescaling of a unitary);
    eigsh scrambles bases inside a degenerate multiplet (momentum +-k), so
    individual eigenvectors are never matched.  A tau jump alone is not a
    criterion: in the saturated regime As ~ theta^-2 makes tau jump at every step.
    Returns dicts with theta_lo/hi, tau_lo/hi, nu_lo/hi (top branch), idx_lo/hi
    (stored top-multiplet columns), overlap, truncated (multiplet fills k_save),
    path_lo/hi.
    """
    files = [(t, p) for t, p in files if np.isfinite(t)]
    out = []
    z0 = np.load(files[0][1])
    for (t0, p0), (t1, p1) in zip(files[:-1], files[1:]):
        z1 = np.load(p1)
        m0, m1 = (eig_multiplets(z["w"], mult_rel_tol)[0] for z in (z0, z1))
        ov = subspace_overlap(_orth(z0["v2"][:, m0], np), _orth(z1["v2"][:, m1], np), np)
        if ov < overlap_tol:
            tau = tau_of(np.array([z0["As"][0], z1["As"][0]]), kind)
            k = z0["v2"].shape[1]
            out.append(dict(theta_lo=t0, theta_hi=t1, tau_lo=tau[0], tau_hi=tau[1],
                            nu_lo=float(z0["Bs"][0]), nu_hi=float(z1["Bs"][0]),
                            idx_lo=m0, idx_hi=m1, overlap=ov, path_lo=p0, path_hi=p1,
                            truncated=len(m0) >= k or len(m1) >= k))
        z0 = z1
    return out


def multiplet_tau_nu(sup, Bop, theta, w, v2_cols):
    """(tau, nu) of each stored member of one multiplet at grid theta, and its W.

    Symmetry partners (momentum +k / -k) are degenerate at every theta with
    IDENTICAL (tau, nu): no arc.  Distinct (tau, nu) means two branches are
    degenerate at this grid point, i.e. the crossing sits inside the multiplet.
    """
    xp = sup.xp
    m2 = sup.mask(theta) ** 2
    W = float(np.mean(np.asarray(w).real))
    taus, nus = [], []
    for j in range(v2_cols.shape[1]):
        u = sup.gather(Bop.from_pauli(xp.asarray(v2_cols[:, j])))
        u = u / xp.linalg.norm(u)
        p = xp.abs(u) ** 2
        N, D = float(to_cpu((m2 * p).sum())), float(to_cpu((m2 * sup.dsq * p).sum()))
        nus.append(W / N)
        taus.append(float(sup.spec.tau(D / N)))
    return np.array(taus), np.array(nus), W


def _bisect(probe, lo, hi, n_iter, rtol):
    """Bisection in log(theta) on the sign of probe(theta)[0] (0 means merged/exact)."""
    s_lo = probe(lo)[0]
    it = 0
    while it < n_iter and (hi - lo) > rtol * max(abs(lo), 1e-30):
        mid = float(np.sqrt(lo * hi))
        s_mid = probe(mid)[0]
        fprint(f"    bisect {it}: theta={mid:.10g}  s={s_mid:+.4e}")
        if s_mid == 0.0:
            lo = hi = mid
            break
        if np.sign(s_mid) == np.sign(s_lo):
            lo = mid
        else:
            hi = mid
        it += 1
    return (float(np.sqrt(lo * hi)) if hi > lo else float(lo)), it


def within_sector_arc(sup, Bop, bracket, mode="nu", refs=None, k_ref=8, k_calc=8,
                      n_iter=32, rtol=1e-11, n_alpha=721, ncv=50, mult_rel_tol=1e-8,
                      max_rel_splitting=1e-6):
    """Locate a within-sector (or full-space) branch crossing and build its arc.

    ``bracket`` needs theta_lo, theta_hi (and nu_lo, nu_hi for mode='nu').
    ``refs`` = (QA, QB): orthonormal branch references on ``sup`` (e.g. the
    stored top multiplets, :func:`stored_vectors`); default: the top multiplets
    of k_ref fresh eigenpairs at the bracket ends.

    mode='nu'       bisect s(theta) = nu_0 - (nu_lo + nu_hi)/2 with the top pair
                    only; at theta* the degenerate subspace S is ranks 0, 1.
    mode='overlap'  at each trial theta compute k_calc eigenpairs, group them in
                    multiplets, label the multiplets A / B by subspace overlap
                    with the references and bisect g = W_A - W_B (g = 0 when A
                    and B fall in one multiplet); S = union of the A and B
                    multiplets at theta*.  k_calc must hold both branches.

    Returns the arc dict plus theta_star, W, W_spread, dim_S, gap_rel,
    exact_degeneracy (gap_rel <= max_rel_splitting; otherwise an avoided
    crossing and the arc is only approximate), proj_norm_A/B, overlap_A/B;
    None if no sign change.
    """
    xp = sup.xp
    seed = 12345 + (0 if sup.K is None else sup.K)
    if refs is None:
        def ref(theta):
            r = solve(sup, Bop, theta, k=k_ref, ncv=ncv, seed=seed)
            free_gpu_memory()
            return _orth(r["vecs"][:, top_multiplet(r["w"], mult_rel_tol)], xp)
        refs = (ref(bracket["theta_lo"]), ref(bracket["theta_hi"]))
    QA, QB = refs
    lo, hi = float(bracket["theta_lo"]), float(bracket["theta_hi"])

    if mode == "nu":
        if abs(bracket["nu_lo"] - bracket["nu_hi"]) < 1e-6:
            fprint("    branch nu values not separated; cannot label the branches")
            return None
        nu_mid = 0.5 * (bracket["nu_lo"] + bracket["nu_hi"])

        def probe(theta):
            r = solve(sup, Bop, theta, k=2, ncv=ncv, seed=seed)
            free_gpu_memory()
            return float(r["Bs"][0]) - nu_mid, r, [0, 1]
    elif mode == "overlap":
        def probe(theta):
            r = solve(sup, Bop, theta, k=k_calc, ncv=ncv, seed=seed)
            free_gpu_memory()
            groups = eig_multiplets(r["w"], mult_rel_tol)
            Qs = [_orth(r["vecs"][:, g], xp) for g in groups]
            iA = int(np.argmax([subspace_overlap(Q, QA, xp) for Q in Qs]))
            iB = int(np.argmax([subspace_overlap(Q, QB, xp) for Q in Qs]))
            if iA == iB:
                return 0.0, r, groups[iA]
            return float(r["w"][groups[iA][0]] - r["w"][groups[iB][0]]), r, sorted(groups[iA] + groups[iB])
    else:
        raise ValueError(f"unknown locate mode {mode!r}")

    s_lo, _, _ = probe(lo)
    s_hi, _, _ = probe(hi)
    fprint(f"    s(theta_lo)={s_lo:+.4e}  s(theta_hi)={s_hi:+.4e}")
    if mode == "overlap" and (s_lo == 0.0 or s_hi == 0.0):
        fprint("    branches already degenerate at a bracket end: symmetry degeneracy, not a crossing")
        return None
    if s_lo * s_hi > 0:
        fprint("    indicator does not change sign; no crossing in this bracket")
        return None
    theta_star, it = _bisect(probe, lo, hi, n_iter, rtol)
    _, r, idx = probe(theta_star)
    QS = _orth(r["vecs"][:, idx], xp)
    W = float(np.mean(r["w"][idx]))
    W_spread = float(np.max(r["w"][idx]) - np.min(r["w"][idx]))

    def project_into_S(Q):
        C = QS.conj().T @ Q
        u_c, sv, _ = xp.linalg.svd(C, full_matrices=False)
        return QS @ u_c[:, 0], float(to_cpu(sv[0]))

    w_a, na = project_into_S(QA)
    w_b, nb = project_into_S(QB)
    w_a = w_a / xp.linalg.norm(w_a)
    w_b = w_b - xp.vdot(w_a, w_b) * w_a
    if float(xp.linalg.norm(w_b)) < 1e-10:
        raise ValueError("branch references collapse onto one direction inside S")
    w_b = w_b / xp.linalg.norm(w_b)
    alpha = np.linspace(0.0, np.pi, n_alpha, endpoint=False)
    arc = degenerate_pair_arc(W, w_a, w_b, r["mask"], sup.dsq, sup.spec.tau, alpha)
    gap_rel = W_spread / max(abs(W), 1e-300)
    arc.update(theta_star=theta_star, W=W, W_spread=W_spread, dim_S=len(idx), gap_rel=gap_rel,
               exact_degeneracy=bool(gap_rel <= max_rel_splitting), proj_norm_A=na,
               proj_norm_B=nb, overlap_A=subspace_overlap(QS, QA, xp),
               overlap_B=subspace_overlap(QS, QB, xp), n_bisect=it, locate_mode=mode)
    return arc
