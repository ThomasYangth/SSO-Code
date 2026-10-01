r"""The soft (sqrt-Lorentzian) nu(tau) frontier, static and Floquet, any sector.

Problem.  Maximise nu = <v|B|v>/<v|v> at fixed slowness.  In the eigenbasis the
detuning Delta^2_{mn} is diagonal on the operator index (m, n), and the soft
filter is

    L_theta = 1/sqrt(1 + theta^2 Delta^2),
    M(theta) = Pi L_theta B L_theta Pi,

with Pi projecting out the trivially slow operators (static: {I, H, ...,
H^{o-1}}; Floquet: {I}, "o1").  The top eigenvector w of M gives the physical
operator v = L_theta w, and

    Bs = nu      = <v|B|v>/<v|v>       = <w|M|w>/<w|L_theta^2|w>
    As = sigma^2 = <v|Delta^2|v>/<v|v>
    W            = top eigenvalue of M (comparable across sectors K).

Sectors.  M commutes with doubled translation, so it is block diagonal in
K = (k_m - k_n) mod L; restricting to one K gives a smooth per-sector curve and
an eigsh on d_K ~ n^2/L entries ("compressed").  span{I, H, ...} lives in K = 0,
so K != 0 forces o = 0.  K = None is the full space (any boundary condition).

theta = infinity.  L_theta -> 1 on the exact commutant (Delta^2 <= tol^2, i.e.
pairs inside degenerate multiplets for lam = 0) and -> 0 elsewhere, so the
problem becomes Pi B Pi on that support, with As = 0.

Raw output (one NPZ per (L, K, theta), original key names): w, As, Bs (length
k_save), v2 (the pre-filter eigenvectors w in the Pauli basis, or a (1, k_save)
placeholder when save_v2=False), theta, lam, basis, E (static) or u, phi
(Floquet), K and k_m (momentum resolved), d_res (theta = inf).
"""

import os
from time import time

import numpy as np

from sso.backend import LinearOperator, eigsh, to_cpu, xp as get_xp
from sso.cli import fprint
from sso.nutau.pauli import EnergyBasisB


def effective_o(K, o):
    """Projector rank actually used: o in K = 0 / full space, 0 in K != 0."""
    return o if K in (None, 0) else 0


def frontier_filename(kind, L, K, theta, lam, k_save, o):
    """Original file pattern: L{L}[_K{K}]_theta{theta}l{lam}k{k_save}o{o}.npz.

    Static full space has no K tag (NuTau_ds_soft); Floquet full space uses Knone.
    """
    if K is None:
        tag = "" if kind == "static" else "_Knone"
    else:
        tag = f"_K{K}"
    return f"L{L}{tag}_theta{float(theta)}l{float(lam)}k{k_save}o{effective_o(K, o)}.npz"


def theta_grid(theta_min, theta_max, theta_num, include_infinity=False):
    """Production grid: geomspace rounded to 4 decimals (+ optional inf)."""
    th = np.round(np.geomspace(theta_min, theta_max, num=theta_num), 4)
    return np.concatenate([th, [np.inf]]) if include_infinity else th


class Support:
    """Index set of one sector problem and its theta-independent ingredients.

    resonant=False: the doubled-momentum sector K (all pairs for K=None).
    resonant=True:  the exact commutant inside sector K (theta = infinity).
    """

    def __init__(self, spec, K, o, lam=0.0, resonant=False, degen_tol=1e-10,
                 usegpu=False):
        xp = get_xp(usegpu)
        n = spec.n
        self.spec, self.K, self.lam, self.resonant = spec, K, lam, resonant
        self.xp, self.usegpu = xp, usegpu
        idx = (spec.resonant_indices(K, lam, degen_tol) if resonant
               else spec.sector_indices(K))
        if idx is None:
            m = np.arange(n)
            dsq = spec.detuning_sq_pairs(m[:, None], m[None, :], lam).reshape(-1)
        else:
            dsq = spec.detuning_sq_pairs(idx // n, idx % n, lam)
        self.idx_np = idx
        self.dim = n * n if idx is None else len(idx)
        self.idx = None if idx is None else xp.asarray(idx)
        self.dsq = xp.asarray(dsq)

        q = spec.diag_ortho(effective_o(K, o))
        self.q = self.qh = self.pos = None
        if q is not None:
            diag = np.arange(n, dtype=np.int64) * (n + 1)
            if idx is None:
                pos = diag
            else:
                pos = np.minimum(np.searchsorted(idx, diag), max(len(idx) - 1, 0))
                present = (len(idx) > 0) & (idx[pos] == diag)
                if not np.any(present):
                    q = None
                elif not np.all(present):
                    raise RuntimeError("support contains only part of the diagonal")
            if q is not None:
                self.pos = xp.asarray(pos)
                self.q = xp.asarray(q.astype(complex))
                self.qh = self.q.conj().T
        self.o = 0 if self.q is None else self.q.shape[1]
        self.work = xp.zeros(n * n, dtype=complex) if idx is not None else None

    def project(self, x):
        """Remove the span of the trivially slow operators (in place, returns x)."""
        if self.q is not None:
            x[self.pos] -= self.q @ (self.qh @ x[self.pos])
        return x

    def scatter(self, u):
        """Compressed vector -> full n^2 E-basis vector."""
        if self.idx is None:
            return u
        full = self.xp.zeros(self.spec.n ** 2, dtype=complex)
        full[self.idx] = u
        return full

    def gather(self, full):
        """Full n^2 E-basis vector -> compressed vector (copy)."""
        full = full.reshape(-1)
        return full.copy() if self.idx is None else full[self.idx]

    def mask(self, theta):
        """L_theta on the support (identically 1 at theta = inf)."""
        if not np.isfinite(theta):
            return self.xp.ones(self.dim)
        return 1.0 / self.xp.sqrt(1.0 + theta ** 2 * self.dsq)

    def operator(self, Bop, theta):
        """(M(theta) as a LinearOperator on the support, mask)."""
        n = self.spec.n
        mask = self.mask(theta)

        def matvec(u):
            x = self.project(u.reshape(-1) * mask)
            if self.idx is None:
                Me = x.reshape(n, n)
            else:
                self.work.fill(0)
                self.work[self.idx] = x
                Me = self.work.reshape(n, n)
            out = Bop.apply(Me).reshape(-1)
            y = (out if self.idx is None else out[self.idx]) * mask
            return self.project(y)

        LO = LinearOperator(self.usegpu)
        return LO((self.dim, self.dim), matvec=matvec, dtype=np.complex128), mask


def top_eigenpairs(op, dim, k, v0=None, tol=1e-8, ncv=50, usegpu=False,
                   dense_thresh=64):
    """The k algebraically largest eigenpairs of a Hermitian LinearOperator.

    Tiny supports (dim <= dense_thresh) are materialised and sent to eigh.
    Results are padded with zeros to exactly k columns.
    """
    xp = get_xp(usegpu)
    if dim == 0:
        return np.zeros(k), xp.zeros((0, k), dtype=complex)
    if dim <= max(dense_thresh, k + 2):
        Md = xp.zeros((dim, dim), dtype=complex)
        e = xp.zeros(dim, dtype=complex)
        for j in range(dim):
            e.fill(0)
            e[j] = 1.0
            Md[:, j] = op.matvec(e)
        w, v = xp.linalg.eigh(0.5 * (Md + Md.conj().T))
    else:
        k_comp = min(k + 1, dim - 1)
        w, v = eigsh(usegpu)(op, k=k_comp, which="LA", v0=v0, tol=tol,
                             ncv=int(min(max(ncv, 2 * k_comp + 2), dim)),
                             maxiter=50 * dim)
    order = xp.argsort(w)[::-1]
    w, v = to_cpu(w[order]).real, v[:, order]
    nk = min(k, v.shape[1])
    w_out = np.zeros(k)
    v_out = xp.zeros((dim, k), dtype=complex)
    w_out[:nk] = w[:nk]
    v_out[:, :nk] = v[:, :nk]
    return w_out, v_out


def solve(support, Bop, theta, k=2, v0=None, ncv=50, seed=12345):
    """Top-k eigenpairs of M(theta) on ``support`` with their (As, Bs).

    Returns dict(w, As, Bs, vecs, mask): eigenvalues (numpy, descending), the
    cleaned unit eigenvectors w in compressed coordinates (device), the
    observables As = <v|Delta^2|v>/<v|v>, Bs = <w|M|w>/<v|v> with v = L w.
    Without a warm start v0 a seeded random start vector is used.
    """
    xp = support.xp
    op, mask = support.operator(Bop, theta)
    dim = support.dim
    if v0 is None and dim > 0:
        rng = np.random.default_rng(seed)
        v0 = xp.asarray(rng.standard_normal(dim) + 1j * rng.standard_normal(dim))
    if v0 is not None:
        v0 = v0 / xp.linalg.norm(v0)
    tol = 1e-8 if np.isfinite(theta) else 1e-10
    w, vecs = top_eigenpairs(op, dim, k, v0=v0, tol=tol, ncv=ncv,
                             usegpu=support.usegpu)
    if support.spec.kind == "static":
        empty_As = support.lam ** 2
    else:
        empty_As = float(abs(1.0 - np.exp(1j * support.lam)) ** 2)
    As = np.full(k, empty_As if dim == 0 else 0.0)
    Bs = np.zeros(k)
    for i in range(k):
        if dim == 0:
            break
        u = support.project(vecs[:, i].copy())
        nrm = float(xp.linalg.norm(u))
        if nrm < 1e-14:
            vecs[:, i] = u
            continue
        u = u / nrm
        vecs[:, i] = u
        v = mask * u
        vv = float(xp.real(xp.vdot(v, v)))
        if vv < 1e-14:
            continue
        As[i] = float(xp.real(xp.vdot(v, support.dsq * v))) / vv
        Bs[i] = float(xp.real(xp.vdot(u, op.matvec(u)))) / vv
    return dict(w=w, As=As, Bs=Bs, vecs=vecs, mask=mask)


def _save(path, spec, support, Bop, res, theta, save_v2):
    k = len(res["w"])
    if save_v2:
        v2 = np.zeros((4 ** spec.L, k), dtype=complex)
        if support.dim > 0:
            for i in range(k):
                v2[:, i] = to_cpu(Bop.to_pauli(support.scatter(res["vecs"][:, i])))
    else:
        v2 = np.zeros((1, k), dtype=complex)
    payload = dict(w=res["w"], As=res["As"], Bs=res["Bs"], v2=v2,
                   theta=theta, lam=support.lam, basis="pauli")
    if spec.kind == "static":
        payload["E"] = spec.values
    else:
        payload["u"] = np.exp(1j * spec.values)
        payload["phi"] = spec.values
    if spec.k is not None:
        payload["K"] = -1 if support.K is None else support.K
        payload["k_m"] = spec.k
    if not np.isfinite(theta):
        payload["d_res"] = support.dim
    np.savez(path, **payload)


def _load(path, support, Bop):
    """Cached (w, As, Bs) and, when v2 was stored, the warm-start vector."""
    z = np.load(path)
    v0 = None
    if z["v2"].shape[0] == 4 ** support.spec.L and support.dim > 0:
        full = Bop.from_pauli(support.xp.asarray(z["v2"][:, 0])).reshape(-1)
        v0 = support.gather(full)
    return dict(w=z["w"], As=z["As"], Bs=z["Bs"]), v0


def frontier_scan(spec, K, thetas, o, outdir, lam=0.0, k_save=2, ncv=50,
                  degen_tol=1e-10, save_v2=True, override=False, usegpu=False):
    """Solve M(theta) for every theta in sector K, one NPZ per theta in ``outdir``.

    Finite theta are warm-started from the previous solution (or from a cached
    file's v2).  Returns (As, Bs, W) arrays of shape (len(thetas), k_save).
    """
    Bop = EnergyBasisB(spec.V, spec.L, usegpu)
    finite = Support(spec, K, o, lam, resonant=False, usegpu=usegpu)
    resonant = None
    out = np.zeros((3, len(thetas), k_save))
    v0 = None
    Ktag = "none" if K is None else K
    for it, theta in enumerate(thetas):
        path = os.path.join(outdir, frontier_filename(spec.kind, spec.L, K, theta,
                                                      lam, k_save, o))
        if np.isfinite(theta):
            sup = finite
        else:
            if resonant is None:
                resonant = Support(spec, K, o, lam, resonant=True,
                                   degen_tol=degen_tol, usegpu=usegpu)
            sup = resonant
        if os.path.exists(path) and not override:
            res, v_cached = _load(path, sup, Bop)
            fprint(f"  K={Ktag} theta={theta:<9.4g} cached")
            if np.isfinite(theta):
                v0 = v_cached
        else:
            t0 = time()
            seed = (12345 if np.isfinite(theta) else 999) + (0 if K is None else K)
            res = solve(sup, Bop, theta, k=k_save,
                        v0=v0 if np.isfinite(theta) else None, ncv=ncv, seed=seed)
            _save(path, spec, sup, Bop, res, theta, save_v2)
            fprint(f"  K={Ktag} theta={theta:<9.4g} dim={sup.dim} o={sup.o} "
                   f"As={res['As']} Bs={res['Bs']} ({time() - t0:.1f}s)")
            if np.isfinite(theta):
                v0 = res["vecs"][:, 0]
        out[0, it], out[1, it], out[2, it] = res["As"], res["Bs"], res["w"]
    return out[0], out[1], out[2]
