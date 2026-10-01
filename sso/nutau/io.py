"""Readers for raw frontier output and for the slim figure data.

Raw theta-grid files follow ``L{L}[_K{K}]_theta{theta}l{lam}k{k_save}o{o}.npz``.
The projector rank ``o`` is always matched explicitly: production directories
hold several o for the same (L, K) (e.g. the Kim-Huse K = 0 o-sweep o = 1..4),
and matching ``o\\d+`` would silently mix them.
"""

import glob
import os
import re

import numpy as np

from sso.config import REPO_ROOT


def grid_files(datadir, L, K, lam, k_save, o, kind="static"):
    """Sorted [(theta, path)] for one (L, K, lam, k_save, o); theta = inf included.

    K=None selects the full-space files (no K tag for static, Knone for Floquet).
    """
    if K is None:
        tag = "" if kind == "static" else "_Knone"
    else:
        tag = f"_K{K}"
    pat = re.compile(rf"L{L}{tag}_theta(inf|[0-9.eE+-]+)l{re.escape(str(float(lam)))}"
                     rf"k{k_save}o{o}\.npz")
    out = []
    for p in glob.glob(os.path.join(datadir, f"L{L}{tag}_theta*.npz")):
        m = pat.fullmatch(os.path.basename(p))
        if m:
            out.append((float(m.group(1)), p))
    out.sort()
    if not out:
        raise FileNotFoundError(f"no files L{L}{tag}_theta*l{lam}k{k_save}o{o}.npz in {datadir}")
    return out


def read_grid(datadir, L, K, lam, k_save, o, kind="static", branch=0):
    """Top-branch summary of a theta grid: dict(theta, As, Bs, W[, d_res]) sorted by theta.

    ``d_res`` (size of the theta = inf resonant subspace; -1 where absent) is
    returned for Floquet grids, where an empty commutant marks nu = 0 as a
    sentinel rather than data.
    """
    rows = []
    for theta, p in grid_files(datadir, L, K, lam, k_save, o, kind):
        z = np.load(p)
        d_res = int(z["d_res"]) if "d_res" in z.files else -1
        rows.append((theta, float(z["As"][branch]), float(z["Bs"][branch]),
                     float(np.real(z["w"][branch])), d_res))
    a = np.array(rows)
    out = dict(theta=a[:, 0], As=a[:, 1], Bs=a[:, 2], W=a[:, 3])
    if kind == "floquet":
        out["d_res"] = a[:, 4].astype(int)
    return out


def load_npz(path):
    """Dict of arrays from a slim-data NPZ (scalars unwrapped)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def read_csv(path):
    """Dict of float columns from a comma-separated file with one header line."""
    with open(path) as f:
        head = f.readline().strip().split(",")
    a = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return {h: a[:, i] for i, h in enumerate(head)}


def save_slim(path, **arrays):
    """Write a slim-data NPZ (directory created)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


def load_arcs(directory, pattern):
    """[(tau, nu, theta_star)] of the arc files matching ``pattern``, each sorted by tau."""
    arcs = []
    for p in sorted(glob.glob(os.path.join(directory, pattern))):
        z = load_npz(p)
        o = np.argsort(z["tau"])
        arcs.append((np.asarray(z["tau"])[o], np.asarray(z["nu"])[o], float(z["theta_star"])))
    return arcs


def raw_root(src=None):
    """Root of the raw compute output: ``src``, else $SSO_OUTPUT, else <repo>/output (never created)."""
    return src or os.environ.get("SSO_OUTPUT") or os.path.join(REPO_ROOT, "output")
