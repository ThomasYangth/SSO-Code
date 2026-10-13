"""Data access for the hx = 0.9 Ising chain (L = 12, K = 0, o4) panels (figure_B (b), figure_E).

frontier(hz) / arcs(hz): theta-grid frontier and within-K0 arcs for hz in HZS;
modul(kind): the energy current Q = sum_x (Y_x Z_{x+1} - Z_x Y_{x+1}) at hz = HZ_R under
the 'gauss' / 'hard' / 'avg' filters (static tau = 1/sqrt(As), As <= AS_FLOOR dropped).
Data: data/nutau_hx0.9_twopanel/ (scripts/extract/nutau_hx09_twopanel.py).
"""

import os

import numpy as np

from sso.config import data_dir
from sso.nutau.conventions import tau_static
from sso.nutau.io import load_arcs, load_npz
from sso.plotting.frontier import grid_curve, tau_sorted

DATA = data_dir("nutau_hx0.9_twopanel")
NU = 1.0 / 9.0
TAU_MAX = 3e4
AS_FLOOR = 1e-14
HZS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
HZ_R = 0.03


def frontier(hz):
    _, t, n, _ = grid_curve(load_npz(os.path.join(DATA, f"grid_hz{hz}.npz")))
    return tau_sorted(t, n)


def arcs(hz):
    return [(t, n) for t, n, _ in load_arcs(DATA, f"arc_hz{hz}_theta*.npz")]


def modul(kind):
    z = load_npz(os.path.join(DATA, f"filt_hz{HZ_R}_YZcur.npz"))
    As, n = z[f"{kind}_As"], z[f"{kind}_nu"]
    t = tau_static(As)
    ok = np.isfinite(t) & np.isfinite(n) & (As > AS_FLOOR)
    return tau_sorted(t[ok], n[ok])
