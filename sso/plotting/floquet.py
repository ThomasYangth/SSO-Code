"""Data access for the kicked-Ising (g = 0.9, J = 1, h = 0.809) panels (figure_B (a), figure_G (b)).

frontier(L): K = 0 frontier (identity-only projector "o1"), Floquet chord tau.
OPERATORS: (key, label, K) of the filtered base operators.
Data: data/floquet_two_panel/ (scripts/extract/nutau_floquet_two_panel.py).
"""

import os

from sso.config import data_dir
from sso.nutau.io import load_npz
from sso.plotting.frontier import grid_curve, tau_sorted

DATA = data_dir("floquet_two_panel")
TAU_MAX = 1e5
TAU_NOISE = 1e8
OPERATORS = [("Zsum", r"$\Sigma_x Z_x$", "0"), ("Z0", r"$Z_1$", "none"),
             ("Xsum", r"$\Sigma_x X_x$", "0")]


def frontier(L):
    _, t, n, _ = grid_curve(load_npz(os.path.join(DATA, f"grid_L{L}_K0.npz")), "floquet")
    return tau_sorted(t, n)
