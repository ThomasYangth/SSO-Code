r"""Loaders for the slim DQT / TI data in ``data/fig3panel/`` (figure_B (c), figure_F), written by
``scripts/extract/dqt_fig3panel.py`` and ``scripts/extract/dqt_lorentz_fronts.py``.

Layout of ``data/fig3panel/``::

    modhydro_fourier_L{L}_n1_g-1.05_h+0.5_COMBINED.npz   nu(tau) front per L, Gaussian window (fig3panel a)
    modhydro_lorentz_L{L}_n1_g-1.05_h+0.5_REFILTER.npz  nu(tau) front per L, Lorentzian filter
                              (scripts/extract/dqt_lorentz_fronts.py; figure_B b)
    correlators_L{L}.npz      t_C, C (HS autocorrelator), t_R, R (RPS norm^2), nprod, nhaar
    grps_L{L}.npz             t, absG = |G_RPS(t1,t2)| on |t| <= 0.15 L^2, nstates
    ti_q_scan/nutau_M{M}_q{q}.dat   TI-ansatz q-scan tables (panel c, sso.tiansatz)
"""

import numpy as np

from sso.config import data_dir
from sso.dqt.combine import combined_filename

FIG = "fig3panel"
LS = tuple(range(16, 24))
INSET_L = 20
TWIN = 0.15            # G_RPS window |t|/L^2 <= TWIN


def load_front(L, base=None):
    """Panel (a): dict with tau_obs, tau_err, nu, nu_err (and N1, N2, M, ...)."""
    base = base or data_dir(FIG)
    return dict(np.load(f"{base}/{combined_filename(L)}"))


def load_lorentz_front(L, base=None):
    """Lorentzian-filter front: dict with width, tau_obs, tau_err, nu, nu_err, n_haar, n_prod."""
    from sso.dqt.refilter import refilter_filename
    base = base or data_dir(FIG)
    return dict(np.load(f"{base}/{refilter_filename('lorentz', L)}"))


def load_correlators(L, base=None):
    """Panel (b): dict(t_C, C, t_R, R, nprod, nhaar); R = ||H_k(t)||^2_RPS."""
    base = base or data_dir(FIG)
    return dict(np.load(f"{base}/correlators_L{L}.npz"))


def load_grps(L=INSET_L, base=None):
    """Panel (b) inset: (t, |G_RPS(t1,t2)|, number of product states)."""
    base = base or data_dir(FIG)
    d = np.load(f"{base}/grps_L{L}.npz")
    return d["t"], d["absG"], int(d["nstates"])


def ti_dir(base=None):
    """Directory of the TI-ansatz q-scan tables (figure_F (b))."""
    return f"{base or data_dir(FIG)}/ti_q_scan"
