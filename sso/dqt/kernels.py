r"""Gaussian time kernels of the s-windowed mode  H_k(s) = exp(-s^2 ad_H^2 / 4) H_k.

    H_k(s) = int dt g_s(t) e^{iHt} H_k e^{-iHt},   g_s(t) = e^{-t^2/s^2} / (sqrt(pi) s)

(g_s has standard deviation s/sqrt2 and Fourier transform e^{-s^2 Omega^2/4}).  Two
windows convolve to G_v with variance v = s^2, which gives the correlator kernels

    N1(s) = int G_{s^2}(tau) C(tau) dtau,       N2(s) = int (-G''_{s^2})(tau) C(tau) dtau,

    G_v(tau) = e^{-tau^2/2v} / sqrt(2 pi v),    -G_v''(tau) = G_v(tau) (v - tau^2) / v^2 .

The overlaps with the bare mode use one window, v = s^2/2:
    V0(s) = <<H_k|H_k(s)>>        = int G_{s^2/2} C,
    V2(s) = <<ad^2 H_k|H_k(s)>>   = int (-G''_{s^2/2}) C.
"""

import numpy as np


def gauss(v, tau):
    """G_v(tau) = exp(-tau^2/2v) / sqrt(2 pi v)."""
    return np.exp(-tau ** 2 / (2.0 * v)) / np.sqrt(2.0 * np.pi * v)


def neg_gpp(v, tau):
    """-G_v''(tau) = G_v(tau) (v - tau^2) / v^2  (Fourier transform Omega^2 e^{-v Omega^2/2})."""
    return gauss(v, tau) * (v - tau ** 2) / v ** 2


def gwin(s, tau):
    """Single time window g_s(tau) = exp(-tau^2/s^2) / (sqrt(pi) s)."""
    return np.exp(-tau ** 2 / s ** 2) / (np.sqrt(np.pi) * s)


# --- Lorentzian filters ------------------------------------------------------------------------
#
# 'lorentz'  (default)  f_m(Omega) = m^2 / (Omega^2 + m^2)            (= 1/(1 + theta Omega^2), theta = 1/m^2)
# 'lorentz2' (comparison only)  f_m^2 = m^4 / (Omega^2 + m^2)^2
# Time kernels (y = m |tau|), in the convention H_k(f) = int dt l(t) e^{iHt} H_k e^{-iHt}:
#     lor_win        (m/2) e^{-y}                                FT f
#     lor2_win       (m/4) (1 + y) e^{-y}                         FT f^2    (= lor_win * lor_win)
#     neg_lor2_win_pp (m^3/4) (1 - y) e^{-y}                      FT Omega^2 f^2
#     lor2_K         (m/96) (15 + 15 y + 6 y^2 + y^3) e^{-y}      FT f^4    (= lor2_win * lor2_win)
#     neg_lor2_Kpp   (m^3/96) (3 + 3 y - y^3) e^{-y}              FT Omega^2 f^4
# For 'lorentz': N1, N2 use lor2_win, neg_lor2_win_pp; V0, a use lor_win; and
# V2 = int (-l'') C = m^2 (C(0) - V0) exactly (Omega^2 f = m^2 - m^2 f; -l'' holds m^2 delta(t)).
# For 'lorentz2': N1, N2 use lor2_K, neg_lor2_Kpp; V0, a, V2 use lor2_win, neg_lor2_win_pp.
# Exponential (not Gaussian) tails: the stored time grid must reach several 1/m
# (see :mod:`sso.dqt.refilter`).


def lor_win(m, tau):
    """Lorentzian window (m/2) e^{-m|tau|}  (Fourier transform m^2/(Omega^2 + m^2))."""
    return 0.5 * m * np.exp(-m * np.abs(tau))


def lor2_win(m, tau):
    """Squared-Lorentzian window (m/4)(1 + m|tau|) e^{-m|tau|}  (FT m^4/(Omega^2+m^2)^2)."""
    y = m * np.abs(tau)
    return 0.25 * m * (1.0 + y) * np.exp(-y)


def neg_lor2_win_pp(m, tau):
    """-(lor2_win)'' = (m^3/4)(1 - m|tau|) e^{-m|tau|}  (FT Omega^2 m^4/(Omega^2+m^2)^2)."""
    y = m * np.abs(tau)
    return 0.25 * m ** 3 * (1.0 - y) * np.exp(-y)


def lor2_K(m, tau):
    """Autocorrelation of lor2_win: (m/96)(15 + 15y + 6y^2 + y^3) e^{-y}  (FT f^4)."""
    y = m * np.abs(tau)
    return m / 96.0 * (15.0 + y * (15.0 + y * (6.0 + y))) * np.exp(-y)


def neg_lor2_Kpp(m, tau):
    """-(lor2_K)'' = (m^3/96)(3 + 3y - y^3) e^{-y}  (FT Omega^2 f^4)."""
    y = m * np.abs(tau)
    return m ** 3 / 96.0 * (3.0 + 3.0 * y - y ** 3) * np.exp(-y)
