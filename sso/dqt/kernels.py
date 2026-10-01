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
