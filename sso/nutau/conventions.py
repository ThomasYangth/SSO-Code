"""Slowness conventions: tau as a function of the detuning variance.

Every frontier point carries ``As = sigma^2 = <v|Delta^2|v>/<v|v>``.

static   Delta_mn = E_m - E_n - lam                      tau = 1/sqrt(As)
Floquet  Delta^2_mn = 4 sin^2((phi_m - phi_n - lam)/2)    tau = 1/(2 arcsin(sqrt(As)/2))

The Floquet "chord" formula inverts 2 sin(1/(2 tau)) = sqrt(As): sqrt(As) is the
RMS chord on the unit circle, converted back to the angle it subtends, so a
single quasi-energy difference omega gives tau = 1/|omega| exactly (as in the
static case).  The chord is bounded by 2, hence tau >= 1/pi for Floquet.
Both return +inf at As = 0 (exact commutant).
"""

import numpy as np


def tau_static(As):
    """Static convention tau = 1/sqrt(As) (inf where As <= 0)."""
    As = np.asarray(As, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(np.maximum(As, 0.0))


def tau_floquet(As):
    """Floquet chord convention tau = 1/(2 arcsin(sqrt(As)/2)) (inf where As <= 0)."""
    As = np.asarray(As, dtype=float)
    half_chord = np.clip(np.sqrt(np.maximum(As, 0.0)) / 2.0, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return 1.0 / (2.0 * np.arcsin(half_chord))


def tau_of(As, kind):
    """tau(As) for ``kind`` in {'static', 'floquet'}."""
    if kind == "static":
        return tau_static(As)
    if kind == "floquet":
        return tau_floquet(As)
    raise ValueError(f"unknown spectrum kind {kind!r}")
