"""Retained-but-not-in-figures features: H_q filter base, mixtures, keep_slowest, overlap / full-space crossings."""
import numpy as np

from sso.nutau.crossing import eig_multiplets, subspace_overlap, within_sector_arc
from sso.nutau.filters import to_energy_basis, width_grid
from sso.nutau.frontier import Support
from sso.nutau.models import (base_operator, full_basis, mixture_matrix, modulated_density,
                              operator_matrix)
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize_static
from sso.operators import ising

KH = ising(1.0, 0.905, 0.809)


def conserved_weight(L, K):
    spec = diagonalize_static(KH, L)
    M = to_energy_basis(operator_matrix(modulated_density(KH, K, L), L, full_basis(L)), spec.V)
    kd = (spec.k[:, None] - spec.k[None, :]) % L
    w2 = np.abs(M) ** 2
    sec = [w2[kd == k].sum() / w2.sum() for k in range(L)]
    zero = np.abs(spec.omega()) <= 1e-9 * np.abs(spec.omega()).max()
    return max(sec), int(np.argmax(sec)), w2[zero].sum() / w2.sum()


def test_modulated_density_sector_and_parity():
    frac, Ka, w0 = conserved_weight(5, 1)
    assert frac > 1 - 1e-10 and Ka in (1, 4) and w0 > 1e-6          # odd L: conserved part
    frac, Ka, w0 = conserved_weight(6, 1)
    assert frac > 1 - 1e-10 and Ka in (1, 5) and w0 < 1e-20          # even L: none
    H = operator_matrix(modulated_density(KH, 0, 4), 4, full_basis(4))
    assert np.allclose(H, operator_matrix(KH, 4, full_basis(4)))


def test_mixture_and_keep_slowest_grid():
    b = full_basis(3)
    M = mixture_matrix("Zsum:0.5,Z0:-2", 3, b)
    assert np.allclose(M, 0.5 * operator_matrix(base_operator("Zsum"), 3, b)
                       - 2 * operator_matrix(base_operator("Z0"), 3, b))
    w = np.array([0.0, 1e-3, 0.5, 2.0])
    assert np.isclose(width_grid(w, 5, "hard", 2.0, 20.0, 1e-12)[0], 5e-4)
    assert np.isclose(width_grid(w, 5, "gauss", 2.0, 20.0, 1e-12, keep_slowest=True)[0], 1e-3 * (1 + 1e-6))


def test_multiplets_and_overlap():
    assert eig_multiplets([1.0, 1.0, 0.5, 0.2, 0.2]) == [[0, 1], [2], [3, 4]]
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((6, 2)))[0]
    assert np.isclose(subspace_overlap(Q, Q, np), 1.0)
    assert np.isclose(subspace_overlap(Q[:, :1], Q[:, 1:], np), 0.0)


def test_full_space_overlap_crossing_matches_sector_crossing():
    """In the full space the K=0 / K=1 hand-over of Kim-Huse L=6 is a within-space
    crossing; overlap-mode bisection must land on the cross-sector theta*."""
    spec = diagonalize_static(KH, 6, momentum=False)
    arc = within_sector_arc(Support(spec, None, 4), EnergyBasisB(spec.V, 6),
                            dict(theta_lo=10.0, theta_hi=12.5893), mode="overlap",
                            k_ref=4, k_calc=4, rtol=1e-9)
    assert abs(arc["theta_star"] / 11.42494912 - 1) < 1e-8
    assert arc["exact_degeneracy"] and arc["cross_frac_N"] < 1e-8
