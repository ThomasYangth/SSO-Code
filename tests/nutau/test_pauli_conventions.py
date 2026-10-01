"""Pauli basis, RPS kernel and tau conventions."""
import numpy as np

from sso.nutau.conventions import tau_floquet, tau_static
from sso.nutau.pauli import EnergyBasisB, matrix_to_pauli, pauli_to_matrix, rps_kernel

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)


def test_pauli_round_trip():
    rng = np.random.default_rng(0)
    O = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    assert np.allclose(pauli_to_matrix(matrix_to_pauli(O, 3), 3), O)


def test_pauli_coefficients_of_a_string():
    c = matrix_to_pauli(np.kron(np.kron(X, Z), Y), 3)
    assert np.isclose(np.abs(c).max(), 1.0) and np.sum(np.abs(c) > 1e-12) == 1


def test_rps_kernel():
    B = rps_kernel(2)
    assert np.isclose(B[0], 1.0) and np.isclose(B.min(), 1 / 9) and len(B) == 16
    assert np.isclose(np.sum(B == 1 / 3), 6)


def test_nu_of_strings_and_hermiticity():
    rng = np.random.default_rng(1)
    V, _ = np.linalg.qr(rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8)))
    Bop = EnergyBasisB(V, 3)
    for O, nu in ((np.kron(np.kron(X, I2), I2), 1 / 3), (np.kron(np.kron(Z, I2), Y), 1 / 9),
                  (np.eye(8), 1.0)):
        assert np.isclose(Bop.nu(V.conj().T @ O @ V), nu)
    a = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    b = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    Ba = Bop.apply(a).copy()
    Bb = Bop.apply(b).copy()
    assert np.isclose(np.vdot(Ba, b), np.vdot(a, Bb))


def test_tau_conventions():
    w = np.array([1e-3, 0.1, 1.0, 3.0])
    assert np.allclose(tau_floquet(4 * np.sin(w / 2) ** 2), 1 / w)
    assert np.allclose(tau_static(w ** 2), 1 / w)
    assert np.isinf(tau_static(0.0)) and np.isinf(tau_floquet(0.0))
