"""Frontier solver vs dense brute force, sectors, theta = inf, and regression values."""
import numpy as np
import pytest

from sso.nutau.frontier import Support, solve
from sso.nutau.models import kicked_ising
from sso.nutau.pauli import EnergyBasisB, matrix_to_pauli, pauli_to_matrix, rps_kernel
from sso.nutau.spectrum import diagonalize_floquet, diagonalize_static
from sso.operators import ising

KH = ising(1.0, 0.905, 0.809)


def dense_B_E(spec):
    """B as an explicit n^2 x n^2 matrix in the energy basis."""
    n, L = spec.n, spec.L
    V = np.asarray(spec.V)
    B = rps_kernel(L)
    cols = []
    for j in range(n * n):
        e = np.zeros(n * n, complex)
        e[j] = 1
        O = V @ e.reshape(n, n) @ V.conj().T
        cols.append((V.conj().T @ pauli_to_matrix(B * matrix_to_pauli(O, L), L) @ V).reshape(-1))
    return np.array(cols).T


@pytest.mark.parametrize("kind", ["static", "floquet"])
def test_full_space_vs_brute_force(kind):
    spec = (diagonalize_static(KH, 4, momentum=False) if kind == "static"
            else diagonalize_floquet(kicked_ising()[0], 4, momentum=False))
    o = 4 if kind == "static" else 1
    theta = 2.0
    n = spec.n
    dsq = spec.detuning_sq().reshape(-1)
    Lm = 1 / np.sqrt(1 + theta ** 2 * dsq)
    q = spec.diag_ortho(o)
    U = np.zeros((n * n, o), complex)
    U[np.arange(n) * (n + 1)] = q
    P = np.eye(n * n) - U @ U.conj().T
    M = P @ np.diag(Lm) @ dense_B_E(spec) @ np.diag(Lm) @ P
    assert np.allclose(M, M.conj().T)
    w, v = np.linalg.eigh(M)
    top = v[:, -1]
    vv = np.sum(np.abs(Lm * top) ** 2)
    res = solve(Support(spec, None, o), EnergyBasisB(spec.V, 4), theta, k=2)
    assert np.isclose(res["w"][0], w[-1], rtol=1e-9)
    assert np.isclose(res["Bs"][0], w[-1] / vv, rtol=1e-8)
    assert np.isclose(res["As"][0], np.sum(np.abs(Lm * top) ** 2 * dsq) / vv, rtol=1e-8)


def test_sectors_cover_full_space():
    spec_k = diagonalize_static(KH, 5)
    spec_f = diagonalize_static(KH, 5, momentum=False)
    assert np.allclose(np.sort(spec_k.values), np.sort(spec_f.values))
    Bk = EnergyBasisB(spec_k.V, 5)
    W = [solve(Support(spec_k, K, 4), Bk, 3.0, k=1)["w"][0] for K in range(5)]
    Wf = solve(Support(spec_f, None, 4), EnergyBasisB(spec_f.V, 5), 3.0, k=1)["w"][0]
    assert np.isclose(max(W), Wf, rtol=1e-8)
    assert sum(Support(spec_k, K, 0).dim for K in range(5)) == 4 ** 5


def test_regression_L6_static():
    spec = diagonalize_static(KH, 6)
    Bop = EnergyBasisB(spec.V, 6)
    r0 = solve(Support(spec, 0, 4), Bop, 1.0, k=2)
    assert np.allclose(r0["As"], [0.4228378267266025, 0.35534410068736033], rtol=1e-9)
    assert np.allclose(r0["Bs"], [0.17198587440175173, 0.09419144741061444], rtol=1e-9)
    r1 = solve(Support(spec, 1, 4), Bop, 10.0, k=2)
    assert np.allclose(r1["w"], [0.07511645575552257, 0.03039974378642629], rtol=1e-9)
    assert np.allclose(r1["Bs"], [0.1299484660869532, 0.052338291163651865], rtol=1e-9)


def test_regression_L6_floquet():
    spec = diagonalize_floquet(kicked_ising()[0], 6)
    Bop = EnergyBasisB(spec.V, 6)
    r = solve(Support(spec, 0, 1), Bop, 3.1072, k=2)
    assert np.allclose(r["As"], [0.02691695216611524, 0.03646421696038944], rtol=1e-9)
    assert np.allclose(r["Bs"], [0.17707490440383217, 0.09417833498477216], rtol=1e-9)
    ri = solve(Support(spec, 0, 1, resonant=True), Bop, np.inf, k=2, seed=999)
    assert np.allclose(ri["Bs"], [0.08080235533861023, 0.05977228179906749], rtol=1e-9)
    assert np.all(ri["As"] < 1e-20)
