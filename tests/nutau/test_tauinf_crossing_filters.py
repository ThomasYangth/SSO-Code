"""tau = inf commutant, the arc formula, and the filters."""
import numpy as np

from sso.nutau.crossing import degenerate_pair_arc, sign_change_brackets
from sso.nutau.conventions import tau_static
from sso.nutau.filters import filter_values, sample_periods
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize_static
from sso.nutau.tauinf import ising_r, power_baseline_nu, tauinf_nu
from sso.operators import ising

KH = ising(1.0, 0.905, 0.809)


def test_tauinf_and_baseline_L6():
    spec = diagonalize_static(KH, 6, momentum=False)
    Bop = EnergyBasisB(spec.V, 6)
    ref = [0.251635153, 0.108781992, 0.06628995634, 0.06627598151, 0.06392145222]
    got = [tauinf_nu(spec, Bop, o)["nu"] for o in range(1, 6)]
    assert np.allclose(got, ref, rtol=1e-8)
    base = power_baseline_nu(spec, Bop, 5)
    assert np.allclose(base, [1, 0.2434923456, 0.09080103882, 0.03662630996,
                              0.02412318209, 0.02349393406], rtol=1e-8)
    assert np.isclose(base[1], ising_r(1.0, 0.905, 0.809))


def test_arc_endpoints_and_brackets():
    rng = np.random.default_rng(2)
    wa, wb = rng.standard_normal(6) + 0j, rng.standard_normal(6) + 0j
    wb -= np.vdot(wa, wb) / np.vdot(wa, wa) * wa
    wa, wb = wa / np.linalg.norm(wa), wb / np.linalg.norm(wb)
    m, d = rng.uniform(0.2, 1, 6), rng.uniform(0.1, 2, 6)
    arc = degenerate_pair_arc(0.3, wa, wb, m, d, tau_static, np.array([0.0, np.pi / 2]))
    Na, Da = np.sum(m ** 2 * np.abs(wa) ** 2), np.sum(m ** 2 * d * np.abs(wa) ** 2)
    assert np.isclose(arc["nu"][0], 0.3 / Na) and np.isclose(arc["tau"][0], np.sqrt(Na / Da))
    th = np.array([1.0, 2.0, 3.0, 4.0])
    br = sign_change_brackets(th, np.array([1, 2, 3, 4.0]), th, np.array([2, 2.5, 2.5, 2.0]))
    assert [(lo, hi) for lo, hi, _, _ in br] == [(2.0, 3.0)]


def test_filters():
    w = np.linspace(-3, 3, 7)
    assert np.allclose(filter_values("avg", w, 1, np, floquet=True), 1)
    assert np.allclose(filter_values("avg", w, 1e-12, np), 1)
    assert np.allclose(filter_values("hard", w, 1.5, np), np.abs(w) <= 1.5)
    N = 7
    direct = np.mean([np.exp(1j * w * t) for t in range(N)], axis=0)
    assert np.allclose(filter_values("avg", w, N, np, floquet=True), direct)
    assert sample_periods(1e6, 70)[0] == 1 and sample_periods(1e6, 70)[-1] == 10 ** 6
