"""Lorentzian time kernels and the re-filtering of stored curves."""
import numpy as np

from sso.dqt.combine import pool_samples
from sso.dqt.exact import exact_reference
from sso.dqt.fourier import run_fourier_cv
from sso.dqt.kernels import lor2_K, lor2_win, lor_win, neg_lor2_Kpp, neg_lor2_win_pp
from sso.dqt.refilter import seed_samples

_trapz = getattr(np, "trapezoid", None) or np.trapz


def test_lorentz_kernels_fourier_transforms():
    """lor_win <-> f = m^2/(W^2+m^2); lor2_win <-> f^2 (and W^2 f^2); lor2_K <-> f^4."""
    m = 0.6
    t = np.linspace(-150, 150, 600001)
    for Om in (0.0, 0.3, 1.1, 2.5):
        f = m ** 2 / (Om ** 2 + m ** 2)
        ft = lambda k: _trapz(k(m, t) * np.cos(Om * t), t)
        assert np.isclose(ft(lor_win), f, atol=2e-8)
        assert np.isclose(ft(lor2_win), f ** 2, atol=1e-9)
        assert np.isclose(ft(neg_lor2_win_pp), Om ** 2 * f ** 2, atol=1e-8)
        assert np.isclose(ft(lor2_K), f ** 4, atol=1e-9)
        assert np.isclose(ft(neg_lor2_Kpp), Om ** 2 * f ** 4, atol=1e-8)


def test_gauss_refilter_reproduces_samples_and_lorentz_matches_ed():
    L, sv = 8, np.array([0.7, 1.4, 2.0, 4.0, 8.0, 12.0])
    run = run_fourier_cv(L, sv, 30, 200, seed=3, log=lambda *a: None)
    N1, N2, V0, V2, A = seed_samples(run, "gauss", sv)
    assert np.allclose(N1, run["N1_samples"], rtol=1e-13) and np.allclose(A, run["a_samples"], rtol=1e-13)
    for kind, c in (("lorentz", 1.0), ("lorentz2", np.sqrt(5.0))):
        ms = c / np.array([2.0, 3.0, 4.0])
        fr = pool_samples(*seed_samples(run, kind, ms, y_C=10, y_P=10), run["b0_samples"],
                          run["b2_samples"], run["GHS"], run["Rg"])
        ed = exact_reference(L, 1.0, -1.05, 0.5, 1, ms, kind=kind)
        for k in ("N1", "N2", "M"):
            assert np.all(np.abs(fr[k] - ed[k]) < 4 * fr[k + "_err"]), (kind, k)
