"""DQT estimators vs exact diagonalization, and seed pooling (CPU, tiny L)."""
import numpy as np
import pytest

from sso.dqt.chebmoment import chebmoment_estimate
from sso.dqt.combine import combine_fourier, g_rps_window, pooled_correlators
from sso.dqt.exact import exact_reference
from sso.dqt.fourier import run_fourier_cv, seed_filename
from sso.dqt.model import mfim_hk

L, J, G, H = 8, 1.0, -1.05, 0.5
SV = np.array([1.0, 2.0, 3.0])
quiet = lambda *a: None


@pytest.fixture(scope="module")
def ref():
    return exact_reference(L, J, G, H, 1, SV)


@pytest.fixture(scope="module")
def run():
    return run_fourier_cv(L, SV, nhaar=24, nprod=150, seed=1, dt=0.1, log=quiet)


def _z(est, err, exact):
    return np.abs(est - exact) / err


def test_fourier_vs_ed(run, ref):
    # N1, N2 within 4 standard errors; plus a loose relative check (DQT error ~2^{-L/2})
    for k in ("N1", "N2", "M"):
        assert np.all(_z(run[k], run[f"{k}_err"], ref[k]) < 4), k
        assert np.allclose(run[k], ref[k], rtol=0.15), k


def test_control_variate_reduces_variance(run):
    assert np.all(run["M_err"][1:] < run["M_direct_err"][1:])


def test_chebmoment_vs_ed(ref):
    H0 = mfim_hk(L, J, G, H, 0)
    Hk = mfim_hk(L, J, G, H, 1)
    est = chebmoment_estimate(H0, Hk, L, SV, nhaar=16, nprod=150, seed=2, nsigma=5.0)
    for k in ("N1", "N2", "M"):
        assert np.all(_z(est[k], est[f"{k}_err"], ref[k]) < 4), k


def test_combine(tmp_path):
    # two seeds on a prefix-extended s grid (as in production)
    runs = [run_fourier_cv(6, [1.0, 2.0], 3, 6, seed=10, dt=0.25, log=quiet),
            run_fourier_cv(6, [1.0, 2.0, 3.0], 3, 6, seed=11, dt=0.25, log=quiet)]
    files = []
    for r, s in zip(runs, (10, 11)):
        f = tmp_path / seed_filename(6, G, H, s)
        np.savez_compressed(f, **r)
        files.append(str(f))
    # single seed: combine reproduces the per-seed estimates exactly
    one = combine_fourier(files[:1])
    for k in ("N1", "N2", "M", "M_err", "tau_obs", "nu", "cc"):
        assert np.allclose(one[k], runs[0][k], rtol=1e-13, atol=0), k
    both = combine_fourier(files)
    assert list(both["n_haar"]) == [6, 6, 3] and list(both["n_prod"]) == [12, 12, 6]
    n1 = np.concatenate([runs[0]["N1_samples"][:, 0], runs[1]["N1_samples"][:, 0]])
    assert np.isclose(both["N1"][0], n1.mean(), rtol=1e-14)
    assert np.allclose(both["nu"][2:], runs[1]["nu"][2:], rtol=1e-13)
    cor = pooled_correlators(files)
    assert cor["nprod"] == 12 and np.isclose(cor["C"][0], np.mean(
        np.concatenate([r["C_samples"][:, 0] for r in runs])).real)
    t, absG, nst = g_rps_window(files, 2.0)
    assert nst == 12 and np.allclose(absG, absG.T)
    assert np.allclose(np.diag(absG)[len(t) // 2], cor["R"][0])


def test_production_grid():
    from sso.dqt.fourier import production_svals
    lens = [len(production_svals(s)) for s in (54, 62, 68, 76, 84, 88, 96, 104)]
    assert lens == [30, 34, 37, 41, 45, 47, 51, 55]
    assert np.array_equal(production_svals(20), production_svals(54)[:13])
