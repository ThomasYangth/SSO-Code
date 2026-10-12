"""Lorentzian nu(tau) fronts of the DQT energy-density mode for data/fig3panel/.

    python scripts/extract/dqt_lorentz_fronts.py [--raw=DIR] [--out=DIR] [--check=DIR]

Applies the Lorentzian filter m^2/(omega^2+m^2) (KrylovChain convention) to the stored raw curves of the
production seeds (seeds with C_samples / m_samples; no new propagation; sso.dqt.refilter
with the Euler-Maclaurin kink correction) on the default width grid
(sso.dqt.refilter.default_widths and COVER: tau_target >= 1, m T_c >= 8.5, m T_p >= 6).

--raw    seed NPZs modhydro_fourier_cv_L*_seed*.npz (default sso.config.output_dir("dqt");
         archived set: /scratch/gpfs/DABANIN/ty1475/KrylovSize/data_modhydro).
--out    default sso.config.data_dir("fig3panel").
--check  optional directory of fronts written by scripts/compute/dqt/refilter.py; the
         fresh fronts are compared with them.
Writes modhydro_lorentz_L{L}_n1_g-1.05_h+0.5_REFILTER.npz (L = 16..23): width, tau_obs,
tau_err, nu, nu_err, N1, N2, M (+ errors), n_haar, n_prod, n_seeds, y_C, y_P.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import data_dir, output_dir
from sso.dqt.combine import seed_files
from sso.dqt.figdata import LS
from sso.dqt.refilter import default_widths, refilter_filename, refilter_front

KEYS = ("width", "tau_obs", "tau_err", "nu", "nu_err", "N1", "N1_err", "N2", "N2_err", "M",
        "M_err", "n_haar", "n_prod", "n_seeds")

args = parse_argv()
raw = find_value(args, "raw", str, None) or output_dir("dqt")
out = find_value(args, "out", str, None) or data_dir("fig3panel")
check = find_value(args, "check", str, None)
for L in LS:
    files = seed_files(raw, L)
    widths, cover = default_widths(files, "lorentz")
    d = refilter_front(files, "lorentz", widths, **cover)
    fn = os.path.join(out, refilter_filename("lorentz", L))
    np.savez_compressed(fn, **{k: d[k] for k in KEYS}, **cover)
    ok = np.isfinite(d["nu"])
    msg = (f"L={L}: {d['n_seeds']} seeds with curves, {ok.sum()} widths, "
           f"tau/L^2 <= {np.nanmax(d['tau_obs']) / L ** 2:.3f}")
    if check:
        c = np.load(os.path.join(check, refilter_filename("lorentz", L)))
        rel = max(float(np.nanmax(np.abs(d[k] / c[k] - 1))) for k in ("tau_obs", "nu", "nu_err"))
        msg += f"; max rel. deviation from {check}: {rel:.1e}"
    fprint(msg + f" -> {fn}")
