"""Build the slim data of fig3panel (data/fig3panel/) from the raw production output.

    python scripts/extract/dqt_fig3panel.py [--raw=DIR] [--ti=DIR] [--out=DIR]

--raw  directory with the seed NPZs modhydro_fourier_cv_L*_seed*.npz (default
       sso.config.output_dir("dqt")); the archived production set lives in
       /scratch/gpfs/DABANIN/ty1475/KrylovSize/data_modhydro on Della.  If that directory
       also holds *_COMBINED.npz fronts, the freshly pooled fronts are checked against them.
--ti   directory with the TI-ansatz q-scan tables nutau_M{M}_q{q}.dat (default
       sso.config.output_dir("ti")); archived in OGH/data.
--out  default sso.config.data_dir("fig3panel").

Writes, per L = 16..23: the pooled nu(tau) front (*_COMBINED.npz, panel a) and
correlators_L{L}.npz (C(t), RPS norm R(t); panel b); grps_L20.npz (|G_RPS| window, inset
of b); ti_q_scan/*.dat (panel c: M = 8..11, q = 0 and every q > 0 common to all M).
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import data_dir, output_dir
from sso.dqt.combine import (combine_fourier, combined_filename, g_rps_window,
                             pooled_correlators, seed_files)
from sso.dqt.figdata import INSET_L, LS, TWIN
from sso.tiansatz import MSTAB_MS, discover_qs, q_tag

args = parse_argv()
raw = find_value(args, "raw", str, None) or output_dir("dqt")
ti = find_value(args, "ti", str, None) or output_dir("ti")
out = find_value(args, "out", str, None) or data_dir("fig3panel")
os.makedirs(out, exist_ok=True)

worst = 0.0
for L in LS:
    files = seed_files(raw, L)
    front = combine_fourier(files)
    np.savez_compressed(os.path.join(out, combined_filename(L)), **front)
    msg = f"L={L}: {len(files)} seeds, {len(front['svals'])} s-values"
    arch = os.path.join(raw, combined_filename(L))
    if os.path.isfile(arch):
        a = np.load(arch)
        rel = max(float(np.max(np.abs(front[k] - a[k]) / np.maximum(np.abs(a[k]), 1e-300)))
                  for k in a.files)
        worst = max(worst, rel)
        msg += f"; max rel. deviation from archived COMBINED = {rel:.1e}"
    cor = pooled_correlators(files)
    np.savez_compressed(os.path.join(out, f"correlators_L{L}.npz"), **cor)
    fprint(msg + f"; curves from {cor['nhaar']} Haar / {cor['nprod']} product states")
t, absG, nst = g_rps_window(seed_files(raw, INSET_L), TWIN * INSET_L ** 2)
np.savez_compressed(os.path.join(out, f"grps_L{INSET_L}.npz"), t=t, absG=absG, nstates=nst)
fprint(f"G_RPS window L={INSET_L}: {len(t)}x{len(t)} from {nst} product states")
if worst > 1e-12:
    raise SystemExit(f"pooled fronts deviate from archived COMBINED by {worst:.1e} > 1e-12")

dst = os.path.join(out, "ti_q_scan")
os.makedirs(dst, exist_ok=True)
qs = [0.0] + discover_qs(ti, MSTAB_MS)
for M in MSTAB_MS:
    for q in qs:
        shutil.copy2(os.path.join(ti, f"nutau_M{M}_q{q_tag(q)}.dat"), dst)
fprint(f"copied {len(MSTAB_MS) * len(qs)} q-scan tables (M={list(MSTAB_MS)}, {len(qs)} q's) -> {dst}")
