"""Re-filter the stored DQT curves (seeds with raw curves) with another filter.

    python scripts/compute/dqt/refilter.py --L=16,...,23 [--kind=lorentz] [--raw=DIR] [--out=DIR]
        [--y_C=..] [--y_P=..] [--tau_min=1.0]      (kind: lorentz (default), lorentz2, gauss)

No new propagation: N1, N2, V0, V2 and the product-state amplitudes are recomputed
from C_samples / m_samples (:mod:`sso.dqt.refilter`).  Width grid and coverage:
sso.dqt.refilter.default_widths / COVER (tau_target from --tau_min up to the largest
width some seed covers).
Writes <out>/modhydro_{kind}_L{L}_n1_g-1.05_h+0.5_REFILTER.npz (keys of
pool_samples plus width, n_seeds, y_C, y_P).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.dqt.combine import _with_curves, seed_files
from sso.dqt.refilter import default_widths, refilter_filename, refilter_front

args = parse_argv()
raw = find_value(args, "raw", str, None) or output_dir("dqt")
out = find_value(args, "out", str, None) or raw
os.makedirs(out, exist_ok=True)
kind = find_value(args, "kind", str, "lorentz")
y_C = find_value(args, "y_C", float, None)
y_P = find_value(args, "y_P", float, None)
tau_min = find_value(args, "tau_min", float, 1.0)
for L in [int(x) for x in find_value(args, "L", str, "16").split(",")]:
    files = _with_curves(seed_files(raw, L))
    widths, cover = default_widths(files, kind, tau_min, y_C=y_C, y_P=y_P)
    d = refilter_front(files, kind, widths, **cover)
    fn = os.path.join(out, refilter_filename(kind, L))
    np.savez_compressed(fn, **cover, **d)
    fprint(f"L={L}: {d['n_seeds']} seeds with curves -> {fn}")
    for w, t, te, nu, ne, nh, npr in zip(d["width"], d["tau_obs"], d["tau_err"], d["nu"],
                                         d["nu_err"], d["n_haar"], d["n_prod"]):
        fprint(f"  w={w:8.4f}  tau={t:8.3f}+-{te:.3f}  nu={nu:.4f}+-{ne:.4f}  haar {nh} prod {npr}")
