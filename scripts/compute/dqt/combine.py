"""Pool seed runs into nu(tau) fronts:  *_COMBINED.npz per L.

    python scripts/compute/dqt/combine.py --L=16,17,...  [--raw=DIR] [--out=DIR]

raw defaults to sso.config.output_dir("dqt") (where fourier_cv.py writes), out to the
same directory.  Prints the per-s table (tau_obs, nu, relative error).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.dqt.combine import combine_fourier, combined_filename, seed_files

args = parse_argv()
raw = find_value(args, "raw", str, None) or output_dir("dqt")
out = find_value(args, "out", str, None) or raw
g = find_value(args, "g", float, -1.05)
h = find_value(args, "h", float, 0.5)
for L in [int(x) for x in find_value(args, "L", str, "16").split(",")]:
    files = seed_files(raw, L, g, h)
    d = combine_fourier(files)
    fn = os.path.join(out, combined_filename(L, g, h))
    np.savez_compressed(fn, **d)
    fprint(f"L={L}: {len(files)} seeds, s_max={d['svals'][-1]:g}, n_prod {d['n_prod'].min()}-"
           f"{d['n_prod'].max()}, n_haar {d['n_haar'].min()}-{d['n_haar'].max()} -> {fn}")
    for s, t, te, nu, ne in zip(d["svals"], d["tau_obs"], d["tau_err"], d["nu"], d["nu_err"]):
        fprint(f"  s={s:6.1f}  tau={t:8.3f}+-{te:.3f}  nu={nu:.4f}+-{ne:.4f}  ({100 * ne / nu:.2f}%)")
