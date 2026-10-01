"""One seed of the Fourier-correlator DQT estimator (Model B ring) -> seed NPZ.

    python scripts/compute/dqt/fourier_cv.py --L=20 --seed=300 --nhaar=4 --nprod=50 \
        --smax=84 [--gpu] [--engine=chebyshev|krylov] [--dt=0.5] [--nsigma=6] \
        [--J=1 --g=-1.05 --h=0.5 --n=1] [--svals=0.7,1.4,...] [--out=DIR]

Writes ``<out>/modhydro_fourier_cv_L{L}_n{n}_g{g}_h{h}_seed{seed}.npz`` (default
out = sso.config.output_dir("dqt")).  ``--smax`` selects the production s grid
(sso.dqt.fourier.production_svals); ``--svals`` gives an explicit list instead.
See slurm/dqt/campaigns.sh for the production seed families.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.dqt.fourier import production_svals, run_fourier_cv, seed_filename

args = parse_argv()
L = find_value(args, "L", int, 12)
seed = find_value(args, "seed", int, 0)
J = find_value(args, "J", float, 1.0)
g = find_value(args, "g", float, -1.05)
h = find_value(args, "h", float, 0.5)
n = find_value(args, "n", int, 1)
if "svals" in args:
    svals = np.array([float(x) for x in args["svals"].split(",")])
else:
    svals = production_svals(find_value(args, "smax", float, 14.0))
res = run_fourier_cv(L, svals, nhaar=find_value(args, "nhaar", int, 4),
                     nprod=find_value(args, "nprod", int, 40), seed=seed, J=J, g=g, h=h, n=n,
                     dt=find_value(args, "dt", float, 0.5),
                     nsigma=find_value(args, "nsigma", float, 6.0),
                     engine=find_value(args, "engine", str, "chebyshev"),
                     usegpu=find_value(args, "gpu", bool, False))
out = find_value(args, "out", str, None) or output_dir("dqt")
os.makedirs(out, exist_ok=True)
fn = os.path.join(out, seed_filename(L, g, h, seed, n))
np.savez_compressed(fn, **res)
fprint(f"saved -> {fn}\nFOURIER_DONE")
