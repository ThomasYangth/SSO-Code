"""Small-L physics check of the full DQT pipeline against exact diagonalization.

    python scripts/compute/dqt/validate_ed.py [--L=10] [--nhaar=40] [--nprod=400] [--dt=0.5]
        [--svals=0.7,1.4,2,3,4,6,8] [--engines=chebyshev,krylov] [--chebmoment] [--gpu]

For each engine: run_fourier_cv (production estimator incl. control variate) and, with
--chebmoment, the Chebyshev-moment estimator; prints estimate, standard error, ED value
and z = (est - ED)/err for N1, N2, M and nu, and writes everything to
output_dir("dqt", "validation")/validate_ed_L{L}_dt{dt}.npz.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.dqt.chebmoment import chebmoment_estimate
from sso.dqt.exact import exact_reference
from sso.dqt.fourier import run_fourier_cv
from sso.dqt.model import MODEL_B, mfim_hk, to_backend

args = parse_argv()
L = find_value(args, "L", int, 10)
nhaar = find_value(args, "nhaar", int, 40)
nprod = find_value(args, "nprod", int, 400)
dt = find_value(args, "dt", float, 0.5)
gpu = find_value(args, "gpu", bool, False)
svals = np.array([float(x) for x in find_value(args, "svals", str, "0.7,1.4,2,3,4,6,8").split(",")])
engines = find_value(args, "engines", str, "chebyshev,krylov").split(",")
J, g, h = MODEL_B["J"], MODEL_B["g"], MODEL_B["h"]

ref = exact_reference(L, J, g, h, 1, svals, usegpu=gpu)
ref["nu"] = ref["M"] / ref["N1"]
out = {f"ed_{k}": v for k, v in ref.items()}
runs = {}
for eng in engines:
    runs[eng] = run_fourier_cv(L, svals, nhaar, nprod, seed=7, J=J, g=g, h=h, dt=dt,
                               engine=eng, usegpu=gpu, log=lambda *a: None)
if find_value(args, "chebmoment", bool, False):
    cm = chebmoment_estimate(to_backend(mfim_hk(L, J, g, h, 0), gpu),
                             to_backend(mfim_hk(L, J, g, h, 1), gpu), L, svals,
                             nhaar, nprod, seed=7, usegpu=gpu)
    runs["chebmoment"] = cm

fprint(f"L={L} dt={dt} nhaar={nhaar} nprod={nprod}  (z = (est-ED)/SEM)")
for name, r in runs.items():
    r = dict(r)
    r["nu_err"] = (r["M"] / r["N1"]) * np.sqrt((r["M_err"] / r["M"]) ** 2 + (r["N1_err"] / r["N1"]) ** 2)
    r["nu"] = r["M"] / r["N1"]
    fprint(f"--- {name}")
    for k in ("N1", "N2", "M", "nu"):
        z = (r[k] - ref[k]) / r[f"{k}_err"]
        rel = r[k] / ref[k] - 1
        fprint(f"  {k:3s} z: " + " ".join(f"{x:+5.1f}" for x in z)
               + "   rel: " + " ".join(f"{x:+.1e}" for x in rel))
        out[f"{name}_{k}"], out[f"{name}_{k}_err"] = r[k], r[f"{k}_err"]
    if "M_direct" in r:
        fprint("  CV variance gain (M_direct_err/M_err)^2: "
               + " ".join(f"{x:.1f}" for x in (r["M_direct_err"] / r["M_err"]) ** 2))
if {"chebyshev", "krylov"} <= set(runs):
    a, b = runs["chebyshev"], runs["krylov"]
    dC = np.abs(a["C_samples"] - b["C_samples"]).max()
    dm = np.abs(a["m_samples"] - b["m_samples"]).max()
    fprint(f"engine agreement (same seed): max|dC|={dC:.2e}  max|dm|={dm:.2e}")
    out["engine_dC"], out["engine_dm"] = dC, dm
fn = os.path.join(output_dir("dqt", "validation"), f"validate_ed_L{L}_dt{dt:g}.npz")
np.savez_compressed(fn, svals=svals, **out)
fprint(f"saved -> {fn}")
