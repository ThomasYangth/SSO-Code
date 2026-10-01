"""Cross-sector (K_a <-> K_b) superposition arcs from stored theta grids.

Replaces NuTau_ksec_crossK.py.  Brackets are the sign changes of
W_Ka - W_Kb on the stored grids (no eigensolve); theta* by false position;
arc nu = W/N(alpha), sigma^2 = D/N(alpha), alpha in [0, pi/2].
Reads   $SSO_OUTPUT/<model>_DSsoftKsec_Data/L{L}_K{K}_theta*l{lam}k{k_save}o{o}.npz
        (o pinned: pows_max+1 for K=0, 0 otherwise)
        (--datadir=<dir> reads the grid from elsewhere, e.g. an archive)
Writes  $SSO_OUTPUT/<model>_DSsoftKsec_Data/crossings/crossK_L{L}_K{Ka}K{Kb}_l{lam}_theta{theta*:.8g}.npz

Example:
    python scripts/compute/nutau/crossing_sectors.py --J=1.0 --hx=0.905 --hz=0.809 \
        --Lmin=6 --Lmax=12 --Ka=0 --Kb=1 --pows_max=3 --k_save=2 --usegpu=True
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.crossing import cross_sector_arc, sign_change_brackets
from sso.nutau.frontier import Support, effective_o
from sso.nutau.io import read_grid
from sso.nutau.models import model_from_args
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize


def main():
    args = parse_argv()
    fprint("arguments:", args)
    kind, model, name = model_from_args(args)
    usegpu = find_value(args, "usegpu", bool, False)
    Lmin = find_value(args, "Lmin", int, 6)
    Lmax = find_value(args, "Lmax", int, Lmin)
    Ka, Kb = find_value(args, "Ka", int, 0), find_value(args, "Kb", int, 1)
    lam = find_value(args, "lam", float, 0.0)
    o = find_value(args, "pows_max", int, 3) + 1
    k_save = find_value(args, "k_save", int, 2)
    override = find_value(args, "override", bool, False)
    datadir = str(args.get("datadir", output_dir(f"{name}_DSsoftKsec_Data")))
    outdir = output_dir(f"{name}_DSsoftKsec_Data", "crossings")

    for L in range(Lmin, Lmax + 1):
        ga = read_grid(datadir, L, Ka, lam, k_save, effective_o(Ka, o))
        gb = read_grid(datadir, L, Kb, lam, k_save, effective_o(Kb, o))
        brackets = sign_change_brackets(ga["theta"], ga["W"], gb["theta"], gb["W"])
        fprint(f"\nL={L}: {len(brackets)} W_K{Ka} - W_K{Kb} sign change(s)")
        if not brackets:
            continue
        spec = diagonalize(kind, model, L, usegpu=usegpu)
        Bop = EnergyBasisB(spec.V, L, usegpu)
        sup_a = Support(spec, Ka, o, lam, usegpu=usegpu)
        sup_b = Support(spec, Kb, o, lam, usegpu=usegpu)
        stem = os.path.join(outdir, f"crossK_L{L}_K{Ka}K{Kb}_l{lam}_theta")
        for lo, hi, g_lo, g_hi in brackets:
            done = [p for p in glob.glob(stem + "*.npz")
                    if np.allclose(np.load(p)["theta_bracket"], [lo, hi], rtol=1e-6, atol=0)]
            if done and not override:
                fprint(f"  bracket [{lo:.6g}, {hi:.6g}] cached: {os.path.basename(done[0])}")
                continue
            fprint(f"  bracket [{lo:.6g}, {hi:.6g}]")
            arc = cross_sector_arc(sup_a, sup_b, Bop, (lo, hi, g_lo, g_hi),
                                   tol=find_value(args, "tol", float, 1e-10),
                                   max_iter=find_value(args, "max_iter", int, 12),
                                   n_alpha=find_value(args, "n_alpha", int, 721))
            th = arc["theta_star"]
            fprint(f"    theta*={th:.10g}  W={arc['W']:.10g}  "
                   f"rel degeneracy {arc['rel_degeneracy']:.2e}  ({arc['n_solves']} solves)")
            np.savez(stem + f"{th:.8g}.npz", **arc, K_a=Ka, K_b=Kb, L=L, lam=lam,
                     theta_bracket=np.array([lo, hi]),
                     endpoint_a=np.array([arc["tau"][0], arc["nu"][0]]),
                     endpoint_b=np.array([arc["tau"][-1], arc["nu"][-1]]),
                     exact_degeneracy=bool(arc["rel_degeneracy"] < 1e-6))


if __name__ == "__main__":
    main()
