"""Within-sector (or full-space) branch-crossing arcs of a stored static theta grid.

Replaces NuTau_soft_crossing.py (both locate modes, sector and full space) and
fill_crossing_ksec_novec.py.

Brackets (--detect):
  jump     the largest upward tau jump of the grid (no stored vectors needed;
           works for save_v2=False grids); branch references recomputed
           (top multiplets of k_ref fresh eigenpairs at the bracket ends)
  overlap  every adjacent pair whose stored top multiplets have subspace
           overlap < --overlap_tol (needs stored v2); references = the stored
           top multiplets; the per-member (tau, nu) of degenerate endpoint
           multiplets is printed (identical = symmetry partners, distinct =
           the crossing is inside the multiplet)
  --theta_bracket=lo,hi overrides detection (references recomputed).
Locate (--mode): nu (bisect nu_0 - nu_mid, top pair) or overlap (branch
tracking with --k_calc eigenpairs per trial theta).

Sector: --K=<int> reads $SSO_OUTPUT/<model>_DSsoftKsec_Data/L{L}_K{K}_theta*,
--K=none the full-space grid $SSO_OUTPUT/<model>[OBC]_DSsoft_Data/L{L}_theta*
(--obc=True for open chains); --datadir=<dir> reads the grid from elsewhere.
Writes <grid dir under $SSO_OUTPUT>/crossings/
    cross_L{L}_K{K}_l{lam}k{k_save}o{o}_theta{theta*:.8g}.npz   (sector)
    cross_L{L}_l{lam}k{k_save}o{o}_theta{theta*:.8g}.npz        (full space)
A bracket whose arc (same theta_bracket) exists is skipped unless --override.

Examples:
    python scripts/compute/nutau/crossing_within.py --J=1.0 --hx=2.0 --hz=0.0 \
        --L=12 --K=0 --pows_max=3 --k_save=2 --theta_bracket=0.3981,0.5012 --usegpu=True
    python scripts/compute/nutau/crossing_within.py --L=10 --K=none --detect=overlap \
        --mode=overlap --k_calc=8 --usegpu=True
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.conventions import tau_static
from sso.nutau.crossing import (detect_crossings, largest_tau_jump, multiplet_tau_nu,
                                stored_vectors, within_sector_arc)
from sso.nutau.frontier import Support, effective_o
from sso.nutau.io import grid_files, read_grid
from sso.nutau.models import model_from_args
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize


def main():
    args = parse_argv()
    fprint("arguments:", args)
    kind, model, name = model_from_args(args)
    if kind != "static":
        raise ValueError("within-sector arcs are run on the static grids")
    usegpu = find_value(args, "usegpu", bool, False)
    L = find_value(args, "L", int, 12)
    K_arg = str(args.get("K", "0"))
    K = None if K_arg == "none" else int(K_arg)
    pbc = not find_value(args, "obc", bool, False)
    lam = find_value(args, "lam", float, 0.0)
    o = effective_o(K, find_value(args, "pows_max", int, 3) + 1)
    k_save = find_value(args, "k_save", int, 2)
    mode = find_value(args, "mode", str, "nu")
    detect = find_value(args, "detect", str, "jump")
    override = find_value(args, "override", bool, False)
    sub = (f"{name}_DSsoftKsec_Data" if K is not None
           else f"{name}{'' if pbc else 'OBC'}_DSsoft_Data")
    datadir = str(args.get("datadir", output_dir(sub)))
    outdir = output_dir(sub, "crossings")
    stem = (f"cross_L{L}_K{K}_" if K is not None else f"cross_L{L}_") + f"l{lam}k{k_save}o{o}_theta"

    g = read_grid(datadir, L, K, lam, k_save, o)
    tau = tau_static(g["As"])
    stored = None
    if "theta_bracket" in args:
        lo, hi = (float(x) for x in str(args["theta_bracket"]).split(","))
        i, j = (int(np.argmin(np.abs(g["theta"] - t))) for t in (lo, hi))
        brackets = [dict(theta_lo=g["theta"][i], theta_hi=g["theta"][j], tau_lo=tau[i],
                         tau_hi=tau[j], nu_lo=g["Bs"][i], nu_hi=g["Bs"][j])]
    elif detect == "jump":
        brackets = [largest_tau_jump(g["theta"], tau, g["Bs"])]
    elif detect == "overlap":
        brackets = detect_crossings(grid_files(datadir, L, K, lam, k_save, o),
                                    overlap_tol=find_value(args, "overlap_tol", float, 0.9))
        stored = True
    else:
        raise ValueError(f"unknown --detect={detect}")
    fprint(f"{len(brackets)} bracket(s)")
    if not brackets:
        return

    spec = diagonalize(kind, model, L, pbc=pbc, momentum=K is not None, usegpu=usegpu)
    Bop = EnergyBasisB(spec.V, L, usegpu)
    sup = Support(spec, K, o, lam, usegpu=usegpu)
    for br in brackets:
        fprint(f"bracket theta in [{br['theta_lo']:.6g}, {br['theta_hi']:.6g}]: "
               f"tau {br['tau_lo']:.4g} -> {br['tau_hi']:.4g}, nu {br['nu_lo']:.4g} -> {br['nu_hi']:.4g}"
               + (f", stored top-multiplet overlap {br['overlap']:.4f}"
                  + (" (multiplet may be truncated by k_save)" if br["truncated"] else "")
                  if stored else ""))
        done = [p for p in glob.glob(os.path.join(outdir, stem + "*.npz"))
                if np.allclose(np.load(p)["theta_bracket"], [br["theta_lo"], br["theta_hi"]],
                               rtol=1e-6, atol=0)]
        if done and not override:
            fprint(f"  cached: {os.path.basename(done[0])}")
            continue
        refs = None
        if stored:
            cols = []
            for side in ("lo", "hi"):
                z = np.load(br[f"path_{side}"])
                v2 = z["v2"][:, br[f"idx_{side}"]]
                if v2.shape[1] > 1:
                    t, nu, _ = multiplet_tau_nu(sup, Bop, br[f"theta_{side}"], z["w"][br[f"idx_{side}"]], v2)
                    spread = (nu.max() - nu.min()) / max(abs(nu.max()), 1e-300)
                    fprint(f"  theta_{side} top multiplet: nu={np.array2string(nu, precision=6)} "
                           f"tau={np.array2string(t, precision=4)} -> "
                           + ("symmetry partners" if spread < 1e-6 else f"distinct branches ({spread:.2e})"))
                cols.append(stored_vectors(sup, Bop, v2))
            refs = tuple(cols)
        arc = within_sector_arc(sup, Bop, br, mode=mode, refs=refs,
                                k_ref=find_value(args, "k_ref", int, 8),
                                k_calc=find_value(args, "k_calc", int, 8),
                                n_alpha=find_value(args, "n_alpha", int, 721),
                                ncv=find_value(args, "ncv", int, 50),
                                max_rel_splitting=find_value(args, "max_rel_splitting", float, 1e-6))
        if arc is None:
            continue
        fprint(f"  theta*={arc['theta_star']:.10g}  W={arc['W']:.10g}  dim S={arc['dim_S']}  "
               f"rel splitting {arc['gap_rel']:.2e}  exact={arc['exact_degeneracy']}  "
               f"|P_S ref| = {arc['proj_norm_A']:.4f}, {arc['proj_norm_B']:.4f}")
        np.savez(os.path.join(outdir, stem + f"{arc['theta_star']:.8g}.npz"), **arc, lam=lam,
                 theta_bracket=np.array([br["theta_lo"], br["theta_hi"]]),
                 grid_tau=np.array([br["tau_lo"], br["tau_hi"]]),
                 grid_nu=np.array([br["nu_lo"], br["nu_hi"]]))


if __name__ == "__main__":
    main()
