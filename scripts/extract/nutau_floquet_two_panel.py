"""Slim data for figures/floquet_two_panel.pdf (kicked Ising g=0.9, J=1, h=0.809).

Reads (raw root: --src, else $SSO_OUTPUT; read-only)
    KickedIsing_g0.9J1.0h0.809_DSFloquet_Data/L{6..12}_K0_theta*l0.0k2o1.npz
    KickedIsing_g0.9J1.0h0.809_FloquetHardCut_Data/{gausscut,hardcut}_L12_{K0_Zsum,K0_Xsum,Knone_Z0}_l0.0.npz
    KickedIsing_g0.9J1.0h0.809_FloquetTrajectory_Data/ftraj_L12_Knone_{Zsum,Z0,Xsum}_N1e+06l0.0.npz
    --ti_dir/nutau_floquet_M{6..12}_g0p9tF1p0_gpures.dat   (TI-ansatz tables, copied verbatim)
Writes data/floquet_two_panel/
    grid_L{L}_K0.npz          theta, As, Bs, W, d_res   (top branch, theta = inf included)
    {gausscut,hardcut}_L12_K{tag}_{op}.npz   width, As, nu, kept
    ftraj_L12_Knone_{op}.npz  N, As_avg, nu_avg
    nutau_floquet_M{M}_g0p9tF1p0_gpures.dat

Usage:
    python scripts/extract/nutau_floquet_two_panel.py --src=<raw root> --ti_dir=<TI data dir>
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import parse_argv
from sso.config import data_dir
from sso.nutau.io import raw_root, read_grid, save_slim

MODEL = "KickedIsing_g0.9J1.0h0.809"
OPS = (("Zsum", "0"), ("Xsum", "0"), ("Z0", "none"))


def main():
    args = parse_argv()
    root = raw_root(args.get("src"))
    out = data_dir("floquet_two_panel")
    for L in range(6, 13):
        g = read_grid(os.path.join(root, f"{MODEL}_DSFloquet_Data"), L, 0, 0.0, 2, 1, kind="floquet")
        save_slim(os.path.join(out, f"grid_L{L}_K0.npz"), **g)
    cut = os.path.join(root, f"{MODEL}_FloquetHardCut_Data")
    trj = os.path.join(root, f"{MODEL}_FloquetTrajectory_Data")
    for op, K in OPS:
        for tag in ("gausscut", "hardcut"):
            z = np.load(os.path.join(cut, f"{tag}_L12_K{K}_{op}_l0.0.npz"))
            save_slim(os.path.join(out, f"{tag}_L12_K{K}_{op}.npz"), width=z["omega_star"],
                      **{k: z[k] for k in ("As", "nu", "kept")})
        z = np.load(os.path.join(trj, f"ftraj_L12_Knone_{op}_N1e+06l0.0.npz"))
        save_slim(os.path.join(out, f"ftraj_L12_Knone_{op}.npz"),
                  **{k: z[k] for k in ("N", "As_avg", "nu_avg")})
    ti = args.get("ti_dir")
    if ti is None:
        raise SystemExit("--ti_dir=<directory of nutau_floquet_M*_g0p9tF1p0_gpures.dat> is required")
    for M in range(6, 13):
        name = f"nutau_floquet_M{M}_g0p9tF1p0_gpures.dat"
        shutil.copy(os.path.join(ti, name), os.path.join(out, name))
    print("wrote", out)


if __name__ == "__main__":
    main()
