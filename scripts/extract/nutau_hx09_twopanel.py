"""Slim data for figures/nutau_hx0.9_twopanel.pdf (Ising J=1, hx=0.9, small hz, L=12).

Reads (raw root: --src, else $SSO_OUTPUT; read-only), for each hz:
    Ising_J1.0X0.9Z{hz}_DSsoftKsec_Data/L12_K0_theta*l0.0k2o4.npz
    Ising_J1.0X0.9Z{hz}_DSsoftKsec_Data/crossings/cross_L12_K0_*.npz   (within-K0 arcs)
    Ising_J1.0X0.9Z0.03_StaticFilter_Data/filt_L12_K0_YZcur_l0.0o4.npz
Writes data/nutau_hx0.9_twopanel/
    grid_hz{hz}.npz            theta, As, Bs, W
    arc_hz{hz}_theta{t}.npz    tau, nu, theta_star
    filt_hz0.03_YZcur.npz      {gauss,hard,avg}_{As,nu}

Usage:
    python scripts/extract/nutau_hx09_twopanel.py --src=<raw root>
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import parse_argv
from sso.config import data_dir
from sso.nutau.io import raw_root, read_grid, save_slim

HZS = ("0.005", "0.01", "0.02", "0.03", "0.05", "0.07", "0.1", "0.15", "0.2")


def main():
    args = parse_argv()
    root = raw_root(args.get("src"))
    out = data_dir("nutau_hx0.9_twopanel")
    for hz in HZS:
        d = os.path.join(root, f"Ising_J1.0X0.9Z{hz}_DSsoftKsec_Data")
        save_slim(os.path.join(out, f"grid_hz{hz}.npz"), **read_grid(d, 12, 0, 0.0, 2, 4))
        for p in sorted(glob.glob(os.path.join(d, "crossings", "cross_L12_K0_*.npz"))):
            z = np.load(p)
            th = float(z["theta_star"])
            save_slim(os.path.join(out, f"arc_hz{hz}_theta{th:.8g}.npz"),
                      tau=z["tau"], nu=z["nu"], theta_star=th)
    z = np.load(os.path.join(root, "Ising_J1.0X0.9Z0.03_StaticFilter_Data",
                             "filt_L12_K0_YZcur_l0.0o4.npz"))
    save_slim(os.path.join(out, "filt_hz0.03_YZcur.npz"),
              **{f"{k}_{q}": z[f"{k}_{q}"] for k in ("gauss", "hard", "avg") for q in ("As", "nu")})
    print("wrote", out)


if __name__ == "__main__":
    main()
