"""Slim data for figures/sector_L12_kimhuse_and_tfim.pdf (Kim-Huse and TFIM, L=12).

Reads (raw root: --src, else $SSO_OUTPUT; read-only), for
model in {Ising_J1.0X0.905Z0.809, Ising_J1.0X2.0Z0.0}:
    {model}_DSsoftKsec_Data/L12_K0_theta*l0.0k2o4.npz, L12_K1_theta*l0.0k2o0.npz
    {model}_DSsoftKsec_Data/crossings/crossK_L12_K0K1_l0.0_theta*.npz   (cross-sector arcs)
    {model}_DSsoftKsec_Data/crossings/cross_L12[_K0]_l0.0k2o4_theta*.npz (within-K0 arcs)
Writes data/sector_L12_kimhuse_and_tfim/
    {model}_grid_K0.npz, {model}_grid_K1.npz   theta, As, Bs, W
    {model}_crossK_theta{theta*}.npz           tau, nu, theta_star
    {model}_withinK0_theta{theta*}.npz         tau, nu, theta_star

Usage:
    python scripts/extract/nutau_sector_L12.py --src=<raw root>
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import parse_argv
from sso.config import data_dir
from sso.nutau.io import raw_root, read_grid, save_slim

MODELS = ("Ising_J1.0X0.905Z0.809", "Ising_J1.0X2.0Z0.0")


def slim_arcs(files, out, prefix):
    for p in files:
        z = np.load(p)
        th = float(z["theta_star"])
        save_slim(os.path.join(out, f"{prefix}_theta{th:.8g}.npz"),
                  tau=z["tau"], nu=z["nu"], theta_star=th)


def main():
    args = parse_argv()
    root = raw_root(args.get("src"))
    out = data_dir("sector_L12_kimhuse_and_tfim")
    for model in MODELS:
        d = os.path.join(root, f"{model}_DSsoftKsec_Data")
        for K, o in ((0, 4), (1, 0)):
            g = read_grid(d, 12, K, 0.0, 2, o)
            save_slim(os.path.join(out, f"{model}_grid_K{K}.npz"), **g)
        c = os.path.join(d, "crossings")
        slim_arcs(glob.glob(os.path.join(c, "crossK_L12_K0K1_l0.0_theta*.npz")), out, f"{model}_crossK")
        slim_arcs(glob.glob(os.path.join(c, "cross_L12_l0.0k2o4_theta*.npz"))
                  + glob.glob(os.path.join(c, "cross_L12_K0_l0.0k2o4_theta*.npz")),
                  out, f"{model}_withinK0")
    print("wrote", out)


if __name__ == "__main__":
    main()
