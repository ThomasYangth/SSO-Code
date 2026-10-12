"""Slim data for figure_G (a): Kim-Huse L = 12 filtered operators (o4).

    python scripts/extract/nutau_kimhuse_filters.py [--src=DIR]

Reads <src>/Ising_J1.0X0.905Z0.809_StaticFilter_Data/filt_L12_<tag>_l0.0o{4|0}.npz for
K0_Zsum, K0_Xsum, Knone_Z0 (slurm/nutau/filter_kimhuse_L12.slurm), K0_OPT1S (the optimal
single-site Pauli sum, o4) and K1_Hdens (H_q, q = 2 pi/L, o0)
(slurm/nutau/filter_kimhuse_onesite_Hq_L12.slurm); src defaults to $SSO_OUTPUT.  Writes
data/kimhuse_three_panel/filt_L12_<tag>.npz with {lorentz,gauss,hard,avg}_{As,nu}, and
onesite_opt_L12_o4.npz (the optimal direction n and nu_inf).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import parse_argv
from sso.config import data_dir
from sso.nutau.io import raw_root, save_slim

KINDS = ("lorentz", "gauss", "hard", "avg")

args = parse_argv()
root = raw_root(args.get("src"))
out = data_dir("kimhuse_three_panel")
d = os.path.join(root, "Ising_J1.0X0.905Z0.809_StaticFilter_Data")
for tag, o in (("K0_Zsum", 4), ("K0_Xsum", 4), ("Knone_Z0", 4), ("K0_OPT1S", 4), ("K1_Hdens", 0)):
    path = os.path.join(d, f"filt_L12_{tag}_l0.0o{o}.npz")
    if not os.path.exists(path):
        print("missing", path)
        continue
    z = np.load(path)
    save_slim(os.path.join(out, f"filt_L12_{tag}.npz"), **{f"{k}_{q}": z[f"{k}_{q}"] for k in KINDS for q in ("As", "nu")})
opt = os.path.join(d, "onesite_opt_L12_o4.npz")
if os.path.exists(opt):
    z = np.load(opt)
    save_slim(os.path.join(out, "onesite_opt_L12_o4.npz"), **{k: z[k] for k in z.files})
print("wrote", out)
