"""Slim data for figures/kimhuse_three_panel.pdf (Kim-Huse J=1, hx=0.905, hz=0.809).

Reads (raw root: --src, else $SSO_OUTPUT; read-only)
    Ising_J1.0X0.905Z0.809_DSsoftKsec_Data/L{6..12}_K0_theta*l0.0k2o4.npz, L{L}_K1_theta*l0.0k2o0.npz
    Ising_J1.0X0.905Z0.809_DSsoftKsec_Data/crossings/crossK_L{L}_K0K1_l0.0_theta*.npz
    tau = inf tables: either the driver output
        Ising_J1.0X0.905Z0.809_TauInf_Data/{tauinf,baseline}_L{L}.csv   (L = 6..14)
    or, with --tauinf_csv=<file> --baseline_csv=<file>, the original merged CSVs
    (benchmark_nutau_ds_tauinf.csv column nu_new_tauinf; baseline_hn_nu.csv).
Writes data/kimhuse_three_panel/
    grid_L{L}_K0.npz, grid_L{L}_K1.npz    theta, As, Bs, W
    crossK_L{L}.npz                       tau, nu, theta_star
    tauinf.csv  (L,o,nu_tauinf)   baseline.csv  (L,n,nu_hn)

Usage:
    python scripts/extract/nutau_kimhuse_three_panel.py --src=<raw root> \
        [--tauinf_csv=<csv> --baseline_csv=<csv>]
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import parse_argv
from sso.config import data_dir
from sso.nutau.io import raw_root, read_csv, read_grid, save_slim

MODEL = "Ising_J1.0X0.905Z0.809"


def main():
    args = parse_argv()
    root = raw_root(args.get("src"))
    out = data_dir("kimhuse_three_panel")
    d = os.path.join(root, f"{MODEL}_DSsoftKsec_Data")
    for L in range(6, 13):
        for K, o in ((0, 4), (1, 0)):
            save_slim(os.path.join(out, f"grid_L{L}_K{K}.npz"), **read_grid(d, L, K, 0.0, 2, o))
        arcs = sorted(glob.glob(os.path.join(d, "crossings", f"crossK_L{L}_K0K1_l0.0_theta*.npz")))
        if len(arcs) != 1:
            raise RuntimeError(f"expected one cross-K arc at L={L}, found {len(arcs)}")
        z = np.load(arcs[0])
        save_slim(os.path.join(out, f"crossK_L{L}.npz"), tau=z["tau"], nu=z["nu"],
                  theta_star=float(z["theta_star"]))

    if "tauinf_csv" in args:
        t = read_csv(args["tauinf_csv"])
        rows = sorted(zip(t["L"].astype(int), t["o"].astype(int), t["nu_new_tauinf"]))
        b = read_csv(args["baseline_csv"])
        base = sorted(zip(b["L"].astype(int), b["n"].astype(int), b["nu_hn"]))
    else:
        rows, base = [], []
        tdir = os.path.join(root, f"{MODEL}_TauInf_Data")
        for L in range(6, 15):
            t = read_csv(os.path.join(tdir, f"tauinf_L{L}.csv"))
            rows += zip(t["L"].astype(int), t["o"].astype(int), t["nu_tauinf"])
            b = read_csv(os.path.join(tdir, f"baseline_L{L}.csv"))
            base += zip(b["L"].astype(int), b["n"].astype(int), b["nu_hn"])
    with open(os.path.join(out, "tauinf.csv"), "w") as f:
        f.write("L,o,nu_tauinf\n")
        f.writelines(f"{L},{o},{nu:.10g}\n" for L, o, nu in sorted(rows))
    with open(os.path.join(out, "baseline.csv"), "w") as f:
        f.write("L,n,nu_hn\n")
        f.writelines(f"{L},{n},{nu:.10g}\n" for L, n, nu in sorted(base))
    print("wrote", out)


if __name__ == "__main__":
    main()
