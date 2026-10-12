"""Slim data for figure_D (TFIM J=1, hx=2.0, hz=0.0): sector frontiers L = 6..12 and tau = inf.

    python scripts/extract/nutau_tfim_frontier.py --src=<root L 6..11> --src12=<root L 12>

Reads (read-only), MODEL = Ising_J1.0X2.0Z0.0:
    <src>/MODEL_DSsoftKsec_Data/L{6..11}_K{0,1}_theta*l0.0k2o{4,0}.npz      (tfim_frontier_L6_11.slurm)
    <src>/MODEL_DSsoftKsec_Data/crossings/{crossK_L*_K0K1_*, cross_L*_K0_l0.0k2o4_*}.npz
                                                                           (tfim_arcs_L6_11.slurm)
    <src12>/MODEL_DSsoftKsec_Data/L12_..., crossings/{crossK_L12_*, cross_L12[_K0]_l0.0k2o4_*}
                                                    (tfim_frontier_L12.slurm, within_tfim_L12.slurm)
    <src>/MODEL_TauInf_Data/{tauinf,baseline}_L{6..12}.csv                 (tfim_tauinf_L6_12.slurm)
src defaults to $SSO_OUTPUT, src12 to src.
Writes data/tfim_frontier/
    grid_L{L}_K0.npz, grid_L{L}_K1.npz    theta, As, Bs, W
    crossK_L{L}.npz, withinK0_L{L}.npz    tau, nu, theta_star
    tauinf.csv (L,o,nu_tauinf)            baseline.csv (L,n,nu_hn)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sso.cli import parse_argv
from sso.config import data_dir
from sso.nutau.io import raw_root, read_csv, read_grid, save_slim

MODEL = "Ising_J1.0X2.0Z0.0"


def one_arc(paths, what, L):
    if len(paths) != 1:
        raise RuntimeError(f"expected one {what} arc at L={L}, found {len(paths)}: {paths}")
    z = np.load(paths[0])
    return dict(tau=z["tau"], nu=z["nu"], theta_star=float(z["theta_star"]))


def main():
    args = parse_argv()
    root = raw_root(args.get("src"))
    root12 = args.get("src12") or root
    out = data_dir("tfim_frontier")
    os.makedirs(out, exist_ok=True)
    for L in range(6, 13):
        d = os.path.join(root12 if L == 12 else root, f"{MODEL}_DSsoftKsec_Data")
        for K, o in ((0, 4), (1, 0)):
            save_slim(os.path.join(out, f"grid_L{L}_K{K}.npz"), **read_grid(d, L, K, 0.0, 2, o))
        c = os.path.join(d, "crossings")
        save_slim(os.path.join(out, f"crossK_L{L}.npz"),
                  **one_arc(glob.glob(os.path.join(c, f"crossK_L{L}_K0K1_l0.0_theta*.npz")), "cross-K", L))
        save_slim(os.path.join(out, f"withinK0_L{L}.npz"),
                  **one_arc(glob.glob(os.path.join(c, f"cross_L{L}_K0_l0.0k2o4_theta*.npz"))
                            + glob.glob(os.path.join(c, f"cross_L{L}_l0.0k2o4_theta*.npz")), "within-K0", L))
    rows, base = [], []
    tdir = os.path.join(root, f"{MODEL}_TauInf_Data")
    for L in range(6, 13):
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
