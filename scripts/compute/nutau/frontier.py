"""Soft nu(tau) frontier on a theta grid, per doubled-momentum sector.

Static (--model=Ising) replaces NuTau_ds_soft_ksec.py (--K=<list>) and
NuTau_ds_soft.py (--K=none, optionally --obc=True); Floquet
(--model=KickedIsing) replaces NuTau_ds_Floquet.py.  One NPZ per (L, K, theta)
with the original names and keys, in

    $SSO_OUTPUT/<model>_DSsoftKsec_Data/   static, momentum sectors
    $SSO_OUTPUT/<model>[OBC]_DSsoft_Data/  static, full space
    $SSO_OUTPUT/<model>_DSFloquet_Data/    Floquet

Projector: static o = pows_max + 1 (span{I..H^pows_max}); Floquet o = 1 (the
identity; --ortho_identity=False for o = 0).  K != 0 always uses o = 0.

Example (Kim-Huse production grid):
    python scripts/compute/nutau/frontier.py --model=Ising --J=1.0 --hx=0.905 \
        --hz=0.809 --Lmin=6 --Lmax=12 --K=0,1,2 --pows_max=3 --k_save=2 \
        --lam=0.0 --theta_min=0.1 --theta_max=10000 --theta_num=51 --usegpu=True
"""

import os
import sys
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.frontier import frontier_scan, theta_grid
from sso.nutau.models import model_from_args
from sso.nutau.spectrum import diagonalize


def main():
    args = parse_argv()
    fprint("arguments:", args)
    kind, model, name = model_from_args(args)
    usegpu = find_value(args, "usegpu", bool, False)
    Lmin = find_value(args, "Lmin", int, 6)
    Lmax = find_value(args, "Lmax", int, Lmin)
    K_arg = str(args.get("K", "0"))
    lam = find_value(args, "lam", float, 0.0)
    k_save = find_value(args, "k_save", int, 2)
    pbc = not find_value(args, "obc", bool, False)
    if kind == "static":
        o = find_value(args, "pows_max", int, 3) + 1
    else:
        o = 1 if find_value(args, "ortho_identity", bool, True) else 0
    thetas = theta_grid(find_value(args, "theta_min", float, 0.1),
                        find_value(args, "theta_max", float, 100.0),
                        find_value(args, "theta_num", int, 21),
                        find_value(args, "include_infinity", bool, False))
    momentum = K_arg != "none"
    if kind == "floquet":
        sub = f"{name}_DSFloquet_Data"
    elif momentum:
        sub = f"{name}_DSsoftKsec_Data"
    else:
        sub = f"{name}{'' if pbc else 'OBC'}_DSsoft_Data"
    outdir = output_dir(sub)
    fprint(f"{name} ({kind}), L={Lmin}..{Lmax}, K={K_arg}, o={o}, lam={lam}, "
           f"{len(thetas)} theta in [{thetas[0]}, {thetas[-1]}] -> {outdir}")

    for L in range(Lmin, Lmax + 1):
        t0 = time()
        spec = diagonalize(kind, model, L, pbc=pbc, momentum=momentum, usegpu=usegpu)
        fprint(f"\nL={L}: diagonalization {time() - t0:.1f}s")
        if K_arg == "none":
            Ks = [None]
        elif K_arg == "all":
            Ks = list(range(L))
        else:
            Ks = [int(x) for x in K_arg.split(",")]
        for K in Ks:
            frontier_scan(spec, K, thetas, o, outdir, lam=lam, k_save=k_save,
                          ncv=find_value(args, "ncv", int, 50),
                          degen_tol=find_value(args, "degen_tol", float, 1e-10),
                          save_v2=find_value(args, "save_v2", bool, True),
                          override=find_value(args, "override", bool, False),
                          usegpu=usegpu)


if __name__ == "__main__":
    main()
