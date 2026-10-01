"""tau = infinity optimum nu_o(inf), o = 1..5, and the H^n baseline, per L.

Replaces tools/benchmark_nutau_ds_tauinf.py and tools/compute_baseline_hn_nu.py
(full-space static spectrum; the commutant is block diagonal in the
degenerate multiplets of H).  Writes, per L,
    $SSO_OUTPUT/<model>_TauInf_Data/tauinf_L{L}.csv   L,o,n_mult,max_d,packed,nu_tauinf,dt
    $SSO_OUTPUT/<model>_TauInf_Data/baseline_L{L}.csv L,n,nu_hn
nu_hn at n = p is the RPS weight of H^p orthogonalised against lower powers.

Example:
    python scripts/compute/nutau/tauinf.py --J=1.0 --hx=0.905 --hz=0.809 --Ls=6,7,8 --usegpu=True
"""

import os
import sys
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from sso.cli import find_value, fprint, parse_argv
from sso.config import output_dir
from sso.nutau.models import model_from_args
from sso.nutau.pauli import EnergyBasisB
from sso.nutau.spectrum import diagonalize_static
from sso.nutau.tauinf import power_baseline_nu, tauinf_nu


def main():
    args = parse_argv()
    fprint("arguments:", args)
    kind, model, name = model_from_args(args)
    if kind != "static":
        raise ValueError("tau = infinity scan is defined for static models")
    usegpu = find_value(args, "usegpu", bool, False)
    Ls = [int(x) for x in str(args.get("Ls", "6")).split(",")]
    o_max = find_value(args, "o_max", int, 5)
    outdir = output_dir(f"{name}_TauInf_Data")
    for L in Ls:
        t0 = time()
        spec = diagonalize_static(model, L, pbc=True, momentum=False, usegpu=usegpu)
        Bop = EnergyBasisB(spec.V, L, usegpu)
        fprint(f"\nL={L}: diagonalization {time() - t0:.1f}s")
        with open(os.path.join(outdir, f"tauinf_L{L}.csv"), "w") as f:
            f.write("L,o,n_mult,max_d,packed,nu_tauinf,dt\n")
            for o in range(1, o_max + 1):
                t0 = time()
                r = tauinf_nu(spec, Bop, o, usegpu=usegpu)
                dt = time() - t0
                fprint(f"  o={o}: nu={r['nu']:.10g}  n_mult={r['n_mult']} max_d={r['max_d']} "
                       f"packed={r['packed']}  ({dt:.1f}s)")
                f.write(f"{L},{o},{r['n_mult']},{r['max_d']},{r['packed']},{r['nu']:.10g},{dt:.3g}\n")
        with open(os.path.join(outdir, f"baseline_L{L}.csv"), "w") as f:
            f.write("L,n,nu_hn\n")
            for p, nu in enumerate(power_baseline_nu(spec, Bop, n_max=o_max)):
                fprint(f"  baseline n={p}: nu={nu:.10g}")
                f.write(f"{L},{p},{nu:.10g}\n")


if __name__ == "__main__":
    main()
