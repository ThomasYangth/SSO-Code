"""theta scan of the mu = log 3 frontier (notes' method) and the mu = inf closed form.

    python run_frontier.py [--N=60] [--n_theta=331]

Writes data/frontier_mulog3_N{N}.npz (theta, lam, nu, tau, norm2, comm2, mu_norm2, plus
nu_exact = nu with the mu-norm summed past N, and nu_lor/tau_lor of the explicit
Lorentzian 1/(1+theta omega^2) on O_0 with the full mu-norm) and data/closed_form_muinf.npz (theta, nu, tau).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import krylovchain as kc


def arg(name, typ, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return typ(a.split("=", 1)[1])
    return default


N = arg("N", int, 60)
thetas = np.concatenate([[0.0], np.geomspace(1e-3, 1e8, arg("n_theta", int, 331))])
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

rows = [kc.solve_frontier(kc.MU_EVN, t, N) for t in thetas]
out = {k: np.array([r[k] for r in rows]) for k in ("lam", "nu", "tau", "norm2", "comm2", "mu_norm2")}
lor = [kc.lorentzian_filter(kc.MU_EVN, t, kc.mu_norm_size(kc.MU_EVN, N)) for t in thetas]
out["nu_lor"] = np.array([r["nu"] for r in lor])
out["tau_lor"] = np.array([r["tau"] for r in lor])
out["nu_exact"] = np.array([kc.solve_frontier(kc.MU_EVN, t, N, mu_norm="exact")["nu"] for t in thetas])
print(f"mu=log3, N={N}: max |nu_truncated/nu_exact - 1| = {np.max(np.abs(out['nu'] / out['nu_exact'] - 1)):.1e}; "
      f"tau {out['tau'].min():.4f}..{out['tau'].max():.4g}, nu {out['nu'].max():.4f}..{out['nu'].min():.3e}, "
      f"nu*tau at the largest theta {out['nu'][-1] * out['tau'][-1]:.8f} (ln 2 = {np.log(2):.8f})")
np.savez(os.path.join(HERE, "data", f"frontier_mulog3_N{N}.npz"), theta=thetas, mu=kc.MU_EVN, N=N, **out)

cf = [kc.closed_form_mu_inf(t) for t in thetas]
np.savez(os.path.join(HERE, "data", "closed_form_muinf.npz"), theta=thetas,
         nu=np.array([r["nu"] for r in cf]), tau=np.array([r["tau"] for r in cf]))
