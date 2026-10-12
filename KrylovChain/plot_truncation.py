"""Plot 2: truncation convergence of the mu = log 3 frontier (notes' method, mu-norm
lambda^2 c~_N^T c~_N).

    python plot_truncation.py      ->  figures/truncation_mulog3.{pdf,png}, data/truncation_mulog3.npz

(a) nu(tau) for N = 1, 3, 5, 7 and the N = 60 baseline; inset nu*tau, where the
    differences are visible.  (N and N+1 coincide for odd N: calR couples only sites of
    equal parity and c~_N has no weight on odd sites.)
(b) |nu_N(tau) / nu_60(tau) - 1| vs N at fixed tau (one curve per tau): theta is
    root-found separately for each N so that tau_N(theta) = tau exactly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import krylovchain as kc

MU, N_BASE = kc.MU_EVN, 60
N_CURVES = (1, 3, 5, 7)
TAUS = (1.0, 3.0, 10.0, 100.0, 1e3, 1e4)
N_ERR = np.arange(1, 41)
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # one-hue ramp


def nu_at_tau(tau, N):
    f = lambda x: np.log(kc.solve_frontier(MU, np.exp(x), N)["tau"] / tau)
    lt = brentq(f, np.log(1e-8), np.log(1e12), xtol=1e-15)
    return kc.solve_frontier(MU, np.exp(lt), N)["nu"]


thetas = np.concatenate([[0.0], np.geomspace(1e-3, 1e8, 221)])
curves = {}
for N in N_CURVES + (N_BASE,):
    r = [kc.solve_frontier(MU, t, N) for t in thetas]
    tau = np.array([x["tau"] for x in r]); nu = np.array([x["nu"] for x in r])
    o = np.argsort(tau)
    curves[N] = (tau[o], nu[o])

base = {t: nu_at_tau(t, N_BASE) for t in TAUS}
err = np.array([[abs(nu_at_tau(t, N) / base[t] - 1) for N in N_ERR] for t in TAUS])
np.savez(os.path.join(HERE, "data", "truncation_mulog3.npz"), taus=TAUS, N=N_ERR, rel_err=err,
         nu_base=np.array([base[t] for t in TAUS]), N_base=N_BASE)
for t, e in zip(TAUS, err):
    print(f"tau={t:7g}: " + " ".join(f"N={n}:{x:.1e}" for n, x in zip(N_ERR, e) if n in (1, 3, 5, 9, 15, 21, 29)))

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "0.35",
                     "xtick.color": "0.25", "ytick.color": "0.25"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9))
ramp = dict(zip(N_CURVES, ["#eb6834", "#eda100", "#1baf7a", "#e87ba4"]))
ins = axA.inset_axes([0.50, 0.55, 0.46, 0.42])
for N in N_CURVES:
    t, n = curves[N]
    axA.plot(t, n, "-", color=ramp[N], lw=1.4, label=f"$N={N}$")
    ins.plot(t, n * t, "-", color=ramp[N], lw=1.2)
t, n = curves[N_BASE]
axA.plot(t, n, "--", color="k", lw=1.0, label=f"$N={N_BASE}$ (baseline)")
ins.plot(t, n * t, "--", color="k", lw=0.9)
axA.set(xscale="log", yscale="log", xlabel=r"$\tau$", ylabel=r"$\nu$")
axA.grid(alpha=0.25, lw=0.4)
axA.legend(fontsize=6.5, frameon=False, loc="lower left")
axA.set_title(r"(a) $\mu=\log 3$", loc="left", fontsize=8)
ins.set_xscale("log")
ins.set_xlabel(r"$\tau$", fontsize=6, labelpad=1)
ins.set_ylabel(r"$\nu\tau$", fontsize=6, labelpad=1)
ins.tick_params(labelsize=5.5, pad=1)
ins.set_xlim(1, t[-1])
ins.grid(alpha=0.25, lw=0.3)

floor = 1e-16
for t_, e, col in zip(TAUS, err, BLUES):
    axB.plot(N_ERR, np.maximum(e, floor), "o-", color=col, ms=2.5, lw=1.0, label=rf"$\tau={t_:g}$")
axB.plot(N_ERR, 0.4 * 3.0 ** -N_ERR.astype(float), ":", color="0.45", lw=1, label=r"$0.4\cdot 3^{-N}$")
axB.set(yscale="log", xlabel=r"$N$", ylabel=r"$|\nu_N/\nu_{60}-1|$ at fixed $\tau$", ylim=(floor / 2, 0.1))
axB.grid(alpha=0.25, lw=0.4)
axB.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
axB.set_title("(b)", loc="left", fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"truncation_mulog3.{ext}"), dpi=200)
print("wrote figures/truncation_mulog3.{pdf,png}")
