"""Plot 1: the nu(tau) frontier of the Krylov chain at mu = log 3 (N = 60, converged).

    python plot_frontier_evn.py      (after run_frontier.py)  ->  figures/frontier_mulog3.{pdf,png}

Also the explicit Lorentzian filter 1/(1+theta omega^2) applied to O_0 (same mu = log 3
norm); it lies within 1.5e-4 of the frontier, so the two curves overlap at this scale.
Dotted guide: the large-tau limit nu = ln 2 / tau.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "data", "frontier_mulog3_N60.npz"))
o = np.argsort(d["tau"])
tau, nu = d["tau"][o], d["nu"][o]

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "0.35",
                     "xtick.color": "0.25", "ytick.color": "0.25"})
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.plot(tau, nu, "-", color="#2a78d6", lw=2, label=r"frontier, $\mu=\log 3$, $N=60$")
ol = np.argsort(d["tau_lor"])
ax.plot(d["tau_lor"][ol], d["nu_lor"][ol], "--", color="#eb6834", lw=1.6,
        label=r"Lorentzian $1/(1+\theta\omega^2)$ on $O_0$")
t = np.geomspace(20, tau[-1], 50)
ax.plot(t, np.log(2) / t, ":", color="0.45", lw=1, label=r"$\ln 2/\tau$")
ax.set(xscale="log", yscale="log", xlabel=r"$\tau$", ylabel=r"$\nu$")
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"frontier_mulog3.{ext}"), dpi=200)
print("wrote figures/frontier_mulog3.{pdf,png}")
