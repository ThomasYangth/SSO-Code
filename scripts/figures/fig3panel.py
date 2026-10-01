r"""Figure: three-panel (two-column) modulated-hydrodynamics figure.

(a) nu = M/N1 vs tau_obs/L^2 for Model B rings L = 16..23 (DQT, control-variate pooled
    fronts); inset: nu vs tau_obs unrescaled.
(b) C(t)/C(0) (solid) and ||H_q(t)||_RPS/||H_q||_RPS (dashed) vs t/L^2; shared discrete L
    colorbar with (a); inset |G_RPS(t1,t2)| = |E_psi m_psi(t1)^* m_psi(t2)| at L = 20,
    |t|/L^2 <= 0.15, log colour scale normalized to its maximum.
(c) Delta nu = nu0 - nu(q) vs tau q^2 from the thermodynamic-limit TI ansatz (Kim-Huse
    chain, span M = 11), one curve per q; opacity = M-stability (M = 8..11).

Reads data/fig3panel (built by scripts/extract/dqt_fig3panel.py); writes
figures/fig3panel.{png,pdf}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm

from sso.config import figure_path
from sso.dqt.figdata import (INSET_L, LS, TWIN, load_correlators, load_front, load_grps,
                             ti_dir)
from sso.tiansatz import discover_qs, dnu_tq2, mstab_family

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 5.8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0})
lcol = dict(zip(LS, plt.cm.viridis(np.linspace(0, 0.92, len(LS)))))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.05),
                                    gridspec_kw=dict(width_ratios=[1, 1, 1.12]))
fig.subplots_adjust(left=0.07, right=0.965, bottom=0.2, top=0.92, wspace=0.55)

# ---------------- (a) nu vs tau_obs/L^2 ----------------
fronts = {L: load_front(L) for L in LS}
for L, d in fronts.items():
    axA.errorbar(d["tau_obs"] / L ** 2, d["nu"], xerr=d["tau_err"] / L ** 2, yerr=d["nu_err"],
                 fmt="o-", ms=1.6, lw=0.8, elinewidth=0.5, capsize=0, color=lcol[L], label=f"$L={L}$")
axA.set_xlabel(r"$\tau/L^2$")
axA.set_ylabel(r"$\nu$")
axA.set_xlim(left=0)
axA.grid(alpha=0.25, lw=0.4)
insA = axA.inset_axes([0.55, 0.53, 0.42, 0.43])
insA.set_box_aspect(1)
insA.set_anchor("NE")
for L, d in fronts.items():
    insA.plot(d["tau_obs"], d["nu"], "-", lw=0.8, color=lcol[L])
insA.set_xlabel(r"$\tau$", fontsize=5.5, labelpad=0.5)
insA.set_ylabel(r"$\nu$", fontsize=5.5, labelpad=0.5)
insA.tick_params(labelsize=4.8, length=1.5, pad=1)
insA.set_xlim(left=0)
insA.grid(alpha=0.25, lw=0.3)

# ---------------- (b) bare correlators vs t/L^2, + G_RPS inset ----------------
for L in LS:
    d = load_correlators(L)
    rn = np.sqrt(d["R"])
    axB.semilogy(d["t_C"] / L ** 2, d["C"] / d["C"][0], "-", lw=0.9, color=lcol[L])
    axB.semilogy(d["t_R"] / L ** 2, rn / rn[0], "--", lw=0.9, color=lcol[L])
axB.plot([], [], "k-", lw=0.9, label=r"$C(t)$")
axB.plot([], [], "k--", lw=0.9, label=r"$\|H_q(t)\|_{\rm RPS}$")
axB.set_xlabel(r"$t/L^2$")
axB.set_ylabel("normalized correlator")
axB.set_xlim(0, TWIN)
axB.set_ylim(1e-3, 1.5)
axB.grid(alpha=0.25, lw=0.4, which="major")
axB.legend(loc="lower left", handlelength=1.8, frameon=False)

tw, G, nst = load_grps(INSET_L)
ins = axB.inset_axes([0.48, 0.51, 0.32, 0.37])
im = ins.imshow(np.clip(G / np.nanmax(G), 1e-4, 1), origin="lower", cmap="magma",
                norm=LogNorm(1e-4, 1), extent=[tw[0] / INSET_L ** 2, tw[-1] / INSET_L ** 2] * 2,
                aspect="equal", interpolation="nearest")
ins.set_xlabel(r"$t_2/L^2$", fontsize=5.5, labelpad=0.5)
ins.set_ylabel(r"$t_1/L^2$", fontsize=5.5, labelpad=0.5)
ins.tick_params(labelsize=4.8, length=1.5, pad=1)
ins.set_xticks([-0.1, 0, 0.1])
ins.set_yticks([-0.1, 0, 0.1])
ins.set_xticklabels(["-.1", "0", ".1"])
ins.set_yticklabels(["-.1", "0", ".1"])
ins.set_title(r"$\langle H_q(t_1),H_q(t_2)\rangle_{\rm RPS}$", fontsize=5.3, pad=1.5)
cax = axB.inset_axes([0.835, 0.51, 0.022, 0.27])
cbi = fig.colorbar(im, cax=cax, ticks=[1e-4, 1e-2, 1])
cbi.ax.tick_params(labelsize=4.5, length=1.2, pad=0.8)
cbi.outline.set_linewidth(0.4)

Lcm = ListedColormap([lcol[L] for L in LS])
Lnorm = BoundaryNorm(np.arange(LS[0] - 0.5, LS[-1] + 1), Lcm.N)
smL = ScalarMappable(norm=Lnorm, cmap=Lcm)
smL.set_array([])
cbL = fig.colorbar(smL, ax=[axA, axB], ticks=list(LS), fraction=0.025, pad=0.015, aspect=30)
cbL.set_label(r"$L$", labelpad=1)
cbL.ax.tick_params(labelsize=5.5, length=0)
cbL.outline.set_linewidth(0.4)

# ---------------- (c) TI ansatz Delta nu vs tau q^2 ----------------
plasma = matplotlib.colormaps["plasma"]
qcmap = LinearSegmentedColormap.from_list("plasma_t", plasma(np.linspace(0.0, 0.85, 256)))
QS = discover_qs(ti_dir())
fam = mstab_family(ti_dir())
qnorm = LogNorm(min(QS), max(QS))
xvis = []
for q in QS:
    if q not in fam:
        continue
    base = qcmap(qnorm(q))
    xk, yk, al = dnu_tq2(fam[q], q)
    if xk.size < 2:
        continue
    xvis.extend(xk[yk >= 1e-6])
    pts = np.column_stack([xk, yk]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = np.tile(np.array(base), (len(segs), 1))
    cols[:, 3] = 0.5 * (al[:-1] + al[1:])
    axC.add_collection(LineCollection(segs, colors=cols, linewidths=1.3, zorder=2))
    axC.scatter(xk, yk, s=3, color=[base], alpha=al, edgecolors="none", zorder=3)
axC.set_xscale("log")
axC.set_yscale("log")
axC.autoscale_view()
axC.set_xlabel(r"$\tau\,q^2$")
axC.set_ylabel(r"$\Delta\nu$")
axC.set_ylim(bottom=1e-6)
axC.set_xlim(0.7 * min(xvis), 1.4 * max(xvis))
axC.grid(True, which="major", alpha=0.25, lw=0.4)
sm = ScalarMappable(norm=qnorm, cmap=qcmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=axC, fraction=0.05, pad=0.02)
cb.set_label(r"$q$", labelpad=1)
cb.ax.tick_params(labelsize=5.5)
cb.outline.set_linewidth(0.4)

for ax, lab in zip((axA, axB, axC), "abc"):
    ax.text(-0.02, 1.03, f"({lab})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
for ext in ("png", "pdf"):
    out = figure_path(f"fig3panel.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved ->", out)
print(f"inset: L={INSET_L}, {nst} product states, |t|/L^2<={TWIN}")
