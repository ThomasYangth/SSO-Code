"""Figure F: diffusive relaxation of the energy-density mode and its finite-q frontier.

(a) Mixed-field Ising ring (J=1, g=-1.05, h=0.5), L = 16..23 (DQT): bare autocorrelator
    C(t)/C(0) (solid) and RPS norm ||H_q(t)||_RPS / ||H_q||_RPS (dashed) vs t/L^2; inset
    |G_RPS(t1, t2)| = |E_psi m_psi(t1)^* m_psi(t2)| at L = 20, |t|/L^2 <= 0.15, normalised
    to its maximum (panel (b) of fig3panel).  No filter (bare time evolution).
(b) Delta nu = nu0 - nu(q) vs tau q^2, Kim-Huse chain in the thermodynamic limit
    (TI ansatz, span M = 11), opacity = M-stability (panel (c) of fig3panel).  Frontier
    (theta-penalised optimisation); no filter.

Each panel (with its colourbar) one third of a 7.0 in double-column width.
Data: data/fig3panel/ (correlators_L*.npz, grps_L20.npz, ti_q_scan/).
Output: figures/figure_F.{pdf,png}
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm

from sso.dqt.figdata import INSET_L, LS, TWIN, load_correlators, load_grps, ti_dir
from sso.plotting.style import layout_row, panel_label, save_figure
from sso.tiansatz import discover_qs, dnu_tq2, mstab_family

SLOT, FIG_H, PANEL_H = 7.0 / 3.0, 2.05, 1.36
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 6


def panel_correlators(fig, ax, cax):
    lcol = dict(zip(LS, plt.cm.viridis(np.linspace(0, 0.92, len(LS)))))
    for L in LS:
        d = load_correlators(L)
        rn = np.sqrt(d["R"])
        ax.semilogy(d["t_C"] / L ** 2, d["C"] / d["C"][0], "-", lw=0.9, color=lcol[L])
        ax.semilogy(d["t_R"] / L ** 2, rn / rn[0], "--", lw=0.9, color=lcol[L])
    ax.plot([], [], "k-", lw=0.9, label=r"$C(t)$")
    ax.plot([], [], "k--", lw=0.9, label=r"$\|H_q(t)\|_{\rm RPS}$")
    ax.set_xlabel(r"$t/L^2$", fontsize=FS_LABEL)
    ax.set_ylabel("normalized correlator", fontsize=FS_LABEL)
    ax.set_xlim(0, TWIN)
    ax.set_ylim(1e-3, 1.5)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, lw=0.4, which="major")
    ax.legend(loc="lower left", fontsize=FS_LEG, handlelength=1.8, frameon=False)
    tw, G, nst = load_grps(INSET_L)
    ins = ax.inset_axes([0.47, 0.50, 0.33, 0.38])
    im = ins.imshow(np.clip(G / np.nanmax(G), 1e-4, 1), origin="lower", cmap="magma",
                    norm=LogNorm(1e-4, 1), extent=[tw[0] / INSET_L ** 2, tw[-1] / INSET_L ** 2] * 2,
                    aspect="equal", interpolation="nearest")
    ins.set_xlabel(r"$t_2/L^2$", fontsize=5, labelpad=0.5)
    ins.set_ylabel(r"$t_1/L^2$", fontsize=5, labelpad=0.5)
    ins.tick_params(labelsize=4.5, length=1.5, pad=1)
    for setter in (ins.set_xticks, ins.set_yticks):
        setter([-0.1, 0, 0.1])
    ins.set_xticklabels(["-.1", "0", ".1"])
    ins.set_yticklabels(["-.1", "0", ".1"])
    ins.set_title(r"$|G_{\rm RPS}(t_1,t_2)|$", fontsize=5, pad=1.5)
    icax = ax.inset_axes([0.83, 0.50, 0.025, 0.38])
    cbi = fig.colorbar(im, cax=icax, ticks=[1e-4, 1e-2, 1])
    cbi.ax.tick_params(labelsize=4.3, length=1.2, pad=0.8)
    cbi.ax.minorticks_off()
    cbi.outline.set_linewidth(0.4)
    cmap = ListedColormap([lcol[L] for L in LS])
    norm = BoundaryNorm(np.arange(LS[0] - 0.5, LS[-1] + 1), cmap.N)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=list(LS)[::2])
    cax.set_title(r"$L$", fontsize=FS_LABEL - 1, pad=2)
    cb.ax.tick_params(labelsize=FS_TICK - 1, length=1.5)
    cb.outline.set_linewidth(0.5)
    print(f"(a) inset: L={INSET_L}, {nst} product states, |t|/L^2 <= {TWIN}")


def panel_tq2(fig, ax, cax):
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
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.1, zorder=2))
        ax.scatter(xk, yk, s=2, color=[base], alpha=al, edgecolors="none", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.autoscale_view()
    ax.set_xlabel(r"$\tau\,q^2$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\Delta\nu$", fontsize=FS_LABEL)
    ax.set_ylim(bottom=1e-6)
    ax.set_xlim(0.7 * min(xvis), 1.4 * max(xvis))
    ax.grid(True, which="major", alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=FS_TICK)
    cb = fig.colorbar(ScalarMappable(norm=qnorm, cmap=qcmap), cax=cax)
    cax.set_title(r"$q$", fontsize=FS_LABEL - 1, pad=2)
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    cb.outline.set_linewidth(0.5)


def main():
    fig = plt.figure(figsize=(2 * SLOT, FIG_H))
    fx, fy = (lambda x: x / (2 * SLOT)), (lambda y: y / FIG_H)
    bottom = 0.46
    axA = fig.add_axes([fx(0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    caxA = fig.add_axes([fx(1.8), fy(bottom), fx(0.07), fy(PANEL_H)])
    axB = fig.add_axes([fx(SLOT + 0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    caxB = fig.add_axes([fx(SLOT + 1.8), fy(bottom), fx(0.07), fy(PANEL_H)])
    panel_correlators(fig, axA, caxA)
    panel_tq2(fig, axB, caxB)
    for ax, lab in ((axA, "(a)"), (axB, "(b)")):
        panel_label(ax, lab, FS_LABEL)
    W = layout_row(fig, [(axA, caxA), (axB, caxB)], bottom, PANEL_H)
    print(f"plot box {W:.3f} x {PANEL_H} in")
    print("wrote", save_figure(fig, "figure_F", png_dpi=300))


if __name__ == "__main__":
    main()
