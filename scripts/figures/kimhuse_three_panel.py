"""Kim-Huse (J=1, hx=0.905, hz=0.809) three-panel figure with a shared L colourbar.

(a) global nu(tau) frontier, L = 6..12: per theta the K=0 (o4) / K=1 (o0)
    sector with the larger W, the hand-over filled by the cross-sector arc,
    drawn as one line per L (tau <= 1e5, static tau = 1/sqrt(As)).
(b) nu_4(inf) vs L = 6..14 (the o = 5 commutant: I..H^4 projected out), with an
    exponential fit; inset log-log with a power-law fit.
(c) nu_n(inf) vs n = o - 1 for even L (solid) and the H^n baseline (dashed,
    drawn at x = n - 1), plus the thermodynamic-limit reference r^d, r = nu(H).

Data: data/kimhuse_three_panel/ (scripts/extract/nutau_kimhuse_three_panel.py).
Output: figures/kimhuse_three_panel.{pdf,png}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FixedLocator, LogLocator, StrMethodFormatter

from sso.config import data_dir
from sso.nutau.io import load_arcs, load_npz, read_csv
from sso.nutau.tauinf import ising_r
from sso.plotting.frontier import sector_frontier
from sso.plotting.style import save_figure

DATA = data_dir("kimhuse_three_panel")
J, HX, HZ = 1.0, 0.905, 0.809
TAU_MAX_A = 1e5
L_ALL = list(range(6, 15))
L_PANEL_A = [L for L in L_ALL if L <= 12]
L_PANEL_C = [L for L in L_ALL if L % 2 == 0]
FS_LABEL, FS_TICK = 9, 8
_base = mpl.colormaps["viridis_r"]
_cmap = LinearSegmentedColormap.from_list("viridis_r", [_base(x) for x in np.linspace(0.0, 1.0, 256)])
COLORS = {L: _cmap(i / max(1, len(L_ALL) - 1)) for i, L in enumerate(L_ALL)}


def by_L(table, xkey, ykey, shift=0, xmin=None):
    """{L: [(x + shift, y), ...]} sorted by x from a CSV column dict."""
    out = {}
    for L, x, y in zip(table["L"].astype(int), table[xkey].astype(int), table[ykey]):
        if xmin is None or x >= xmin:
            out.setdefault(L, []).append((x + shift, y))
    return {L: sorted(v) for L, v in out.items()}


def main():
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.05))

    for L in L_PANEL_A:
        fr = sector_frontier(load_npz(os.path.join(DATA, f"grid_L{L}_K0.npz")),
                             load_npz(os.path.join(DATA, f"grid_L{L}_K1.npz")),
                             load_arcs(DATA, f"crossK_L{L}.npz"), tau_max=TAU_MAX_A)
        axA.plot(*fr["union"], "-", color=COLORS[L], linewidth=1.2, zorder=2)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    axA.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    axA.set_ylim(0.02, 0.4)

    tauinf = by_L(read_csv(os.path.join(DATA, "tauinf.csv")), "o", "nu_tauinf", shift=-1)
    Ls_b = sorted(L for L in tauinf if any(n == 4 for n, _ in tauinf[L]))
    nu_b = np.array([dict(tauinf[L])[4] for L in Ls_b])
    Ls_b = np.array(Ls_b, float)
    slope, intercept = np.polyfit(Ls_b, np.log(nu_b), 1)
    rate, pref = -slope, np.exp(intercept)
    Lf = np.linspace(Ls_b.min() - 0.4, Ls_b.max() + 0.4, 100)
    axB.plot(Lf, pref * np.exp(-rate * Lf), "--", color="0.45", linewidth=0.9, zorder=1,
             label=fr"$\propto e^{{-{rate:.2f} L}}$")
    for L, v in zip(Ls_b, nu_b):
        axB.scatter([L], [v], s=16, color=COLORS[int(L)], zorder=3, edgecolors="none")
    axB.set_yscale("log")
    axB.set_xlabel(r"$L$", fontsize=FS_LABEL)
    axB.set_ylabel(r"$\nu_4(\infty)$", fontsize=FS_LABEL)
    axB.set_xticks(list(range(int(Ls_b.min()), int(Ls_b.max()) + 1, 2)))
    axB.legend(fontsize=7, frameon=False, loc="upper right")
    print(f"(b) nu_4(inf) = {pref:.4g} exp(-{rate:.4g} L)")

    sp, ip = np.polyfit(np.log(Ls_b), np.log(nu_b), 1)
    axins = axB.inset_axes([0.16, 0.12, 0.30, 0.30])
    Lf2 = np.linspace(Ls_b.min(), Ls_b.max(), 100)
    axins.plot(Lf2, np.exp(ip) * Lf2 ** sp, "--", color="0.45", linewidth=0.8, zorder=1)
    for L, v in zip(Ls_b, nu_b):
        axins.scatter([L], [v], s=14, color=COLORS[int(L)], zorder=3, edgecolors="none")
    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.tick_params(labelsize=5.5, length=2, pad=1)
    axins.set_xticks([6, 10, 14])
    axins.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axins.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    axins.yaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05]))
    axins.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axins.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    axins.text(0.05, 0.07, fr"$\propto L^{{-{-sp:.2f}}}$", transform=axins.transAxes,
               fontsize=6, ha="left", va="bottom", color="0.3")

    base = by_L(read_csv(os.path.join(DATA, "baseline.csv")), "n", "nu_hn", shift=-1, xmin=1)
    for L in sorted(L for L in tauinf if L in L_PANEL_C):
        pts = tauinf[L]
        axC.plot([p[0] for p in pts], [p[1] for p in pts], "-o", markersize=2.6,
                 linewidth=1.0, color=COLORS[L])
        if L in base:
            axC.plot([p[0] for p in base[L]], [p[1] for p in base[L]], "--s", markersize=2.2,
                     linewidth=0.8, color=COLORS[L], alpha=0.5)
    d_ref = np.arange(1, 6)
    axC.plot(d_ref - 1, ising_r(J, HX, HZ) ** d_ref, "k:", linewidth=1.3, zorder=6)
    axC.set_yscale("log")
    axC.set_ylim(0.003, 0.35)
    axC.set_xlabel(r"$n$", fontsize=FS_LABEL)
    axC.set_ylabel(r"$\nu_n(\infty)$", fontsize=FS_LABEL)
    axC.set_xticks([0, 1, 2, 3, 4])

    for ax in (axA, axB, axC):
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 6.0),
                                              numticks=12))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    plt.subplots_adjust(left=0.085, right=0.895, bottom=0.26, top=0.95, wspace=0.46)

    disc = ListedColormap([COLORS[L] for L in L_ALL])
    bounds = [L_ALL[0] - 0.5] + [L + 0.5 for L in L_ALL]
    sm = mpl.cm.ScalarMappable(cmap=disc, norm=BoundaryNorm(bounds, len(L_ALL)))
    sm.set_array([])
    cax = fig.add_axes([0.915, 0.26, 0.016, 0.69])
    cbar = fig.colorbar(sm, cax=cax, ticks=L_ALL, spacing="proportional")
    cbar.set_label(r"$L$", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    print("wrote", save_figure(fig, "kimhuse_three_panel"))


if __name__ == "__main__":
    main()
