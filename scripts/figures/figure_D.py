"""Figure A: Kim-Huse frontier, sector-resolved (L = 12) and vs L with the tau -> inf plateau.

(a) Kim-Huse (J=1, hx=0.905, hz=0.809), L = 12: sector-resolved nu(tau) frontier, the
    left panel of sector_L12_kimhuse_and_tfim (k=0 / k=1 / cross-sector mix; faded =
    losing branches; dashed = k=0 plateau; right axis = effective size -log_3 nu).
(b) global nu(tau) frontier for L = 6..12 (panel (a) of kimhuse_three_panel); inset:
    nu(inf) (= nu_4(inf), I..H^4 removed) vs L = 6..14 with an exponential fit (panel (b) of kimhuse_three_panel,
    without its power-law inset).  Shared discrete L colourbar.

Each panel occupies one third of a 7.0 in double-column width.
Data: data/sector_L12_kimhuse_and_tfim/, data/tfim_frontier/ (scripts/extract/nutau_tfim_frontier.py).
Output: figures/figure_D.{pdf,png}
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, StrMethodFormatter

from sso.config import data_dir
from sso.nutau.io import load_arcs, load_npz, read_csv
from sso.plotting.frontier import sector_frontier
from sso.plotting.style import layout_row, panel_label, save_figure


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sector = _load("sector_L12_kimhuse_and_tfim")

DATA_KH = data_dir("tfim_frontier")
SLOT = 7.0 / 3.0
PANEL_H = 1.36   # plot height (inches); widths are equalised by layout_row
FIG_H = 2.05
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 6.5
L_ALL = list(range(6, 13))
L_FRONT = [L for L in L_ALL if L <= 12]
_base = mpl.colormaps["viridis_r"]
_cmap = LinearSegmentedColormap.from_list("viridis_r", [_base(x) for x in np.linspace(0.0, 1.0, 256)])
COLORS = {L: _cmap(i / max(1, len(L_ALL) - 1)) for i, L in enumerate(L_ALL)}


def panel_sector(ax):
    sector.draw_panel(ax, sector.MODELS[1])
    handles = [Line2D([0], [0], color=sector.C_K0, lw=1.8, label=r"$k=0$"),
               Line2D([0], [0], color=sector.C_K1, lw=1.8, label=r"$k=1$"),
               Line2D([0], [0], color=sector.C_MIX, lw=1.8, label="mix")]
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="lower left",
              handlelength=1.4, borderaxespad=0.3)


def panel_frontier_vs_L(ax):
    for L in L_FRONT:
        fr = sector_frontier(load_npz(os.path.join(DATA_KH, f"grid_L{L}_K0.npz")),
                             load_npz(os.path.join(DATA_KH, f"grid_L{L}_K1.npz")),
                             load_arcs(DATA_KH, f"crossK_L{L}.npz"),
                             load_arcs(DATA_KH, f"withinK0_L{L}.npz"), tau_max=1e5)
        ax.plot(*fr["union"], "-", color=COLORS[L], linewidth=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.set_ylim(0.08, 0.4)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 6.0), numticks=12))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))

    tab = read_csv(os.path.join(DATA_KH, "tauinf.csv"))
    sel = tab["o"].astype(int) == 5                      # nu_4(inf): the o = 5 commutant
    Ls = tab["L"].astype(int)[sel]
    nu4 = tab["nu_tauinf"][sel]
    order = np.argsort(Ls)
    Ls, nu4 = Ls[order].astype(float), nu4[order]
    ins = ax.inset_axes([0.58, 0.55, 0.39, 0.36])
    for L, v in zip(Ls, nu4):
        ins.scatter([L], [v], s=9, color=COLORS[int(L)], zorder=3, edgecolors="none")
    ins.set_xticks([6, 9, 12])
    ins.set_ylim(0.1190, 0.1205)
    ins.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.119, 0.120]))
    ins.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ins.tick_params(labelsize=5.5, length=2, pad=1)
    ins.set_xlabel(r"$L$", fontsize=6, labelpad=0.5)
    ins.set_title(r"$\nu(\infty)$", fontsize=6, pad=1.5)
    ins.grid(alpha=0.3, lw=0.4)
    print("(b) inset: nu_4(inf) = " + ", ".join(f"{v:.5f}" for v in nu4))


def main():
    fig = plt.figure(figsize=(2 * SLOT, FIG_H))
    fx, fy = (lambda x: x / (2 * SLOT)), (lambda y: y / FIG_H)
    bottom, height = 0.46, PANEL_H
    axA = fig.add_axes([fx(0.6), fy(bottom), fx(1.2), fy(height)])
    axB = fig.add_axes([fx(SLOT + 0.5), fy(bottom), fx(1.2), fy(height)])
    cax = fig.add_axes([fx(SLOT + 1.8), fy(bottom), fx(0.07), fy(height)])
    panel_sector(axA)
    panel_frontier_vs_L(axB)
    disc = ListedColormap([COLORS[L] for L in L_ALL])
    sm = mpl.cm.ScalarMappable(cmap=disc, norm=BoundaryNorm([L_ALL[0] - 0.5] + [L + 0.5 for L in L_ALL],
                                                            len(L_ALL)))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=L_ALL[::2])
    cb.set_label(r"$L$", fontsize=FS_LABEL, labelpad=1)
    cb.ax.tick_params(labelsize=FS_TICK - 1, length=1.5)
    cb.outline.set_linewidth(0.5)
    for ax, lab in ((axA, "(a)"), (axB, "(b)")):
        panel_label(ax, lab, FS_LABEL)
    W = layout_row(fig, [(axA, None), (axB, cax)], bottom, height)
    print(f"plot box {W:.3f} x {height:.2f} in")
    print("wrote", save_figure(fig, "figure_D", png_dpi=300))


if __name__ == "__main__":
    main()
