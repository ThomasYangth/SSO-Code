"""Figure C: tau -> inf plateau nu_n(inf) vs the number n of projected-out powers of H.

Kim-Huse chain (J=1, hx=0.905, hz=0.809): nu_n(inf) vs n = o - 1 (I, H, ..., H^n removed)
for even L = 6..14 (solid, circles) and the H^n baseline (dashed, squares, drawn at
x = n - 1), with the thermodynamic-limit reference r^d, r = nu(H) (dotted).  Panel (c) of
kimhuse_three_panel.  No filter is involved.

One panel, one third of a 7.0 in double-column width.
Data: data/kimhuse_three_panel/{tauinf,baseline}.csv.
Output: figures/figure_C.{pdf,png}
"""

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
from matplotlib.ticker import LogLocator, StrMethodFormatter

from sso.config import data_dir
from sso.nutau.io import read_csv
from sso.nutau.tauinf import ising_r
from sso.plotting.style import layout_row, save_figure

DATA = data_dir("kimhuse_three_panel")
J, HX, HZ = 1.0, 0.905, 0.809
SLOT, FIG_H, PANEL_H = 7.0 / 3.0, 2.05, 1.36
FS_LABEL, FS_TICK = 8, 7
L_ALL = list(range(6, 15))
L_EVEN = [L for L in L_ALL if L % 2 == 0]
_base = mpl.colormaps["viridis_r"]
_cmap = LinearSegmentedColormap.from_list("viridis_r", [_base(x) for x in np.linspace(0.0, 1.0, 256)])
COLORS = {L: _cmap(i / max(1, len(L_ALL) - 1)) for i, L in enumerate(L_ALL)}


def by_L(table, xkey, ykey, shift=0, xmin=None):
    out = {}
    for L, x, y in zip(table["L"].astype(int), table[xkey].astype(int), table[ykey]):
        if xmin is None or x >= xmin:
            out.setdefault(L, []).append((x + shift, y))
    return {L: sorted(v) for L, v in out.items()}


def main():
    fig = plt.figure(figsize=(SLOT, FIG_H))
    ax = fig.add_axes([0.2, 0.2, 0.55, 0.65])
    cax = fig.add_axes([0.8, 0.2, 0.03, 0.65])
    tauinf = by_L(read_csv(os.path.join(DATA, "tauinf.csv")), "o", "nu_tauinf", shift=-1)
    base = by_L(read_csv(os.path.join(DATA, "baseline.csv")), "n", "nu_hn", shift=-1, xmin=1)
    for L in sorted(L for L in tauinf if L in L_EVEN):
        pts = tauinf[L]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", markersize=2.4, linewidth=1.0,
                color=COLORS[L])
        if L in base:
            ax.plot([p[0] for p in base[L]], [p[1] for p in base[L]], "--s", markersize=2.0,
                    linewidth=0.8, color=COLORS[L], alpha=0.5)
    d_ref = np.arange(1, 6)
    ax.plot(d_ref - 1, ising_r(J, HX, HZ) ** d_ref, "k:", linewidth=1.2, zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(0.003, 0.35)
    ax.set_xlabel(r"$n$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu_n(\infty)$", fontsize=FS_LABEL)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    disc = ListedColormap([COLORS[L] for L in L_ALL])
    sm = mpl.cm.ScalarMappable(cmap=disc, norm=BoundaryNorm([L_ALL[0] - 0.5] + [L + 0.5 for L in L_ALL],
                                                            len(L_ALL)))
    cb = fig.colorbar(sm, cax=cax, ticks=L_EVEN)
    cax.set_title(r"$L$", fontsize=FS_LABEL - 1, pad=2)
    cb.ax.tick_params(labelsize=FS_TICK - 1, length=1.5)
    cb.outline.set_linewidth(0.5)
    W = layout_row(fig, [(ax, cax)], 0.42, PANEL_H)
    print(f"plot box {W:.3f} x {PANEL_H} in")
    print("wrote", save_figure(fig, "figure_C", png_dpi=300))


if __name__ == "__main__":
    main()
