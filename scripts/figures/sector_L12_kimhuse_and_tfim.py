"""Sector-resolved nu(tau) frontier at L=12: Kim-Huse (left) and TFIM hx=2, hz=0 (right).

Solid = global optimum over the K=0 (o4) and K=1 (o0) sectors (winner by W),
teal k=0, orange k=1, olive "mix" = cross-sector superposition arc; faded =
each sector's losing branch (the TFIM within-K0 arc lands in the faded k=0
branch); dashed = k=0 tau -> inf plateau (largest-theta K=0 point); right
axis = effective size -log_3 nu.  Static tau = 1/sqrt(As).

Data: data/sector_L12_kimhuse_and_tfim/ (scripts/extract/nutau_sector_L12.py).
Output: figures/sector_L12_kimhuse_and_tfim.{pdf,png}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, StrMethodFormatter

from sso.config import data_dir
from sso.nutau.io import load_arcs, load_npz
from sso.plotting.frontier import sector_frontier
from sso.plotting.style import save_figure

DATA = data_dir("sector_L12_kimhuse_and_tfim")
TAU_MAX = 1e5
C_K0, C_K1 = "#009E73", "#E69F00"
C_MIX = tuple(0.5 * (np.array(to_rgb(C_K0)) + np.array(to_rgb(C_K1))))
FADE_ALPHA = 0.35
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 6.5
MODELS = ("Ising_J1.0X0.905Z0.809", "Ising_J1.0X2.0Z0.0")
COLOR = {0: C_K0, 1: C_K1, "mix": C_MIX}


def draw_panel(ax, model):
    fr = sector_frontier(load_npz(os.path.join(DATA, f"{model}_grid_K0.npz")),
                         load_npz(os.path.join(DATA, f"{model}_grid_K1.npz")),
                         load_arcs(DATA, f"{model}_crossK_theta*.npz"),
                         load_arcs(DATA, f"{model}_withinK0_theta*.npz"),
                         tau_max=TAU_MAX)
    for ln in fr["lines"]:
        solid = ln["solid"]
        z = (5 if ln["sector"] == "mix" else 4) if solid else 2
        ax.plot(ln["x"], ln["y"], "-", color=COLOR[ln["sector"]], lw=1.8 if solid else 1.3,
                alpha=1.0 if solid else FADE_ALPHA, zorder=z)
    ax.axhline(fr["nu_inf0"], color=C_K0, ls="--", lw=1.0, alpha=0.9, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.9 * min(fr["t0"].min(), fr["t1"].min()), TAU_MAX)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ln3 = np.log(3.0)
    sec = ax.secondary_yaxis("right", functions=(lambda v: -np.log(v) / ln3,
                                                 lambda s: 3.0 ** (-s)))
    sec.set_ylabel("Effective Size", fontsize=FS_LABEL)
    sec.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    sec.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    sec.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    sec.tick_params(labelsize=FS_TICK)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 1.75))
    for ax, model in zip(axes, MODELS):
        draw_panel(ax, model)
    handles = [Line2D([0], [0], color=C_K0, lw=1.8, label=r"$k=0$"),
               Line2D([0], [0], color=C_K1, lw=1.8, label=r"$k=1$"),
               Line2D([0], [0], color=C_MIX, lw=1.8, label="mix")]
    axes[0].legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="lower left",
                   handlelength=1.4, borderaxespad=0.3)
    plt.subplots_adjust(left=0.14, right=0.925, bottom=0.26, top=0.96, wspace=0.92)
    print("wrote", save_figure(fig, "sector_L12_kimhuse_and_tfim", png_dpi=200, pdf_dpi=200))


if __name__ == "__main__":
    main()
