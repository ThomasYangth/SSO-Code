"""Figure H: the DMRG attempt (unsuccessful approach) -- epsilon-constrained DMRG nu(tau)
vs the exact frontier.

(a) L = 12 (Ising J=1, hx=0.905, hz=0.809, OBC): DMRG at chi = 128, 256, 512 and the exact
    (ED) frontier (attempt_dmrg_L12).
(b) L = 12, 18, 24, 30 x chi = 128, 256, 512 vs tau/L^2 (attempt_dmrg_multiL, without its
    exact L = 12 frontier, which is shown in (a)).
tau = 1/sqrt(<C_H^2>), nu = <M>.  Only rows with plotted = 1 (frozen chi-wall tails
omitted).  The two L = 12 step-8 points at chi = 256 / 512 come from completed (converged)
chi-refinement runs whose MPS was not written (values from the log); they are drawn like
every other converged point.  No filter is involved.

Each panel one third of a 7.0 in double-column width.
Data: data/attempt_dmrg/{dmrg_points,ed_front_L12}.csv (docs/dmrg_attempt.md).
Output: figures/figure_H.{pdf,png}
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from sso import config
from sso.plotting.style import layout_row, panel_label, save_figure

SLOT, FIG_H, PANEL_H = 7.0 / 3.0, 2.05, 1.36
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 5.8
FRONT = "#8a6a1e"
CHIS = (128, 256, 512)
COL12 = {128: "#9cc0e6", 256: "#3d7dbf", 512: "#123a63"}
LS = (12, 18, 24, 30)
CMAP = {12: plt.cm.Blues, 18: plt.cm.Greens, 24: plt.cm.Oranges, 30: plt.cm.Purples}
SHADE = {128: 0.5, 256: 0.72, 512: 0.95}


def load(name):
    with open(config.data_dir("attempt_dmrg", name)) as f:
        return list(csv.DictReader(f))


def series(pts, L, chi):
    s = sorted((float(r["eps"]), float(r["nu"])) for r in pts
               if int(r["L"]) == L and int(r["chi"]) == chi)
    return [1 / np.sqrt(e) for e, _ in s], [m for _, m in s]


def panel_L12(ax, pts, front):
    ax.plot([1 / np.sqrt(e) for e, _ in front], [m for _, m in front], ls=":", lw=1.6,
            color=FRONT, zorder=3)
    for chi in CHIS:
        x, y = series(pts, 12, chi)
        ax.plot(x, y, "-o", color=COL12[chi], lw=1.2, ms=2.8, mec="white", mew=0.4, zorder=5)
    ax.set_xlim(2, 26)
    ax.set_ylim(0.20, 0.31)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", color="#e2ded6", lw=0.5, zorder=0)
    handles = [Line2D([], [], ls=":", lw=1.6, color=FRONT, label="exact frontier")]
    handles += [Line2D([], [], color=COL12[c], lw=1.2, marker="o", ms=2.8, mec="white", mew=0.4,
                       label=rf"$\chi={c}$") for c in CHIS]
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="lower left", handlelength=1.8,
              labelspacing=0.2, borderaxespad=0.3)
    ax.set_title(r"$L=12$", fontsize=FS_LABEL - 1, pad=2)


def panel_multiL(ax, pts):
    for L in LS:
        for chi in CHIS:
            x, y = series(pts, L, chi)
            if not x:
                continue
            c = CMAP[L](SHADE[chi])
            x = np.asarray(x) / L ** 2
            ax.plot(x, y, "-o", color=c, lw=1.1, ms=2.6 if len(x) > 2 else 3.6, mec="white",
                    mew=0.4, zorder=4 + chi // 128)
    ax.set_xscale("log")
    ax.set_ylim(0.195, 0.315)
    ax.set_xlim(0.004, 0.18)
    ax.xaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05, 0.1]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\tau/L^2$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", color="#e2ded6", lw=0.5, zorder=0)
    hL = [Line2D([], [], color=CMAP[L](0.75), lw=1.4, label=f"$L={L}$") for L in LS]
    leg = ax.legend(handles=hL, fontsize=FS_LEG, frameon=False, loc="lower left", handlelength=1.6,
                    labelspacing=0.2, borderaxespad=0.3)
    ax.add_artist(leg)
    hC = [Line2D([], [], color=plt.cm.Greys(SHADE[c]), lw=1.4, label=rf"$\chi={c}$") for c in CHIS]
    ax.legend(handles=hC, fontsize=FS_LEG, frameon=False, loc="upper right", handlelength=1.6,
              labelspacing=0.2, borderaxespad=0.3)


def main():
    pts = [r for r in load("dmrg_points.csv") if r["plotted"] == "1"]
    front = sorted((float(r["eps"]), float(r["nu"])) for r in load("ed_front_L12.csv"))
    fig = plt.figure(figsize=(2 * SLOT, FIG_H))
    fx, fy = (lambda x: x / (2 * SLOT)), (lambda y: y / FIG_H)
    bottom = 0.46
    axA = fig.add_axes([fx(0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    axB = fig.add_axes([fx(SLOT + 0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    panel_L12(axA, pts, front)
    panel_multiL(axB, pts)
    for ax, lab in ((axA, "(a)"), (axB, "(b)")):
        panel_label(ax, lab, FS_LABEL)
    W = layout_row(fig, [(axA, None), (axB, None)], bottom, PANEL_H)
    print(f"plot box {W:.3f} x {PANEL_H} in")
    print("wrote", save_figure(fig, "figure_H", png_dpi=300))


if __name__ == "__main__":
    main()
