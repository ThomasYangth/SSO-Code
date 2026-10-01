"""hx=0.9 Ising chain, L=12, K=0 (o4): frontier vs hz, and filtered energy current.

LEFT   nu(tau) frontier for hz in {0.005..0.2} (colour = hz), broken at the
       branch-crossing tau gap and bridged by the within-K0 arc; inset
       Delta nu = 1/9 - nu vs tau hz^2.
RIGHT  hz = 0.03: frontier plus the current Q = sum_x (Y_x Z_{x+1} - Z_x Y_{x+1})
       filtered in the eigenbasis of H(hz) by a Gaussian, a hard cutoff and
       the running time average (sinc); gold star = bare Q at tau = 1/(2 hz),
       nu = 1/9 (its exact variance is 4 hz^2).  Static tau = 1/sqrt(As);
       filter points with As <= 1e-14 dropped.

Data: data/nutau_hx0.9_twopanel/ (scripts/extract/nutau_hx09_twopanel.py).
Output: figures/nutau_hx0.9_twopanel.{pdf,png}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from sso.config import data_dir
from sso.nutau.conventions import tau_static
from sso.nutau.io import load_arcs, load_npz
from sso.plotting.frontier import break_gaps, grid_curve, tau_sorted
from sso.plotting.style import save_figure

DATA = data_dir("nutau_hx0.9_twopanel")
NU = 1.0 / 9.0
TAU_MAX = 3e4
AS_FLOOR = 1e-14
HZS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
HZ_R = 0.03


def frontier(hz):
    _, t, n, _ = grid_curve(load_npz(os.path.join(DATA, f"grid_hz{hz}.npz")))
    return tau_sorted(t, n)


def arcs(hz):
    return [(t, n) for t, n, _ in load_arcs(DATA, f"arc_hz{hz}_theta*.npz")]


def modul(kind):
    z = load_npz(os.path.join(DATA, f"filt_hz{HZ_R}_YZcur.npz"))
    As, n = z[f"{kind}_As"], z[f"{kind}_nu"]
    t = tau_static(As)
    ok = np.isfinite(t) & np.isfinite(n) & (As > AS_FLOOR)
    return tau_sorted(t[ok], n[ok])


def main():
    plasma = matplotlib.colormaps["plasma"]
    cmap = LinearSegmentedColormap.from_list("pl", [plasma(x) for x in np.linspace(0.12, 0.88, 256)])
    norm = LogNorm(vmin=min(HZS), vmax=max(HZS))
    fig = plt.figure(figsize=(4.6, 1.95))
    axL = fig.add_axes([0.100, 0.235, 0.28, 0.66])
    cax = fig.add_axes([0.390, 0.235, 0.013, 0.66])
    axR = fig.add_axes([0.620, 0.235, 0.28, 0.66])

    for hz in HZS:
        c = cmap(norm(hz))
        t, n = frontier(hz)
        k = t <= TAU_MAX
        axL.plot(*break_gaps(t[k], n[k]), "-", color=c, lw=1.0)
        for at, an in arcs(hz):
            g = at <= TAU_MAX
            axL.plot(at[g], an[g], "-", color=c, lw=1.0)
    axL.axhline(NU, color="0.6", ls=":", lw=0.8)
    axL.set_xscale("log")
    axL.set_xlim(0.5, TAU_MAX)
    axL.set_ylim(0.0, 0.35)
    axL.set_xlabel(r"$\tau$", fontsize=9)
    axL.set_ylabel(r"$\nu$", fontsize=9)
    axL.tick_params(labelsize=7)
    axL.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axL.xaxis.set_major_formatter(LogFormatterMathtext())
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cax.set_title(r"$h_z$", fontsize=7, pad=2)
    cbar.ax.tick_params(labelsize=6)

    axi = axL.inset_axes([0.50, 0.53, 0.43, 0.44])
    for hz in HZS:
        c = cmap(norm(hz))
        t, n = frontier(hz)
        dn = NU - n
        g = (t <= TAU_MAX) & (dn > 0)
        axi.plot(*break_gaps(t[g] * hz ** 2, dn[g]), "-", color=c, lw=0.8)
        for at, an in arcs(hz):
            da = NU - an
            gg = (at <= TAU_MAX) & (da > 0)
            axi.plot(at[gg] * hz ** 2, da[gg], "-", color=c, lw=0.8)
    axi.set_xscale("log")
    axi.set_yscale("log")
    axi.set_xlabel(r"$\tau h_z^2$", fontsize=6, labelpad=1)
    axi.set_ylabel(r"$\Delta\nu$", fontsize=6, labelpad=1)
    axi.tick_params(labelsize=5, pad=1)

    t, n = frontier(HZ_R)
    k = t <= TAU_MAX
    axR.plot(*break_gaps(t[k], n[k]), "-", color="k", lw=1.3, label="frontier")
    for at, an in arcs(HZ_R):
        g = at <= TAU_MAX
        axR.plot(at[g], an[g], "-", color="k", lw=1.3)
    for kind, lab, col in (("gauss", "Gaussian", "#d62728"),
                           ("hard", r"hard ($\omega$)", "#1f77b4"),
                           ("avg", "sinc", "#2ca02c")):
        tm, nm = modul(kind)
        g = tm <= TAU_MAX
        axR.plot(tm[g], nm[g], "--", color=col, lw=1.3, label=lab)
    tbare = 1.0 / (2 * HZ_R)
    axR.scatter([tbare], [NU], marker="*", s=80, color="gold", edgecolors="k",
                linewidths=0.5, zorder=6)
    axR.text(tbare, 0.1045, r"bare $Q$", fontsize=6, ha="center", va="center", color="#b8860b")
    axR.axhline(NU, color="0.6", ls=":", lw=0.8)
    axR.set_xscale("log")
    axR.set_xlim(3.0, TAU_MAX)
    axR.set_ylim(0.07, 0.12)
    axR.set_xlabel(r"$\tau$", fontsize=9)
    axR.set_ylabel(r"$\nu$", fontsize=9)
    axR.tick_params(labelsize=7)
    axR.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axR.xaxis.set_major_formatter(LogFormatterMathtext())
    axR.legend(fontsize=5.5, loc="lower left", framealpha=0.9, handlelength=1.6,
               labelspacing=0.25)
    print("wrote", save_figure(fig, "nutau_hx0.9_twopanel", png_dpi=150))


if __name__ == "__main__":
    main()
