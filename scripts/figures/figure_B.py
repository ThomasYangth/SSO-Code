"""Figure B: weak-integrability breaking, diffusive energy mode, and Floquet filters (all Lorentzian).

(b) hx = 0.9 Ising chain, L = 12, K = 0 (o4): nu(tau) frontier for hz = 0.005..0.2
    (left panel of nutau_hx0.9_twopanel, frontier broken at the branch-crossing gap and
    bridged by the within-K0 arc).  Inset (replacing the Delta nu vs tau hz^2 collapse):
    hz = 0.03 frontier and the energy current Q = sum_x (Y_x Z_{x+1} - Z_x Y_{x+1}) under
    the Lorentzian filter m^2/(omega^2 + m^2) = 1/(1 + theta omega^2) (KrylovChain
    convention; used throughout this figure) in the eigenbasis of H; star = bare Q at
    tau = 1/(2 hz), nu = 1/9, labelled Q.  Static tau = 1/sqrt(As); filter points with As <= 1e-14 dropped.
(c) nu vs tau_obs/L^2 of the energy-density mode H_q (n = 1) of the mixed-field Ising
    ring (J=1, g=-1.05, h=0.5), L = 16..23, under the Lorentzian filter (DQT; the stored
    production curves re-filtered by sso.dqt.refilter, which reaches tau/L^2 ~ 0.15-0.17);
    inset nu vs tau unrescaled.  Layout of panel (a) of fig3panel (which used the Gaussian
    window).
(a) kicked Ising (g = 0.9, J = 1, h = 0.809), L = 12: frontier (identity-only projector)
    and Sum Z_x (K = 0), Z_1 (all momenta), Sum X_x (K = 0) under the Lorentzian filter
    (panel (a) of floquet_two_panel with only this filter).  Floquet chord tau; points with
    tau >= 1e8 dropped.

Each panel (with its colourbar) occupies one third of a 7.0 in double-column width.
Data: data/nutau_hx0.9_twopanel/ (filt_hz0.03_YZcur_lorentz.npz), data/fig3panel/ (modhydro_lorentz_*_REFILTER.npz, scripts/extract/dqt_lorentz_fronts.py),
      data/floquet_two_panel/ (lorentzcut_L12_*.npz); lorentz data from
      slurm/nutau/filter_lorentz_L12.slurm via the nutau extractors (--lorentz_src).
Output: figures/figure_B.{pdf,png}
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, StrMethodFormatter

from sso.config import data_dir
from sso.dqt.figdata import LS, load_lorentz_front
from sso.nutau.conventions import tau_floquet, tau_static
from sso.nutau.io import load_npz
from sso.plotting.frontier import break_gaps, tau_sorted
from sso.plotting.style import layout_row, panel_label, save_figure


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hx = _load("nutau_hx0.9_twopanel")
fl = _load("floquet_two_panel")

SLOT, FIG_H = 7.0 / 3.0, 2.05
PANEL_H = 1.36   # plot height (inches); widths are equalised by layout_row
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 6
C_FILT = "#2a78d6"
_PL = matplotlib.colormaps["plasma"]
OP_COLOUR = {"Zsum": _PL(0.80), "Z0": _PL(0.50), "Xsum": _PL(0.15)}   # plasma, as in (a); brightest = closest to the frontier


def lorentz_static():
    z = load_npz(os.path.join(hx.DATA, f"filt_hz{hx.HZ_R}_YZcur_lorentz.npz"))
    As, n = z["lorentz_As"], z["lorentz_nu"]
    t = tau_static(As)
    ok = np.isfinite(t) & np.isfinite(n) & (As > hx.AS_FLOOR)
    return tau_sorted(t[ok], n[ok])


def lorentz_floquet(op, K):
    z = load_npz(os.path.join(fl.DATA, f"lorentzcut_L12_K{K}_{op}.npz"))
    tau, nu = tau_floquet(z["As"]), z["nu"]
    ok = np.isfinite(tau) & np.isfinite(nu) & (tau < fl.TAU_NOISE)
    return tau_sorted(tau[ok], nu[ok])


def panel_hz(fig, ax, cax):
    plasma = matplotlib.colormaps["plasma"]
    cmap = LinearSegmentedColormap.from_list("pl", [plasma(x) for x in np.linspace(0.12, 0.88, 256)])
    norm = LogNorm(vmin=min(hx.HZS), vmax=max(hx.HZS))
    for hz in hx.HZS:
        c = cmap(norm(hz))
        t, n = hx.frontier(hz)
        k = t <= hx.TAU_MAX
        ax.plot(*break_gaps(t[k], n[k]), "-", color=c, lw=1.0)
        for at, an in hx.arcs(hz):
            g = at <= hx.TAU_MAX
            ax.plot(at[g], an[g], "-", color=c, lw=1.0)
    ax.axhline(hx.NU, color="0.6", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, hx.TAU_MAX)
    ax.set_ylim(0.0, 0.35)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cax.set_title(r"$h_z$", fontsize=FS_LABEL - 1, pad=2)
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    cb.outline.set_linewidth(0.5)

    ins = ax.inset_axes([0.47, 0.44, 0.50, 0.44])
    t, n = hx.frontier(hx.HZ_R)
    k = t <= hx.TAU_MAX
    ins.plot(*break_gaps(t[k], n[k]), "-", color="k", lw=1.1, label="frontier")
    for at, an in hx.arcs(hx.HZ_R):
        g = at <= hx.TAU_MAX
        ins.plot(at[g], an[g], "-", color="k", lw=1.1)
    tm, nm = lorentz_static()
    g = tm <= hx.TAU_MAX
    ins.plot(tm[g], nm[g], "--", color=C_FILT, lw=1.1, label="Lorentzian")
    tb = 1.0 / (2 * hx.HZ_R)
    ins.scatter([tb], [hx.NU], marker="*", s=30, color="gold", edgecolors="k", linewidths=0.4, zorder=6)
    ins.axhline(hx.NU, color="0.6", ls=":", lw=0.6)
    ins.set_xscale("log")
    ins.set_xlim(3.0, hx.TAU_MAX)
    ins.set_ylim(0.075, 0.118)
    ins.tick_params(labelsize=5, length=1.5, pad=1)
    ins.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ins.xaxis.set_major_formatter(LogFormatterMathtext())
    ins.xaxis.set_minor_formatter(NullFormatter())
    ins.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.08, 0.10]))
    ins.set_title(rf"$h_z={hx.HZ_R}$", fontsize=5.5, pad=1.5)
    ins.text(tb, hx.NU - 0.0035, r"$Q$", fontsize=5.5, ha="center", va="top")
    ins.legend(fontsize=4.8, frameon=False, loc="lower left", handlelength=1.6,
               borderaxespad=0.2, labelspacing=0.2)


def panel_dqt(fig, ax, cax):
    lcol = dict(zip(LS, plt.cm.viridis(np.linspace(0, 0.92, len(LS)))))
    fronts = {}
    for L in LS:
        d = load_lorentz_front(L)
        ok = np.isfinite(d["nu"])
        fronts[L] = {k: d[k][ok] for k in ("tau_obs", "tau_err", "nu", "nu_err")}
    for L, d in fronts.items():
        ax.errorbar(d["tau_obs"] / L ** 2, d["nu"], xerr=d["tau_err"] / L ** 2, yerr=d["nu_err"],
                    fmt="o-", ms=1.4, lw=0.8, elinewidth=0.5, capsize=0, color=lcol[L])
    ax.set_xlabel(r"$\tau/L^2$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.set_xlim(left=0)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, lw=0.4)
    ins = ax.inset_axes([0.50, 0.48, 0.47, 0.49])
    ins.set_box_aspect(1)
    ins.set_anchor("NE")
    for L, d in fronts.items():
        ins.plot(d["tau_obs"], d["nu"], "-", lw=0.8, color=lcol[L])
    ins.set_xlabel(r"$\tau$", fontsize=5.5, labelpad=0.5)
    ins.set_ylabel(r"$\nu$", fontsize=5.5, labelpad=0.5)
    ins.tick_params(labelsize=5, length=1.5, pad=1)
    ins.set_xlim(left=0)
    ins.grid(alpha=0.25, lw=0.3)
    cmap = ListedColormap([lcol[L] for L in LS])
    norm = BoundaryNorm(np.arange(LS[0] - 0.5, LS[-1] + 1), cmap.N)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=list(LS)[::2])
    cax.set_title(r"$L$", fontsize=FS_LABEL - 1, pad=2)
    cb.ax.tick_params(labelsize=FS_TICK - 1, length=1.5)
    cb.outline.set_linewidth(0.5)


def panel_floquet(ax):
    bt, bn = fl.frontier(12)
    m = bt <= fl.TAU_MAX
    ax.plot(bt[m], bn[m], "-", color="black", linewidth=1.6, zorder=5)
    y_all, tau_lo = [bn[m]], bt.min()
    for op, _, K in fl.OPERATORS:
        tau, nu = lorentz_floquet(op, K)
        k = tau <= fl.TAU_MAX
        ax.plot(tau[k], nu[k], "--", color=OP_COLOUR[op], linewidth=1.2,
                zorder=6 if op == "Zsum" else 4)          # Sum Z_x drawn over the frontier
        y_all.append(nu[k])
        tau_lo = min(tau_lo, tau.min())
    y = np.concatenate(y_all)
    y = y[np.isfinite(y) & (y > 0)]
    ax.set_ylim(y.min() * 0.80, y.max() * 1.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tau_lo * 0.75, fl.TAU_MAX * 1.3)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=15))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1.0, 1e2, 1e4]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    handles = [Line2D([], [], color=OP_COLOUR[op], lw=1.2, ls="--", label=lab)
               for op, lab, _ in fl.OPERATORS]
    leg = ax.legend(handles=handles, title="Lorentzian filter of", title_fontsize=FS_LEG,
                    fontsize=FS_LEG, frameon=False, loc="upper right", handlelength=1.8,
                    borderaxespad=0.3, labelspacing=0.25)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], color="black", lw=1.6, label="frontier")], fontsize=FS_LEG,
              frameon=False, loc="lower left", handlelength=1.8, borderaxespad=0.3)


def main():
    fig = plt.figure(figsize=(3 * SLOT, FIG_H))
    fx, fy = (lambda x: x / (3 * SLOT)), (lambda y: y / FIG_H)
    bottom, height = 0.46, PANEL_H
    axA = fig.add_axes([fx(0.5), fy(bottom), fx(1.2), fy(height)])
    axB = fig.add_axes([fx(SLOT + 0.5), fy(bottom), fx(1.2), fy(height)])
    caxB = fig.add_axes([fx(SLOT + 1.8), fy(bottom), fx(0.07), fy(height)])
    axC = fig.add_axes([fx(2 * SLOT + 0.5), fy(bottom), fx(1.2), fy(height)])
    caxC = fig.add_axes([fx(2 * SLOT + 1.8), fy(bottom), fx(0.07), fy(height)])
    panel_floquet(axA)
    panel_hz(fig, axB, caxB)
    panel_dqt(fig, axC, caxC)
    for ax, lab in ((axA, "(a)"), (axB, "(b)"), (axC, "(c)")):
        panel_label(ax, lab, FS_LABEL)
    W = layout_row(fig, [(axA, None), (axB, caxB), (axC, caxC)], bottom, height)
    print(f"plot box {W:.3f} x {height:.2f} in")
    print("wrote", save_figure(fig, "figure_B", png_dpi=300))


if __name__ == "__main__":
    main()
