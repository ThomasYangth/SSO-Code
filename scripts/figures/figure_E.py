"""Figure E: hx = 0.9 Ising chain (L = 12, K = 0, o4): weak-integrability-breaking collapse
and a comparison of filters of the energy current.

(a) Delta nu = 1/9 - nu vs tau hz^2 for hz = 0.005..0.2 (frontier grids and within-K0
    arcs; the inset of nutau_hx0.9_twopanel's left panel).  No filter.
(b) hz = 0.03: frontier and the current Q = sum_x (Y_x Z_{x+1} - Z_x Y_{x+1}) under four
    filters in the eigenbasis of H -- explicit filter comparison: Lorentzian
    m^2/(omega^2+m^2) (the default filter), Gaussian exp(-omega^2/2s^2), hard cutoff
    1[|omega| <= w], and the running time average ("sinc"); star = bare Q at
    tau = 1/(2 hz), nu = 1/9.
(c) nu_frontier(tau) - nu_filter(tau) at equal tau, log scale.  The frontier is
    evaluated at each filter point's tau from the within-K0 arc (721 points) on the arc's
    tau range and from a cubic spline in (log tau, log nu) of the theta grid elsewhere;
    leave-one-out tests give an interpolation error <~ 5e-7 relative, below the smallest
    gap (Lorentzian, ~7e-6 relative).
Static tau = 1/sqrt(As); filter points with As <= 1e-14 dropped; tau <= 3e4.

Each panel one third of a 7.0 in double-column width.
Data: data/nutau_hx0.9_twopanel/ (grid_hz*, arc_hz*, filt_hz0.03_YZcur{,_lorentz}.npz).
Output: figures/figure_E.{pdf,png}
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.interpolate import CubicSpline

from sso.nutau.conventions import tau_static
from sso.nutau.io import load_npz
from sso.plotting.frontier import break_gaps, tau_sorted
from sso.plotting.style import layout_row, panel_label, save_figure


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hx = _load("nutau_hx0.9_twopanel")
SLOT, FIG_H, PANEL_H = 7.0 / 3.0, 2.05, 1.36
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 6
FILTERS = [("lorentz", "Lorentzian", "#2a78d6", (0, (5, 1.6))),
           ("gauss", "Gaussian", "#eb6834", (0, (3, 1.2, 1, 1.2))),
           ("hard", "hard", "#1baf7a", (0, (1.2, 1.2))),
           ("avg", "sinc", "#eda100", (0, (6, 1.5, 1.2, 1.5, 1.2, 1.5)))]


def filter_curve(kind):
    if kind == "lorentz":
        z = load_npz(os.path.join(hx.DATA, f"filt_hz{hx.HZ_R}_YZcur_lorentz.npz"))
        As, n = z["lorentz_As"], z["lorentz_nu"]
        t = tau_static(As)
        ok = np.isfinite(t) & np.isfinite(n) & (As > hx.AS_FLOOR)
        return tau_sorted(t[ok], n[ok])
    return hx.modul(kind)


def frontier_function(hz):
    """nu_frontier(tau): arc on its tau range, cubic spline (log-log) of the grid elsewhere."""
    t, n = hx.frontier(hz)
    arcs = hx.arcs(hz)
    if arcs:
        at, an = arcs[0]
        o = np.argsort(at)
        at, an = at[o], an[o]
        lo, hi = at[0], at[-1]
        up = t > hi
        lowb = t < lo
        cs_up = CubicSpline(np.log(t[up]), np.log(n[up]))
        cs_lo = CubicSpline(np.log(t[lowb]), np.log(n[lowb])) if lowb.sum() >= 4 else None

        def f(x):
            x = np.asarray(x, float)
            out = np.full_like(x, np.nan)
            a = (x >= lo) & (x <= hi)
            out[a] = np.exp(np.interp(np.log(x[a]), np.log(at), np.log(an)))
            u = x > hi
            out[u] = np.exp(cs_up(np.log(x[u])))
            if cs_lo is not None:
                b = x < lo
                out[b] = np.exp(cs_lo(np.log(x[b])))
            return out
        return f
    cs = CubicSpline(np.log(t), np.log(n))
    return lambda x: np.exp(cs(np.log(np.asarray(x, float))))


def panel_collapse(fig, ax, cax):
    plasma = matplotlib.colormaps["plasma"]
    cmap = LinearSegmentedColormap.from_list("pl", [plasma(x) for x in np.linspace(0.12, 0.88, 256)])
    norm = LogNorm(vmin=min(hx.HZS), vmax=max(hx.HZS))
    for hz in hx.HZS:
        c = cmap(norm(hz))
        t, n = hx.frontier(hz)
        dn = hx.NU - n
        g = (t <= hx.TAU_MAX) & (dn > 0)
        ax.plot(*break_gaps(t[g] * hz ** 2, dn[g]), "-", color=c, lw=1.0)
        for at, an in hx.arcs(hz):
            da = hx.NU - an
            gg = (at <= hx.TAU_MAX) & (da > 0)
            ax.plot(at[gg] * hz ** 2, da[gg], "-", color=c, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau h_z^2$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\Delta\nu = 1/9 - \nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_minor_formatter(NullFormatter())
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cax.set_title(r"$h_z$", fontsize=FS_LABEL - 1, pad=2)
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    cb.outline.set_linewidth(0.5)


def panel_filters(ax):
    t, n = hx.frontier(hx.HZ_R)
    k = t <= hx.TAU_MAX
    ax.plot(*break_gaps(t[k], n[k]), "-", color="k", lw=1.3, zorder=3)
    for at, an in hx.arcs(hx.HZ_R):
        g = at <= hx.TAU_MAX
        ax.plot(at[g], an[g], "-", color="k", lw=1.3, zorder=3)
    for kind, lab, col, ls in FILTERS:
        tm, nm = filter_curve(kind)
        g = tm <= hx.TAU_MAX
        ax.plot(tm[g], nm[g], linestyle=ls, color=col, lw=1.2, zorder=5 if kind == "lorentz" else 4)
    tb = 1.0 / (2 * hx.HZ_R)
    ax.scatter([tb], [hx.NU], marker="*", s=45, color="gold", edgecolors="k", linewidths=0.4, zorder=6)
    ax.text(tb, hx.NU - 0.0025, r"$Q$", fontsize=FS_LEG + 0.5, ha="center", va="top")
    ax.axhline(hx.NU, color="0.6", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(3.0, hx.TAU_MAX)
    ax.set_ylim(0.07, 0.118)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_title(rf"$h_z = {hx.HZ_R}$", fontsize=FS_LABEL - 1, pad=2)
    handles = [Line2D([], [], color="k", lw=1.3, label="frontier")]
    handles += [Line2D([], [], color=col, lw=1.2, linestyle=ls, label=lab) for _, lab, col, ls in FILTERS]
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="lower left", handlelength=2.2,
              labelspacing=0.2, borderaxespad=0.3)


def panel_difference(ax):
    f = frontier_function(hx.HZ_R)
    for kind, lab, col, ls in FILTERS:
        tm, nm = filter_curve(kind)
        g = (tm <= hx.TAU_MAX) & (tm >= 3.0)
        d = f(tm[g]) - nm[g]
        ok = np.isfinite(d) & (d > 0)
        ax.plot(tm[g][ok], d[ok], linestyle=ls, color=col, lw=1.2)
        if np.any(np.isfinite(d) & (d <= 0)):
            print(f"  {kind}: {np.sum(d <= 0)} points with nu_filter >= nu_frontier (dropped on log axis)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.0, hx.TAU_MAX)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu_{\rm frontier} - \nu_{\rm filter}$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.grid(True, which="major", alpha=0.25, lw=0.4)


def main():
    fig = plt.figure(figsize=(3 * SLOT, FIG_H))
    fx, fy = (lambda x: x / (3 * SLOT)), (lambda y: y / FIG_H)
    bottom = 0.46
    axA = fig.add_axes([fx(0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    caxA = fig.add_axes([fx(1.8), fy(bottom), fx(0.07), fy(PANEL_H)])
    axB = fig.add_axes([fx(SLOT + 0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    axC = fig.add_axes([fx(2 * SLOT + 0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    panel_collapse(fig, axA, caxA)
    panel_filters(axB)
    panel_difference(axC)
    for ax, lab in ((axA, "(a)"), (axB, "(b)"), (axC, "(c)")):
        panel_label(ax, lab, FS_LABEL)
    W = layout_row(fig, [(axA, caxA), (axB, None), (axC, None)], bottom, PANEL_H)
    print(f"plot box {W:.3f} x {PANEL_H} in")
    print("wrote", save_figure(fig, "figure_E", png_dpi=300))


if __name__ == "__main__":
    main()
