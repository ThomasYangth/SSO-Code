"""Figure G: filters vs frontier for a Hamiltonian system, and finite-size ED vs the
thermodynamic-limit TI ansatz for Floquet and Hamiltonian systems.

(a) Kim-Huse chain (J=1, hx=0.905, hz=0.809), L = 12: global frontier coloured by sector
    (k = 0 o4 / k = 1 o0 / the cross-sector mix arc; only the winning segments), and two
    operators under the Lorentzian filter m^2/(omega^2+m^2):
      * sum_x n.sigma_x along the single-site Pauli direction n maximising nu(tau = inf)
        after removing I..H^3 (scripts/compute/nutau/optimal_onesite.py; L = 12:
        n = (0.7125, 0, -0.7017), nu_inf = 0.0209 vs 0.0187 for Sum Z alone),
      * the longest-wavelength energy-density mode H_q, q = 2 pi / L (K = 1; nothing
        projected; no conserved part at even L).
    Static tau = 1/sqrt(As); points with As <= 1e-14 dropped.
(b) kicked Ising (g=0.9, J=1, h=0.809): ED K = 0 frontier (identity removed, o1) on rings
    L = 6..12 (solid, running minimum of nu in tau) vs the TI ansatz at support M = 6..12
    (dashed); Floquet chord tau.  floquet_two_panel (b).
(c) Kim-Huse: ED K = 0 frontier with I..H^3 removed (o4), L = 6..12, vs the static TI
    ansatz with H, H^2, H^3 removed (o4; "restricted true powers": the H^k strings of
    support <= M + 1), M = 6..12.  The truncated powers leave the TI curves 2-5 % above ED
    at small tau.  Static tau.
Panels (b), (c) involve no filter.

Each panel (with its colourbars) one third of a 7.0 in double-column width.
Data: data/kimhuse_three_panel/ (grid_L*_K*, crossK_L12, filt_L12_{K0_OPT1S,K1_Hdens},
      onesite_opt_L12_o4, ti_rtp_o4/),
      data/floquet_two_panel/ (grid_L*_K0, nutau_floquet_M*_g0p9tF1p0_gpures.dat).
Output: figures/figure_G.{pdf,png}
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, StrMethodFormatter

from sso.config import data_dir
from sso.nutau.conventions import tau_floquet, tau_static
from sso.nutau.io import load_arcs, load_npz
from sso.plotting.frontier import grid_curve, sector_frontier, tau_sorted
from sso.plotting.style import layout_row, panel_label, save_figure
from sso.tiansatz import read_ti_table

KH = data_dir("kimhuse_three_panel")
FL = data_dir("floquet_two_panel")
SLOT, FIG_H, PANEL_H = 7.0 / 3.0, 2.05, 1.36
FS_LABEL, FS_TICK, FS_LEG = 8, 7, 5.6
TAU_MAX = 1e5
_PL = matplotlib.colormaps["plasma"]
C_K0, C_K1 = "#009E73", "#E69F00"          # sector colours of figure_A (a)
SECTOR_COL = {0: C_K0, 1: C_K1, "mix": tuple(0.5 * (np.array(matplotlib.colors.to_rgb(C_K0))
                                                     + np.array(matplotlib.colors.to_rgb(C_K1))))}
OPS = [("K0_OPT1S", r"$\Sigma_x \hat n\!\cdot\!\vec\sigma_x$ (opt.)", _PL(0.15)),
       ("K1_Hdens", r"$H_q$, $q=2\pi/L$", _PL(0.55))]
LS_ED, MS_TI = list(range(6, 13)), list(range(6, 13))


def discrete(name, values, lo, hi):
    base = plt.get_cmap(name)
    cols = [base(p) for p in np.linspace(hi, lo, len(values))]
    edges = np.concatenate([[values[0] - 0.5], (np.array(values[:-1]) + np.array(values[1:])) / 2,
                            [values[-1] + 0.5]])
    return cols, ListedColormap(cols), BoundaryNorm(edges, len(values))


def log_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel_filters(ax):
    fr = sector_frontier(load_npz(os.path.join(KH, "grid_L12_K0.npz")),
                         load_npz(os.path.join(KH, "grid_L12_K1.npz")),
                         load_arcs(KH, "crossK_L12.npz"), tau_max=TAU_MAX)
    ys, tlo = [], np.inf
    for ln in fr["lines"]:
        if not ln["solid"]:
            continue
        ax.plot(ln["x"], ln["y"], "-", color=SECTOR_COL[ln["sector"]], lw=1.6, zorder=4)
        ys.append(np.asarray(ln["y"]))
        tlo = min(tlo, np.min(ln["x"]))
    opt = load_npz(os.path.join(KH, "onesite_opt_L12_o4.npz"))
    nx, ny, nz = opt["sum_n"]
    for tag, lab, col in OPS:
        z = load_npz(os.path.join(KH, f"filt_L12_{tag}.npz"))
        As, nu = z["lorentz_As"], z["lorentz_nu"]
        t = tau_static(As)
        ok = np.isfinite(t) & np.isfinite(nu) & (As > 1e-14) & (t <= TAU_MAX)
        t, n = tau_sorted(t[ok], nu[ok])
        ax.plot(t, n, linestyle=(0, (4, 1.5)), color=col, lw=1.3, zorder=5)
        ys.append(n)
        tlo = min(tlo, t.min())
    y = np.concatenate(ys)
    y = y[np.isfinite(y) & (y > 0)]
    ax.set_ylim(y.min() * 0.8, y.max() * 1.25)
    ax.set_xlim(tlo * 0.75, TAU_MAX * 1.3)
    log_axes(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1.0, 1e2, 1e4]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    h1 = [Line2D([], [], color=SECTOR_COL[k], lw=1.6, label=lab)
          for k, lab in ((0, r"$k=0$"), (1, r"$k=1$"), ("mix", "mix"))]
    leg = ax.legend(handles=h1, fontsize=FS_LEG, frameon=False, loc="upper right", handlelength=1.6,
                    labelspacing=0.15, borderaxespad=0.25, title="frontier", title_fontsize=FS_LEG)
    ax.add_artist(leg)
    h2 = [Line2D([], [], color=c, lw=1.3, linestyle=(0, (4, 1.5)), label=lab) for _, lab, c in OPS]
    ax.legend(handles=h2, fontsize=FS_LEG, frameon=False, loc="lower left", handlelength=2.2,
              labelspacing=0.15, borderaxespad=0.25, title="Lorentzian filter of", title_fontsize=FS_LEG)
    print(f"(a) optimal one-site direction n = ({nx:+.4f}, {ny:+.4f}, {nz:+.4f}), nu_inf = {float(opt['sum_nu']):.5g}")


def panel_ed_ti(fig, ax, caxes, ed, ti, tauf, xlim, ylim, xticks, yticks):
    colsM, cmapM, normM = discrete("autumn", MS_TI, 0.0, 0.75)
    colsL, cmapL, normL = discrete("winter", LS_ED, 0.0, 1.0)
    for L, col in zip(LS_ED, colsL):
        t, n = ed(L)
        ax.plot(t, np.minimum.accumulate(n), "-", color=col, lw=1.1, zorder=2 + L / 100)
    for M, col in zip(MS_TI, colsM):
        t, n = ti(M)
        ax.plot(t, n, "--", color=col, lw=1.2, dashes=(4, 1.6), zorder=3 + M / 100)
    log_axes(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(yticks))
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\nu$", fontsize=FS_LABEL)
    handles = [Line2D([], [], color="0.35", lw=1.1, ls="-", label="finite ring (ED)"),
               Line2D([], [], color="0.35", lw=1.2, ls="--", dashes=(4, 1.6), label="TI ansatz")]
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="lower left", handlelength=2.0,
              labelspacing=0.2, borderaxespad=0.3)
    for cax, cmap, norm, vals, lab in ((caxes[0], cmapM, normM, MS_TI, r"$M$"),
                                       (caxes[1], cmapL, normL, LS_ED, r"$L$")):
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=vals[::2])
        cb.ax.tick_params(labelsize=FS_TICK - 1.5, length=1.5, pad=1)
        cb.outline.set_linewidth(0.5)
        cax.set_title(lab, fontsize=FS_LABEL - 1.5, pad=1.5)


def floquet_ed(L):
    _, t, n, _ = grid_curve(load_npz(os.path.join(FL, f"grid_L{L}_K0.npz")), "floquet")
    return tau_sorted(t, n)


def floquet_ti(M):
    _, c = read_ti_table(os.path.join(FL, f"nutau_floquet_M{M}_g0p9tF1p0_gpures.dat"))
    return tau_sorted(tau_floquet(c["sigma2"]), c["nu"])


def static_ed(L):
    _, t, n, _ = grid_curve(load_npz(os.path.join(KH, f"grid_L{L}_K0.npz")))
    return tau_sorted(t, n)


def static_ti(M):
    _, c = read_ti_table(os.path.join(KH, "ti_rtp_o4", f"nutau_rtp_M{M}_o4.dat"))
    ok = c["accepted"] > 0 if "accepted" in c else np.ones(len(c["nu"]), bool)
    return tau_sorted(tau_static(c["sigma2"][ok]), c["nu"][ok])


def main():
    fig = plt.figure(figsize=(3 * SLOT, FIG_H))
    fx, fy = (lambda x: x / (3 * SLOT)), (lambda y: y / FIG_H)
    bottom = 0.46
    axA = fig.add_axes([fx(0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    axB = fig.add_axes([fx(SLOT + 0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    caxB = (fig.add_axes([fx(SLOT + 1.8), fy(bottom + 0.8), fx(0.07), fy(0.5)]),
            fig.add_axes([fx(SLOT + 1.8), fy(bottom), fx(0.07), fy(0.5)]))
    axC = fig.add_axes([fx(2 * SLOT + 0.5), fy(bottom), fx(1.2), fy(PANEL_H)])
    caxC = (fig.add_axes([fx(2 * SLOT + 1.8), fy(bottom + 0.8), fx(0.07), fy(0.5)]),
            fig.add_axes([fx(2 * SLOT + 1.8), fy(bottom), fx(0.07), fy(0.5)]))
    panel_filters(axA)
    panel_ed_ti(fig, axB, caxB, floquet_ed, floquet_ti, tau_floquet, (0.8, 20.0), (0.1, 0.34),
                [1, 2, 5, 10, 20], [0.1, 0.15, 0.2, 0.3])
    panel_ed_ti(fig, axC, caxC, static_ed, static_ti, tau_static, (0.4, 20.0), (0.03, 0.34),
                [0.5, 1, 2, 5, 10, 20], [0.03, 0.05, 0.1, 0.2, 0.3])
    for ax, lab in ((axA, "(a)"), (axB, "(b)"), (axC, "(c)")):
        panel_label(ax, lab, FS_LABEL)
    W = layout_row(fig, [(axA, None), (axB, caxB), (axC, caxC)], bottom, PANEL_H)
    print(f"plot box {W:.3f} x {PANEL_H} in")
    print("wrote", save_figure(fig, "figure_G", png_dpi=300))


if __name__ == "__main__":
    main()
