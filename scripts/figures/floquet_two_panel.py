"""Kicked Ising (g=0.9, J=1, h=0.809) nu(tau): frontier vs explicit filters, and ED vs TI ansatz.

(a) L=12: K=0 frontier (black, identity-only projector "o1") and the base
    operators Sum Z_x (K=0), Z_1 (all momenta), Sum X_x (K=0) under a Gaussian
    filter (solid), a hard frequency cutoff (dash-dot) and the N-period time
    average ("sinc", dashed); Sum Z_x + Gaussian highlighted.  Filter points
    with tau >= 1e8 (As at machine epsilon) are dropped.
(b) ED frontier, rings L = 6..12 (solid, winter; running minimum of nu in
    tau) vs the translation-invariant ansatz at support cutoff M = 6..12
    (dashed, autumn; tau recomputed from its sigma^2 column).
All tau are the Floquet chord tau = 1/(2 arcsin(sqrt(As)/2)).

Data: data/floquet_two_panel/ (scripts/extract/nutau_floquet_two_panel.py).
Output: figures/floquet_two_panel.{pdf,png}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

from sso.config import data_dir
from sso.nutau.conventions import tau_floquet
from sso.nutau.io import load_npz
from sso.plotting.frontier import grid_curve, tau_sorted
from sso.plotting.style import decade_label, save_figure
from sso.tiansatz import read_ti_table

DATA = data_dir("floquet_two_panel")
TAU_MAX = 1e5
TAU_NOISE = 1e8
OPERATORS = [("Zsum", r"$\Sigma_x Z_x$", "0"), ("Z0", r"$Z_1$", "none"),
             ("Xsum", r"$\Sigma_x X_x$", "0")]
COLOUR = {"Zsum": "#e63946", "Z0": "#f4a261", "Xsum": "#1d3557"}
FILTERS = [("gauss", "Gaussian", "-"), ("hard", "hard", (0, (4.5, 1.5, 1.2, 1.5))),
           ("avg", "sinc", (0, (4, 2)))]
BEST = ("Zsum", "gauss")
BG_ALPHA = 0.6
TAU_MIN_B, TAU_MAX_B, NU_MIN_B, NU_MAX_B = 0.8, 20.0, 0.1, 0.34


def frontier(L):
    _, t, n, _ = grid_curve(load_npz(os.path.join(DATA, f"grid_L{L}_K0.npz")), "floquet")
    return tau_sorted(t, n)


def filter_curve(op, K, fam):
    if fam == "avg":
        z = load_npz(os.path.join(DATA, f"ftraj_L12_Knone_{op}.npz"))
        return tau_sorted(tau_floquet(z["As_avg"]), z["nu_avg"])
    tag = "gausscut" if fam == "gauss" else "hardcut"
    z = load_npz(os.path.join(DATA, f"{tag}_L12_K{K}_{op}.npz"))
    tau, nu = tau_floquet(z["As"]), z["nu"]
    good = np.isfinite(tau) & np.isfinite(nu) & (tau < TAU_NOISE)
    return tau_sorted(tau[good], nu[good])


def ti_curve(M):
    _, c = read_ti_table(os.path.join(DATA, f"nutau_floquet_M{M}_g0p9tF1p0_gpures.dat"))
    return tau_sorted(tau_floquet(c["sigma2"]), c["nu"])


def draw_a(fig, ax, x_leg, y_top, lab_fs, tick_fs, leg_fs):
    bt, bn = frontier(12)
    data = {(op, fam): filter_curve(op, K, fam) for op, _, K in OPERATORS for fam, _, _ in FILTERS}
    y_all, tau_lo = [], bt.min()
    for op, _, _ in OPERATORS:
        for fam, _, ls in FILTERS:
            if (op, fam) == BEST:
                continue
            tau, nu = data[(op, fam)]
            k = tau <= TAU_MAX
            ax.plot(tau[k], nu[k], linestyle=ls, color=COLOUR[op], linewidth=0.85,
                    alpha=BG_ALPHA, zorder=2)
            y_all.append(nu[k])
            tau_lo = min(tau_lo, tau.min())
    m = bt <= TAU_MAX
    ax.plot(bt[m], bn[m], "-", color="black", linewidth=1.8, zorder=5)
    y_all.append(bn[m])
    tau, nu = data[BEST]
    k = tau <= TAU_MAX
    ax.plot(tau[k], nu[k], "-", color=COLOUR[BEST[0]], linewidth=1.8, zorder=6)

    y = np.concatenate(y_all)
    y = y[np.isfinite(y) & (y > 0)]
    ax.set_ylim(y.min() * 0.80, y.max() * 1.18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\nu$", fontsize=lab_fs)
    ax.set_xlabel(r"$\tau$", fontsize=lab_fs)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=15))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xlim(tau_lo * 0.75, TAU_MAX * 1.3)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10)), numticks=15))
    ax.xaxis.set_minor_formatter(NullFormatter())
    decades = [10.0 ** e for e in range(int(np.floor(np.log10(tau_lo))),
                                        int(np.ceil(np.log10(TAU_MAX))) + 1)]
    decades = [t for t in decades if tau_lo * 0.75 <= t <= TAU_MAX]
    ax.set_xticks(decades)
    ax.set_xticklabels([decade_label(t) for t in decades], fontsize=tick_fs)
    ax.legend(handles=[Line2D([], [], color="black", linewidth=1.8, label="frontier")],
              loc="upper right", fontsize=leg_fs, frameon=False, handlelength=2.0,
              handletextpad=0.5, borderaxespad=0.5)

    fig_w, fig_h = fig.get_size_inches()
    box = dict(fontsize=leg_fs, title_fontsize=leg_fs + 0.5, frameon=True, fancybox=False,
               edgecolor="0.6", framealpha=1.0, borderpad=0.45, labelspacing=0.3,
               handletextpad=0.5, borderaxespad=0.0)
    h1 = [Line2D([], [], color="0.35", linestyle=ls, linewidth=1.0, label=lab)
          for _, lab, ls in FILTERS]
    h2 = [Line2D([], [], color=COLOUR[op], linewidth=2.4 if op == BEST[0] else 1.6, label=lab)
          for op, lab, _ in OPERATORS]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def size_of(leg):
        bb = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        return bb.width, bb.height

    widths = []
    for kw in (dict(handles=h1, title="filter", handlelength=2.6),
               dict(handles=h2, title="base operator", handlelength=1.6)):
        probe = fig.legend(loc="upper left", bbox_to_anchor=(x_leg, y_top), **kw, **box)
        widths.append(size_of(probe)[0])
        probe.remove()
    w = max(widths)
    leg1 = fig.legend(handles=h1, title="filter", loc="upper left",
                      bbox_to_anchor=(x_leg, y_top - 1.0, w, 1.0), mode="expand",
                      handlelength=2.6, **box)
    leg1._legend_box.align = "left"
    _, h1_fig = size_of(leg1)
    leg2 = fig.legend(handles=h2, title="base operator", loc="upper left",
                      bbox_to_anchor=(x_leg, y_top - h1_fig - 0.06 / fig_h - 1.0, w, 1.0),
                      mode="expand", handlelength=1.6, **box)
    leg2._legend_box.align = "left"


def discrete_cmap(name, values, lo, hi):
    base = plt.get_cmap(name)
    cols = [base(p) for p in np.linspace(hi, lo, len(values))]
    edges = np.concatenate([[values[0] - 0.5], (np.array(values[:-1]) + np.array(values[1:])) / 2,
                            [values[-1] + 0.5]])
    return cols, ListedColormap(cols), BoundaryNorm(edges, len(values))


def draw_b(fig, ax, cbar_x, cbar_y, cbar_w, cbar_h, lab_fs, tick_fs, leg_fs):
    Ms, Ls = list(range(6, 13)), list(range(6, 13))
    colsM, cmapM, normM = discrete_cmap("autumn", Ms, 0.0, 0.75)
    colsL, cmapL, normL = discrete_cmap("winter", Ls, 0.0, 1.0)
    for L, col in zip(Ls, colsL):
        t, n = frontier(L)
        ax.plot(t, np.minimum.accumulate(n), "-", color=col, lw=1.2, zorder=2 + L / 100)
    for M, col in zip(Ms, colsM):
        ax.plot(*ti_curve(M), "--", color=col, lw=1.3, dashes=(4, 1.6), zorder=3 + M / 100)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(TAU_MIN_B, TAU_MAX_B)
    ax.set_ylim(NU_MIN_B, NU_MAX_B)
    ax.set_xlabel(r"$\tau$", fontsize=lab_fs)
    ax.set_ylabel(r"$\nu$", fontsize=lab_fs)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=15))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0), numticks=15))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=tick_fs)
    handles = [Line2D([], [], color="0.35", lw=1.2, ls="-", label="Finite Ring"),
               Line2D([], [], color="0.35", lw=1.3, ls="--", dashes=(4, 1.6), label="TI Ansatz")]
    ax.legend(handles=handles, fontsize=leg_fs, frameon=False, loc="lower left",
              handlelength=2.0, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    fig_h = fig.get_size_inches()[1]
    gap = 0.20 / fig_h
    h_each = (cbar_h - gap) / 2
    for y0, cmap, norm, vals, lab in ((cbar_y + h_each + gap, cmapM, normM, Ms, r"$M$"),
                                      (cbar_y, cmapL, normL, Ls, r"$L$")):
        cax = fig.add_axes([cbar_x, y0, cbar_w, h_each])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=vals)
        cb.ax.tick_params(labelsize=tick_fs - 1.5, length=2, pad=1.5)
        cb.outline.set_linewidth(0.6)
        cb.ax.set_title(lab, fontsize=lab_fs - 2, pad=2)


def main():
    lab_fs, tick_fs, leg_fs = 11, 9, 7.5
    fig_w, fig_h = 7.0, 2.75
    ax_b, ax_h = 0.46, 2.06
    a_l, a_w = 0.72, 1.95
    b_l, b_w = 4.46, 2.08
    fig = plt.figure(figsize=(fig_w, fig_h))
    fx, fy = (lambda x: x / fig_w), (lambda y: y / fig_h)
    ax_a = fig.add_axes([fx(a_l), fy(ax_b), fx(a_w), fy(ax_h)])
    ax_b_ = fig.add_axes([fx(b_l), fy(ax_b), fx(b_w), fy(ax_h)])
    draw_a(fig, ax_a, x_leg=fx(a_l + a_w + 0.10), y_top=fy(ax_b + ax_h),
           lab_fs=lab_fs, tick_fs=tick_fs, leg_fs=leg_fs)
    draw_b(fig, ax_b_, cbar_x=fx(b_l + b_w + 0.10), cbar_y=fy(ax_b), cbar_w=fx(0.11),
           cbar_h=fy(ax_h - 0.18), lab_fs=lab_fs, tick_fs=tick_fs, leg_fs=leg_fs)
    for ax, lab in ((ax_a, "(a)"), (ax_b_, "(b)")):
        pos = ax.get_position()
        fig.text(pos.x0 - fx(0.45), pos.y1 + fy(0.04), lab, fontsize=lab_fs,
                 va="bottom", ha="left")
    print("wrote", save_figure(fig, "floquet_two_panel"))


if __name__ == "__main__":
    main()
