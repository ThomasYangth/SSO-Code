"""DMRG attempt (unsuccessful approach, not a paper figure): ν vs τ/L² for
L = 12, 18, 24, 30 and bond dimensions χ = 128, 256, 512.

Data: data/attempt_dmrg/dmrg_points.csv (built by
scripts/extract/dmrg_points.py) — ε = ⟨C_H²⟩ and ν = ⟨M⟩ of the converged
ε-ladder MPS; τ = 1/√ε (static convention). Only rows with plotted=1 are
drawn (frozen χ-wall tail steps are omitted, they sit on top of the first wall
point). Hollow markers: completed runs whose MPS was not saved (value read
from the log). Dotted: the exact L = 12 front (data/attempt_dmrg/ed_front_L12.csv)
rescaled by 12².

Writes figures/attempt_dmrg_multiL_tauOverL2.png (and .pdf).
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sso import config  # noqa: E402

LS = (12, 18, 24, 30)
CHIS = (128, 256, 512)
CMAP = {12: plt.cm.Blues, 18: plt.cm.Greens, 24: plt.cm.Oranges, 30: plt.cm.Purples}
SHADE = {128: 0.5, 256: 0.72, 512: 0.95}
FRONT = "#8a6a1e"


def load(name):
    with open(config.data_dir("attempt_dmrg", name)) as f:
        return list(csv.DictReader(f))


def main():
    pts = [r for r in load("dmrg_points.csv") if r["plotted"] == "1"]
    front = sorted((float(r["eps"]), float(r["nu"])) for r in load("ed_front_L12.csv"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.edgecolor": "#4a4f56", "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.grid(True, which="both", color="#e6e3dd", lw=0.7, zorder=0)
    ax.plot([1 / np.sqrt(e) / 12**2 for e, _ in front], [m for _, m in front],
            ls=":", lw=2.2, color=FRONT, zorder=2)

    for L in LS:
        for chi in CHIS:
            s = sorted((float(r["eps"]), float(r["nu"]), r["saved"] == "1")
                       for r in pts if int(r["L"]) == L and int(r["chi"]) == chi)
            if not s:
                continue
            c = CMAP[L](SHADE[chi])
            x = [1 / np.sqrt(e) / L**2 for e, _, _ in s]
            y = [m for _, m, _ in s]
            ms = 5.5 if len(s) > 2 else 8
            z = 4 + chi // 128
            ax.plot(x, y, "-", color=c, lw=1.8, zorder=z)
            for xi, yi, (_, _, saved) in zip(x, y, s):
                ax.plot(xi, yi, "o", ms=ms, zorder=z + 0.5,
                        mfc=c if saved else "white", mec="white" if saved else c,
                        mew=1.0 if saved else 1.8)

    ax.set_xscale("log")
    ax.set_xlim(0.0018, 0.28)
    ax.set_ylim(0.195, 0.315)
    ax.xaxis.set_major_locator(FixedLocator([0.002, 0.004, 0.008, 0.02, 0.05, 0.1, 0.2]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.set_xlabel(r"$\tau / L^2$,   $\tau = 1/\sqrt{\langle C_H^2\rangle}$")
    ax.set_ylabel(r"$\nu = \langle M\rangle$   (Pauli weight)")
    ax.set_title(r"DMRG attempt: $\nu$ vs $\tau/L^2$,  $L=12,18,24,30$ $\times$ $\chi=128,256,512$",
                 fontsize=12.5, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    hL = [Line2D([0], [0], color=CMAP[L](0.75), lw=2.6, marker="o", ms=6, mec="white") for L in LS]
    lL = ["L = %d" % L for L in LS]
    hL.append(Line2D([0], [0], ls=":", lw=2.2, color=FRONT))
    lL.append(r"$L=12$ exact front")
    leg1 = ax.legend(hL, lL, frameon=False, fontsize=11, loc="lower left",
                     title="colour = size", title_fontsize=10)
    ax.add_artist(leg1)
    hC = [Line2D([0], [0], color=plt.cm.Greys(SHADE[c]), lw=2.6, marker="o", ms=6, mec="white")
          for c in CHIS]
    lC = ["χ = %d" % c for c in CHIS]
    hC.append(Line2D([0], [0], ls="none", marker="o", ms=6, mfc="white", mec="#555", mew=1.8))
    lC.append("MPS not saved (log value)")
    ax.legend(hC, lC, frameon=False, fontsize=11, loc="upper right",
              title="brightness = χ", title_fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = config.figure_path("attempt_dmrg_multiL_tauOverL2." + ext)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()
