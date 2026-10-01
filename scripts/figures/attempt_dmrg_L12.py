"""DMRG attempt (unsuccessful approach, not a paper figure): the L = 12
ν(τ) front from the ε-constrained DMRG at χ = 128, 256, 512 against the exact
(ED) optimum.

Model: Ising H = Σ ZZ + 0.905 Σ X + 0.809 Σ Z, OBC. τ = 1/√⟨C_H²⟩ (static
convention), ν = ⟨M⟩. Data: data/attempt_dmrg/dmrg_points.csv (rows L=12,
plotted=1) and data/attempt_dmrg/ed_front_L12.csv. χ = 128 is one consistent
ladder (anneal=0, job 13325137_0, steps 1–10); the χ = 256 / 512 points are
one-step runs warm-started from a χ = 128 step-8 state (see
docs/dmrg_attempt.md). Hollow markers: completed runs whose MPS was not
saved (value from the log).

Writes figures/attempt_dmrg_L12.png (and .pdf).
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sso import config  # noqa: E402

COL = {128: "#9cc0e6", 256: "#3d7dbf", 512: "#123a63", "front": "#a9761a"}


def load(name):
    with open(config.data_dir("attempt_dmrg", name)) as f:
        return list(csv.DictReader(f))


def main():
    pts = [r for r in load("dmrg_points.csv") if r["plotted"] == "1" and r["L"] == "12"]
    front = sorted((float(r["eps"]), float(r["nu"])) for r in load("ed_front_L12.csv"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.edgecolor": "#4a4f56", "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.grid(True, which="major", color="#e2ded6", lw=0.8, zorder=0)
    ax.plot([1 / np.sqrt(e) for e, _ in front], [m for _, m in front], ls=":", lw=2.4,
            color=COL["front"], zorder=3, label="exact front (ED optimum)")
    handles = [Line2D([0], [0], ls=":", lw=2.4, color=COL["front"])]
    labels = ["exact front (ED optimum)"]
    for chi in (128, 256, 512):
        s = sorted((float(r["eps"]), float(r["nu"]), r["saved"] == "1")
                   for r in pts if int(r["chi"]) == chi)
        x = [1 / np.sqrt(e) for e, _, _ in s]
        y = [m for _, m, _ in s]
        ax.plot(x, y, "-", color=COL[chi], lw=2, zorder=5)
        for xi, yi, (_, _, saved) in zip(x, y, s):
            ax.plot(xi, yi, "o", ms=6.5, zorder=5.5, mfc=COL[chi] if saved else "white",
                    mec="white" if saved else COL[chi], mew=1.2 if saved else 2.0)
        handles.append(Line2D([0], [0], color=COL[chi], lw=2, marker="o", ms=6.5, mec="white", mew=1.2))
        labels.append(r"$\chi=%d$" % chi)
    handles.append(Line2D([0], [0], ls="none", marker="o", ms=6.5, mfc="white", mec="#555", mew=2.0))
    labels.append("MPS not saved (log value)")

    ax.set_xlim(2, 26)
    ax.set_ylim(0.20, 0.31)
    ax.set_xlabel(r"$\tau = 1/\sqrt{\langle C_H^2\rangle}$   (longer timescale $\rightarrow$)")
    ax.set_ylabel(r"$\nu = \langle M\rangle$   (Pauli weight)")
    ax.set_title(r"DMRG attempt: $\nu(\tau)$ front, $L=12$   (Ising $J{=}1,\,h_x{=}0.905,\,h_z{=}0.809$, OBC)",
                 fontsize=12.5, pad=12)
    ax.legend(handles, labels, frameon=False, fontsize=11.5, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = config.figure_path("attempt_dmrg_L12." + ext)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()
