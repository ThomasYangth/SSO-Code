"""Figure output and small axis helpers shared by the figure scripts."""

import numpy as np

from sso.config import figure_path


def save_figure(fig, name, png_dpi=300, pdf_dpi=None):
    """Write figures/<name>.pdf and figures/<name>.png; returns the pdf path."""
    pdf = figure_path(f"{name}.pdf")
    fig.savefig(pdf, **({} if pdf_dpi is None else dict(dpi=pdf_dpi)))
    fig.savefig(figure_path(f"{name}.png"), dpi=png_dpi)
    return pdf


def decade_label(t):
    """'1', '10' or '$10^{e}$' for a power of ten."""
    e = int(round(np.log10(t)))
    return {0: "1", 1: "10"}.get(e, rf"$10^{{{e}}}$")
