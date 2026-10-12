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


def layout_row(fig, panels, bottom, height, margin=0.04, gap=0.22, cbar_gap=0.08,
               cbar_w=0.07, iterations=3, stack_gap=0.22):
    """Place a row of panels with equal plot width, filling the figure width.

    ``panels`` is a list of (ax, cax-or-None); cax is a colourbar axis kept ``cbar_gap``
    to the right of its plot, or a tuple of colourbar axes stacked top to bottom over the
    plot height (gap ``stack_gap``).  The decorations of each panel (tick and axis labels,
    secondary axes, colourbar and its labels, panel label) are measured, and the common
    plot width W solves  sum(left_i + W + right_i) + (n-1) gap + 2 margin = figure width.
    All sizes in inches.  Returns W.
    """
    fig_w, fig_h = fig.get_size_inches()
    n = len(panels)
    W = None
    for _ in range(iterations):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ext = []
        for ax, cax in panels:
            caxes = [] if cax is None else (list(cax) if isinstance(cax, (tuple, list)) else [cax])
            boxes = [ax.get_tightbbox(r)] + [c.get_tightbbox(r) for c in caxes]
            x0 = min(b.x0 for b in boxes) / fig.dpi
            x1 = max(b.x1 for b in boxes) / fig.dpi
            p = ax.get_position()
            ext.append((p.x0 * fig_w - x0, x1 - p.x1 * fig_w))
        W = (fig_w - 2 * margin - (n - 1) * gap - sum(l + rr for l, rr in ext)) / n
        x = margin
        for (ax, cax), (le, re) in zip(panels, ext):
            x += le
            ax.set_position([x / fig_w, bottom / fig_h, W / fig_w, height / fig_h])
            if cax is not None:
                caxes = list(cax) if isinstance(cax, (tuple, list)) else [cax]
                h = (height - (len(caxes) - 1) * stack_gap) / len(caxes)
                for j, c in enumerate(caxes):
                    y0 = bottom + height - (j + 1) * h - j * stack_gap
                    c.set_position([(x + W + cbar_gap) / fig_w, y0 / fig_h, cbar_w / fig_w, h / fig_h])
            x += W + re + gap
    return W


def panel_label(ax, label, fontsize=8):
    """'(a)'-style label a fixed distance (in points) left of the plot's top-left corner."""
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-28, 3),
                textcoords="offset points", fontsize=fontsize, ha="left", va="bottom")
