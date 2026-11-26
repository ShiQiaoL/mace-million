"""Shared style of all figures of the paper (Nature Communications: 180 mm double column, sans-serif 5-7 pt, bold
lower-case panel letters, vector output).  One colour per entity in every figure:

    exact   MACE-MPA-0, exact three-pass evaluation (teacher)        navy
    student distilled MACE student (this work, focal series)         green
    stock   MACE-MPA-0, stock LAMMPS ML-IAP evaluation               grey
    nep     NEP89 in GPUMD                                           orange
    classic classical force fields                                   light brown-grey
    dft     DFT reference                                            black
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
W2 = 180 * MM            # double-column width
W1 = 88 * MM             # single-column width
C = dict(exact="#1f4e79", student="#2e8b57", student2="#8fcfa9", stock="#7f7f7f", nep="#e08a1e", classic="#a89f91",
         dft="#000000", line="#5f5e5a", ref="#b0b0b0")
BASE, SECOND, TICK, LETTER = 7, 6, 6, 8
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..")


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": BASE, "legend.fontsize": SECOND,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "axes.titlepad": 4, "axes.labelpad": 2,
        "mathtext.default": "regular", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })


def letter(ax, s, dx=-0.0, dy=1.0, fig=None, x_mm=None):
    """Bold lower-case panel letter at the top-left corner of the panel (outside the axes box)."""
    f = fig or ax.figure
    bb = ax.get_position()
    left = ax.get_tightbbox(f.canvas.get_renderer()).transformed(f.transFigure.inverted()).x0 if x_mm is None \
        else bb.x0 - x_mm * MM / f.get_figwidth()
    f.text(left, bb.y1 + 2.2 * MM / f.get_figheight(), s, fontsize=LETTER, fontweight="bold", va="bottom", ha="left")


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=BASE, fontweight="normal")


def check(fig, name):
    """Geometric check: overlapping text boxes and text outside the figure are reported."""
    r = fig.canvas.get_renderer()
    ticks = set()
    for ax in fig.axes:                     # tick labels beyond the axis limits exist but are not drawn
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo <= t.get_loc() <= hi):
                    ticks.update((t.label1, t.label2))
    ts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
          if t.get_text().strip() and t.get_visible() and t not in ticks]
    bad = [(a.get_text()[:25], b.get_text()[:25]) for i, (a, ba) in enumerate(ts) for b, bb in ts[i + 1:] if ba.overlaps(bb)]
    out = [t.get_text()[:25] for t, b in ts if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1 + 1 or b.y1 > fig.bbox.y1 + 1]
    msg = f"{name}: {len(bad)} text overlaps {bad[:6]}; {len(out)} outside {out[:6]}"
    print(msg.encode("ascii", "replace").decode())


def save(fig, name):
    fig.canvas.draw()
    check(fig, name)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    print("wrote", name)
