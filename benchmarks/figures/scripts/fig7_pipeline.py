#!/usr/bin/env python
"""Fig. 7 (overview): the verified LAMMPS + ML-IAP + three-pass MACE pipeline (paper/PIPELINE_verified.md), with the
modules changed in this work in green and the measured gains as badges.  Vector PDF via matplotlib.
Numbers: PROGRESS.md (jobs 01/72/83/84 and the limit scans).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GREEN = dict(fc="#EAF3DE", ec="#3B6D11", tc="#1F4A08", sc="#3B6D11")
GRAY = dict(fc="#F3F2ED", ec="#9A9890", tc="#3C3C39", sc="#6B6A64")
BADGE = dict(fc="#3B6D11", tc="white")
CONT = "#B4B2A9"
LINE = "#5F5E5A"
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5})


def box(ax, x, y, w, h, title, sub=None, style=GRAY, r=0.12, fs=7.5, ss=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=style["fc"], ec=style["ec"], lw=0.8))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=style["tc"], weight="bold")
    else:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=fs, color=style["tc"], weight="bold")
        ax.text(x + w / 2, y + h * 0.31, sub, ha="center", va="center", fontsize=ss, color=style["sc"], linespacing=1.15)


def badge(ax, cx, y, text, fs=6.3):
    ax.text(cx, y, text, ha="center", va="center", fontsize=fs, color=BADGE["tc"], weight="bold", linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.9", fc=BADGE["fc"], ec="none"))


def arrow(ax, p, q, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=LINE, shrinkA=0, shrinkB=0))


def line(ax, p, q, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-", lw=lw, color=LINE, shrinkA=0, shrinkB=0))


def container(ax, x, y, w, h, label=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2", fc="none", ec=CONT, lw=0.8, ls=(0, (4, 3))))
    if label:
        ax.text(x + 0.18, y + h - 0.13, label, ha="left", va="top", fontsize=6.8, color="#6B6A64", style="italic")


def panel_title(ax, y, letter, text):
    ax.text(0.3, y, letter, fontsize=10, weight="bold", va="top")
    ax.text(0.75, y, text, fontsize=8.5, va="top")


def main():
    fig = plt.figure(figsize=(7.2, 6.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 17.3)
    ax.axis("off")
    X0, X1 = 0.4, 17.6

    # ================================================================ a: one MD step
    panel_title(ax, 17.15, "a", "One MD step in LAMMPS (Kokkos; every array stays on the GPU)")
    ya, h = 14.35, 1.1
    ym = ya + h / 2
    box(ax, X0, ya, 2.45, h, "Initial integrate", "half step; NPT box")
    box(ax, 3.05, ya, 2.75, h, "Ghosts, neighbours", "forward comm, or rebuild\nevery ~16 steps")
    px, pw = 6.3, 8.55
    container(ax, px, ya - 0.3, pw, h + 0.85, "pair compute (ML-IAP)")
    box(ax, px + 0.25, ya, 2.35, h, "Pair data", "filter pairs by r$_\\mathregular{max}$", GREEN)
    box(ax, px + 2.85, ya, 2.5, h, "Force call", "Python, panel b", GREEN)
    box(ax, px + 5.6, ya, 2.7, h, "Assemble", "f$_{ij}$ → f$_i$, virial, E$_i$")
    box(ax, 15.15, ya, 2.45, h, "Reverse comm,\nfinal integrate", "fixes, thermo, dump", fs=7.2)
    arrow(ax, (X0 + 2.45, ym), (3.05, ym))
    arrow(ax, (5.8, ym), (px + 0.25, ym))
    arrow(ax, (px + 2.6, ym), (px + 2.85, ym))
    arrow(ax, (px + 5.35, ym), (px + 5.6, ym))
    arrow(ax, (px + 8.3, ym), (15.15, ym))
    yrail = ya - 0.75
    line(ax, (16.375, ya), (16.375, yrail)); line(ax, (16.375, yrail), (1.625, yrail)); arrow(ax, (1.625, yrail), (1.625, ya))
    ax.text(9.0, yrail - 0.12, "next step", ha="center", va="top", fontsize=6.3, color=GRAY["sc"])
    badge(ax, px + 1.45, ya + h + 0.85, "no host copies, arrays grow with headroom\nML-IAP overhead 0.25 → 0.002 s per step")
    badge(ax, px + 6.6, ya + h + 0.85, "MD step 0.43 → 0.28 s\n(10$^6$ water atoms, r$_\\mathregular{max}$ 4.5 Å student)")

    # ================================================================ b: the force call
    panel_title(ax, 12.15, "b", "Force call: exact three-pass evaluation of the two-layer MACE model (this work)")
    yb, hb = 9.35, 1.4
    labels = [("Prepare", "sort pairs by receiver;\nchunks; int32 guard"),
              ("Pass A", "layer 1 forward,\nno autograd → H$^{(1)}$"),
              ("Exchange H$^{(1)}$", "forward comm\nto ghost atoms"),
              ("Pass B", "layer 2 + backward\n→ f$_{ij}$ and G = ∂E/∂H$^{(1)}$"),
              ("Exchange G", "reverse comm\nghosts → owners"),
              ("Pass C", "layer 1 recomputed,\nbackward with G → f$_{ij}$"),
              ("Return", "f$_{ij}$, E$_i$; force guard;\nGPU atomic add")]
    n, gap = len(labels), 0.28
    bw = (X1 - X0 - (n - 1) * gap) / n
    for k, (t, s) in enumerate(labels):
        x = X0 + k * (bw + gap)
        box(ax, x, yb, bw, hb, t, s, GREEN, ss=6.1)
        if k:
            arrow(ax, (x - gap, yb + hb / 2), (x, yb + hb / 2))
    yc0 = yb - 1.45
    container(ax, X0, yc0, X1 - X0, 1.2)
    ax.text(X0 + 0.22, yc0 + 0.85, "resident, all atoms:", fontsize=6.5, color="#3C3C39", va="center", weight="bold")
    ax.text(X0 + 3.0, yc0 + 0.85, "positions (fp64) · int32 pair list · H$^{(1)}$ (512 per atom) · G · pair forces",
            fontsize=6.5, color="#3C3C39", va="center")
    ax.text(X0 + 0.22, yc0 + 0.35, "one chunk at a time:", fontsize=6.5, color=GREEN["tc"], va="center", weight="bold")
    ax.text(X0 + 3.0, yc0 + 0.35, "edge messages · node features · autograd graph  (freed after each chunk; padded to fixed shapes for CUDA graphs)",
            fontsize=6.5, color=GREEN["tc"], va="center")
    yB = yb + hb + 0.62
    badge(ax, 3.2, yB, "memory 1.55 MB → 4.6 kB per atom (×340 less)")
    badge(ax, 9.0, yB, "28.6 M atoms on one H200 (exact MACE-MPA-0)")
    badge(ax, 14.7, yB, "kernels ×2.5: species-block skip, tabulated radial,\nCUDA graphs, torch.compile", 6.0)

    # ================================================================ c: outer loop
    panel_title(ax, 6.85, "c", "Outer loop: the exact model supervises a distilled student on the fly (this work)")
    yc, hc = 3.55, 1.35
    cl = [("Build student", "three-pass .pt from\ncurrent weights"),
          ("MD segment", "student drives\n100 ps NPT"),
          ("Rerun checks", "teacher and student\nrecompute dumped frames"),
          ("Pass?", "force RMSE, worst atoms,\nvirial pressure, safety stops"),
          ("Restart", "next segment from\nsegment_end.restart")]
    cgap = 0.42
    cw = (X1 - X0 - 4 * cgap) / 5
    xc = [X0 + k * (cw + cgap) for k in range(5)]
    for k, (t, s) in enumerate(cl):
        box(ax, xc[k], yc, cw, hc, t, s, GREEN, ss=6.1)
        if k:
            arrow(ax, (xc[k] - cgap, yc + hc / 2), (xc[k], yc + hc / 2))
    ax.text(xc[4] - cgap / 2, yc + hc / 2 + 0.2, "yes", ha="center", va="bottom", fontsize=6.2, color=GRAY["sc"])
    # fail branch
    rx, rw, ry = xc[3] - 0.25, cw + 0.5, yc - 1.95
    box(ax, rx, ry, rw, hc, "Carve, label, fine-tune", "7 Å clusters around the worst atoms,\nbonds completed; teacher labels", GREEN, ss=6.1)
    arrow(ax, (xc[3] + cw / 2, yc), (xc[3] + cw / 2, ry + hc))
    ax.text(xc[3] + cw / 2 + 0.12, yc - 0.3, "no", ha="left", va="center", fontsize=6.2, color=GRAY["sc"])
    ylo = ry - 0.3
    line(ax, (xc[3] + cw / 2, ry), (xc[3] + cw / 2, ylo)); line(ax, (xc[3] + cw / 2, ylo), (xc[0] + cw / 2, ylo)); arrow(ax, (xc[0] + cw / 2, ylo), (xc[0] + cw / 2, yc))
    yhi = yc + hc + 0.28
    line(ax, (xc[4] + cw / 2, yc + hc), (xc[4] + cw / 2, yhi)); line(ax, (xc[4] + cw / 2, yhi), (xc[1] + cw / 2, yhi)); arrow(ax, (xc[1] + cw / 2, yhi), (xc[1] + cw / 2, yc + hc))
    yC = yhi + 0.62
    badge(ax, 3.2, yC, "student ×18 faster than the exact model\n(1/4.9 of NEP89 on the same GPU)")
    badge(ax, 9.0, yC, "checks ≈ 1.5% of the run time")
    badge(ax, 14.7, yC, "production 4.3 M atom-steps/s at 10$^6$ atoms\n(stock model: out of memory above 86 k atoms)", 6.0)

    # ================================================================ legend
    yl = 0.42
    ax.add_patch(FancyBboxPatch((X0, yl - 0.18), 0.45, 0.36, boxstyle="round,pad=0,rounding_size=0.08", fc=GREEN["fc"], ec=GREEN["ec"], lw=0.8))
    ax.text(X0 + 0.6, yl, "changed or added in this work", va="center", fontsize=6.8, color="#3C3C39")
    ax.add_patch(FancyBboxPatch((5.4, yl - 0.18), 0.45, 0.36, boxstyle="round,pad=0,rounding_size=0.08", fc=GRAY["fc"], ec=GRAY["ec"], lw=0.8))
    ax.text(6.0, yl, "unchanged LAMMPS / MACE code", va="center", fontsize=6.8, color="#3C3C39")
    badge(ax, 14.7, yl, "badges: measured on one NVIDIA H200, water, 300 K", 6.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig7_pipeline.{ext}"), dpi=300)
    print("wrote fig7_pipeline.pdf/png")


if __name__ == "__main__":
    main()
