#!/usr/bin/env python
"""Fig. 1: why MACE runs out of memory and how the three-pass evaluation removes it.
 (a) GPU memory per atom (water, MACE-MPA-0, r_max 6 A): stock evaluation 1.55 MB/atom (job01 slope), exact three-pass
     4.57 kB/atom resident + transient per chunk (LIMIT_RESULTS), NEP89 7.4 KiB/atom (measured)
 (b) schematic of the receiver-chunked three-pass evaluation (passes A, B, C; resident vs per-chunk arrays)
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C_RES, C_CH, C_PASS = "#1f4e79", "#aed6f1", "#fdebd0"


def box(ax, x, y, w, h, text, fc, ec="k", fs=6.5, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, c="k", cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=c, connectionstyle=cs))


def main():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(7.2, 2.9))
    a = fig.add_axes([0.07, 0.17, 0.25, 0.72])
    lab = ["MACE-MPA-0\nstock", "MACE-MPA-0\nthree-pass\n(resident)", "NEP89"]
    val = [1.55e6, 4.57e3, 7.37 * 1024]
    a.bar(range(3), val, color=["#7f8c8d", C_RES, "#e67e22"], width=0.6)
    a.set_yscale("log")
    a.set_ylim(1e3, 1e7)
    a.set_xticks(range(3))
    a.set_xticklabels(lab, fontsize=6.3)
    a.set_ylabel("GPU memory per atom (bytes)")
    for i, (v, t) in enumerate(zip(val, ["1.55 MB", "4.6 kB", "7.5 kB"])):
        a.text(i, v * 1.4, t, ha="center", fontsize=6.5)
    a.set_title("a  memory per atom (water)", loc="left", fontsize=8)

    b = fig.add_axes([0.37, 0.02, 0.62, 0.92])
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    b.axis("off")
    b.text(0.0, 0.97, "b  receiver-chunked three-pass evaluation (exact)", fontsize=8, va="top")
    b.text(0.01, 0.86, "resident, all atoms:", fontsize=6.5, style="italic")
    resident = [("positions $\\mathbf{r}$\n(fp64)", 0.01), ("neighbour list\n(int32)", 0.205),
                ("$H^{(1)}$\n(512 per atom)", 0.40), ("$G=\\partial E/\\partial H^{(1)}$", 0.595),
                ("forces $\\mathbf{F}$", 0.79)]
    for t, x in resident:
        box(b, x, 0.72, 0.18, 0.11, t, "white", ec=C_RES, lw=1.0, fs=5.8)
    ys = [0.47, 0.26, 0.05]
    texts = ["A  layer 1, forward only\n$H^{(1)}[\\mathcal{C}] \\leftarrow \\Phi_1(\\mathcal{E}(\\mathcal{C}))$",
             "B  layer 2 + backward\n$\\partial e_2/\\partial \\mathbf{r}_{ij}\\ \\rightarrow\\ \\mathbf{F}$;"
             "  $\\partial e_2/\\partial H^{(1)}\\ \\rightarrow\\ G$",
             "C  recompute layer 1 + backward\nupstream $\\partial\\varepsilon^{(1)}/\\partial h+G\\ \\rightarrow\\ \\mathbf{F}$"]
    for y, t in zip(ys, texts):
        box(b, 0.01, y, 0.44, 0.15, t, C_PASS, fs=6.3)
        box(b, 0.52, y + 0.005, 0.30, 0.14, "per chunk $\\mathcal{C}$ only:\nedge messages, node features,\nautograd graph",
            C_CH, fs=5.6)
        arrow(b, (0.45, y + 0.075), (0.52, y + 0.075))
        b.text(0.835, y + 0.075, "freed after\neach chunk", fontsize=5.6, va="center")
    arrow(b, (0.23, 0.47), (0.23, 0.41))
    arrow(b, (0.23, 0.26), (0.23, 0.20))
    arrow(b, (0.49, 0.72), (0.2, 0.62), c=C_RES, cs="arc3,rad=0.1")
    b.text(0.36, 0.64, "reads / accumulates", fontsize=5.6, color=C_RES)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_concept.{ext}"), dpi=300)
    print("wrote fig1_concept")


if __name__ == "__main__":
    main()
