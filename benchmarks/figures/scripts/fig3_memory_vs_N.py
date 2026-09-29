#!/usr/bin/env python
"""Fig. 3: peak GPU memory vs system size on one 141-GB H200 (water).  Sources:
  stock MACE-MPA-0 through LAMMPS ML-IAP, fp32 (cuEquivariance fused) and fp64 .... job01 (results/REPORT_TABLES.md, torch peak allocated)
  exact three-pass MACE-MPA-0, standalone, fp32 chunk 16384/8192 and fp64 chunk 4096 ... LIMIT_RESULTS_three-pass-MACE.md (job10-12)
  NEP89 (GPUMD) ........................................................................ 7.37 KiB/atom measured (2026-09-21)
  MACE student V5 in LAMMPS (torch peak allocated) ...................................... job83
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

stock32 = np.array([[192, .46], [1536, 2.60], [5184, 8.31], [12288, 19.35], [24000, 37.49], [41472, 64.50],
                    [65856, 102.14], [86016, 133.25]])
stock64 = np.array([[192, 1.59], [1536, 11.69], [5184, 39.10], [12288, 92.46], [15360, 115.54]])
exact32 = np.array([[1.54e6, 27.8], [18.69e6, 106.0], [24.0e6, 130.3], [25.47e6, 137.2], [28.58e6, 141.2]])
exact64 = np.array([[12.29e6, 118.0], [14.22e6, 134.8], [15.27e6, 143.8]])
student = np.array([[15360, 0.8], [122880, 3.5], [983040, 4.8], [7864320, 20.4]])
NEP_KIB = 7.37


def main():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.axhline(141, color="k", lw=0.8, ls="--")
    ax.text(3e5, 165, "H200 memory (141 GB)", fontsize=6.5)
    n = np.logspace(2, np.log10(1.86e7), 50)
    ax.plot(n, n * NEP_KIB * 1024 / 1e9, "-", color="#e67e22", lw=1.2, label="NEP89 (7.4 KiB/atom)")
    ax.plot(*stock32.T, "^-", color="#7f8c8d", ms=3.5, lw=1.2, label="MACE-MPA-0, stock (fp32)")
    ax.plot(*stock64.T, "v--", color="#7f8c8d", ms=3.5, lw=1.0, label="MACE-MPA-0, stock (fp64)")
    ax.plot([98304], [141], "x", color="#7f8c8d", ms=7)
    ax.annotate("OOM", (98304, 141), xytext=(3, -12), textcoords="offset points", fontsize=6.5, color="#7f8c8d")
    ax.plot(*exact32.T, "s-", color="#1f4e79", ms=3.5, lw=1.2, label="MACE-MPA-0, exact three-pass (fp32)")
    ax.plot(*exact64.T, "s--", color="#1f4e79", mfc="white", ms=3.5, lw=1.0, label="MACE-MPA-0, exact three-pass (fp64)")
    ax.plot(*student.T, "o-", color="#2e86c1", ms=3.5, lw=1.2, label="MACE student (model only)")
    ax.annotate("28.6M atoms", (28.58e6, 141.2), xytext=(-40, -22), textcoords="offset points", fontsize=6.5,
                color="#1f4e79", arrowprops=dict(arrowstyle="-", lw=0.5, color="#1f4e79"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 5e7)
    ax.set_ylim(0.3, 400)
    ax.set_xlabel("atoms")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.text(0.98, 0.04, "stock: 1.55 MB/atom\nexact: 4.57 kB/atom resident (fp32)", transform=ax.transAxes, ha="right",
            fontsize=6.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_memory_vs_N.{ext}"), dpi=300)
    print("wrote fig3_memory_vs_N")


if __name__ == "__main__":
    main()
