#!/usr/bin/env python
"""Fig. 4: throughput vs system size on one H200 (water, 300 K).  Sources:
  exact three-pass MACE-MPA-0, standalone, CUDA graphs + fast layers ............ job25 (steady s/step)
  exact three-pass MACE-MPA-0 inside LAMMPS (1M and 8.96M atoms) ................ job38/job47 (CUDA graphs)
  stock MACE-MPA-0 LAMMPS ML-IAP (fp32, cuEquivariance), OOM above 86,016 atoms .. job08 / job01
  NEP89 in GPUMD, student V5 in LAMMPS, 3 repeats each ........................... job83 (paper/data/e2_scaling_job83.txt)
  FLOP ceiling of the exact model (45 MFLOP/atom-step; FP32 peak / 3xTF32) ...... THEORY_speed_ceiling_mace-mpa0_H200.md
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/figures/scripts
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..")

N_exact = np.array([24000, 41472, 98304, 192000, 421824, 786432, 1536000, 3000000, 6291456, 12288000])
t_exact = np.array([0.20, 0.30, 0.56, 1.15, 2.47, 4.63, 9.04, 17.62, 36.69, 72.21])
N_stock, t_stock = np.array([12288, 41472]), np.array([0.0788, 0.2388])
lmp_exact = {983040: 2.25e5}          # 2.15-2.34e5 in LAMMPS with CUDA graphs (job38/47), midpoint


def e2():
    rows = {}
    for ln in open(os.path.join(DATA, "e2_scaling_job83.txt")):
        m = re.search(r"(NEP89|V5) N=(\d+).*?(?:= |steady )([\d.e+]+) atom", ln)
        if m:
            rows.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append(float(m.group(3)))
    return {k: (np.array(sorted(v)), np.array([np.mean(v[n]) for n in sorted(v)]),
                np.array([np.std(v[n]) for n in sorted(v)])) for k, v in rows.items()}


def main():
    d = e2()
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhspan(1.5e6, 1.9e6, color="#1f4e79", alpha=0.12, lw=0)
    ax.text(1.2e4, 1.65e6, "FLOP ceiling of exact MPA-0", fontsize=6.5, color="#1f4e79", va="center")
    n, v, s = d["NEP89"]
    ax.errorbar(n, v, yerr=s, fmt="D-", color="#e67e22", ms=4, lw=1.2, label="NEP89 (GPUMD)")
    n, v, s = d["V5"]
    ax.errorbar(n, v, yerr=s, fmt="o-", color="#2e86c1", ms=4, lw=1.2, label="MACE student (this work)")
    ax.plot(N_exact, N_exact / t_exact, "s-", color="#1f4e79", ms=3.5, lw=1.2, label="MACE-MPA-0, exact three-pass")
    ax.plot(list(lmp_exact), list(lmp_exact.values()), "s", mfc="white", color="#1f4e79", ms=5,
            label="  … inside LAMMPS")
    ax.plot(N_stock, N_stock / t_stock, "^-", color="#7f8c8d", ms=4, lw=1.2, label="MACE-MPA-0, stock ML-IAP")
    ax.plot([N_stock[-1] * 2.07], [N_stock[-1] / t_stock[-1]], "x", color="#7f8c8d", ms=7)
    ax.annotate("OOM", (N_stock[-1] * 2.07, N_stock[-1] / t_stock[-1]), xytext=(4, -10), textcoords="offset points",
                fontsize=6.5, color="#7f8c8d")
    ax.axvline(8e6, color="grey", ls=":", lw=0.8)
    ax.text(7.3e6, 6e5, "8M-atom cap", fontsize=6, color="grey", ha="right", rotation=90, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e3, 2e7)
    ax.set_ylim(6e4, 2.5e8)
    ax.set_xlabel("atoms")
    ax.set_ylabel("throughput (atom-steps s$^{-1}$)")
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / 1e6 * 86400 * 1e-6 * 1e3 / 1e3,
                                                   lambda x: x * 1e6 / 86400 * 1e6 / 1e6))
    sec.set_ylabel("ns day$^{-1}$ at 1 fs for 10$^{6}$ atoms")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=6.3, loc="upper center", ncol=2, frameon=False, columnspacing=0.8)
    ax.set_title("One NVIDIA H200 (141 GB), water, 300 K", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_speed_vs_N.{ext}"), dpi=300)
    print("wrote fig4_speed_vs_N")


if __name__ == "__main__":
    main()
