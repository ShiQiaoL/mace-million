#!/usr/bin/env python
"""Fig. 5: distilled students.
 (a) accuracy-speed plane for water: force RMSE against DFT (20 frames; job75, job82a) vs throughput at ~1e6 atoms on one
     H200 (teacher 2.25e5 in LAMMPS; students from jobs 73/76/74/83; NEP89 job83).  DFT (VASP GPU, 192 atoms in 111 s,
     job71) as a vertical line.  The l_max = 1 student was unstable in MD (job74) and is marked.
 (b) on-the-fly supervision: force RMSE and |virial pressure difference| (student vs teacher, worst checked frame) per
     segment, 1.92e5-atom run (V1, 3 segments) and 1,536-atom NPT run (r_max 4.0 A student, 10 segments).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/figures/scripts
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..")

d1 = json.load(open(os.path.join(DATA, "dft_water", "vs_dft.json")))
d2 = json.load(open(os.path.join(DATA, "dft_water", "vs_dft_V4V5.json")))
cl = next(iter(json.load(open(os.path.join(DATA, "dft_water", "classical.json"))).values()))
F = lambda d, k: 1e3 * d[k]["f_rmse"]
POINTS = [  # label, throughput (atom-steps/s at ~1e6 atoms), force RMSE vs DFT, colour, marker
    ("MACE-MPA-0 (teacher, exact)", 2.25e5, F(d1, "mace-mpa-0-medium.model"), "#1f4e79", "s"),
    ("student $r_{\\max}$ 4.5 Å", 3.56e6, F(d1, "V1_16x1_l2_c2_r45.model"), "#2e86c1", "o"),
    ("  + supervision", 3.56e6, F(d1, "student_c2.model"), "#85c1e9", "o"),
    ("student $r_{\\max}$ 4.25 Å", 3.97e6, F(d2, "V5_16x1_l2_c2_r425.model"), "#2e86c1", "D"),
    ("student $r_{\\max}$ 4.0 Å", 4.69e6, F(d2, "V4_16x1_l2_c2_r40.model"), "#2e86c1", "^"),
    ("  + supervision", 4.69e6, F(d2, "student_c10.model"), "#85c1e9", "^"),
    ("student $\\ell_{\\max}$ 1 (unstable)", 4.5e6, F(d1, "V3_16x1_l1_c2_r45.model"), "#c0392b", "x"),
    ("NEP89", 1.96e7, F(d1, "NEP89"), "#e67e22", "D"),
]


def main():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    a.text(0.98, 0.97, "DFT (VASP, GPU): 1.7 atom-steps s$^{-1}$,\noff scale to the left", transform=a.transAxes,
           fontsize=5.8, ha="right", va="top")
    for lab, x, y, c, m in POINTS:
        a.scatter([x], [y], color=c, marker=m, s=28, zorder=3, label=lab)
    a.axvspan(3.94e6, 2.5e7, color="#2e86c1", alpha=0.07, lw=0)
    a.text(4.2e6, 12, "≥ 1/5 of\nNEP89", fontsize=6, color="#2e86c1")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(1e5, 5e7)
    a.set_ylim(10, 1000)
    a.set_xlabel("throughput at $10^{6}$ atoms, one H200 (atom-steps s$^{-1}$)")
    a.set_ylabel("water force RMSE vs DFT (meV Å$^{-1}$)")
    a.grid(True, which="major", alpha=0.3)
    a.legend(fontsize=5.8, frameon=False, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 0.93))
    a.set_title("a  accuracy against cost", loc="left", fontsize=8)
    b = ax[1]
    for fn, lab, c, m in (("al_state_V1_192k.json", "1.92×10$^{5}$ atoms, $r_{\\max}$ 4.5 Å", "#2e86c1", "o"),
                          ("al_state_V4_npt300.json", "1,536 atoms, $r_{\\max}$ 4.0 Å", "#1f4e79", "^")):
        h = json.load(open(os.path.join(DATA, fn)))["history"]
        cyc = [x["cycle"] for x in h]
        b.plot(cyc, [1e3 * x["worst"]["f_rmse"] for x in h], m + "-", color=c, ms=3.5, lw=1.0, label=lab)
    b.axhline(35, color="grey", ls="--", lw=0.7)
    b.text(0.0, 35.5, "threshold", fontsize=6, color="grey", ha="left", va="bottom")
    b.set_xlabel("supervised segment")
    b.set_ylabel("force RMSE vs teacher (meV Å$^{-1}$)")
    b.set_ylim(15, 40)
    b.legend(fontsize=6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    b.set_title("b  on-the-fly supervision", loc="left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig5_students.{ext}"), dpi=300)
    print("wrote fig5_students")


if __name__ == "__main__":
    main()
