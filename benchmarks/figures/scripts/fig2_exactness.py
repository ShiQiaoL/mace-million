#!/usr/bin/env python
"""Fig. 2: the three-pass evaluation reproduces MACE-MPA-0.
 (a) largest force deviations: fp64 three-pass vs original on all 1.53e7 atoms (6.7e-12, LIMIT_RESULTS); fp32 three-pass
     vs original (6.2e-5); the original model's own fp32-fp64 difference (2e-5); three-pass in LAMMPS vs stock fp64
     ML-IAP (2.2e-5, job35); chunk 512 vs 12288 (1.7e-6, job82a)
 (b) NVE total energy, 384 water atoms, 1 fs: three-pass 100 ps and stock 10 ps (job82a/82b)
 (c) O-O and O-H radial distribution functions in NPT, original vs three-pass (job35)
"""
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/figures/scripts
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..")


def thermo(path):
    rows, cols, on = [], None, False
    for ln in open(path, errors="replace"):
        w = ln.split()
        if w[:2] == ["Step", "Atoms"]:
            cols, on = w, True
            continue
        if on:
            try:
                rows.append([float(x) for x in w])
            except ValueError:
                on = False
    r = np.array(rows)
    return r[:, cols.index("Step")], r[:, cols.index("TotEng")]


def main():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # (a)
    lab = ["fp64 three-pass vs original\n(all 1.5e7 atoms)", "fp32 three-pass vs original", "original: fp32 vs fp64",
           "three-pass in LAMMPS\nvs stock fp64", "chunk 512 vs 12,288"]
    val = [6.7e-12, 6.2e-5, 2e-5, 2.2e-5, 1.7e-6]
    col = ["#1f4e79", "#2e86c1", "#7f8c8d", "#2e86c1", "#2e86c1"]
    a = ax[0]
    y = np.arange(len(val))[::-1]
    a.barh(y, val, color=col, height=0.6, left=1e-13)
    a.set_xscale("log")
    a.set_xlim(1e-13, 1e-2)
    a.set_yticks(y)
    a.set_yticklabels(lab, fontsize=5.8)
    a.set_xlabel("max $|\\Delta F|$ (eV Å$^{-1}$)")
    for yi, v in zip(y, val):
        a.text(v * 2.0, yi, f"{v:.1e}", va="center", fontsize=5.8)
    a.set_title("a  force deviations", loc="left", fontsize=8)
    # (b)
    b = ax[1]
    for name, colr, lab_ in (("threepass", "#1f4e79", "three-pass"), ("stock", "#7f8c8d", "stock")):
        s, e = thermo(os.path.join(DATA, "nve", f"{name}_log.lammps"))
        b.plot(s * 1e-3, (e - e[0]) / 384 * 1e3, color=colr, lw=0.8, label=lab_)
    b.set_xlabel("time (ps)")
    b.set_ylabel("$E_\\mathrm{tot}-E_\\mathrm{tot}(0)$ (meV atom$^{-1}$)")
    b.set_ylim(-1.0, 1.0)
    b.legend(fontsize=6.5, frameon=False, loc="lower right")
    b.text(0.03, 0.95, "drift (eV atom$^{-1}$ ps$^{-1}$):\nthree-pass $3.1\\times10^{-6}$ (100 ps)\nstock $1.6\\times10^{-5}$ (10 ps)",
           transform=b.transAxes, fontsize=5.8, va="top")
    b.set_title("b  NVE, 384 atoms, 1 fs", loc="left", fontsize=8)
    # (c)
    c = ax[2]
    d = np.load(os.path.join(DATA, "rdf_job35_original_vs_threepass.npz"))
    r = d["r"]
    for k, off, nm in (("O-O", 0.0, "O–O"), ("O-H", 2.0, "O–H + 2")):
        c.plot(r, d[f"0_{k}"] + off, color="#7f8c8d", lw=1.6, label="original" if k == "O-O" else None)
        c.plot(r, d[f"1_{k}"] + off, color="#1f4e79", lw=0.8, ls="--", label="three-pass" if k == "O-O" else None)
        c.text(5.2, off + 1.25, nm, fontsize=6.5)
    c.set_xlim(1.5, 6)
    c.set_ylim(0, 5.5)
    c.set_xlabel("$r$ (Å)")
    c.set_ylabel("$g(r)$")
    c.legend(fontsize=6.5, frameon=False, loc="upper right")
    c.set_title("c  NPT water, 300 K", loc="left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_exactness.{ext}"), dpi=300)
    print("wrote fig2_exactness")


if __name__ == "__main__":
    main()
