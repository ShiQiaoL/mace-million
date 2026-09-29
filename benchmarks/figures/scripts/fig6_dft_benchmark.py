#!/usr/bin/env python
"""Fig. 6: force and energy errors against DFT (held-out frames) for teacher, students, NEP89 and classical force
fields on water, SiO2 and CuNi.  Data: paper/data/dft_<system>/{vs_dft.json, classical*.json} (job75, classical_eval).

    python fig6_dft_benchmark.py   ->  paper/figures/fig6_dft_benchmark.{pdf,png}
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/figures/scripts
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..")

# model key in the JSON -> (label, colour); one row per model class, in plotting order
SYSTEMS = [
    ("water", "Water", {"mace-mpa-0-medium.model": "teacher", "V1_16x1_l2_c2_r45.model": "student",
                        "student_c2.model": "student + OTF", "NEP89": "NEP89"},
     [("classical.json", "SPC/Fw")]),
    ("sio2", "SiO$_2$", {"mace-mpa-0-medium.model": "teacher", "V1_16x1_l2_c2_r45.model": "student", "NEP89": "NEP89"},
     [("classical_vashishta.json", "Vashishta"), ("classical_tersoff.json", "Tersoff")]),
    ("cuni", "CuNi", {"mace-mpa-0-medium.model": "teacher", "V1_16x1_l2_c2_r45.model": "student", "NEP89": "NEP89"},
     [("classical.json", "EAM")]),
]
COLOURS = {"teacher": "#1f4e79", "student": "#2e86c1", "student + OTF": "#85c1e9", "NEP89": "#e67e22",
           "classical": "#7f8c8d"}


def load(system, keys, classical):
    d = json.load(open(os.path.join(DATA, f"dft_{system}", "vs_dft.json")))
    rows = [(lab, d[k]) for k, lab in keys.items() if k in d]
    for fn, lab in classical:
        c = json.load(open(os.path.join(DATA, f"dft_{system}", fn)))
        rows.append((lab, next(iter(c.values()))))
    return rows


def main():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey="row")
    for j, (sysname, title, keys, classical) in enumerate(SYSTEMS):
        rows = load(sysname, keys, classical)
        labels = [r[0] for r in rows]
        col = [COLOURS.get(l, COLOURS["classical"]) for l in labels]
        x = np.arange(len(rows))
        for i, (metric, scale, ylab) in enumerate([("f_rmse", 1e3, "force RMSE vs DFT (meV Å$^{-1}$)"),
                                                   ("e_rmse_meV_atom", 1.0, "energy RMSE vs DFT (meV atom$^{-1}$)")]):
            ax = axes[i, j]
            v = [r[1][metric] * scale for r in rows]
            ax.bar(x, v, color=col, width=0.7)
            for xi, vi in zip(x, v):
                ax.text(xi, vi * 1.08, f"{vi:.0f}" if vi >= 10 else f"{vi:.2f}", ha="center", va="bottom", fontsize=6)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha="right")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"{title} ({rows[0][1]['n_frames']} DFT frames)")
    for ax in axes[0]:
        ax.set_ylim(10, 3000)
    for ax in axes[1]:
        ax.set_ylim(0.2, 200)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig6_dft_benchmark.{ext}"), dpi=300)
    print("wrote", os.path.join(OUT, "fig6_dft_benchmark.pdf"))


if __name__ == "__main__":
    main()
