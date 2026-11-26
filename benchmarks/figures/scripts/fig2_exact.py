#!/usr/bin/env python
"""Fig. 2: the exact three-pass evaluation removes the memory limit without changing the model.
 a  peak GPU memory vs atoms on one H200 (stock: job01; exact three-pass: limit scans; student: job83; NEP89: 7.37 KiB/atom)
 b  largest force deviations (fp64/fp32 replica checks; LAMMPS parity job35; chunk invariance job82a)
 c  NVE total energy, 384 water atoms, 1 fs (job82a/82b)
 d  O-O and O-H radial distribution functions, NPT water (job35)
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import paperstyle as ps

ps.apply()
stock32 = np.array([[192, .46], [1536, 2.60], [5184, 8.31], [12288, 19.35], [24000, 37.49], [41472, 64.50], [65856, 102.14], [86016, 133.25]])
exact32 = np.array([[1.54e6, 27.8], [18.69e6, 106.0], [24.0e6, 130.3], [25.47e6, 137.2], [28.58e6, 141.2]])
exact64 = np.array([[12.29e6, 118.0], [14.22e6, 134.8], [15.27e6, 143.8]])
student = np.array([[15360, 0.8], [122880, 3.5], [983040, 4.8], [7864320, 20.4]])


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


fig = plt.figure(figsize=(ps.W2, 118 * ps.MM))
gs = fig.add_gridspec(2, 12, left=0.075, right=0.985, top=0.925, bottom=0.085, hspace=0.62, wspace=2.6)
a = fig.add_subplot(gs[0, :7]); b = fig.add_subplot(gs[0, 8:]); c = fig.add_subplot(gs[1, :6]); d = fig.add_subplot(gs[1, 7:])

# ---- a
a.axhline(141, color=ps.C["ref"], lw=0.8, ls=(0, (4, 3)))
a.text(1.3e2, 160, "GPU memory, 141 GB", fontsize=ps.SECOND, color=ps.C["line"], va="bottom")
n = np.logspace(2, np.log10(1.86e7), 40)
a.plot(n, n * 7.37 * 1024 / 1e9, color=ps.C["nep"])
a.plot(*stock32.T, "-^", color=ps.C["stock"])
a.plot([98304], [141], "x", color=ps.C["stock"], ms=5, mew=1.1)
a.plot(*exact32.T, "-s", color=ps.C["exact"])
a.plot(*exact64.T, "--s", color=ps.C["exact"], mfc="white", lw=0.9)
a.plot(*student.T, "-o", color=ps.C["student"])
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e2, 6e7); a.set_ylim(0.3, 400)
a.set_xlabel("Atoms"); a.set_ylabel("Peak GPU memory (GB)")
a.text(1.3e2, 14, "MACE-MPA-0, stock\n1.55 MB per atom", color=ps.C["stock"], fontsize=ps.SECOND, ha="left", va="bottom")
a.text(8.0e4, 190, "out of memory", fontsize=ps.SECOND, color=ps.C["stock"], ha="right", va="bottom")
a.text(1.25e6, 36, "MACE-MPA-0, three-pass\n4.6 kB per atom resident", color=ps.C["exact"], fontsize=ps.SECOND, ha="right", va="bottom")
a.text(9.5e6, 190, "fp64", color=ps.C["exact"], fontsize=ps.SECOND, ha="right", va="bottom")
a.text(9.5e6, 9.5, "MACE student", color=ps.C["student"], fontsize=ps.SECOND, ha="left", va="center")
a.text(2.0e5, 0.62, "NEP89", color=ps.C["nep"], fontsize=ps.SECOND, ha="left", va="center")
a.text(5.0e7, 40, "28.6 M atoms\n(fp32)", fontsize=ps.SECOND, color=ps.C["exact"], ha="right", va="top")
ps.title(a, "Three-pass evaluation lowers memory per added atom 340-fold")

# ---- b  (points on a log axis, no bars)
lab = ["fp64 three-pass vs original\n(all 1.5 × 10$^7$ atoms)", "fp32 three-pass vs original", "original model,\nfp32 vs fp64",
       "three-pass in LAMMPS\nvs stock fp64", "chunks of 512\nvs 12,288"]
val = [6.7e-12, 6.2e-5, 2e-5, 2.2e-5, 1.7e-6]
col = [ps.C["exact"], ps.C["exact"], ps.C["stock"], ps.C["exact"], ps.C["exact"]]
y = np.arange(len(val))[::-1]
for yi, v, cc in zip(y, val, col):
    b.plot([v], [yi], "o", color=cc, ms=4.5)
    b.text(v * 3, yi, f"{v:.1e}".replace("e-0", " × 10$^{-").replace("e-", " × 10$^{-") + "}$", va="center", fontsize=ps.SECOND, color=cc)
b.set_xscale("log"); b.set_xlim(1e-13, 3e-1); b.set_ylim(-0.6, len(val) - 0.4)
b.set_yticks(y); b.set_yticklabels(lab, fontsize=ps.SECOND)
b.set_xticks([1e-12, 1e-9, 1e-6, 1e-3])
b.set_xlabel("Largest force deviation (eV Å$^{-1}$)")
b.tick_params(axis="y", length=0)
ps.title(b, "Deviations stay at round-off level")

# ---- c
for name, cc, lb, lw in (("threepass", ps.C["exact"], "three-pass, 100 ps", 0.7), ("stock", ps.C["stock"], "stock, 10 ps", 0.9)):
    s, e = thermo(os.path.join(ps.DATA, "nve", f"{name}_log.lammps"))
    c.plot(s * 1e-3, (e - e[0]) / 384 * 1e3, color=cc, lw=lw, label=lb, zorder=3 if name == "stock" else 2)
c.set_xlabel("Time (ps)"); c.set_ylabel("Energy change (meV per atom)")
c.set_ylim(-0.6, 0.9); c.set_xlim(0, 100)
c.legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)
c.text(99, -0.5, "drift 3.1 × 10$^{-6}$ eV per atom per ps", fontsize=ps.SECOND, color=ps.C["exact"], ha="right")
ps.title(c, "Total energy is conserved over 100 ps")

# ---- d
r = np.load(os.path.join(ps.DATA, "rdf_job35_original_vs_threepass.npz"))
x = r["r"]
for k, off, nm in (("O-O", 0.0, "O–O"), ("O-H", 2.0, "O–H (shifted by 2)")):
    d.plot(x, r[f"0_{k}"] + off, color=ps.C["stock"], lw=2.2, label="original" if k == "O-O" else None)
    d.plot(x, r[f"1_{k}"] + off, color=ps.C["exact"], lw=0.8, ls=(0, (3, 2)), label="three-pass" if k == "O-O" else None)
    d.text(5.95, off + 1.22, nm, fontsize=ps.SECOND, ha="right", color=ps.C["line"])
d.set_xlim(1.5, 6); d.set_ylim(0, 5.4); d.set_xlabel("Distance (Å)"); d.set_ylabel("Radial distribution function")
d.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
ps.title(d, "Liquid structure is unchanged")

fig.canvas.draw()
for ax, s in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
    ps.letter(ax, s)
ps.save(fig, "fig2_exact")
