#!/usr/bin/env python
"""Fig. 3: throughput of the exact model and of the supervised students on one H200.
 a  throughput vs atoms (water): NEP89/GPUMD and student (job83, 3 repeats), exact three-pass (job25; LAMMPS job38/47),
    stock ML-IAP (job08), FLOP ceiling of the exact model
 b  accuracy against cost at ~1e6 atoms: force RMSE vs DFT (water, job75/82a) against throughput (jobs 73/76/83)
 c  on-the-fly supervision: force RMSE of the student vs the teacher per supervised segment (job64, job77/78)
 d  production run, 983,040 water atoms NPT: rate per 100-ps segment and density (job84; updated as segments finish)
"""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import paperstyle as ps

ps.apply()
N_exact = np.array([24000, 41472, 98304, 192000, 421824, 786432, 1536000, 3000000, 6291456, 12288000])
t_exact = np.array([0.20, 0.30, 0.56, 1.15, 2.47, 4.63, 9.04, 17.62, 36.69, 72.21])
N_stock, t_stock = np.array([12288, 41472]), np.array([0.0788, 0.2388])
PROD = [dict(seg=1, rate=3.418e6, rate_all=983040 * 1e5 / 29168, rho=0.9059), dict(seg=2, rate=4.185e6, rate_all=983040 * 1e5 / 23840, rho=0.9080)]


def e2():
    rows = {}
    for ln in open(os.path.join(ps.DATA, "e2_scaling_job83.txt")):
        m = re.search(r"(NEP89|V5) N=(\d+).*?(?:= |steady )([\d.e+]+) atom", ln)
        if m:
            rows.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append(float(m.group(3)))
    return {k: (np.array(sorted(v)), np.array([np.mean(v[n]) for n in sorted(v)])) for k, v in rows.items()}


fig = plt.figure(figsize=(ps.W2, 122 * ps.MM))
gs = fig.add_gridspec(2, 12, left=0.075, right=0.985, top=0.925, bottom=0.085, hspace=0.62, wspace=3.2)
a = fig.add_subplot(gs[0, :6]); b = fig.add_subplot(gs[0, 7:]); c = fig.add_subplot(gs[1, :6]); d = fig.add_subplot(gs[1, 7:])

# ---- a
d2 = e2()
a.axhspan(1.5e6, 1.9e6, color=ps.C["exact"], alpha=0.13, lw=0)
a.plot(*d2["NEP89"], "-D", color=ps.C["nep"])
a.plot(*d2["V5"], "-o", color=ps.C["student"], lw=1.4, ms=4)
a.plot(N_exact, N_exact / t_exact, "-s", color=ps.C["exact"], ms=3)
a.plot([983040], [2.25e5], "s", color=ps.C["exact"], mfc="white", ms=4.5)
a.plot(N_stock, N_stock / t_stock, "-^", color=ps.C["stock"])
a.plot([98304], [N_stock[-1] / t_stock[-1]], "x", color=ps.C["stock"], ms=5, mew=1.1)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(8e3, 2e7); a.set_ylim(5e4, 6e7)
a.set_xlabel("Atoms"); a.set_ylabel("Throughput (atom-steps s$^{-1}$)")
a.text(1.9e7, 2.6e7, "NEP89", color=ps.C["nep"], fontsize=ps.SECOND, ha="right", va="bottom")
a.text(1.9e7, 5.0e6, "MACE student", color=ps.C["student"], fontsize=ps.SECOND, ha="right", va="bottom")
a.text(1.9e7, 1.35e6, "arithmetic ceiling of exact MACE-MPA-0", color=ps.C["exact"], fontsize=ps.SECOND, ha="right", va="top")
a.text(1.9e7, 9.0e4, "MACE-MPA-0, three-pass", color=ps.C["exact"], fontsize=ps.SECOND, ha="right", va="top")
a.text(1.25e6, 2.9e5, "in LAMMPS", color=ps.C["exact"], fontsize=ps.SECOND, ha="left", va="bottom")
a.text(1.0e4, 2.3e5, "MACE-MPA-0, stock", color=ps.C["stock"], fontsize=ps.SECOND, ha="left", va="bottom")
a.text(1.15e5, 1.25e5, "out of memory", color=ps.C["stock"], fontsize=ps.SECOND, ha="left", va="top")
sec = a.secondary_yaxis("right", functions=(lambda y: y * 86400e-12, lambda x: x / 86400e-12))
sec.set_ylabel("ns per day for 10$^6$ atoms at 1 fs"); sec.spines["right"].set_visible(True)
ps.title(a, "Students run 18 times faster than the exact model")

# ---- b
d1 = json.load(open(os.path.join(ps.DATA, "dft_water", "vs_dft.json")))
dv = json.load(open(os.path.join(ps.DATA, "dft_water", "vs_dft_V4V5.json")))
F = lambda dd, k: 1e3 * dd[k]["f_rmse"]
pts = [("teacher", 2.25e5, F(d1, "mace-mpa-0-medium.model"), ps.C["exact"], "s", "MACE-MPA-0 (teacher)", (7, 0), "left"),
       ("s45", 3.56e6, F(d1, "V1_16x1_l2_c2_r45.model"), ps.C["student"], "o", "students, cutoff\n4.5, 4.25, 4.0 Å", (-7, 9), "right"),
       ("s425", 3.97e6, F(dv, "V5_16x1_l2_c2_r425.model"), ps.C["student"], "o", None, None, None),
       ("s40", 4.69e6, F(dv, "V4_16x1_l2_c2_r40.model"), ps.C["student"], "o", None, None, None),
       ("s45s", 3.56e6, F(d1, "student_c2.model"), ps.C["student2"], "o", None, None, None),
       ("s40s", 4.69e6, F(dv, "student_c10.model"), ps.C["student2"], "o", "after supervision", (4, -12), "left"),
       ("nep", 1.96e7, F(d1, "NEP89"), ps.C["nep"], "D", "NEP89", (-7, 0), "right")]
for _, x, y, cc, m, lab, off, ha in pts:
    b.plot([x], [y], m, color=cc, ms=5, mec="white" if cc == ps.C["student2"] else cc, mew=0.4)
    if lab:
        b.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=ps.SECOND, color=cc if cc != ps.C["student2"] else ps.C["student"],
                   ha=ha, va="center")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(1e5, 5e7); b.set_ylim(15, 1200)
b.set_yticks([20, 50, 100, 200, 500, 1000]); b.set_yticklabels(["20", "50", "100", "200", "500", "1,000"])
b.set_xlabel("Throughput at 10$^6$ atoms (atom-steps s$^{-1}$)"); b.set_ylabel("Force error against DFT (meV Å$^{-1}$)")
b.text(0.03, 0.95, "lower and further right is better", transform=b.transAxes, fontsize=ps.SECOND, color=ps.C["line"], va="top")
ps.title(b, "Speed is gained at almost no cost in accuracy")

# ---- c
for fn, lab, cc, m in (("al_state_V1_192k.json", "192,000 atoms, cutoff 4.5 Å", ps.C["student"], "o"),
                       ("al_state_V4_npt300.json", "1,536 atoms, cutoff 4.0 Å", ps.C["student"], "^")):
    h = json.load(open(os.path.join(ps.DATA, fn)))["history"]
    c.plot([x["cycle"] + 1 for x in h], [1e3 * x["worst"]["f_rmse"] for x in h], "-" + m, color=cc,
           mfc="white" if m == "^" else cc, label=lab)
c.axhline(35, color=ps.C["ref"], lw=0.8, ls=(0, (4, 3)))
c.text(10.4, 35.6, "retraining threshold", fontsize=ps.SECOND, color=ps.C["line"], ha="right", va="bottom")
c.set_xlim(0.5, 10.5); c.set_ylim(15, 40); c.set_xticks(range(1, 11))
c.set_xlabel("Supervised segment"); c.set_ylabel("Force error against teacher (meV Å$^{-1}$)")
c.legend(loc="upper right", bbox_to_anchor=(1.0, 0.74), handlelength=1.8)
ps.title(c, "Supervision lowers the student's error segment by segment")

# ---- d
seg = np.array([p["seg"] for p in PROD])
d.plot(seg, [p["rate"] / 1e6 for p in PROD], "-o", color=ps.C["student"], label="molecular dynamics only")
d.plot(seg, [p["rate_all"] / 1e6 for p in PROD], "--o", color=ps.C["student"], mfc="white", label="including teacher checks")
d.set_xlim(0.5, 10.5); d.set_xticks(range(1, 11)); d.set_ylim(0, 5)
d.set_xlabel("100-ps segment of the 983,040-atom run"); d.set_ylabel("Throughput (10$^6$ atom-steps s$^{-1}$)")
d.legend(loc="lower right", handlelength=1.8)
ps.title(d, "Production run: above 4 × 10$^6$ after equilibration")

fig.canvas.draw()
for ax, s in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
    ps.letter(ax, s)
ps.save(fig, "fig3_throughput")
