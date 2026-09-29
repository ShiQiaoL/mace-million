#!/usr/bin/env python
"""Fig. 4: accuracy against DFT on held-out frames for water, SiO2 and CuNi (points on log axes, no bars).
Data: paper/data/dft_<system>/{vs_dft.json, vs_dft_r40.json, classical*.json} (jobs 75, 79, 82a, 89, 90).
Top row: force RMSE; bottom row: energy RMSE (mean offset removed).  One column per system, shared y axes.
"""
import json
import os

import matplotlib.pyplot as plt

import paperstyle as ps

ps.apply()
J = lambda *p: json.load(open(os.path.join(ps.DATA, *p)))
first = lambda d: next(iter(d.values()))
SYS = [
    ("Water, 20 frames of 192 atoms", "water",
     [("teacher", J("dft_water", "vs_dft.json")["mace-mpa-0-medium.model"], "exact", "s"),
      ("student\n4.5 Å", J("dft_water", "vs_dft.json")["V1_16x1_l2_c2_r45.model"], "student", "o"),
      ("student\n4.25 Å", J("dft_water", "vs_dft_V4V5.json")["V5_16x1_l2_c2_r425.model"], "student", "o"),
      ("NEP89", J("dft_water", "vs_dft.json")["NEP89"], "nep", "D"),
      ("SPC/Fw", first(J("dft_water", "classical.json")), "classic", "v")]),
    ("SiO$_2$, 20 frames of 192 atoms", "sio2",
     [("teacher", J("dft_sio2", "vs_dft.json")["mace-mpa-0-medium.model"], "exact", "s"),
      ("student\n4.5 Å", J("dft_sio2", "vs_dft.json")["V1_16x1_l2_c2_r45.model"], "student", "o"),
      ("student\n4.0 Å", first(J("dft_sio2", "vs_dft_r40.json")), "student", "o"),
      ("NEP89", J("dft_sio2", "vs_dft.json")["NEP89"], "nep", "D"),
      ("Vashishta", first(J("dft_sio2", "classical_vashishta.json")), "classic", "v"),
      ("Tersoff", first(J("dft_sio2", "classical_tersoff.json")), "classic", "v")]),
    ("CuNi, 4 frames of 108 atoms", "cuni",
     [("teacher", J("dft_cuni", "vs_dft.json")["mace-mpa-0-medium.model"], "exact", "s"),
      ("student\n4.5 Å", J("dft_cuni", "vs_dft.json")["V1_16x1_l2_c2_r45.model"], "student", "o"),
      ("student\n4.0 Å", first(J("dft_cuni", "vs_dft_r40.json")), "student", "o"),
      ("NEP89", J("dft_cuni", "vs_dft.json")["NEP89"], "nep", "D"),
      ("EAM", first(J("dft_cuni", "classical.json")), "classic", "v")]),
]
fig, axes = plt.subplots(2, 3, figsize=(ps.W2, 98 * ps.MM), sharey="row",
                         gridspec_kw=dict(left=0.075, right=0.99, top=0.92, bottom=0.10, hspace=0.18, wspace=0.12, width_ratios=[5, 6.4, 5]))
for j, (ttl, _, rows) in enumerate(SYS):
    for i, (key, scale, ylab, lim) in enumerate((("f_rmse", 1e3, "Force error (meV Å$^{-1}$)", (12, 3000)),
                                                 ("e_rmse_meV_atom", 1.0, "Energy error (meV per atom)", (0.3, 200)))):
        ax = axes[i, j]
        teacher = rows[0][1][key] * scale
        ax.axhline(teacher, color=ps.C["exact"], lw=0.6, ls=(0, (2, 2)), alpha=0.6)
        for x, (lab, r, ent, m) in enumerate(rows):
            v = r[key] * scale
            ax.plot([x], [v], m, color=ps.C[ent], ms=5.5 if ent == "student" else 4.8)
            ax.text(x, v * 1.22, f"{v:.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=ps.SECOND, color=ps.C[ent])
        ax.set_yscale("log"); ax.set_ylim(*lim); ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xticks(range(len(rows)))
        if i == 0:
            ax.set_xticklabels([]); ps.title(ax, ttl)
            ax.set_yticks([20, 100, 1000]); ax.set_yticklabels(["20", "100", "1,000"])
        else:
            ax.set_xticklabels([r[0] for r in rows], fontsize=ps.SECOND, rotation=0)
            ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
        if j == 0:
            ax.set_ylabel(ylab)
        ax.tick_params(axis="x", length=0)
axes[0, 0].text(0.03, 0.06, "against DFT; lower is better", transform=axes[0, 0].transAxes, fontsize=ps.SECOND, color=ps.C["line"])
fig.canvas.draw()
for j, s in enumerate("abc"):
    ps.letter(axes[0, j], s, x_mm=(12 if j == 0 else 4.5))
ps.save(fig, "fig4_accuracy")
