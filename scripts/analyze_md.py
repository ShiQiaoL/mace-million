#!/usr/bin/env python
"""
Analyse LAMMPS thermo output.
  nve : energy-conservation drift  ->  slope of TotEng vs time (eV/atom/ps) and RMS fluctuation of TotEng
        python analyze_md.py nve log.lammps --natoms 1536 --dt-ps 0.0005
  npt : density average over the LAST run block (production) with block-averaging error
        python analyze_md.py npt log.lammps --skip-frac 0.0
"""
import argparse
import re

import numpy as np


def read_thermo_blocks(path):
    blocks = []
    cur = None
    for line in open(path, errors="replace"):
        if re.match(r"^\s*Step\s+", line):
            cur = {"cols": line.split(), "rows": []}
            blocks.append(cur)
            continue
        if cur is not None:
            parts = line.split()
            if len(parts) == len(cur["cols"]):
                try:
                    cur["rows"].append([float(x) for x in parts])
                    continue
                except ValueError:
                    pass
            if parts and parts[0] == "Loop":
                cur = None
    return [(b["cols"], np.array(b["rows"])) for b in blocks if len(b["rows"]) > 1]


def block_average(x, nblocks=5):
    n = len(x) // nblocks
    if n < 2:
        return x.mean(), x.std() / np.sqrt(max(len(x), 1))
    means = np.array([x[i * n:(i + 1) * n].mean() for i in range(nblocks)])
    return means.mean(), means.std(ddof=1) / np.sqrt(nblocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["nve", "npt"])
    ap.add_argument("log")
    ap.add_argument("--natoms", type=int, default=None)
    ap.add_argument("--dt-ps", type=float, default=0.0005)
    ap.add_argument("--skip-frac", type=float, default=0.0)
    a = ap.parse_args()
    blocks = read_thermo_blocks(a.log)
    if not blocks:
        raise SystemExit("no thermo blocks found")
    cols, rows = blocks[-1]
    step = rows[:, cols.index("Step")]
    if a.mode == "nve":
        et = rows[:, cols.index("TotEng")]
        n = a.natoms or int(rows[0, cols.index("Atoms")]) if "Atoms" in cols else a.natoms
        t = (step - step[0]) * a.dt_ps
        slope, icpt = np.polyfit(t, et, 1)
        print(f"steps {int(step[0])}..{int(step[-1])}  ({t[-1]:.3f} ps)  N={n}")
        print(f"TotEng drift = {slope:+.4e} eV/ps  = {slope/n:+.4e} eV/atom/ps")
        print(f"TotEng RMS fluctuation about linear fit = {np.std(et-(slope*t+icpt)):.4e} eV  ({np.std(et-(slope*t+icpt))/n:.3e} eV/atom)")
        if "Temp" in cols:
            T = rows[:, cols.index("Temp")]
            print(f"Temp mean {T.mean():.1f} K  (start {T[0]:.1f}, end {T[-1]:.1f})")
    else:
        rho = rows[:, cols.index("Density")]
        k = int(len(rho) * a.skip_frac)
        m, err = block_average(rho[k:])
        print(f"production block: steps {int(step[k])}..{int(step[-1])}, {len(rho)-k} samples")
        print(f"density = {m:.4f} +/- {err:.4f} g/cm3   (min {rho[k:].min():.4f}, max {rho[k:].max():.4f})")
        if "Temp" in cols:
            print(f"Temp = {rows[k:, cols.index('Temp')].mean():.1f} K ; Press = {rows[k:, cols.index('Press')].mean():.1f} bar")
        if len(blocks) > 1:
            c0, r0 = blocks[0]
            rho0 = r0[:, c0.index("Density")]
            print(f"equilibration block: density start {rho0[0]:.4f} -> end {rho0[-1]:.4f} g/cm3 over {len(rho0)} samples")


if __name__ == "__main__":
    main()
