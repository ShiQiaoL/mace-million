#!/usr/bin/env python
"""Start structures for the generality tests: a small base cell for distillation (extxyz) and replicated
LAMMPS data files (atom_style atomic, types in the order of --elems) for the large-system runs.

    python build_general.py --system sio2 --out-dir structures_general --reps 2 35
    python build_general.py --system cuni --out-dir structures_general --reps 4 63

sio2 : ideal beta-cristobalite (Fd-3m, a = 7.16 A, Si 8a, O 16c; 24 atoms/cell, 2.17 g/cm3), Si-O 1.55 A
cuni : random fcc Cu0.5Ni0.5 (a = 3.565 A, Vegard between Cu 3.615 and Ni 3.524; 4 atoms/cell)
"""
import argparse
import os

import numpy as np
from ase.build import bulk
from ase.io import write
from ase.spacegroup import crystal

SYSTEMS = {
    "sio2": dict(elems=["Si", "O"]),
    "cuni": dict(elems=["Cu", "Ni"]),
}


def unit(system, rng):
    if system == "sio2":
        return crystal(["Si", "O"], basis=[(0, 0, 0), (0.125, 0.125, 0.125)], spacegroup=227, cellpar=[7.16] * 3 + [90] * 3)
    if system == "cuni":
        return bulk("Cu", "fcc", a=3.565, cubic=True)
    raise ValueError(system)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", nargs="+", type=int, default=[2], help="replication n (n x n x n unit cells) per output")
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    elems = SYSTEMS[a.system]["elems"]
    u = unit(a.system, rng)
    for n in a.reps:
        at = u.repeat((n, n, n))
        if a.system == "cuni":                      # random 50/50 occupation of the fcc sites
            z = np.full(len(at), 29)
            z[rng.permutation(len(at))[: len(at) // 2]] = 28
            at.numbers = z
        at.pbc = True
        rho = at.get_masses().sum() / 0.602214076 / at.get_volume()
        stem = os.path.join(a.out_dir, f"{a.system}_{n}x{n}x{n}_N{len(at)}")
        if len(at) <= 5000:
            write(stem + ".extxyz", at)
        write(stem + ".data", at, format="lammps-data", specorder=elems, atom_style="atomic", masses=True)
        d = at.get_all_distances(mic=True) if len(at) <= 3000 else None
        dmin = float(d[d > 0].min()) if d is not None else float("nan")
        print(f"{stem}: {len(at)} atoms, box {at.cell.lengths().round(2)}, rho {rho:.3f} g/cm3, d_min {dmin:.3f} A", flush=True)
    print("elems (LAMMPS type order):", " ".join(elems))


if __name__ == "__main__":
    main()
