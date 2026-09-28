#!/usr/bin/env python
"""
Energy/force parity checks.

  ref   : compute an ASE/MACE reference (FP64, e3nn, GPU) for a LAMMPS data file -> ref.npz
          python compare_forces.py ref --model mace-mpa-0-medium.model --data water64_2x2x2_N1536.data --out ref_N1536.npz
  dump  : compare a LAMMPS forces.dump (sorted by id) + log.lammps step-0 PotEng with ref.npz or with another dump
          python compare_forces.py dump --a case_fp64/forces.dump --b ref_N1536.npz --log-a case_fp64/log.lammps
          python compare_forces.py dump --a case_fp32/forces.dump --b case_fp64/forces.dump --log-a ... --log-b ...
Reports dE (eV, eV/atom), RMS(dF), max|dF| (eV/A), and relative force error. Tolerance guidance:
  FP64 LAMMPS vs ASE reference: |dF|max < 1e-5 eV/A (cuEq vs e3nn summation order) ; FP32: RMS(dF) ~ 1e-3 .. 1e-2 eV/A.
"""
import argparse
import os
import re

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np  # noqa: E402


def read_dump_first_frame(path):
    ids, types, xyz, f = [], [], [], []
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]
            n = int(lines[i - 5].strip()) if "NUMBER OF ATOMS" in lines[i - 6] else None
            j = i + 1
            while j < len(lines) and not lines[j].startswith("ITEM:"):
                parts = lines[j].split()
                rec = dict(zip(cols, parts))
                ids.append(int(rec["id"]))
                types.append(int(rec["type"]))
                xyz.append([float(rec["x"]), float(rec["y"]), float(rec["z"])])
                f.append([float(rec["fx"]), float(rec["fy"]), float(rec["fz"])])
                j += 1
            break
        i += 1
    order = np.argsort(ids)
    return np.array(ids)[order], np.array(types)[order], np.array(xyz)[order], np.array(f)[order]


def read_log_step0_pe(path):
    if not path or not os.path.exists(path):
        return None
    txt = open(path, errors="replace").read()
    m = re.search(r"^\s*Step\s+.*?$\n(.*?)$", txt, re.M)
    if not m:
        return None
    header = re.search(r"^\s*Step\s+.*$", txt, re.M).group(0).split()
    first = m.group(1).split()
    try:
        return float(first[header.index("PotEng")])
    except (ValueError, IndexError):
        return None


def do_ref(a):
    import torch
    from ase import io
    from mace.calculators import MACECalculator

    try:
        atoms = io.read(a.data, format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
    except TypeError:
        atoms = io.read(a.data, format="lammps-data", style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
    calc = MACECalculator(model_paths=a.model, device="cuda" if torch.cuda.is_available() else "cpu",
                          default_dtype="float64", enable_cueq=False)
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    np.savez(a.out, energy=e, forces=f, positions=atoms.get_positions(), numbers=atoms.get_atomic_numbers())
    print(f"reference: N={len(atoms)} E={e:.6f} eV  Fmax={np.abs(f).max():.4f} eV/A  -> {a.out}")


def load_side(path, log):
    if path.endswith(".npz"):
        d = np.load(path)
        return float(d["energy"]), d["forces"], d["positions"]
    ids, types, xyz, f = read_dump_first_frame(path)
    return read_log_step0_pe(log), f, xyz


def do_dump(a):
    ea, fa, xa = load_side(a.a, a.log_a)
    eb, fb, xb = load_side(a.b, a.log_b)
    assert fa.shape == fb.shape, f"shape mismatch {fa.shape} vs {fb.shape}"
    if xa is not None and xb is not None:
        dx = np.abs(xa - xb).max()
        if dx > 1e-6:
            print(f"WARNING: positions differ by up to {dx:.3e} A (different frame or wrapped coordinates?)")
    d = fa - fb
    n = len(fa)
    rms = np.sqrt((d ** 2).mean())
    fmax = np.abs(d).max()
    rel = np.linalg.norm(d) / np.linalg.norm(fb)
    print(f"N = {n}")
    if ea is not None and eb is not None:
        print(f"dE = {ea-eb:+.6e} eV  ({(ea-eb)/n:+.3e} eV/atom)   E_a={ea:.6f} E_b={eb:.6f}")
    print(f"RMS(dF) = {rms:.3e} eV/A   max|dF| = {fmax:.3e} eV/A   rel = {rel:.3e}")
    worst = int(np.argmax(np.abs(d).max(axis=1)))
    print(f"worst atom index (0-based, sorted by id) = {worst}: Fa={fa[worst]} Fb={fb[worst]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--model", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--log-a", default=None)
    d.add_argument("--log-b", default=None)
    a = ap.parse_args()
    do_ref(a) if a.cmd == "ref" else do_dump(a)


if __name__ == "__main__":
    main()
